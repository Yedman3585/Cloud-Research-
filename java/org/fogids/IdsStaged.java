package org.fogids;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.*;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.sdn.overbooking.*;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.*;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * Epoch-based online execution of staged tasks (smokescreen scenario, Python module
 * fogids.smokescreen): every task first runs a CPU stage on its gateway (feature
 * extraction, or the header pass when raw packets are shipped), then, optionally,
 * a second CPU stage on the gateway or an ancestor, reached over the uplinks with the
 * given number of bytes (features or sampled raw packets). The controller computes
 * the work of each stage; this driver only executes them, with iFogSim's
 * processor-sharing CPUs and FIFO uplinks.
 *
 * Protocol (JSON lines; requests on stdout, replies on stdin, logging on stderr):
 *   -> {"type":"epoch","t":..,"epoch":k,"tasks":[ids],"completed":[[id,finish_s],..]}
 *   <- {"assignments":[[id, gateway_mi, node, bytes, node_mi],..],"delay_s":d}
 *      node null: the task is dropped; node_mi 0: no second stage
 *   -> {"type":"end","t":..,"results":[{"id","node","finish_s"},..]}
 * Work is passed in MI and executed at SCALE times the resolution (Tuple lengths are
 * integers): MIPS and MI are both multiplied by SCALE, which leaves times unchanged.
 */
public final class IdsStaged extends SimEntity {
    static final long SCALE = 1000;
    // CPU update interval: completions are noticed at this granularity (IdsOnline uses 0.01 s;
    // deadlines of 0.3 s with tightly packed gateways need a finer one).
    static final double SCHEDULING_INTERVAL_S = 0.001;
    static final int RELEASE = 91001, EPOCH = 91002, APPLY = 91003, FINISH = 91004;
    static final String APP = "ids-staged";

    final Map<String, FogDevice> devices = new LinkedHashMap<>();
    final Map<String, JSONObject> nodeSpec = new LinkedHashMap<>();
    final Map<Integer, JSONObject> tasks = new LinkedHashMap<>();
    final Map<Integer, Object[]> plan = new HashMap<>();  // id -> {node, bytes, node_mi}
    final Map<Integer, Double> finished = new HashMap<>();
    int decided = 0;
    final List<Integer> released = new ArrayList<>();
    final List<double[]> completedSince = new ArrayList<>();
    final double epoch, finishAt, lastRelease;
    final BufferedReader in;
    final PrintStream out;
    final Application app;
    int epochIndex = 0;
    boolean ended = false;

    IdsStaged(JSONObject instance, double drain, BufferedReader in, PrintStream out) throws Exception {
        super("ids-staged");
        this.in = in;
        this.out = out;
        this.epoch = num(instance, "epoch_s");
        double maxRelease = 0;
        for (Object o : (JSONArray) instance.get("tasks")) {
            JSONObject t = (JSONObject) o;
            tasks.put(((Number) t.get("id")).intValue(), t);
            maxRelease = Math.max(maxRelease, num(t, "release_s"));
        }
        lastRelease = maxRelease;
        finishAt = Math.max(num(instance, "horizon_s"), maxRelease) + drain;
        buildTopology((JSONArray) instance.get("nodes"));
        app = Application.createApplication(APP, getId());
        for (FogDevice d : devices.values()) {
            d.setControllerId(getId());
            double mips = d.getHost().getTotalMips();
            app.getModules().add(new AppModule(FogUtils.generateEntityId(), "ids_" + d.getName(), APP, getId(),
                mips, 128, 1000, 1000, "Xen", new IdsBatch.InferenceScheduler(mips), new HashMap<>()));
        }
    }

    static double num(JSONObject o, String key) {
        return ((Number) o.get(key)).doubleValue();
    }

    void buildTopology(JSONArray nodes) throws Exception {
        Map<String, Integer> level = Map.of("cloud", 0, "fog", 1, "edge", 2);
        for (Object o : nodes) {
            JSONObject n = (JSONObject) o;
            String name = (String) n.get("name");
            nodeSpec.put(name, n);
            boolean root = n.get("parent") == null;
            double bw = root ? 1e12 : num(n, "uplink_bytes_s");
            long mips = Math.round(num(n, "mips") * SCALE);
            List<Pe> pes = Collections.singletonList(new Pe(0, new PeProvisionerOverbooking(mips)));
            PowerHost host = new PowerHost(FogUtils.generateEntityId(),
                new RamProvisionerSimple((int) num(n, "memory_mb")), new BwProvisionerOverbooking(10000000),
                1000000, pes, new StreamOperatorScheduler(pes),
                new FogLinearPowerModel(num(n, "busy_w"), num(n, "idle_w")));
            FogDeviceCharacteristics ch = new FogDeviceCharacteristics("x86", "Linux", "Xen", host, 0, 0, 0, 0, 0);
            FogDevice d = new FogDevice(name, ch, new AppModuleAllocationPolicy(Collections.singletonList(host)),
                new LinkedList<Storage>(), SCHEDULING_INTERVAL_S, bw, bw, 0, 0);
            d.setUplinkLatency(root ? 0 : num(n, "uplink_latency_s"));
            d.setLevel(level.get((String) n.get("tier")));
            devices.put(name, d);
        }
        for (FogDevice d : devices.values()) {
            Object parent = nodeSpec.get(d.getName()).get("parent");
            if (parent == null) { d.setParentId(-1); continue; }
            FogDevice p = devices.get((String) parent);
            d.setParentId(p.getId());
            p.getChildrenIds().add(d.getId());
            p.getChildToLatencyMap().put(d.getId(), d.getUplinkLatency());
        }
    }

    @Override public void startEntity() {
        for (FogDevice d : devices.values()) {
            sendNow(d.getId(), FogEvents.APP_SUBMIT, app);
            sendNow(d.getId(), FogEvents.LAUNCH_MODULE, app.getModuleByName("ids_" + d.getName()));
        }
        // Releases are scheduled before epochs so that a task released exactly on an
        // epoch boundary is included in that epoch (same-time events run in order).
        for (Map.Entry<Integer, JSONObject> e : tasks.entrySet())
            send(getId(), num(e.getValue(), "release_s"), RELEASE, e.getKey());
        scheduleEpoch(1);
        send(getId(), finishAt, FINISH);
    }

    void scheduleEpoch(int k) {
        send(getId(), Math.max(0, k * epoch - CloudSim.clock()), EPOCH, k);
    }

    @Override public void processEvent(SimEvent ev) {
        try {
            switch (ev.getTag()) {
                case RELEASE: released.add((Integer) ev.getData()); break;
                case EPOCH: onEpoch((Integer) ev.getData()); break;
                case APPLY: onApply((JSONArray) ev.getData()); break;
                case CloudSimTags.CLOUDLET_RETURN: onComplete((Tuple) ev.getData(), ev.getSource()); break;
                case FINISH: onFinish(); break;
                default: break;
            }
        } catch (IOException | org.json.simple.parser.ParseException error) {
            throw new RuntimeException(error);
        }
    }

    @SuppressWarnings("unchecked")
    void onEpoch(int k) throws IOException, org.json.simple.parser.ParseException {
        if (ended) return;
        if (!released.isEmpty()) {
            JSONObject request = new JSONObject();
            request.put("type", "epoch");
            request.put("t", CloudSim.clock());
            request.put("epoch", k);
            JSONArray ids = new JSONArray();
            ids.addAll(released);
            request.put("tasks", ids);
            JSONArray done = new JSONArray();
            for (double[] c : completedSince) {
                JSONArray pair = new JSONArray();
                pair.add((int) c[0]);
                pair.add(c[1]);
                done.add(pair);
            }
            request.put("completed", done);
            out.println(request.toJSONString());
            out.flush();
            String line = in.readLine();
            if (line == null) throw new IOException("Controller closed the protocol stream");
            JSONObject reply = (JSONObject) new JSONParser().parse(line);
            JSONArray assignments = (JSONArray) reply.get("assignments");
            validate(assignments);
            double delay = reply.get("delay_s") == null ? 0 : num(reply, "delay_s");
            if (delay < 0) throw new IllegalArgumentException("Negative decision delay");
            send(getId(), delay, APPLY, assignments);
            released.clear();
            completedSince.clear();
        }
        if (k * epoch <= lastRelease + epoch) scheduleEpoch(k + 1);
    }

    void validate(JSONArray assignments) {
        Set<Integer> expectedIds = new HashSet<>(released), seen = new HashSet<>();
        for (Object o : assignments) {
            JSONArray a = (JSONArray) o;
            int id = ((Number) a.get(0)).intValue();
            if (!expectedIds.contains(id) || !seen.add(id))
                throw new IllegalArgumentException("Unexpected or duplicate assignment for task " + id);
            String node = (String) a.get(2);
            if (node == null) continue;
            String hop = (String) tasks.get(id).get("gateway");
            while (hop != null && !hop.equals(node)) hop = (String) nodeSpec.get(hop).get("parent");
            if (hop == null) throw new IllegalArgumentException("Node " + node + " is not eligible for task " + id);
            if (((Number) a.get(1)).doubleValue() < 0 || ((Number) a.get(3)).doubleValue() < 0
                || ((Number) a.get(4)).doubleValue() < 0)
                throw new IllegalArgumentException("Negative work or size for task " + id);
        }
        if (!seen.equals(expectedIds)) throw new IllegalArgumentException("Every released task must be assigned");
    }

    Tuple tuple(int id, int stage, String node, double mi, long bytes) {
        long length = Math.max(1, (long) Math.ceil(mi * SCALE));
        Tuple t = new Tuple(APP, 2 * id + stage, Tuple.UP, length, 1, bytes, 1, new UtilizationModelFull(),
            new UtilizationModelFull(), new UtilizationModelFull());
        t.setUserId(getId());
        t.setTupleType("S" + stage + "_" + id);
        t.setSrcModuleName("ready_features");
        t.setDestModuleName("ids_" + node);
        t.setActualTupleId(2 * id + stage);
        return t;
    }

    void onApply(JSONArray assignments) {
        for (Object o : assignments) {
            JSONArray a = (JSONArray) o;
            int id = ((Number) a.get(0)).intValue();
            String node = (String) a.get(2);
            if (node == null) continue;  // dropped
            String gateway = (String) tasks.get(id).get("gateway");
            plan.put(id, new Object[] {node, ((Number) a.get(3)).longValue(), ((Number) a.get(4)).doubleValue()});
            sendNow(devices.get(gateway).getId(), FogEvents.TUPLE_ARRIVAL,
                tuple(id, 0, gateway, ((Number) a.get(1)).doubleValue(), 0));
        }
        decided += assignments.size();
        checkDone();
    }

    void checkDone() {
        if (!ended && decided == tasks.size() && finished.size() == plan.size()) send(getId(), 0, FINISH);
    }

    void onComplete(Tuple tuple, int source) {
        int id = tuple.getCloudletId() / 2, stage = tuple.getCloudletId() % 2;
        Object[] p = plan.get(id);
        String gateway = (String) tasks.get(id).get("gateway");
        String node = stage == 0 ? gateway : (String) p[0];
        if (source != devices.get(node).getId())
            throw new IllegalStateException("Task " + id + " stage " + stage + " executed outside " + node);
        double nodeMi = (Double) p[2];
        if (stage == 0 && nodeMi > 0) {
            // The data is at the gateway: the second stage starts there and travels up.
            sendNow(devices.get(gateway).getId(), FogEvents.TUPLE_ARRIVAL,
                tuple(id, 1, (String) p[0], nodeMi, (Long) p[1]));
            return;
        }
        finished.put(id, CloudSim.clock());
        completedSince.add(new double[] {id, CloudSim.clock()});
        checkDone();
    }

    @SuppressWarnings("unchecked")
    void onFinish() {
        if (ended) return;
        ended = true;
        JSONObject end = new JSONObject();
        end.put("type", "end");
        end.put("t", CloudSim.clock());
        JSONArray results = new JSONArray();
        for (int id : tasks.keySet()) {
            JSONObject r = new JSONObject();
            r.put("id", id);
            Object[] p = plan.get(id);
            r.put("node", p == null ? null : p[0]);
            r.put("finish_s", finished.get(id));
            results.add(r);
        }
        end.put("results", results);
        JSONObject energy = new JSONObject();
        for (FogDevice d : devices.values()) energy.put(d.getName(), d.getEnergyConsumption());
        end.put("energy_j", energy);
        out.println(end.toJSONString());
        out.flush();
        CloudSim.stopSimulation();
    }

    @Override public void shutdownEntity() { }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("instance.json drain_seconds");
        // Keep stdout for the protocol; iFogSim and CloudSim log through System.out.
        PrintStream protocol = new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        System.setOut(System.err);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        JSONObject instance = (JSONObject) new JSONParser().parse(
            new String(Files.readAllBytes(Paths.get(args[0])), StandardCharsets.UTF_8));
        CloudSim.init(1, Calendar.getInstance(), false, 0.000001);
        Log.disable();
        Logger.ENABLED = false;
        new IdsStaged(instance, Double.parseDouble(args[1]), in, protocol);
        CloudSim.startSimulation();
        CloudSim.stopSimulation();
    }
}
