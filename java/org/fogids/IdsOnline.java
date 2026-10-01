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
 * Epoch-based online execution of an IDS-PLACE instance.
 *
 * Protocol (JSON lines; this process writes requests to stdout and reads replies
 * from stdin; all simulator logging goes to stderr):
 *   -> {"type":"epoch","t":..,"epoch":k,"tasks":[ids],"completed":[[id,finish_s],..],
 *       "nodes":{name:{"outstanding":n,"resident":[models, least recently used first]}}}
 *   <- {"assignments":[[id,node,model],..],"delay_s":d}
 *   -> {"type":"end","t":..,"results":[..],"energy_j":{node:joules}}
 * An epoch message is sent only when tasks were released since the previous one.
 * Assignments take effect after the reported decision delay. A cold start (model
 * not resident) is modelled as load_s seconds of extra CPU work on the target node;
 * residency is least-recently-used within node memory.
 */
public final class IdsOnline extends SimEntity {
    static final int RELEASE = 91001, EPOCH = 91002, APPLY = 91003, FINISH = 91004;
    static final String APP = "ids-place";

    final Map<String, FogDevice> devices = new LinkedHashMap<>();
    final Map<String, JSONObject> nodeSpec = new LinkedHashMap<>();
    final Map<String, JSONObject> modelSpec = new LinkedHashMap<>();
    final Map<Integer, JSONObject> tasks = new LinkedHashMap<>();
    final Map<String, LinkedHashSet<String>> resident = new HashMap<>();
    final Map<String, Integer> outstanding = new HashMap<>();
    final Map<Integer, String[]> assigned = new HashMap<>();
    final Map<Integer, Double> dispatched = new HashMap<>(), finished = new HashMap<>();
    final Map<Integer, Boolean> coldStart = new HashMap<>();
    final List<Integer> released = new ArrayList<>();
    final List<double[]> completedSince = new ArrayList<>();
    final double epoch, finishAt, lastRelease;
    final BufferedReader in;
    final PrintStream out;
    final Application app;
    int epochIndex = 0;
    boolean ended = false;

    IdsOnline(JSONObject instance, double drain, BufferedReader in, PrintStream out) throws Exception {
        super("ids-online");
        this.in = in;
        this.out = out;
        this.epoch = num(instance, "epoch_s");
        for (Object o : (JSONArray) instance.get("models")) {
            JSONObject m = (JSONObject) o;
            modelSpec.put((String) m.get("name"), m);
        }
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
            long mips = Math.round(num(n, "mips"));
            List<Pe> pes = Collections.singletonList(new Pe(0, new PeProvisionerOverbooking(mips)));
            PowerHost host = new PowerHost(FogUtils.generateEntityId(),
                new RamProvisionerSimple((int) num(n, "memory_mb")), new BwProvisionerOverbooking(10000000),
                1000000, pes, new StreamOperatorScheduler(pes),
                new FogLinearPowerModel(num(n, "busy_w"), num(n, "idle_w")));
            FogDeviceCharacteristics ch = new FogDeviceCharacteristics("x86", "Linux", "Xen", host, 0, 0, 0, 0, 0);
            FogDevice d = new FogDevice(name, ch, new AppModuleAllocationPolicy(Collections.singletonList(host)),
                new LinkedList<Storage>(), 0.01, bw, bw, 0, 0);
            d.setUplinkLatency(root ? 0 : num(n, "uplink_latency_s"));
            d.setLevel(level.get((String) n.get("tier")));
            devices.put(name, d);
            outstanding.put(name, 0);
            LinkedHashSet<String> models = new LinkedHashSet<>();
            for (Object r : (JSONArray) n.get("resident")) models.add((String) r);
            resident.put(name, models);
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
            JSONObject nodes = new JSONObject();
            for (String name : devices.keySet()) {
                JSONObject state = new JSONObject();
                state.put("outstanding", outstanding.get(name));
                JSONArray models = new JSONArray();
                models.addAll(resident.get(name));
                state.put("resident", models);
                nodes.put(name, state);
            }
            request.put("nodes", nodes);
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
        Set<Integer> expected = new HashSet<>(released), seen = new HashSet<>();
        for (Object o : assignments) {
            JSONArray a = (JSONArray) o;
            int id = ((Number) a.get(0)).intValue();
            String node = (String) a.get(1), model = (String) a.get(2);
            if (!expected.contains(id) || !seen.add(id))
                throw new IllegalArgumentException("Unexpected or duplicate assignment for task " + id);
            if (!modelSpec.containsKey(model)) throw new IllegalArgumentException("Unknown model " + model);
            String hop = (String) tasks.get(id).get("gateway");
            while (hop != null && !hop.equals(node)) hop = (String) nodeSpec.get(hop).get("parent");
            if (hop == null) throw new IllegalArgumentException("Node " + node + " is not eligible for task " + id);
        }
        if (!seen.equals(expected)) throw new IllegalArgumentException("Every released task must be assigned");
    }

    void onApply(JSONArray assignments) {
        for (Object o : assignments) {
            JSONArray a = (JSONArray) o;
            int id = ((Number) a.get(0)).intValue();
            String node = (String) a.get(1), model = (String) a.get(2);
            JSONObject task = tasks.get(id), m = modelSpec.get(model);
            double mips = devices.get(node).getHost().getTotalMips();
            boolean cold = touch(node, model);
            double work = num(m, "fixed_mi") + num(m, "mi_per_flow") * num(task, "n_flows")
                + (cold ? num(m, "load_s") * mips : 0);
            Tuple tuple = new Tuple(APP, id, Tuple.UP, (long) Math.ceil(work), 1,
                ((Number) task.get("input_bytes")).longValue(), 1, new UtilizationModelFull(),
                new UtilizationModelFull(), new UtilizationModelFull());
            tuple.setUserId(getId());
            tuple.setTupleType("IDS_" + id);
            tuple.setSrcModuleName("ready_features");
            tuple.setDestModuleName("ids_" + node);
            tuple.setActualTupleId(id);
            assigned.put(id, new String[] {node, model});
            dispatched.put(id, CloudSim.clock());
            coldStart.put(id, cold);
            outstanding.merge(node, 1, Integer::sum);
            sendNow(devices.get((String) task.get("gateway")).getId(), FogEvents.TUPLE_ARRIVAL, tuple);
        }
    }

    /** Marks a model as most recently used on a node; returns true on a cold start. */
    boolean touch(String node, String model) {
        LinkedHashSet<String> loaded = resident.get(node);
        boolean cold = !loaded.remove(model);
        loaded.add(model);
        double memory = num(nodeSpec.get(node), "memory_mb");
        while (used(loaded) > memory && loaded.size() > 1) {
            Iterator<String> oldest = loaded.iterator();
            oldest.next();
            oldest.remove();
        }
        return cold;
    }

    double used(Collection<String> models) {
        double sum = 0;
        for (String m : models) sum += num(modelSpec.get(m), "memory_mb");
        return sum;
    }

    void onComplete(Tuple tuple, int source) {
        int id = tuple.getCloudletId();
        String node = assigned.get(id)[0];
        if (source != devices.get(node).getId())
            throw new IllegalStateException("Task " + id + " executed outside node " + node);
        finished.put(id, CloudSim.clock());
        outstanding.merge(node, -1, Integer::sum);
        completedSince.add(new double[] {id, CloudSim.clock()});
        if (finished.size() == tasks.size() && !ended) send(getId(), 0, FINISH);
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
            String[] a = assigned.get(id);
            r.put("node", a == null ? null : a[0]);
            r.put("model", a == null ? null : a[1]);
            r.put("dispatch_s", dispatched.get(id));
            r.put("finish_s", finished.get(id));
            r.put("cold_start", coldStart.getOrDefault(id, false));
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
        new IdsOnline(instance, Double.parseDouble(args[1]), in, protocol);
        CloudSim.startSimulation();
        CloudSim.stopSimulation();
    }
}
