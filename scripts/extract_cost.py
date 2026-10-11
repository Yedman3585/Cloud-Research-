"""Measure the CPU cost of CICIoT2023-style feature extraction per packet.

IDS-PLACE treats feature extraction at the gateway as a fixed delay
(``traffic.feature_s_per_flow``). In practice the gateway parses every packet and
updates window statistics before a record can be classified, on the same CPU that
runs IDS inference. This script times that work on the local machine, single thread:

* ``dpkt`` — packet parsing with dpkt and per-window statistics in Python, the way the
  CICIoT2023 feature extractor works (Ethernet → IP → TCP/UDP/ICMP headers, flags,
  protocol indicators, sizes, inter-arrival times, aggregated over windows of N
  packets into the 39 features of the 2024 release);
* ``numpy`` — the same statistics computed vectorised from pre-parsed header fields,
  a stand-in for an optimised (compiled) extractor; it excludes parsing.

Packets are synthetic (seeded): a mix of TCP, UDP and ICMP frames with sizes drawn
between 60 and 1514 bytes. Parsing cost depends on the header path and the Python
overhead per packet, not on payload content.

    python scripts/extract_cost.py               # prints and writes artifacts/calibration/extract.json
"""
import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (10, 100)  # packets per record: 10 for most labels, 100 for DDoS, DoS and Mirai


def synthetic_packets(n, seed=7):
    import dpkt
    rng = random.Random(seed)
    frames = []
    for i in range(n):
        kind = rng.random()
        size = int(60 + rng.random() ** 3 * 1454)
        src, dst = struct.pack('>I', rng.getrandbits(32)), struct.pack('>I', rng.getrandbits(32))
        if kind < 0.6:
            l4 = dpkt.tcp.TCP(sport=rng.randrange(1024, 65535), dport=rng.choice([80, 443, 22, 23, 25, 53, 6667, 8080]),
                              flags=rng.choice([0x02, 0x10, 0x18, 0x11, 0x04, 0x12]), data=b'')
            proto = dpkt.ip.IP_PROTO_TCP
        elif kind < 0.9:
            l4 = dpkt.udp.UDP(sport=rng.randrange(1024, 65535), dport=rng.choice([53, 67, 123, 5353]), data=b'')
            proto = dpkt.ip.IP_PROTO_UDP
        else:
            l4 = dpkt.icmp.ICMP(type=8, data=dpkt.icmp.ICMP.Echo(id=1, seq=i, data=b''))
            proto = dpkt.ip.IP_PROTO_ICMP
        payload = bytes(max(0, size - 14 - 20 - len(bytes(l4))))
        l4.data = payload if proto != dpkt.ip.IP_PROTO_ICMP else dpkt.icmp.ICMP.Echo(id=1, seq=i, data=payload)
        ip = dpkt.ip.IP(src=src, dst=dst, p=proto, ttl=rng.choice([64, 128, 255]), data=l4)
        ip.len = len(bytes(ip))
        eth = dpkt.ethernet.Ethernet(src=b'\x00' * 6, dst=b'\x11' * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
        frames.append((i * 1e-4 + rng.random() * 1e-5, bytes(eth)))
    return frames


PORTS = {'HTTP': {80, 8080}, 'HTTPS': {443}, 'DNS': {53}, 'Telnet': {23}, 'SMTP': {25}, 'SSH': {22}, 'IRC': {6667}}


def extract_dpkt(frames, window):
    """Parse every frame with dpkt and aggregate windows of ``window`` packets into
    CICIoT2023-style records. Returns the records (to keep the work observable)."""
    import dpkt
    records, acc, last_ts = [], [], None
    for ts, buf in frames:
        eth = dpkt.ethernet.Ethernet(buf)
        f = dict.fromkeys(('fin', 'syn', 'rst', 'psh', 'ack', 'ece', 'cwr', 'tcp', 'udp', 'icmp', 'igmp', 'arp',
                           'ipv', 'llc', 'dhcp', *PORTS), 0)
        header, ttl, proto = 14, 0, 0
        ip = eth.data
        if isinstance(ip, dpkt.ip.IP):
            f['ipv'], f['llc'] = 1, 1
            header += ip.hl * 4
            ttl, proto = ip.ttl, ip.p
            l4 = ip.data
            if isinstance(l4, dpkt.tcp.TCP):
                f['tcp'] = 1
                header += l4.off * 4
                flags = l4.flags
                f['fin'], f['syn'], f['rst'] = flags & 1, (flags >> 1) & 1, (flags >> 2) & 1
                f['psh'], f['ack'], f['ece'], f['cwr'] = (flags >> 3) & 1, (flags >> 4) & 1, (flags >> 6) & 1, (flags >> 7) & 1
                ports = {l4.sport, l4.dport}
            elif isinstance(l4, dpkt.udp.UDP):
                f['udp'] = 1
                header += 8
                ports = {l4.sport, l4.dport}
                f['dhcp'] = int(bool(ports & {67, 68}))
            else:
                f['icmp'] = int(isinstance(l4, dpkt.icmp.ICMP))
                ports = set()
            for name, numbers in PORTS.items():
                f[name] = int(bool(ports & numbers))
        iat = 0.0 if last_ts is None else ts - last_ts
        last_ts = ts
        acc.append((header, proto, ttl, len(buf), iat, f))
        if len(acc) == window:
            sizes = [a[3] for a in acc]
            mean = sum(sizes) / window
            var = sum((s - mean) ** 2 for s in sizes) / window
            span = sum(a[4] for a in acc)
            rec = {'Header_Length': sum(a[0] for a in acc) / window, 'Protocol Type': sum(a[1] for a in acc) / window,
                   'Time_To_Live': sum(a[2] for a in acc) / window, 'Rate': window / span if span > 0 else 0.0,
                   'Tot sum': sum(sizes), 'Min': min(sizes), 'Max': max(sizes), 'AVG': mean, 'Std': math.sqrt(var),
                   'Tot size': mean, 'IAT': span / window, 'Number': window, 'Variance': var}
            for key in acc[0][5]:
                rec[key] = sum(a[5][key] for a in acc) / window
            rec['ack_count'] = sum(a[5]['ack'] for a in acc)
            rec['syn_count'] = sum(a[5]['syn'] for a in acc)
            rec['fin_count'] = sum(a[5]['fin'] for a in acc)
            rec['rst_count'] = sum(a[5]['rst'] for a in acc)
            records.append(rec)
            acc = []
    return records


def extract_numpy(fields, window):
    """Vectorised window statistics from pre-parsed per-packet fields (no parsing)."""
    import numpy as np
    n = (len(fields) // window) * window
    x = fields[:n].reshape(-1, window, fields.shape[1])
    sizes = x[:, :, 0]
    out = np.concatenate([x.mean(axis=1), sizes.min(axis=1, keepdims=True), sizes.max(axis=1, keepdims=True),
                          sizes.std(axis=1, keepdims=True), sizes.sum(axis=1, keepdims=True)], axis=1)
    return out


def timed(fn, *args, repeats=5):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--packets', type=int, default=20000)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'calibration' / 'extract.json')
    a = p.parse_args()
    frames = synthetic_packets(a.packets)
    result = {'machine': platform.machine(), 'platform': platform.platform(), 'python': platform.python_version(),
              'packets': a.packets, 'per_packet_us': {}}
    for window in WINDOWS:
        t = timed(extract_dpkt, frames, window)
        result['per_packet_us'][f'dpkt_w{window}'] = 1e6 * t / a.packets
    try:
        import numpy as np
        fields = np.random.default_rng(0).random((a.packets, 30)).astype(np.float32)
        for window in WINDOWS:
            t = timed(extract_numpy, fields, window)
            result['per_packet_us'][f'numpy_w{window}'] = 1e6 * t / a.packets
    except ImportError:
        pass
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
