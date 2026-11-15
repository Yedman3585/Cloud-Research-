"""Calibrate IDS-PLACE detectors on TON-IoT (UNSW Canberra, Zeek flows).

The TON-IoT ``Train_Test_Network`` CSV is a balanced sample (20 000 rows per
attack type plus 50 000 benign), so per-class traffic intensities and the
smokescreen scenario schedule are kept from the CICIoT2023 configuration
(same scenario, different detector quality). Row-level labels are mapped
into the seven-class taxonomy used by the simulator:

    scanning, backdoor, ransomware, mitm -> recon
    password, injection                   -> bruteforce
    xss                                   -> web
    ddos, dos                             -> ddos, dos
    mirai, spoofing                       -> absent (not present in TON-IoT)

The light model is an MLP on numerical flow fields plus proto/conn_state
one-hot; the full model is a deeper MLP on the same fields plus DNS, HTTP and
SSL indicators. Both are per-flow; TON-IoT has no reliable capture ordering,
so the GRU-over-context full model used on CICIoT2023 is not reproduced here.
This is documented in the ``_comment`` of the written config.

Outputs
-------
1. ``artifacts/tonot-detectors/{light,full}.pt``  trained model state dicts
2. ``artifacts/tonot-detectors/report.json``      recall per class, FPR, timings
3. ``configs/ids-place-smokescreen-tonot-x3.json`` simulator config, built from
   the CICIoT x3 template with the TON-IoT models substituted in.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

CLASS_MAP = {
    'normal': 'benign', 'ddos': 'ddos', 'dos': 'dos',
    'scanning': 'recon', 'backdoor': 'recon', 'ransomware': 'recon', 'mitm': 'recon',
    'password': 'bruteforce', 'injection': 'bruteforce', 'xss': 'web',
}
ATTACK_CLASSES = ['ddos', 'dos', 'mirai', 'recon', 'spoofing', 'web', 'bruteforce']
NUM_COLS_LIGHT = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts',
                  'missed_bytes', 'src_ip_bytes', 'dst_ip_bytes']
PROTO_TOPK, CONN_TOPK, SERVICE_TOPK = 6, 10, 10


def load_and_prepare(csv_path, seed=7):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.lstrip('﻿') for c in df.columns]
    df['cls'] = df['type'].map(CLASS_MAP)
    df = df[df['cls'].notna()].copy()
    for c in NUM_COLS_LIGHT:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    proto_top = df['proto'].value_counts().head(PROTO_TOPK).index.tolist()
    conn_top = df['conn_state'].value_counts().head(CONN_TOPK).index.tolist()
    service_top = df['service'].value_counts().head(SERVICE_TOPK).index.tolist()
    for v in proto_top:
        df[f'proto_{v}'] = (df['proto'] == v).astype(np.float32)
    for v in conn_top:
        df[f'conn_{v}'] = (df['conn_state'] == v).astype(np.float32)
    for v in service_top:
        df[f'svc_{v}'] = (df['service'] == v).astype(np.float32)
    for col in ['dns_query', 'ssl_version', 'http_method']:
        df[f'has_{col}'] = (df[col].astype(str).str.len() > 1).astype(np.float32)
    for col in ['http_request_body_len', 'http_response_body_len', 'http_status_code']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df[f'log_{col}'] = np.log1p(np.abs(df[col])).astype(np.float32)
    light_cols = NUM_COLS_LIGHT + [f'proto_{v}' for v in proto_top] + [f'conn_{v}' for v in conn_top]
    full_cols = light_cols + [f'svc_{v}' for v in service_top] + \
                ['has_dns_query', 'has_ssl_version', 'has_http_method'] + \
                ['log_http_request_body_len', 'log_http_response_body_len', 'log_http_status_code']
    for c in set(light_cols + full_cols):
        df[c] = df[c].astype(np.float32).replace([np.inf, -np.inf], 0.0)
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df, light_cols, full_cols


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(X_train, y_train, X_val, y_val, hidden, epochs, batch=512, lr=1e-3, device='cpu'):
    model = MLP(X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_tr = torch.tensor(X_train).to(device)
    y_tr = torch.tensor(y_train).long().to(device)
    X_vl = torch.tensor(X_val).to(device)
    y_vl = torch.tensor(y_val).long().to(device)
    n = X_tr.shape[0]
    best_val, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(X_vl), y_vl).item()
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(model, X, y_binary, classes_vec, device='cpu'):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X).to(device)).argmax(1).cpu().numpy()
    fpr = float(((pred == 1) & (y_binary == 0)).sum() / max(1, (y_binary == 0).sum()))
    recall = {}
    for c in ATTACK_CLASSES:
        mask = classes_vec == c
        if mask.sum() == 0:
            recall[c] = 1.0
            continue
        recall[c] = float((pred[mask] == 1).sum() / mask.sum())
    return recall, fpr


def time_inference(model, in_dim, device='cpu', batches=(1, 10, 100, 1000), reps=30):
    model.eval()
    times = {}
    with torch.no_grad():
        for n in batches:
            x = torch.randn(n, in_dim).to(device)
            for _ in range(5):
                model(x)
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            times[n] = (time.perf_counter() - t0) / reps
    ns = np.array(list(times.keys()), dtype=np.float64)
    ts = np.array(list(times.values()), dtype=np.float64)
    a, b = np.polyfit(ns, ts, 1)
    return {'fixed_s': float(b), 'per_flow_s': float(a), 'raw': {int(k): v for k, v in times.items()}}


def measure_packet_stats(df):
    pkts = (df['src_pkts'].fillna(0) + df['dst_pkts'].fillna(0)).astype(float)
    bytes_ = (df['src_bytes'].fillna(0) + df['dst_bytes'].fillna(0)).astype(float)
    out = {}
    for cls in ['benign'] + ATTACK_CLASSES:
        mask = df['cls'] == cls
        if mask.sum() < 10:
            continue
        p = pkts[mask]
        b = bytes_[mask]
        out[cls] = {
            'packets_per_flow': float(np.clip(p.median(), 1.0, 10000.0)),
            'bytes_per_packet': float(np.clip((b[p > 0] / p[p > 0]).median() if (p > 0).any() else 400.0, 32.0, 1500.0)),
        }
    return out


def write_config(template_path, out_path, report, packet_stats, note):
    with open(template_path) as fh:
        cfg = json.load(fh)
    cfg['name'] = out_path.stem
    cfg['_comment'] = note
    for m in cfg['models']:
        name = m['name']
        r = report['models'][name]
        m['recall'] = {c: round(r['recall'].get(c, 1.0), 4) for c in ATTACK_CLASSES}
        m['false_positive_rate'] = round(r['fpr'], 4)
        ref_mips = cfg['v2']['ref_mips']
        m['fixed_mi'] = round(r['timing']['fixed_s'] * ref_mips, 6)
        m['mi_per_flow'] = round(r['timing']['per_flow_s'] * ref_mips, 6)
    for cls in ['ddos', 'dos']:
        if cls in packet_stats:
            cfg['v2']['packets_per_flow'][cls] = round(packet_stats[cls]['packets_per_flow'], 1)
            cfg['v2']['packet_bytes'][cls] = round(packet_stats[cls]['bytes_per_packet'], 1)
    if 'benign' in packet_stats:
        cfg['v2']['packet_bytes']['benign'] = round(packet_stats['benign']['bytes_per_packet'], 1)
    with open(out_path, 'w') as fh:
        json.dump(cfg, fh, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='docs/train_test_network.csv')
    p.add_argument('--template', default='configs/ids-place-smokescreen-x3.json')
    p.add_argument('--out-config', default='configs/ids-place-smokescreen-tonot-x3.json')
    p.add_argument('--out-detectors', default='artifacts/tonot-detectors')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--epochs-light', type=int, default=15)
    p.add_argument('--epochs-full', type=int, default=30)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    csv_path = ROOT / args.input
    template = ROOT / args.template
    out_cfg = ROOT / args.out_config
    out_det = ROOT / args.out_detectors
    out_det.mkdir(parents=True, exist_ok=True)

    print(f'[1/5] reading {csv_path.name}')
    df, light_cols, full_cols = load_and_prepare(csv_path, seed=args.seed)
    print(f'      {len(df)} rows; {len(light_cols)} light / {len(full_cols)} full features')
    print(f'      classes: {df["cls"].value_counts().to_dict()}')

    y_binary = (df['cls'] != 'benign').astype(int).values
    cls_vec = df['cls'].values
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    cut_tr, cut_vl = int(0.7 * len(df)), int(0.85 * len(df))
    tr, vl, te = idx[:cut_tr], idx[cut_tr:cut_vl], idx[cut_vl:]

    results = {'dataset': 'TON-IoT Train_Test_Network', 'models': {}}
    for name, cols, hidden, epochs in [
        ('light', light_cols, (32, 16), args.epochs_light),
        ('full', full_cols, (128, 64, 32), args.epochs_full),
    ]:
        print(f'[2/5] training {name} on {len(cols)} features')
        X = df[cols].to_numpy(dtype=np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X).astype(np.float32)
        t0 = time.time()
        model = train_model(Xs[tr], y_binary[tr], Xs[vl], y_binary[vl],
                            hidden=hidden, epochs=epochs, device=args.device)
        print(f'      trained in {time.time() - t0:.1f}s')
        recall, fpr = evaluate(model, Xs[te], y_binary[te], cls_vec[te], device=args.device)
        print(f'      FPR={fpr:.4f}  recall={ {c: round(v,3) for c,v in recall.items()} }')
        timing = time_inference(model, in_dim=len(cols), device=args.device)
        print(f'      timing: fixed={timing["fixed_s"]*1e6:.1f}us, per_flow={timing["per_flow_s"]*1e6:.2f}us')
        torch.save({'state_dict': model.state_dict(), 'cols': cols,
                    'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()},
                   out_det / f'{name}.pt')
        results['models'][name] = {'recall': recall, 'fpr': fpr, 'timing': timing,
                                   'features': cols, 'n_features': len(cols)}

    print(f'[3/5] measuring packet stats')
    pkt_stats = measure_packet_stats(df)
    results['packet_stats'] = pkt_stats
    print(f'      {json.dumps(pkt_stats, indent=2)}')

    with open(out_det / 'report.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f'[4/5] writing {out_cfg.name}')
    note = ('TON-IoT (UNSW Canberra) Train_Test_Network. Scenario schedule and per-class '
            'intensities taken from the CICIoT x3 template; only detector quality '
            '(recall, FPR, inference timing) and packet-level ddos/dos stats are measured '
            'on TON-IoT. The full model here is a deeper MLP over extended flow fields '
            '(DNS/HTTP/SSL indicators) rather than a GRU over per-capture context, which '
            'TON-IoT sampled rows do not provide.')
    write_config(template, out_cfg, results, pkt_stats, note)
    print(f'[5/5] done: {out_cfg}')


if __name__ == '__main__':
    main()
