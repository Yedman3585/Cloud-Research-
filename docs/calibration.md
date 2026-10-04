# Calibration on CICIoT2023 (milestone M4)

The IDS-PLACE configs ship placeholder model costs and recalls. This step replaces
them with values measured on real data and real hardware. Code:
[`scripts/calibrate.py`](../scripts/calibrate.py). Output: a hardware profile
`configs/calibration-<profile>.json` and generator configs
`configs/ids-place-{small,medium,large}-calibrated.json` (`"calibrated": true`).
Only model parameters (`fixed_mi`, `mi_per_flow`, `memory_mb`, `load_s`, `recall`) and
`traffic.bytes_per_flow` change; topology, traffic and deadlines stay as in the base configs.

## Data

CICIoT2023 (Neto et al., *Sensors* 2023, 23(13):5941), feature CSV files. Two releases
are supported: the original one (shuffled `part-*.csv` files with a `label` column and 46
features) and the 2024 one used here (one folder per label, `*.pcap.csv` files without a
label column, 39 features; the label is the folder name, `Benign_Final` is benign).
Each row aggregates a window of packets (`Number` = 10 in the 2024 release) and plays the
role of one *flow* in IDS-PLACE.
The 34 original labels are grouped into the seven attack classes of IDS-PLACE plus benign,
following the dataset's own 8-class grouping:

| IDS-PLACE class | CICIoT2023 labels |
|---|---|
| `benign` | BenignTraffic |
| `ddos` | DDoS-* (12 labels) |
| `dos` | DoS-* (4) |
| `mirai` | Mirai-* (3) |
| `recon` | Recon-* (4), VulnerabilityScan |
| `spoofing` | DNS_Spoofing, MITM-ArpSpoofing |
| `web` | SqlInjection, XSS, CommandInjection, BrowserHijacking, Backdoor_Malware, Uploading_Attack |
| `bruteforce` | DictionaryBruteForce |

The data set is heavily imbalanced (DDoS and DoS dominate; web and brute-force labels have
a few thousand rows in total). `prepare` keeps at most `--cap` rows (default 20 000) per
**original** label, so rare labels are kept in full and frequent ones are capped. In the
2024 layout every capture file of a label contributes a seeded random subset of
ceil(cap / files) rows. `--drop` removes feature columns; `Time_To_Live` and `IAT` are
candidates, since they can identify the capture rather than the attack. The raw files and the sample
live in `data/`, which is not tracked by Git.

## Models

Both models classify a flow into the 8 classes; inputs pass through a signed `log1p`
and standardization with training-split statistics, which is part of the model so that
timing includes it.

- `light`: MLP F → 64 → 32 → 8, with F input features (39 in the 2024 release).
- `full`: two 1D convolutions over the feature vector (32 and 64 channels, kernel 3),
  max-pooling, an LSTM (64 units) over the pooled positions, and a 64 → 8 head.

Training: stratified 70/15/15 split, AdamW (lr 1e-3, weight decay 1e-4), batch 1024,
class-weighted cross-entropy, 15 (`light`) and 10 (`full`) epochs, the epoch with the
best validation balanced accuracy is kept. Seed 2023. Training may use MPS/CUDA;
results on GPU backends are not bit-reproducible.

**Recall used by the simulator** is *detection* recall on the test split: the fraction
of an attack class's flows predicted as any attack class. Confusing two attack classes
still raises an alert, which matches the missed-detection metric of IDS-PLACE. The
8-class recall, balanced accuracy and the benign false-positive rate are reported too.

## Inference cost

Each model is timed in a fresh process with one CPU thread (`torch.set_num_threads(1)`),
on test rows, for windows of 1–1000 flows (batch = window). For each size: 10 warm-up
calls, then at least 50 calls and 0.3 s; the median is used. A least-squares line
`time = fixed + per_flow × flows` gives the two parameters. Cold start (`load_s`) is the
time to load the checkpoint, build the model and run the first inference, with the
PyTorch runtime already loaded. `memory_mb` is the larger of the weight size and the
increase of the process peak RSS from before loading to after the largest window,
rounded up to whole MB; the shared runtime itself is not counted.

Conversion to simulator units: `fixed_mi = fixed × ref_mips`, `mi_per_flow = per_flow ×
ref_mips`, where `ref_mips` is the rating assigned to one core of the measuring machine.
Default: the fog-node MIPS of the base configs (8000), i.e. a fog node is modelled as one
such core, an edge gateway (2000 MIPS) as a quarter of it and the cloud as eight. This is
a modelling assumption, recorded in the profile; a second profile measured on other
hardware tests its sensitivity.

`bytes_per_flow` becomes the size of one float32 feature vector (4 × F bytes, 156 for 39 features).

## Results

<!-- results:begin -->
Not yet measured. Run `python scripts/calibrate.py all --profile mac`.
<!-- results:end -->

## Reproduce

```bash
pip install -e '.[calibration]'
# CICIoT2023 CSV folder unpacked as data/CICIoT2023/ (one subfolder per label)
python scripts/calibrate.py all --profile mac
fogids simulate --config configs/ids-place-small-calibrated.json
```

Individual steps: `prepare`, `train [--model light|full]`, `time`, `configs`.
Intermediate files go to `artifacts/calibration/`.

## Limitations

- Per-flow classification of precomputed features; feature extraction cost at the gateway
  is not measured here (`traffic.feature_s_per_flow` is unchanged).
- Timings come from a general-purpose CPU, not from an edge device; the edge/fog/cloud
  speed ratios remain configuration parameters.
- Rows of CICIoT2023 are split at random, not by capture; recall may be optimistic.
