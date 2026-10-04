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
a few thousand rows in total). `prepare` keeps about `--cap` rows (default 20 000) per
**original** label, so rare labels are kept in full and frequent ones are capped. Rows of a
2024 file are in capture order (consecutive rows are markedly closer in feature space than
random pairs), so sampling keeps that order: every capture file of a label contributes
seeded, randomly chosen, non-overlapping **blocks of 200 consecutive rows**, about
ceil(cap / files) rows per file. `--drop` removes feature columns; `Time_To_Live` and `IAT` are
candidates, since they can identify the capture rather than the attack. The raw files and the sample
live in `data/`, which is not tracked by Git.

## Models

Both models classify a flow into the 8 classes; inputs pass through a signed `log1p`
and standardization with training-split statistics, which is part of the model so that
timing includes it. F is the number of features (39 in the 2024 release).

- `light`: MLP F → 64 → 32 → 8 on the flow record alone.
- `full`: a context model. Its input is the flow record and the K − 1 = 15 records before
  it in the same capture. Each record is embedded (F → 64), a GRU (64 units) runs over the
  16 embeddings, and a head (128 → 64 → 8) reads the final GRU state together with the
  embedding of the flow itself.

Version 1 used a per-record CNN-LSTM as `full`; it was no more accurate than `light`
(see Findings), which is why `full` now uses context.

Training: blocks (not rows) are split 70/15/15 per class, so neighbouring records never
end up on both sides of the split. Both models are trained and scored on the same
targets: records with 15 predecessors in their block. AdamW (lr 1e-3, weight decay 1e-4),
batch 1024, class-weighted cross-entropy, 15 (`light`) and 10 (`full`) epochs, the epoch
with the best validation balanced accuracy is kept. Seed 2023. Training may use MPS/CUDA;
results on GPU backends are not bit-reproducible.

**Mixed traffic.** A context taken from one capture is cleaner than what a gateway sees,
where other hosts' traffic is interleaved. During training each context record (never
the flow itself) is replaced, with a probability drawn from U(0, 0.5) per batch, by a
random training record of any class. The `full` model is scored twice: with clean context
and with half of the context replaced (`--eval-rho 0.5`). **The second, more pessimistic
number goes into the configs.**

**Recall used by the simulator** is *detection* recall on the test split: the fraction
of an attack class's flows predicted as any attack class. Confusing two attack classes
still raises an alert, which matches the missed-detection metric of IDS-PLACE. The
8-class recall, balanced accuracy and the benign false-positive rate are reported too.

## Inference cost

Each model is timed in a fresh process with one CPU thread (`torch.set_num_threads(1)`),
on test rows, for windows of 1–1000 flows (batch = window; for `full` every flow carries
its 16-record context, so the cost per flow includes the context). For each size: 10 warm-up
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
Profile `mac`: Apple M2, macOS-13.0-arm64-arm-64bit, Python 3.12.14, PyTorch 2.11.0, one CPU thread, reference 8000 MIPS.
Sample: 559800 rows from 309 CSV files, at most 20000 per original label.

| Model | Records | Params | fixed, µs | per flow, µs | R² | fixed_mi | mi_per_flow | load_s | memory_mb |
|---|---|---|---|---|---|---|---|---|---|
| light | 1 | 4904 | 25.7 | 0.149 | 0.9927 | 0.205694 | 0.001194 | 0.0057 | 5 |
| full | 16 | 36296 | 161.3 | 12.341 | 0.9987 | 1.290782 | 0.09873 | 0.0025 | 35 |

Test blocks: detection recall per class, balanced 8-class accuracy, benign false-positive rate:

| Model | ddos | dos | mirai | recon | spoofing | web | bruteforce | balanced acc. | benign FPR |
|---|---|---|---|---|---|---|---|---|---|
| light | 1.0000 | 1.0000 | 1.0000 | 0.8549 | 0.8503 | 0.8708 | 0.8565 | 0.7158 | 0.2559 |
| full, clean context | 1.0000 | 1.0000 | 1.0000 | 0.8713 | 0.9050 | 0.9679 | 0.9502 | 0.8655 | 0.1056 |
| full, 50% of context replaced (used) | 1.0000 | 1.0000 | 1.0000 | 0.8436 | 0.8627 | 0.9169 | 0.9087 | 0.8044 | 0.1268 |

Rows per class in the sample: benign 20000, ddos 240000, dos 80000, mirai 60000, recon 82200, spoofing 40000, web 24600, bruteforce 13000.
<!-- results:end -->

## Findings of version 1 (per-record `full`, profile `mac`, 5 October 2026)

1. **Per-flow features cap detection quality, not model capacity.** The CNN-LSTM (`full`,
   9× the parameters) is no better than the MLP (`light`): balanced accuracy 0.727 vs
   0.728, detection recall within 0.7 points on every class. A gradient-boosting
   classifier on the same sample and split reaches 0.767 (0.732 without `Time_To_Live`
   and `IAT`), so both networks sit close to what single-row features allow. The hard
   classes (recon, spoofing, web, bruteforce) overlap with benign traffic row by row; a
   binary booster trades a 5 % benign false-positive rate for 0.59–0.82 recall on them.
2. **Cost differs by three orders of magnitude.** One window of 1000 flows takes 0.17 ms
   with `light` and 130 ms with `full` on one M2 core. `full` is not linear for small
   windows (≈54 µs/flow up to 10 flows, ≈125 µs/flow from 20 flows), so the fit is
   driven by large windows and its intercept is clipped to zero.
3. **Consequence for IDS-PLACE.** With these two models a scheduler should always pick
   `light`: it is as accurate and practically free. The model-choice dimension of the
   problem only matters if the heavier model gains detection quality, e.g. by using
   context across consecutive flow records instead of single rows. This led to the
   context model of version 2. A feasibility run on one capture file per label (cloud
   CPU, about 6 000 targets per label, block split) gave balanced accuracy 0.74 → 0.89
   and benign false positives 26 % → 6 % with 16 records of clean context, and 0.84 /
   16 % with half of the context replaced; 4 records gained less (0.80), 32 no more (0.90). The medium instance
   loads edge and fog to 0.25 with `full` (1.04 with the placeholders) and to 0.0004
   with `light`.
4. The benign false-positive rate (≈28 % for both networks at the class-weighted
   operating point) is not yet part of the simulator's metrics.

## Reproduce

```bash
pip install -e '.[calibration]'
# CICIoT2023 CSV folder unpacked as data/CICIoT2023/ (one subfolder per label)
python scripts/calibrate.py all --profile mac
fogids simulate --config configs/ids-place-small-calibrated.json
```

Individual steps: `prepare`, `train [--model light|full]`, `time`, `configs`.
The sample goes to `data/ciciot2023-sample.npz`, intermediate files to `artifacts/calibration/`.

## Limitations

- Per-flow classification of precomputed features; feature extraction cost at the gateway
  is not measured here (`traffic.feature_s_per_flow` is unchanged).
- Timings come from a general-purpose CPU, not from an edge device; the edge/fog/cloud
  speed ratios remain configuration parameters.
- Rows of CICIoT2023 are split at random, not by capture; recall may be optimistic.
