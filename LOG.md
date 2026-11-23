# Lab notebook

One entry per work session. Newest at the top. Record numbers exactly, including failures.

Template:
```
## YYYY-MM-DD
- Did:
- Result (numbers):
- Problems / fixes:
- Next:
```

---

## 2026-11-16 (timing)
- Did: `run.sh time` (20 warm-up batches, then timed batches; stock baseline model, 44.3M parameters; no validation).
- Result: MPS 9.157 s/batch (300 batches in 2747.0 s) -> full 120,000-batch run about 305 h, 87x slower than the 4090 reference (3.5 h). CPU 40.086 s/batch (100 batches in 4008.6 s) -> about 1336 h. Loss fell from 754.9 to 83.0 over 300 MPS batches, so training works on MPS.
- Decision: no full training on the Mac. Even a reduced 20,000-batch experiment run would take about 51 h per run on MPS. Retrain and experiments go on a rented GPU. The Mac stays for eval, analysis and short debug runs.
- Note: the whole test took about 2 h (my estimate of "a few minutes" was wrong).
- Log: `logs/time_20261116_171918.log`, `results/timing.csv`.
- Next: set up a rented RTX 4090 and run the stock retrain.

## 2026-11-16 (PER breakdown)
- Did: wrote `scripts/per_breakdown.py` (group PER = total edits / total reference phonemes). Ran it on `results/baseline_val_greedy.csv`.
- Result by corpus: Freq words 3.94% (225 trials), Switchboard 10.61% (1037), Random 14.49% (39), Openwebtext 14.60% (80), Harvard 16.35% (45).
- Result by time: sessions up to day 377 (35 sessions): 8.99% (3270 / 36356 phonemes). The six 2025 sessions (days 543 to 636): 18.84% (949 / 5036). Worst: t15.2025.03.14 30.77%, t15.2025.01.10 24.24%, t15.2025.03.30 21.61%.
- Caveat: corpus mix differs between sessions, so the time trend and the corpus effect are not separated yet.
- Outputs: `results/baseline_val_greedy_by_corpus.csv`, `_by_session.csv`, `_breakdown.png`.
- Next: timing (`run.sh time`).

## 2026-11-16 (full eval)
- Did: `run.sh eval` on the full val split, pretrained checkpoint, greedy CTC, no LM, MPS.
- Result: **aggregate val PER 10.19%** over 1426 trials in 41 sessions. Upstream reports 10.1%, so the baseline is reproduced. Per-trial results in `results/baseline_val_greedy.csv`, log `logs/eval_20261116_163228.log`.
- Per-session range: 1.40% (t15.2023.11.17, t15.2023.11.19) to 30.77% (t15.2025.03.14). The 2025 sessions are mostly worse (24.24%, 9.24%, 30.77%, 14.01%, 21.61%, 21.40%), consistent with drift late in the dataset.
- Problem / fix: first attempt failed with `ModuleNotFoundError: editdistance`. VS Code's selected interpreter (`VScode` env) set `VIRTUAL_ENV` and put its bin first on PATH, which overrode `conda run`. `run.sh` now strips the active venv before calling conda.
- Next: per-corpus and per-day breakdown figure.

## 2026-11-16 (first eval)
- Did: `run.sh download` (Safari had auto-unzipped both zips in ~/Downloads; script now moves the folders in). `run.sh check`: torch 2.14.1, MPS available, 45 sessions, pretrained baseline found.
- Result: `run.sh eval-quick` on MPS: session t15.2023.08.13, 35 val trials, PER 8.00% (greedy CTC, no LM). Only one session ran because the first session (t15.2023.08.11) has no val file.
- Next: `run.sh eval` on the full val split.

## 2026-11-16 (setup run)
- Did: `bash run.sh all`. Env `b2txt25` built OK.
- Problem: `download_data.py` failed with HTTP 401 Unauthorized from Dryad on the file download (the file listing worked).
- Fix: download the two zips in the browser; `run.sh download` now unzips them from ~/Downloads.

## 2026-11-16 (run.sh)
- Did: added `run.sh` so every step runs inside the `b2txt25` conda env with one command and a log file.
- Next: `bash run.sh all` in Terminal.

## 2026-11-16 (later)
- Did: checked whether full training is realistic on the M1 Max (32 GB). Wrote `mac_patches.py`, `time_training.py`, `mac_train.py`.
- Result: all three run end to end on synthetic data (CPU, tiny model). Real timing still to do.
- Expectation before measuring: much slower than a 4090. Published comparisons put Macs at roughly 3 to 4x slower than a similarly priced NVIDIA GPU, the M2 Max (faster than the M1 Max) at 14 FP32 TFLOPS vs 83 for the 4090, and Apple GPU kernel launch latency about 35x higher. That last one hurts GRUs, which launch many small kernels per time step.
- Next: SETUP steps 2 to 6, then decide Mac overnight vs rented GPU.

## 2026-11-16
- Did: chose the project (reproduce Brain-to-Text '25 baseline, then one audio-ML experiment). Wrote RESOURCES, SETUP, BACKGROUND and a project plan. Read the upstream repo. Wrote `scripts/mac_greedy_eval.py`.
- Result: eval script runs end to end on synthetic data with a random model (CPU). Not yet run on real data.
- Next: SETUP.md steps 2 to 5.
