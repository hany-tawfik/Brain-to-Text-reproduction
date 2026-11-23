# Setup

The upstream repo assumes Ubuntu and NVIDIA GPUs. On the Mac you can do everything except the full-speed training and the n-gram language model. Those go on a rented Linux GPU box (Part B).

Requires conda (miniforge recommended).

## Quick path: run.sh

`run.sh` does everything below inside the `b2txt25` conda env and logs each run to `logs/`:
```bash
bash run.sh all          # setup + download + check + quick eval + full eval
bash run.sh time         # training speed on MPS and CPU
bash run.sh train sanity 2000
```
Run `bash run.sh` with no arguments to see all commands. The manual steps below are the same thing, one at a time.

## Part A. MacBook

### 1. Clone this repo and cd into it

### 2. Clone the upstream repo
```bash
git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text.git
```
Tested against upstream commit `980a5b7` (2025-07-28).

### 3. Create the environment (Mac version)
The repo's `setup.sh` installs CUDA PyTorch, which does not work on macOS. Use this instead:
```bash
conda create -n b2txt25 python=3.10 -y
conda activate b2txt25
pip install --upgrade pip
pip install torch torchvision torchaudio        # macOS wheels include MPS (Apple GPU)
pip install numpy==2.1.2 pandas==2.3.0 matplotlib==3.10.1 scipy==1.15.2 \
    scikit-learn==1.6.1 tqdm==4.67.1 g2p_en==2.1.0 h5py==3.13.0 \
    omegaconf==2.3.0 editdistance==0.8.1 jupyter==1.1.1 redis==5.2.1
cd nejm-brain-to-text && pip install -e . && cd ..
```
Skip `bitsandbytes` (CUDA only). Check:
```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### 4. Download the data (11.6 GB)
Needs about 12 GB free disk (plus the zips while unzipping).

The repo's `download_data.py` fails with HTTP 401 (tested 2026-11-16): Dryad now refuses scripted file downloads. Download in a browser instead from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), Files section:
- `t15_copyTask_neuralData.zip` (11.05 GB)
- `t15_pretrained_rnn_baseline.zip` (485 MB)

Leave them in `~/Downloads` and run `bash run.sh download`, which unzips them into the right place. Or by hand:
```bash
unzip t15_copyTask_neuralData.zip -d data/       # should create data/hdf5_data_final/
unzip t15_pretrained_rnn_baseline.zip -d data/   # should create data/t15_pretrained_rnn_baseline/
```
Check: `ls data/hdf5_data_final | wc -l` should print 45.

### 5. Reproduce baseline PER on the Mac
```bash
cd model_training
python ../../scripts/mac_greedy_eval.py --max_sessions 2    # quick smoke test
python ../../scripts/mac_greedy_eval.py                     # full val split
```
Writes `results/baseline_val_greedy.csv` and prints aggregate PER. Record it in LOG.md.

If MPS throws an unsupported-op error, rerun with `--cpu`. It is slower but fine for ~1,500 trials.

### 6. Time training on this Mac (M1 Max, 32 GB)
```bash
cd model_training
python ../../scripts/time_training.py --device mps --batches 300
python ../../scripts/time_training.py --device cpu --batches 100
```
Each prints seconds per batch and an estimate for the full 120,000-batch run, and appends to `results/timing.csv`. Copy both summaries into LOG.md.

Decision rule:
- Estimate under ~12 h: run the full reproduction overnight on the Mac and skip the rental.
- Estimate over ~12 h (expected): use the Mac for reduced runs (step 7), rent a GPU once for the full reproduction (Part B).

If the dataloader crashes on macOS, add `--workers 0`.

### 7. Reduced training runs on the Mac
```bash
python ../../scripts/mac_train.py --batches 2000  --name sanity          # minutes, checks the pipeline
python ../../scripts/mac_train.py --batches 20000 --name overnight_20k   # one night
python ../../scripts/mac_train.py --batches 5000  --name finetune_5k \
    --init ../data/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint
```
Outputs land in `results/runs/<name>/`. The learning-rate schedule is rescaled to the shorter budget. For experiments, give the baseline and every variant the same `--batches` and the same seeds.

`scripts/mac_patches.py` makes this work without editing upstream files: it disables `torch.compile` and fused AdamW, forces float32, and moves the model to MPS.

### Known Mac issues with the upstream code
- `rnn_trainer.py` and `evaluate_model_helpers.py` hard-code `torch.autocast(device_type="cuda", ...)` with bfloat16. On Mac, either set `use_amp: false` in the yaml or change the device type. `evaluate_model.py` also sends inputs as bfloat16, which breaks on CPU without autocast. `mac_greedy_eval.py` avoids both by running in float32.
- `rnn_trainer.py` picks CUDA or CPU only, wraps the model in `torch.compile`, and uses fused AdamW. `scripts/mac_patches.py` handles all of these at runtime.
- The upstream checkpoint loader (`init_from_checkpoint`) restores optimizer state and loads without `map_location`, so it breaks on a Mac. `mac_train.py --init` loads weights only instead.
- The language model (Kaldi/WeNet based, `setup_lm.sh`) builds for x86 Linux. Do not try it on the Mac.

## Part B. Rented Linux GPU (full retrain)

1. Rent one RTX 4090 (24 GB VRAM) with Ubuntu 22.04 and at least 32 GB RAM and 60 GB disk. Vast.ai or similar. Budget about 5 hours.
2. On the box:
```bash
git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text.git
cd nejm-brain-to-text
# install miniconda if missing, then:
./setup.sh && conda activate b2txt25
python download_data.py
cd model_training
# rnn_args.yaml uses gpu_number: '1'. On a single-GPU box change it to '0'
python train_model.py      # ~3.5 h, target val PER ~10.1%
```
3. Copy `trained_models/baseline_rnn/` and the training log back to `results/` on the Mac before shutting the box down. Rented boxes delete everything on termination.

### Optional: WER with the 1-gram LM on the same box
Needs Redis, cmake, gcc and the `b2txt25_lm` env (`./setup_lm.sh`). Follow `model_training/README.md` sections "Start redis server", "Start language model", "Evaluate". OPT 6.7B needs ~12.4 GB VRAM, which a 4090 has. Skip the 5-gram (~300 GB RAM). For a stronger LM on modest hardware, try LightBeam (see RESOURCES.md).
