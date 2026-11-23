# Resources

Gathered 2026-11-16. Check links before relying on numbers; benchmarks move.

## Data

| Dataset | What | Size | Link |
|---|---|---|---|
| Card et al. 2024 (B2T '25) | 10,948 sentences, 45 sessions over 20 months, one participant with ALS (T15), 256 electrodes in speech motor cortex, 512 features at 20 ms bins. Includes pretrained baseline checkpoint | 11.59 GB (neural data zip 11.05 GB, baseline 485 MB) | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85) |
| Willett et al. 2023 (B2T '24) | Earlier speech BCI dataset (participant T12) plus pretrained 3-gram and 5-gram language models | see page | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) |

### Card dataset details (from the repo and its CSV)
- Features per electrode: threshold crossings (-4.5 RMS) and spike band power. 256 electrodes x 2 = 512
- Arrays (64 electrodes each): ventral 6v, area 4, 55b, dorsal 6v. Feature order is in the repo README
- Split: 8,234 train sentences, 2,928 val+test. Test labels are hidden (Kaggle scoring, 1,450 sentences)
- Corpora: Switchboard 7,672, frequent words 2,050, 50-word 1,091, OpenWebText 172, Harvard 98, random 79
- Post-implant days 25 to 636
- Format: one folder per session with `data_train.hdf5`, `data_val.hdf5`, `data_test.hdf5`. Loader: `load_h5py_file()` in `model_training/evaluate_model_helpers.py`

## Code

| Repo | Why | Link |
|---|---|---|
| nejm-brain-to-text | Official B2T '25 baseline: PyTorch GRU, training, evaluation, n-gram LM | [GitHub](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) |
| brainaudio (LightBeam) | CTC decoder with LLM fusion using ~10 GB RAM instead of ~320 GB. The realistic way to get WER without a server | [GitHub](https://github.com/ebrahimfeghhi/brainaudio) |
| E2ESpeechDecoding | End-to-end Conformer to characters on B2T '25. Reference for a Conformer encoder | [GitHub](https://github.com/owaismujtaba/E2ESpeechDecoding) |

### Baseline facts (from the repo)
- Model: day-specific 512x512 input layers (softsign), 5-layer GRU with 768 units, patch size 14 / stride 4, 41 output classes (40 phonemes incl. silence + CTC blank)
- Loss: CTC. Optimizer: AdamW, cosine LR, 120,000 batches of 64
- Augmentation: white noise, constant offset, temporal jitter, Gaussian smoothing
- Training time: ~3.5 h on one RTX 4090. Reported val PER: 10.1%
- Tested on Ubuntu 22.04, two RTX 4090s, 512 GB RAM
- Language model: 1-gram (in repo), 3-gram (~60 GB RAM), 5-gram (~300 GB RAM), plus OPT 6.7B rescoring (~12.4 GB VRAM)

## Benchmarks and results

| Benchmark | Baseline | Best known | Notes |
|---|---|---|---|
| B2T '24 (Willett data) | 9.7% WER | 5.77% WER (DCoND-LIFT) | Gains came from ensembles merged by a fine-tuned LLM, diphone targets, better LR schedules. Transformers, Mamba, CNNs did not clearly beat the RNN |
| B2T '25 (Card data) | 6.67% WER | 5.77% WER reported by LightBeam; lower on some test sets with extra correction | Kaggle competition ran from Aug 2025 for about five months |

Key lesson from '24: lower phoneme error rate does not always mean lower word error rate after the language model.

## Papers to read, in order

1. Card et al. 2024, *An Accurate and Rapidly Calibrating Speech Neuroprosthesis*, NEJM. [Link](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132). The dataset and baseline reproduced here
2. Willett et al. 2024, *Brain-to-Text Benchmark '24: Lessons Learned*. [arXiv](https://arxiv.org/abs/2412.17227). What worked and what did not
3. Wandelt et al. 2022, *Decoding grasp and speech signals from the cortical grasp circuit in a tetraplegic human*, Neuron. [DOI](https://doi.org/10.1016/j.neuron.2022.03.009). Speech decoding from a non-classical cortical area (supramarginal gyrus)
4. *Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding*. [arXiv](https://arxiv.org/html/2507.02800v2). Evidence that time masking helps on this data
5. *BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding*, 2026. [arXiv](https://arxiv.org/html/2603.13321v1). Pretrained speech model as decoder
6. *LightBeam: An Accurate and Memory-Efficient CTC Decoder for Speech Neuroprostheses*. [alphaXiv](https://www.alphaxiv.org/abs/2603.14002)
7. *End-to-End Intracortical Speech Decoding from Neural Activity*, 2026. [arXiv](https://arxiv.org/pdf/2605.24313). Conformer results and per-session drift analysis


## Broader tools (later)

- BEND-BCI: benchmark of 23 neural decoders across motor, visual, speech and spatial tasks. [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.21.739953.full.pdf)
- BCIJelly: BCI datasets, decoders and neuromorphic deployment in one Python framework. [alphaXiv](https://www.alphaxiv.org/abs/2608.13576.md)
- FALCON: few-shot, cross-session decoder stability benchmark (relevant to day-to-day drift)

## Compute

| Option | What it is good for | Cost |
|---|---|---|
| MacBook Pro (Apple Silicon) | Data exploration, pretrained inference, greedy PER, short training runs on MPS | free |
| Rented RTX 4090 (e.g. Vast.ai) | Full baseline retrain (~3.5 h), 1-gram LM + OPT | ~$0.40/h, check live prices |
| Rented A100 | Same, more VRAM | ~$0.68/h, check live prices |
| Kaggle notebooks | Free T4/P100, ~30 h/week reported, 12 h session cap | free, slower |
| Colab free | T4 when available, no guaranteed quota | free, unreliable |

Prices from [gpuperhour.com](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), snapshot 2026-11-16.

## Context
- [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25/overview)
- [IEEE Spectrum on the competition](https://spectrum.ieee.org/speech-bci-machine-learning-competition)
