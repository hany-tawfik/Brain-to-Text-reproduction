"""
Train the baseline GRU on a Mac with a reduced batch budget.

Uses the upstream trainer unchanged (patched at runtime by mac_patches.py).
The learning-rate schedule is rescaled to the shorter run so the cosine decay still completes.

Usage (from nejm-brain-to-text/model_training, b2txt25 env active):
  # short sanity run, ~minutes
  python ../../scripts/mac_train.py --batches 2000 --name sanity
  # overnight run
  python ../../scripts/mac_train.py --batches 20000 --name overnight_20k
  # fine-tune the pretrained baseline instead of training from scratch
  python ../../scripts/mac_train.py --batches 5000 --name finetune_5k \
      --init ../data/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint

Outputs go to ../../results/runs/<name>/ (training_log, best checkpoint, val metrics).
For fair comparisons, give the baseline and every variant the same --batches and seeds.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_patches import build_trainer, pick_device  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--config", default="rnn_args.yaml")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None, help="default: batches/10")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--runs_dir", default="../../results/runs")
    a = p.parse_args()

    args = OmegaConf.load(a.config)
    out = os.path.join(a.runs_dir, a.name)
    if os.path.exists(out):
        sys.exit(f"{out} already exists. Pick a new --name.")
    args.mode = "train"
    args.output_dir = out
    args.checkpoint_dir = os.path.join(out, "checkpoint")

    b = a.batches
    args.num_training_batches = b
    args.lr_decay_steps = b
    args.lr_decay_steps_day = b
    warm = max(1, min(1000, b // 20))
    args.lr_warmup_steps = warm
    args.lr_warmup_steps_day = warm
    args.batches_per_val_step = a.val_every or max(1, b // 10)
    args.batches_per_train_log = max(1, min(200, b // 50))

    if a.data_dir:
        args.dataset.dataset_dir = a.data_dir
    if a.seed is not None:
        args.seed = a.seed
        args.dataset.seed = a.seed
    if a.workers is not None:
        args.dataset.num_dataloader_workers = a.workers
    args.init_from_checkpoint = False  # upstream resume path restores optimizer state and needs CUDA

    device = pick_device(a.device)
    trainer = build_trainer(args, device)
    if a.init:
        # Fine-tune: load weights only, fresh optimizer and schedule
        import torch
        ckpt = torch.load(a.init, map_location="cpu", weights_only=False)
        state = {k.replace("module.", "").replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        trainer.model.load_state_dict(state)
        trainer.model.to(device)
        trainer.logger.info(f"Initialized weights from {a.init}")
    OmegaConf.save(args, os.path.join(out, "args_used.yaml"))
    trainer.train()


if __name__ == "__main__":
    main()
