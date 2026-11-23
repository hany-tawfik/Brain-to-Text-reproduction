"""
Time baseline training steps on this machine and estimate the full 120,000-batch run.

Runs the exact baseline training step (augmentation, GRU, CTC loss, backward, clip, AdamW)
for a few warm-up batches and then N timed batches. No validation, no checkpoints kept.

Usage (from nejm-brain-to-text/model_training, b2txt25 env active):
  python ../../scripts/time_training.py --device mps --batches 300
  python ../../scripts/time_training.py --device cpu --batches 100

Result is appended to ../../results/timing.csv. Copy the summary into LOG.md.
"""

import argparse
import csv
import datetime
import os
import platform
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_patches import build_trainer, pick_device, sync  # noqa: E402

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

FULL_RUN_BATCHES = 120_000
RTX4090_HOURS = 3.5  # upstream README figure for the full run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--batches", type=int, default=300, help="timed batches")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--config", default="rnn_args.yaml")
    p.add_argument("--data_dir", default=None, help="override dataset.dataset_dir")
    p.add_argument("--workers", type=int, default=None, help="override dataloader workers")
    p.add_argument("--out", default="../../results/timing.csv")
    a = p.parse_args()

    device = pick_device(a.device)
    args = OmegaConf.load(a.config)
    tmp = tempfile.mkdtemp(prefix="b2t_timing_")
    args.mode = "train"
    args.output_dir = os.path.join(tmp, "run")
    args.checkpoint_dir = os.path.join(tmp, "run", "checkpoint")
    args.num_training_batches = a.warmup + a.batches
    if a.data_dir:
        args.dataset.dataset_dir = a.data_dir
    if a.workers is not None:
        args.dataset.num_dataloader_workers = a.workers

    trainer = build_trainer(args, device)
    model, ctc = trainer.model, trainer.ctc_loss
    ps, st = args.model.patch_size, args.model.patch_stride
    model.train()

    t0 = None
    for i, batch in enumerate(trainer.train_loader):
        if i == a.warmup:
            sync(device)
            t0 = time.perf_counter()
        trainer.optimizer.zero_grad()
        x = batch["input_features"].to(device)
        y = batch["seq_class_ids"].to(device)
        n = batch["n_time_steps"].to(device)
        yl = batch["phone_seq_lens"].to(device)
        d = batch["day_indicies"].to(device)
        x, n = trainer.transform_data(x, n, "train")
        lens = ((n - ps) / st + 1).to(torch.int32)
        logits = model(x, d)
        loss = ctc(torch.permute(logits.log_softmax(2), [1, 0, 2]), y, lens, yl).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip_value)
        trainer.optimizer.step()
        trainer.learning_rate_scheduler.step()
        if i % 50 == 0:
            print(f"batch {i}: loss {loss.item():.2f}")
    sync(device)
    elapsed = time.perf_counter() - t0
    shutil.rmtree(tmp, ignore_errors=True)

    sec_per_batch = elapsed / a.batches
    full_hours = sec_per_batch * FULL_RUN_BATCHES / 3600
    print("\n=== Timing summary ===")
    print(f"device: {device} | torch {torch.__version__} | {platform.platform()}")
    print(f"{a.batches} batches in {elapsed:.1f} s -> {sec_per_batch:.3f} s/batch")
    print(f"Estimated full run ({FULL_RUN_BATCHES:,} batches): {full_hours:.1f} h "
          f"(RTX 4090 reference: {RTX4090_HOURS} h, so {full_hours / RTX4090_HOURS:.1f}x slower)")
    print("Validation passes are not included; add roughly 5 to 10% for them.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "device", "torch", "batches", "sec_per_batch", "est_full_hours"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), str(device),
                    torch.__version__, a.batches, f"{sec_per_batch:.4f}", f"{full_hours:.2f}"])
    print(f"Appended to {a.out}")


if __name__ == "__main__":
    main()
