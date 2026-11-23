"""
Mac-friendly evaluation of the pretrained Brain-to-Text '25 baseline RNN.

What it does
  - Loads the pretrained GRU baseline from Dryad (t15_pretrained_rnn_baseline)
  - Runs it on the validation split using Apple GPU (MPS) or CPU, in float32
  - Greedy CTC decoding (argmax, drop blanks, collapse repeats)
  - Reports phoneme error rate (PER) overall and per session
  - Saves per-trial predictions to a CSV

What it skips
  - Redis, the n-gram language model and OPT rescoring (those need Linux and a CUDA GPU).
    PER is the number you can trust on a Mac. Expect roughly 10% if everything is right
    (the repo reports 10.1% aggregate val PER for a freshly trained baseline).

Status (2026-11-16): run on the full val split on MPS: 10.19% aggregate PER (1426 trials, 41 sessions).

Usage (from the repo's model_training folder, with the b2txt25 env active):
  cd nejm-brain-to-text/model_training
  python ../../scripts/mac_greedy_eval.py \
      --model_path ../data/t15_pretrained_rnn_baseline \
      --data_dir ../data/hdf5_data_final \
      --csv_path ../data/t15_copyTaskData_description.csv \
      --out ../../results/baseline_val_greedy.csv
"""

import argparse
import os
import sys

import editdistance
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Repo modules live in model_training/ (the current directory when run as documented)
sys.path.insert(0, os.getcwd())
from rnn_model import GRUDecoder  # noqa: E402
from data_augmentations import gauss_smooth  # noqa: E402
from evaluate_model_helpers import load_h5py_file, LOGIT_TO_PHONEME  # noqa: E402


def pick_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def greedy_ctc(logits: np.ndarray) -> list[int]:
    ids = np.argmax(logits, axis=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="../data/t15_pretrained_rnn_baseline")
    p.add_argument("--data_dir", default="../data/hdf5_data_final")
    p.add_argument("--csv_path", default="../data/t15_copyTaskData_description.csv")
    p.add_argument("--out", default="../../results/baseline_val_greedy.csv")
    p.add_argument("--cpu", action="store_true", help="force CPU even if MPS is available")
    p.add_argument("--max_sessions", type=int, default=0, help="0 = all; use 2 for a quick smoke test")
    args = p.parse_args()

    device = pick_device(args.cpu)
    print(f"Device: {device}")

    margs = OmegaConf.load(os.path.join(args.model_path, "checkpoint/args.yaml"))
    sessions = list(margs["dataset"]["sessions"])

    model = GRUDecoder(
        neural_dim=margs["model"]["n_input_features"],
        n_units=margs["model"]["n_units"],
        n_days=len(sessions),
        n_classes=margs["dataset"]["n_classes"],
        rnn_dropout=margs["model"]["rnn_dropout"],
        input_dropout=margs["model"]["input_network"]["input_layer_dropout"],
        n_layers=margs["model"]["n_layers"],
        patch_size=margs["model"]["patch_size"],
        patch_stride=margs["model"]["patch_stride"],
    )
    ckpt = torch.load(
        os.path.join(args.model_path, "checkpoint/best_checkpoint"),
        map_location="cpu",
        weights_only=False,
    )
    state = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.to(device).eval()

    csv_df = pd.read_csv(args.csv_path)
    tf = margs["dataset"]["data_transforms"]

    rows = []
    total_edits, total_len = 0, 0
    used = sessions[: args.max_sessions] if args.max_sessions else sessions
    for session in used:
        f = os.path.join(args.data_dir, session, "data_val.hdf5")
        if not os.path.exists(f):
            continue
        data = load_h5py_file(f, csv_df)
        day_idx = sessions.index(session)
        s_edits, s_len = 0, 0
        for t in range(len(data["neural_features"])):
            x = torch.tensor(data["neural_features"][t][None], dtype=torch.float32, device=device)
            with torch.no_grad():
                x = gauss_smooth(
                    inputs=x,
                    device=device,
                    smooth_kernel_std=tf["smooth_kernel_std"],
                    smooth_kernel_size=tf["smooth_kernel_size"],
                    padding="valid",
                )
                logits = model(x=x, day_idx=torch.tensor([day_idx], device=device))
                if isinstance(logits, tuple):
                    logits = logits[0]
            pred = greedy_ctc(logits[0].float().cpu().numpy())
            true = [int(i) for i in data["seq_class_ids"][t][: data["seq_len"][t]]]
            ed = editdistance.eval(pred, true)
            s_edits += ed
            s_len += len(true)
            rows.append(
                {
                    "session": session,
                    "trial": int(data["trial_num"][t]),
                    "corpus": data["corpus"][t],
                    "sentence": data["sentence_label"][t],
                    "true_phonemes": " ".join(LOGIT_TO_PHONEME[i] for i in true),
                    "pred_phonemes": " ".join(LOGIT_TO_PHONEME[i] for i in pred),
                    "edits": ed,
                    "n_true": len(true),
                }
            )
        total_edits += s_edits
        total_len += s_len
        print(f"{session}: {len(data['neural_features'])} trials, PER {100 * s_edits / max(s_len, 1):.2f}%")

    print(f"\nAggregate val PER (greedy, no LM): {100 * total_edits / max(total_len, 1):.2f}%")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved per-trial results to {args.out}")


if __name__ == "__main__":
    main()
