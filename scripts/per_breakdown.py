"""PER breakdown by corpus and by session (post-implant day) from a per-trial eval CSV.

Usage (from the project root):
    python scripts/per_breakdown.py [--in results/baseline_val_greedy.csv]

Writes results/<stem>_by_corpus.csv, results/<stem>_by_session.csv and results/<stem>_breakdown.png.
PER for a group is total edits / total reference phonemes (same as the aggregate), not a mean of trial PERs.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DESC = ROOT / "nejm-brain-to-text/data/t15_copyTaskData_description.csv"

BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def per_table(df, by):
    g = df.groupby(by).agg(trials=("edits", "size"), edits=("edits", "sum"), phonemes=("n_true", "sum"))
    g["per"] = 100 * g["edits"] / g["phonemes"]
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results/baseline_val_greedy.csv"))
    args = ap.parse_args()
    inp = Path(args.inp)
    stem = inp.with_suffix("")

    df = pd.read_csv(inp)
    total = 100 * df["edits"].sum() / df["n_true"].sum()

    corpus = per_table(df, "corpus").sort_values("per")
    corpus.to_csv(f"{stem}_by_corpus.csv", index=False)

    sess = per_table(df, "session")
    days = pd.read_csv(DESC)[["Date", "Post-implant day"]].drop_duplicates()
    days["session"] = "t15." + days["Date"].str.replace("-", ".")
    sess = sess.merge(days[["session", "Post-implant day"]], on="session", how="left")
    sess = sess.rename(columns={"Post-implant day": "day"}).sort_values("day")
    sess.to_csv(f"{stem}_by_session.csv", index=False)

    print(f"Aggregate PER: {total:.2f}%  ({len(df)} trials, {len(sess)} sessions)\n")
    print("By corpus:")
    print(corpus.to_string(index=False, float_format="%.2f"))
    print("\nBy session:")
    print(sess.to_string(index=False, float_format="%.2f"))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.6, 1]},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Panel A: PER per session over time
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.scatter(sess["day"], sess["per"], s=12 + sess["trials"] * 1.2, color=BLUE,
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax1.axhline(total, color=INK2, linestyle="--", linewidth=1)
    ax1.text(sess["day"].max() * 0.62, total + 0.6, f"aggregate {total:.2f}%", color=INK2, fontsize=9)
    for _, r in sess.nlargest(3, "per").iterrows():
        ax1.annotate(r["session"].removeprefix("t15."), (r["day"], r["per"]), xytext=(6, 2),
                     textcoords="offset points", fontsize=8, color=INK2)
    ax1.set_xlabel("Post-implant day")
    ax1.set_ylabel("PER (%)")
    ax1.set_ylim(0, max(sess["per"].max() * 1.12, total * 2))
    ax1.set_title("PER per validation session (dot size = trials)", loc="left", fontsize=11)

    # Panel B: PER per corpus
    ax2.grid(axis="x", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.barh(corpus["corpus"], corpus["per"], color=BLUE, height=0.6)
    for i, r in enumerate(corpus.itertuples()):
        ax2.text(r.per + 0.3, i, f"{r.per:.1f}%  (n={r.trials})", va="center", fontsize=8, color=INK2)
    ax2.set_xlim(0, corpus["per"].max() * 1.6)
    ax2.set_xlabel("PER (%)")
    ax2.set_title("PER per corpus", loc="left", fontsize=11)

    fig.suptitle("Pretrained baseline, greedy CTC, no LM: validation PER", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = f"{stem}_breakdown.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
