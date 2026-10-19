"""Plot bar charts of graph count per score for a PatternBoost run's train/test data.

Usage:
    python tools/plot_score_histograms.py <run_dir>

Reads <run_dir>/train_data.pkl and <run_dir>/test_data.pkl (lists of DataPoint
objects with a `.score` attribute) and saves score_histogram_train.png and
score_histogram_test.png into <run_dir>.
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_scores(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [dp.score for dp in data]


def plot_score_histogram(scores, title, out_path):
    counts = Counter(scores)
    xs = sorted(counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=xs, y=[counts[x] for x in xs], color="steelblue", edgecolor="black", ax=ax)
    ax.set_xlabel("score")
    ax.set_ylabel("number of graphs")
    ax.set_title(f"{title} (n={len(scores)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}  ({len(xs)} distinct scores, range {xs[0]}-{xs[-1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    train_path = args.run_dir / "train_data.pkl"
    test_path = args.run_dir / "test_data.pkl"

    if train_path.exists():
        train_scores = load_scores(train_path)
        plot_score_histogram(train_scores, "train_data.pkl score distribution", args.run_dir / "score_histogram_train.png")
    else:
        print(f"skip: {train_path} not found")

    if test_path.exists():
        test_scores = load_scores(test_path)
        plot_score_histogram(test_scores, "test_data.pkl score distribution", args.run_dir / "score_histogram_test.png")
    else:
        print(f"skip: {test_path} not found")


if __name__ == "__main__":
    main()
