"""Create lightweight artifacts (summary JSON, loss curves, notes) for a PoseBERT run.

Usage: python3 -m pose_bert.scripts.summarize_run <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_epoch_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in metrics if row.get("type") == "epoch"]


def _best_epoch(epoch_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in epoch_rows if row.get("val_loss") is not None]
    if valid:
        return min(valid, key=lambda row: row["val_loss"])
    if epoch_rows:
        return min(epoch_rows, key=lambda row: row["train_loss"])
    return None


def _make_loss_curve(epoch_rows: list[dict[str, Any]], run_dir: Path) -> None:
    if not epoch_rows:
        return

    xs = [row["epoch"] for row in epoch_rows]
    train_ys = [row["train_loss"] for row in epoch_rows]
    val_rows = [row for row in epoch_rows if row.get("val_loss") is not None]
    val_xs = [row["epoch"] for row in val_rows]
    val_ys = [row["val_loss"] for row in val_rows]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 12,
        "legend.fontsize": 7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.4), constrained_layout=True)
    ax.plot(xs, train_ys, label="Train", color="#1f77b4", marker="o", markersize=3)
    if val_xs:
        ax.plot(val_xs, val_ys, label="Validation", color="#d62728", marker="s", markersize=3)

    best = _best_epoch(epoch_rows)
    if best is not None and best.get("val_loss") is not None:
        ax.scatter(
            [best["epoch"]], [best["val_loss"]],
            s=60, facecolors="none", edgecolors="#d62728", linewidths=1.5, zorder=5,
            label=f"Best val (epoch {best['epoch']}, {best['val_loss']:.3f})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Loss (MSE, cm$^2$)")
    fig.suptitle("PoseBERT Pre-training Loss")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="lower"))
    ax.set_xlim(left=min(xs) - 0.5)
    ax.legend(frameon=False, loc="best", handlelength=1.5, borderpad=0.3, labelspacing=0.3)

    fig.savefig(run_dir / "loss_curve.pdf")
    fig.savefig(run_dir / "loss_curve.png", dpi=300)
    plt.close(fig)


def _write_note_template(run_dir: Path) -> None:
    note_path = run_dir / "notes.md"
    if note_path.exists():
        return
    note_path.write_text(
        "# Run Notes\n\n"
        "- Hypothesis: \n"
        "- Outcome: \n"
        "- Next step: \n"
    )


def summarize_run(run_dir: Path, init_note: bool = True) -> dict[str, Any]:
    config = _load_json(run_dir / "config.json")
    metrics = _load_jsonl(run_dir / "metrics.jsonl")
    epoch_rows = _extract_epoch_rows(metrics)
    best = _best_epoch(epoch_rows)
    last = epoch_rows[-1] if epoch_rows else None

    summary = {
        "run_name": run_dir.name,
        "epochs_completed": last["epoch"] if last else 0,
        "best_epoch": best["epoch"] if best else None,
        "best_val_loss": best.get("val_loss") if best else None,
        "last_train_loss": last.get("train_loss") if last else None,
        "last_val_loss": last.get("val_loss") if last else None,
        "model_size": {
            "d_model": config.get("d_model"),
            "num_layers": config.get("num_layers"),
            "nhead": config.get("nhead"),
            "window_size": config.get("window_size"),
            "stride": config.get("stride"),
        },
        "optimization": {
            "lr": config.get("lr"),
            "batch_size": config.get("batch_size"),
            "num_epochs_planned": config.get("num_epochs"),
            "mask_ratio": config.get("mask_ratio"),
            "min_span": config.get("min_span"),
            "max_span": config.get("max_span"),
            "seed": config.get("seed"),
            "val_split": config.get("val_split"),
        },
    }

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _make_loss_curve(epoch_rows, run_dir)
    if init_note:
        _write_note_template(run_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PoseBERT run folder")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-init-note", action="store_true",
                        help="Do not create notes.md.")
    args = parser.parse_args()

    summary = summarize_run(args.run_dir, init_note=not args.no_init_note)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
