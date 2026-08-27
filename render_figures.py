#!/usr/bin/env python3
"""Render Study 22 figures from aggregate-only public source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "pfs5": "#D55E00",
    "age_asa": "#0072B2",
    "physiology3": "#009E73",
    "asa_only": "#777777",
}
LABELS = {
    "pfs5": "PFS (5 predictors)",
    "age_asa": "Age + ASA",
    "physiology3": "Physiology only",
    "asa_only": "ASA only",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def discrimination(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    source = data["figure_source_data"]
    for name in ("pfs5", "age_asa", "physiology3", "asa_only"):
        roc = source[name]["roc"]
        metric = data["models"][name]["out_of_fold"]["auc"]
        axes[0].plot(
            roc["fpr"],
            roc["tpr"],
            color=COLORS[name],
            lw=2 if name in ("pfs5", "age_asa") else 1.3,
            label=f"{LABELS[name]} {metric:.3f}",
        )
        pr = source[name]["precision_recall"]
        ap = data["models"][name]["out_of_fold"]["average_precision"]
        axes[1].plot(
            pr["recall"],
            pr["precision"],
            color=COLORS[name],
            lw=2 if name in ("pfs5", "age_asa") else 1.3,
            label=f"{LABELS[name]} {ap:.3f}",
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="A  ROC curves")
    axes[1].axhline(
        data["cohort"]["event_prevalence"], color="#999999", ls="--", lw=1
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="B  Precision-recall curves")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "discrimination.pdf")
    fig.savefig(output / "discrimination.png")
    plt.close(fig)


def calibration(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for ax, name in zip(axes, ("pfs5", "age_asa")):
        groups = data["figure_source_data"][name]["calibration_groups"]
        predicted = np.array([group["mean_predicted_probability"] for group in groups])
        observed = np.array([group["observed_event_proportion"] for group in groups])
        lower = np.array([group["wilson_ci_95"][0] for group in groups])
        upper = np.array([group["wilson_ci_95"][1] for group in groups])
        ax.plot([0, 0.35], [0, 0.35], "--", color="#999999", lw=1)
        ax.errorbar(
            predicted,
            observed,
            yerr=[observed - lower, upper - observed],
            fmt="o",
            color=COLORS[name],
            capsize=2.5,
        )
        ax.set_title(LABELS[name])
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 0.35)
    axes[0].set_ylabel("Observed event proportion")
    fig.tight_layout()
    fig.savefig(output / "calibration.pdf")
    fig.savefig(output / "calibration.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("aggregate_results.json"))
    parser.add_argument("--output", type=Path, default=Path("rendered_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    style()
    discrimination(data, args.output)
    calibration(data, args.output)
    print(f"Rendered aggregate-only figures in {args.output}")


if __name__ == "__main__":
    main()
