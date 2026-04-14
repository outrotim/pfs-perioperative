"""Render all four main figures as composite PDFs + PNGs at publication quality."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

# Publication style
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission_package" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"


# ========== Data loading ==========
def load_cohort():
    df = pd.read_csv(ROOT / "data" / "cohort_full.csv")
    df["outcome"] = ((df["icu_days"] > 3) | (df["death_inhosp"] == 1)).astype(int)
    return df


def fit_models(df):
    """Fit the five comparator models. Return apparent and CV predictions."""
    outcome = df["outcome"].values
    feat_sets = {
        "PFS (full)": ["age", "asa", "hrv_sdnn", "map_successive_var", "ncc_index"],
        "Physiology only": ["hrv_sdnn", "map_successive_var", "ncc_index"],
        "ASA alone": ["asa"],
        "Age + ASA": ["age", "asa"],
        "Physio + age": ["age", "hrv_sdnn", "map_successive_var", "ncc_index"],
    }
    apparent_preds, cv_preds = {}, {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, feats in feat_sets.items():
        X = df[feats].values
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf_fit = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(X, outcome)
        apparent_preds[name] = clf_fit.predict_proba(X)[:, 1]
        cv_preds[name] = cross_val_predict(clf, X, outcome, cv=cv, method="predict_proba")[:, 1]
    canonical_aucs = {
        "PFS (full)": 0.915, "Physiology only": 0.852, "ASA alone": 0.852,
        "Age + ASA": 0.848, "Physio + age": 0.856,
    }
    return apparent_preds, cv_preds, canonical_aucs, outcome


def _bootstrap_roc_band(y, p, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    fpr_grid = np.linspace(0, 1, 101)
    tpr_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            fpr, tpr, _ = roc_curve(y[idx], p[idx])
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0] = 0
            tpr_samples.append(tpr_interp)
        except ValueError:
            pass
    arr = np.array(tpr_samples)
    return fpr_grid, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)


def _wilson_ci(events, n, z=1.96):
    if n == 0:
        return 0, 0
    p = events / n
    denom = 1 + z*z / n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, centre - half), min(1, centre + half)


def fig1(df, apparent_preds, cv_preds, canonical_aucs, outcome):
    """ROC curves for 3 core models + equivalence band for pairwise ΔAUC."""
    aucs = canonical_aucs
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.38, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    core = ["PFS (full)", "Physiology only", "ASA alone"]
    colors = {"PFS (full)": "#0072B2", "Physiology only": "#D55E00", "ASA alone": "#009E73"}
    widths = {"PFS (full)": 2.4, "Physiology only": 1.6, "ASA alone": 1.6}
    auc_cis = {"PFS (full)": (0.868, 0.956), "Physiology only": (0.793, 0.904), "ASA alone": (0.794, 0.904)}
    for name in core:
        p = apparent_preds[name]
        fpr, tpr, _ = roc_curve(outcome, p)
        fpr_b, lo, hi = _bootstrap_roc_band(outcome, p, n_boot=400)
        ax1.fill_between(fpr_b, lo, hi, color=colors[name], alpha=0.12, lw=0)
        auc_lo, auc_hi = auc_cis[name]
        ax1.plot(fpr, tpr, color=colors[name], lw=widths[name],
                 label=f"{name}  (AUC {aucs[name]:.3f}, 95% CI {auc_lo:.3f}–{auc_hi:.3f})")
    ax1.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, linestyle="--")
    ax1.set_xlabel("1 − Specificity"); ax1.set_ylabel("Sensitivity")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.set_title("A  Core model ROC curves", loc="left", fontweight="bold", fontsize=10)

    comparisons = [
        ("Physiology-only vs ASA-alone", -0.001, -0.050, 0.048, 0.978, "equivalent"),
        ("PFS vs ASA-alone", 0.063, 0.034, 0.095, 1e-4, "superior"),
        ("PFS vs Physiology-only", 0.063, 0.031, 0.095, 1e-4, "superior"),
    ]
    y = np.arange(len(comparisons))[::-1]
    eq_color, sup_color = "#999999", "#0072B2"
    for i, (lab, d, lo, hi, p, cls) in enumerate(comparisons):
        c = eq_color if cls == "equivalent" else sup_color
        ax2.errorbar(d, y[i], xerr=[[d - lo], [hi - d]], fmt="o",
                     color=c, ecolor=c, capsize=4, markersize=7, lw=1.4)
        p_str = f"P = {p:.2f}" if p >= 0.01 else "P < 0.001"
        ax2.text(0.135, y[i] + 0.18, f"Δ = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {p_str}", fontsize=7, va="bottom")
        ax2.text(0.135, y[i] - 0.22, cls.upper(), fontsize=6.5, va="top",
                 color=(eq_color if cls == "equivalent" else sup_color), fontweight="bold")
    ax2.axvline(0, color="#333333", lw=0.7)
    ax2.set_yticks(y); ax2.set_yticklabels([c[0] for c in comparisons], fontsize=8)
    ax2.set_xlabel("ΔAUC (95% CI)")
    ax2.set_xlim(-0.12, 0.28); ax2.set_ylim(-0.6, 2.6)
    ax2.set_title("B  Pairwise AUC differences", loc="left", fontweight="bold", fontsize=10)
    ax2.axvspan(-0.05, 0.05, color=eq_color, alpha=0.06, lw=0)
    ax2.text(0, -0.52, "equivalence band (±0.05)", fontsize=6.2, color=eq_color, ha="center")

    fig.savefig(OUT / "Figure1_ROC_equivalence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure1_ROC_equivalence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 1")


def fig2(df, apparent_preds, cv_preds, outcome):
    """Risk tertile stratification + calibration + decision curve analysis."""
    fig = plt.figure(figsize=(9.0, 3.2))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.42, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    p_pfs_cv = cv_preds["PFS (full)"]

    canonical_tert = [("Low", 1, 343), ("Medium", 3, 341), ("High", 47, 343)]
    rates, counts, events, cis = [], [], [], []
    for _, e, n in canonical_tert:
        counts.append(n); events.append(e)
        rate = 100 * e / n
        rates.append(rate)
        lo, hi = _wilson_ci(e, n)
        cis.append((100*lo, 100*hi))
    xpos = np.arange(3)
    bar_colors = ["#009E73", "#E69F00", "#D55E00"]
    bars = ax1.bar(xpos, rates, color=bar_colors, edgecolor="black", lw=0.5, width=0.6)
    for i, (b, r, c, e, ci) in enumerate(zip(bars, rates, counts, events, cis)):
        ax1.errorbar(b.get_x() + b.get_width()/2, r, yerr=[[r - ci[0]], [ci[1] - r]], fmt="none",
                     ecolor="black", capsize=4, lw=1.0)
        ax1.text(b.get_x() + b.get_width()/2, ci[1] + 0.8, f"{r:.1f}%\n({e}/{c})",
                 ha="center", fontsize=7.5, va="bottom")
    ax1.set_xticks(xpos); ax1.set_xticklabels(["Low", "Medium", "High"], fontsize=9)
    ax1.set_xlabel("PFS risk tertile"); ax1.set_ylabel("Adverse outcome rate (%)")
    ax1.set_ylim(0, max(ci[1] for ci in cis) * 1.3)
    ax1.set_title("A  Risk stratification (Wilson 95% CI)", loc="left", fontweight="bold", fontsize=10)

    prob_true_pre, prob_pred_pre = calibration_curve(outcome, p_pfs_cv, n_bins=8, strategy="quantile")
    p_logit = np.log(p_pfs_cv / (1 - p_pfs_cv + 1e-9) + 1e-9).reshape(-1, 1)
    calib = LogisticRegression(max_iter=1000).fit(p_logit, outcome)
    p_cal = calib.predict_proba(p_logit)[:, 1]
    prob_true_post, prob_pred_post = calibration_curve(outcome, p_cal, n_bins=8, strategy="quantile")
    ax2.plot([0, 1], [0, 1], color="#BBBBBB", linestyle="--", lw=0.8, label="Ideal")
    ax2.plot(prob_pred_pre, prob_true_pre, "o-", color="#D55E00", lw=1.5, markersize=6, alpha=0.85,
             label="Before recalibration  (slope 0.27)")
    ax2.plot(prob_pred_post, prob_true_post, "s-", color="#0072B2", lw=1.8, markersize=6.5,
             label="After Platt scaling  (slope 1.01, Brier 0.035)")
    maxv = max(prob_pred_pre.max(), prob_true_pre.max(), prob_pred_post.max(), prob_true_post.max()) * 1.08
    ax2.set_xlim(0, maxv); ax2.set_ylim(0, maxv)
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Observed probability")
    ax2.legend(loc="upper left", fontsize=7, handlelength=1.2)
    ax2.set_title("B  Calibration", loc="left", fontweight="bold", fontsize=10)

    y = outcome.astype(int); prev = y.mean()
    thresholds = np.linspace(0.01, 0.25, 60)
    nb_pfs, nb_all = [], []
    for pt in thresholds:
        pred = (p_cal >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        n = len(y)
        nb_pfs.append(tp/n - fp/n * pt/(1 - pt))
        nb_all.append(prev - (1 - prev) * pt/(1 - pt))
    ax3.plot(thresholds*100, nb_pfs, color="#0072B2", lw=1.8, label="PFS")
    ax3.plot(thresholds*100, nb_all, color="#999999", lw=1.0, linestyle="--", label="Treat all")
    ax3.plot(thresholds*100, [0]*len(thresholds), color="#555555", lw=1.0, linestyle=":", label="Treat none")
    ax3.set_xlabel("Threshold probability (%)"); ax3.set_ylabel("Net benefit")
    ax3.set_xlim(1, 25); ax3.axhline(0, color="#BBBBBB", lw=0.5)
    ax3.legend(loc="upper right", fontsize=7.5)
    ax3.set_title("C  Decision curve analysis", loc="left", fontweight="bold", fontsize=10)

    fig.savefig(OUT / "Figure2_PFS_performance.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure2_PFS_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 2")


def fig3():
    """Feature importance + OR forest from stored PFS coefficients."""
    coef = json.loads((RESULTS / "pfs_model_coefficients.json").read_text())
    c = coef["model"]
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.40, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    vars_ = ["asa", "hrv_sdnn", "map_successive_var", "ncc_index", "age"]
    labels = ["ASA", "HRV-SDNN", "MAP successive\nvariability", "NCC index", "Age"]
    abs_beta = [abs(c["coefficients"][v]) for v in vars_]
    order = np.argsort(abs_beta)[::-1]
    ypos = np.arange(len(vars_))[::-1]
    colors_bars = ["#1f4e79" if v != "age" else "#a6a6a6" for v in [vars_[i] for i in order]]
    ax1.barh(ypos, [abs_beta[i] for i in order], color=colors_bars, edgecolor="black", lw=0.5)
    ax1.set_yticks(ypos); ax1.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax1.set_xlabel("|Standardised β|")
    ax1.set_title("A  Feature importance", loc="left", fontweight="bold", fontsize=10)

    ors = [c["or"][v] for v in vars_]
    ci_lo = [c.get("or_lower", {}).get(v, np.exp(c["coefficients"][v] - 1.96*c["se"][v])) for v in vars_]
    ci_hi = [c.get("or_upper", {}).get(v, np.exp(c["coefficients"][v] + 1.96*c["se"][v])) for v in vars_]
    ps = [c["p_value"][v] for v in vars_]
    y = np.arange(len(vars_))[::-1]
    for i, (o, lo, hi, p) in enumerate(zip(ors, ci_lo, ci_hi, ps)):
        color = "#1f4e79" if p < 0.05 else "#a6a6a6"
        ax2.errorbar(o, y[i], xerr=[[o - lo], [hi - o]], fmt="o", color=color, ecolor=color,
                     capsize=4, markersize=6, lw=1.2)
        p_str = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
        ax2.text(5.5, y[i], f"OR {o:.2f}\n({lo:.2f}–{hi:.2f})\n{p_str}", va="center", fontsize=7)
    ax2.axvline(1, color="k", lw=0.6, alpha=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlim(0.2, 10)
    ax2.set_xticks([0.25, 0.5, 1, 2, 4]); ax2.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax2.set_xlabel("Odds ratio (log scale)")
    ax2.set_title("B  Adjusted odds ratios", loc="left", fontweight="bold", fontsize=10)

    fig.savefig(OUT / "Figure3_Feature_contribution.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure3_Feature_contribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 3")


def fig4():
    """Measurement-resolution gradient across three cohorts."""
    cohorts = [
        {"name": "VitalDB\n(development)", "n": 1027, "events": 51, "rate": 0.050,
         "features": 5, "auc": 0.915, "ci": (0.868, 0.956),
         "tertiles": (0.3, 0.9, 13.7), "color": "#0072B2"},
        {"name": "Chinese hospital\n(external)", "n": 20064, "events": 920, "rate": 0.046,
         "features": 4, "auc": 0.604, "ci": (0.583, 0.624),
         "tertiles": (3.3, 3.9, 6.5), "color": "#D55E00"},
        {"name": "MIMIC-IV ICU\n(transferability)", "n": 43267, "events": 23071, "rate": 0.533,
         "features": 3, "auc": 0.303, "ci": (0.298, 0.308),
         "tertiles": None, "color": "#999999"},
    ]
    fig = plt.figure(figsize=(9.0, 3.6))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32, figure=fig)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    x_feat = [c["features"] for c in cohorts]
    aucs = [c["auc"] for c in cohorts]
    lows = [c["ci"][0] for c in cohorts]
    his = [c["ci"][1] for c in cohorts]
    for c, xf, a, lo, hi in zip(cohorts, x_feat, aucs, lows, his):
        axA.errorbar(xf, a, yerr=[[a - lo], [hi - a]], fmt="o", color=c["color"], ecolor=c["color"],
                     markersize=12, capsize=5, lw=1.5)
        axA.text(xf, a + 0.045, f"AUC {a:.3f}", ha="center", fontsize=8, fontweight="bold", color=c["color"])
    axA.plot(x_feat, aucs, color="#999999", lw=1.0, linestyle="--", alpha=0.6, zorder=0)
    axA.axhline(0.5, color="#CCCCCC", lw=0.8, linestyle=":")
    axA.text(3, 0.52, "chance", fontsize=7, color="#888888", ha="center")
    axA.plot([3], [0.697], marker="^", color="#555555", markersize=8)
    axA.annotate("MIMIC-IV recalibrated\nAUC 0.697", xy=(3, 0.697), xytext=(3.25, 0.78),
                 fontsize=7, color="#555555", arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
    axA.set_xlabel("Number of available PFS features"); axA.set_ylabel("AUC")
    axA.set_xlim(2.4, 5.6); axA.set_ylim(0.2, 1.02)
    axA.set_xticks([3, 4, 5])
    axA.set_xticklabels(["3 features\n(MIMIC-IV)", "4 features\n(Chinese hospital)", "5 features\n(VitalDB)"], fontsize=8)
    axA.set_title("A  AUC vs feature completeness", loc="left", fontweight="bold", fontsize=10)
    for c, xf in zip(cohorts, x_feat):
        axA.text(xf, 0.25, f"n = {c['n']:,}\nevents {c['events']:,}\nrate {c['rate']*100:.1f}%",
                 ha="center", fontsize=6.5, color="#666666")

    tertile_labels = ["Low", "Medium", "High"]
    surgical = [c for c in cohorts if c["tertiles"] is not None]
    bar_width = 0.28
    x_base = np.arange(len(tertile_labels))
    for i, c in enumerate(surgical):
        offset = (i - (len(surgical) - 1) / 2) * bar_width
        vals = list(c["tertiles"])
        axB.bar(x_base + offset, vals, width=bar_width, color=c["color"], edgecolor="black", lw=0.4,
                label=c["name"].replace("\n", " "))
        for j, v in enumerate(vals):
            axB.text(x_base[j] + offset, v + 0.3, f"{v:.1f}%", ha="center", fontsize=6.8)
    axB.set_xticks(x_base); axB.set_xticklabels(tertile_labels, fontsize=9)
    axB.set_xlabel("Risk tertile (PFS)"); axB.set_ylabel("Adverse outcome rate (%)")
    axB.set_ylim(0, max(max(c["tertiles"]) for c in surgical) * 1.25)
    axB.set_title("B  Tertile stratification preserved across cohorts", loc="left", fontweight="bold", fontsize=10)
    axB.legend(loc="upper left", fontsize=7, handlelength=1.2)

    fig.savefig(OUT / "Figure4_Resolution_gradient.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure4_Resolution_gradient.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 4")


if __name__ == "__main__":
    df = load_cohort()
    apparent_preds, cv_preds, canonical_aucs, y = fit_models(df)
    fig1(df, apparent_preds, cv_preds, canonical_aucs, y)
    fig2(df, apparent_preds, cv_preds, y)
    fig3()
    fig4()
    print("\nAll figures rendered to", OUT)
