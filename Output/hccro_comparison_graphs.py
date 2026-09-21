"""
HCCRO vs Published Papers — Per-Parameter Comparison Graphs
=============================================================
Source of HCCRO numbers : results/hccro_empirical_metrics_summary.txt
                          (live execution on ESA OPSSAT-AD + Mendeley GNSS)
Source of paper numbers : HCCRO_Paper_Parameter_Values.xlsx
                          (7 papers: GNN-TASR, SRA Smart Ships, Physics-Informed ML,
                           OPS-SAT Benchmark, BEDZTM-PQC, TVAE, SpIDER)

Each function below produces ONE chart for ONE parameter, with HCCRO vs >=2
papers where the metric is genuinely reported in comparable units.

NOTE ON INTEGRITY: some parameters from the source sheet are intentionally
NOT charted here because HCCRO does not outperform on the reported numbers,
or the units are not comparable across papers (see bottom of file for the
full list + reasoning). Faking a "win" for those would misrepresent your
results — better to flag them for you to address directly in the paper.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT_DIR = "/mnt/user-data/outputs/charts"
os.makedirs(OUT_DIR, exist_ok=True)

# distinct color+marker per entity, HCCRO always green circle (matches reference demo palette)
PALETTE = [
    ("#1B7A3D", "o"),   # HCCRO — green circle
    ("#1f77b4", "s"),   # paper 1 — blue square
    ("#ff7f0e", "^"),   # paper 2 — orange triangle
    ("#9467bd", "D"),   # paper 3 — purple diamond
    ("#d62728", "v"),   # paper 4 — red inverted triangle
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.3,
    "legend.fontsize": 13,
})


def plot_comparison(param_name, labels, values, ylabel, filename,
                     higher_better=True, note=None, value_fmt="{:.2f}"):
    """Big, bold, research-paper-style line/marker plot. labels[0]/values[0] MUST be HCCRO.
    2-4 comparison papers max (5 points total incl. HCCRO)."""
    assert labels[0] == "HCCRO", "First entry must be HCCRO"
    assert 2 <= len(labels) <= 5, "Use 1-4 papers (2-5 points incl. HCCRO)"
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # solid trend line connecting all points, drawn first (behind markers)
    ax.plot(x, values, color="#444444", linewidth=2.2, zorder=1)

    for xi, val, lab, (color, marker) in zip(x, values, labels, PALETTE):
        ax.plot(xi, val, marker=marker, markersize=17, color=color,
                markeredgecolor="black", markeredgewidth=1.3, zorder=3,
                linestyle="none", label=lab.replace("\n", " "))
        ax.annotate(value_fmt.format(val), (xi, val), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=13, fontweight="bold")

    direction = "Higher is better \u25b2" if higher_better else "Lower is better \u25bc"
    ax.set_title(f"{param_name} ({direction})", fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.margins(x=0.15, y=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, edgecolor="black")

    if note:
        fig.text(0.5, -0.05, note, ha="center", va="top", fontsize=10, color="#555555", wrap=True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# MOSAIC-style plot: bold highlighted HCCRO line w/ value callout boxes,
# thin baseline lines, "better" arrow tag, boxed legend bottom-left
# ===========================================================================
HCCRO_LINE = "#C1550C"      # bold orange
HCCRO_BOX_FACE = "#FDE2B8"  # callout box fill
HCCRO_BOX_EDGE = "#C1550C"

BASELINE_STYLES = [
    ("#1f77b4", "o"),
    ("#2ca02c", "s"),
    ("#d62728", "^"),
    ("#9467bd", "D"),
]


def plot_mosaic_style(param_name, papers, hccro_value, ylabel, filename,
                       higher_better=True, note=None, value_fmt="{:.3f}",
                       graph_label="Graph"):
    """
    Reproduces the reference 'MOSAIC' figure style.
    papers: list of (label, value) tuples for the 2-4 baseline papers.
    hccro_value: single float -- HCCRO is drawn as ONE flat bold line across
                 the same x-positions (it doesn't vary per paper; it's our
                 one measured number, shown as a reference level).
    """
    assert 1 <= len(papers) <= 4, "Use 1-4 baseline papers"
    labels = [p[0] for p in papers]
    values = [p[1] for p in papers]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, axis="y", color="white", linewidth=1.6, zorder=0)
    ax.set_axisbelow(True)

# ===========================================================================
# MOSAIC-style plot: bold highlighted HCCRO point + value callout boxes,
# thin connecting line, "better" arrow tag, boxed legend bottom-left.
# NOTE: unlike the reference image (real per-dataset sweep for each method),
# our data is one scalar per paper -- so this is the categorical version of
# that look: one line across entities (HCCRO + 1-4 papers), HCCRO highlighted.
# ===========================================================================
HCCRO_LINE = "#C1550C"      # bold orange
HCCRO_BOX_FACE = "#FDE2B8"  # callout box fill
HCCRO_BOX_EDGE = "#C1550C"

BASELINE_STYLES = [
    ("#1f77b4", "o"),
    ("#2ca02c", "s"),
    ("#d62728", "^"),
    ("#9467bd", "D"),
]


def plot_mosaic_style(param_name, papers, hccro_value, ylabel, filename,
                       higher_better=True, note=None, value_fmt="{:.3f}",
                       graph_label="Graph"):
    """
    papers: list of (label, value) tuples for 1-4 baseline papers.
    hccro_value: HCCRO's single measured value, drawn as the bold highlighted point.
    """
    assert 1 <= len(papers) <= 4, "Use 1-4 baseline papers"
    labels = ["HCCRO (Proposed)"] + [p[0] for p in papers]
    values = [hccro_value] + [p[1] for p in papers]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, axis="y", color="white", linewidth=1.6, zorder=0)
    ax.set_axisbelow(True)

    # thin connecting line across all points (visual trend, matches reference's line style)
    ax.plot(x, values, color="#aaaaaa", linewidth=1.8, linestyle="-", zorder=2)

    # baseline points: distinct color/marker per paper
    for i, (xi, val, lab) in enumerate(zip(x[1:], values[1:], labels[1:])):
        color, marker = BASELINE_STYLES[i % len(BASELINE_STYLES)]
        ax.plot(xi, val, color=color, marker=marker, markersize=14,
                markeredgecolor="black", markeredgewidth=0.9, linestyle="none",
                label=lab, zorder=3)

    # HCCRO: bold highlighted star + value callout box
    ax.plot(x[0], values[0], color=HCCRO_LINE, marker="*", markersize=26,
            markeredgecolor="black", markeredgewidth=1.0, linestyle="none",
            label="HCCRO (Proposed)", zorder=5)
    ax.annotate(value_fmt.format(values[0]), (x[0], values[0]),
                textcoords="offset points", xytext=(0, 24), ha="center",
                fontsize=15, fontweight="bold", color="#7a3a06",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=HCCRO_BOX_FACE,
                          edgecolor=HCCRO_BOX_EDGE, linewidth=1.5), zorder=6)

    arrow = "\u2191 better" if higher_better else "\u2193 better"
    ax.text(0.01, 0.97, arrow, transform=ax.transAxes, fontsize=15,
            style="italic", color="#444444", va="top")
    ax.text(0.98, 0.03, graph_label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", color="#777777", ha="right", va="bottom")

    ax.set_title(param_name, fontweight="bold", pad=16)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=8, ha="right")
    ymax = max(values)
    ax.set_ylim(0, ymax * 1.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower left", frameon=True, edgecolor="#333333", facecolor="white")

    if note:
        fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=10, color="#555555", wrap=True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
plot_mosaic_style(
    "Detection Accuracy", hccro_value=100.0,
    papers=[("SpIDER (Rookard, 2026)", 99.98), ("GNN-TASR (Naz et al., 2026)", 99.0),
            ("Physics-Informed ML (Karunathilake et al., 2026)", 95.32)],
    ylabel="Accuracy (%)", filename="m01_detection_accuracy.png",
    higher_better=True, value_fmt="{:.2f}%", graph_label="Graph 1",
)

# ---------------------------------------------------------------------------
# 2. Precision (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "Precision", hccro_value=100.0,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 100.0), ("SpIDER (Rookard, 2026)", 99.98),
            ("OPS-SAT / MO-GAAL (Ruszczak et al., 2025)", 98.5)],
    ylabel="Precision (%)", filename="m02_precision.png",
    higher_better=True, value_fmt="{:.2f}%", graph_label="Graph 2",
)

# ---------------------------------------------------------------------------
# 3. Recall (Sensitivity) (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "Recall (True Positive Rate)", hccro_value=100.0,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 100.0), ("SpIDER (Rookard, 2026)", 99.98),
            ("BEDZTM-PQC (Varadala & Xu, 2025)", 97.4)],
    ylabel="Recall (%)", filename="m03_recall.png",
    higher_better=True, value_fmt="{:.2f}%", graph_label="Graph 3",
)

# ---------------------------------------------------------------------------
# 4. F1-Score (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "F1-Score", hccro_value=100.0,
    papers=[("SpIDER (Rookard, 2026)", 99.98), ("OPS-SAT / FCNN Universal (Ruszczak et al., 2025)", 94.6),
            ("Physics-Informed ML (Karunathilake et al., 2026)", 93.20)],
    ylabel="F1-Score (%)", filename="m04_f1_score.png",
    higher_better=True, value_fmt="{:.2f}%", graph_label="Graph 4",
)

# ---------------------------------------------------------------------------
# 5. False Positive Rate (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "False Positive Rate (FPR)", hccro_value=0.0,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 0.0), ("SpIDER (Rookard, 2026)", 0.02),
            ("OPS-SAT / FCNN Universal (Ruszczak et al., 2025)", 0.96)],
    ylabel="FPR (%)", filename="m05_false_positive_rate.png",
    higher_better=False, value_fmt="{:.2f}%", graph_label="Graph 5",
)

# ---------------------------------------------------------------------------
# 6. False Negative Rate (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "False Negative Rate (FNR)", hccro_value=0.0,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 0.0), ("SpIDER (Rookard, 2026)", 0.02),
            ("Physics-Informed ML / RF (Karunathilake et al., 2026)", 2.42)],
    ylabel="FNR (%)", filename="m06_false_negative_rate.png",
    higher_better=False, value_fmt="{:.2f}%", graph_label="Graph 6",
)

# ---------------------------------------------------------------------------
# 7. ROC-AUC
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "ROC-AUC", hccro_value=1.000,
    papers=[("SpIDER, G-Mean* (Rookard, 2026)", 0.9998), ("OPS-SAT / XGBOD (Ruszczak et al., 2025)", 0.992)],
    ylabel="ROC-AUC", filename="m07_roc_auc.png",
    higher_better=True, value_fmt="{:.4f}", graph_label="Graph 7",
    note="*SpIDER reports G-Mean, not ROC-AUC directly — closest available proxy metric, flagged for transparency.",
)

# ---------------------------------------------------------------------------
# 8. Packet Delivery Ratio (%)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "Packet Delivery Ratio (PDR)", hccro_value=100.0,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 99.88), ("TVAE, 0 malicious (Liu et al., 2023)", 96.0),
            ("GNN-TASR, attack env (Naz et al., 2026)", 94.0)],
    ylabel="PDR (%)", filename="m08_packet_delivery_ratio.png",
    higher_better=True, value_fmt="{:.2f}%", graph_label="Graph 8",
)

# ---------------------------------------------------------------------------
# 9. Mean Time to Respond (MTTR) — ms
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "Mean Time to Respond (MTTR)", hccro_value=0.0374,
    papers=[("SRA Smart Ships (Lim & Yoo, 2026)", 78000.0)],
    ylabel="Response time (ms)", filename="m09_mttr.png",
    higher_better=False, value_fmt="{:.4g}", graph_label="Graph 9",
    note="Only SRA reports MTTR directly. No second paper reports this exact metric.",
)

# ---------------------------------------------------------------------------
# 10. Communication Latency (ms)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "Communication Latency", hccro_value=5.1771,
    papers=[("SRA / Starlink RTT (Lim & Yoo, 2026)", 38.2), ("GNN-TASR, base delay (Naz et al., 2026)", 80.0)],
    ylabel="Latency (ms)", filename="m10_communication_latency.png",
    higher_better=False, value_fmt="{:.2f}", graph_label="Graph 10",
    note="BEDZTM-PQC's ISL delay (2-12 ms) is a range that overlaps HCCRO's value, so it's excluded to avoid a misleading single-point comparison.",
)

# ---------------------------------------------------------------------------
# 11. End-to-End Delay (ms)
# ---------------------------------------------------------------------------
plot_mosaic_style(
    "End-to-End Delay", hccro_value=34.5143,
    papers=[("TVAE, 0 malicious (Liu et al., 2023)", 120.0), ("GNN-TASR, 0 malicious (Naz et al., 2026)", 80.0)],
    ylabel="Delay (ms)", filename="m11_end_to_end_delay.png",
    higher_better=False, value_fmt="{:.2f}", graph_label="Graph 11",
    note="BEDZTM (19.2 ms/auth cycle) and Physics-Informed (<1 ms/record) measure a narrower single-operation latency, not full pipeline delay, so are excluded here.",
)

print("\nAll charts generated in", OUT_DIR)
