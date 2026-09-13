"""Generate publication-ready LinkedIn figures from Small-Mind result artifacts.

Run from the repository root with:
    python scripts/generate_linkedin_figures.py

The script intentionally keeps historical, corrected, and final-rebalanced results
separate. See each data structure's source comment for provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "linkedin"

# Consistent, colorblind-friendly identity across all three figures.
NAVY = "#17324D"
TEAL = "#007C83"
CORAL = "#D95F59"
GOLD = "#C58B19"
SLATE = "#526477"
INK = "#17212B"
MUTED = "#667788"
GRID = "#DCE3E8"
PAPER = "#FAFBFC"

FOOTER_PMB = (
    "Small-Mind • PMB: 688 adversarial probes • github.com/arrogance231/small-mind-companion"
)


def load_metrics(relative_path: str) -> dict[str, float]:
    """Load canonical aggregate metrics, preserving source precision."""
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    return {
        "UAR": float(metrics["uar"] * 100),
        "pra_lenient": float(metrics["pra_lenient"] * 100),
    }


# Graph 1 sources:
# - A/D/B/E/F canonical aggregate JSON under results/.
# - The final-rebalanced B value is the corrected value in the final section
#   of docs/proper_scale_results.md (the older B_sft metrics.json is stale).
PROGRESSION = [
    ("Raw model", "Base • no memory", load_metrics("results/v0.1/A_raw/metrics.json")),
    ("Raw + memory", "Base • hybrid memory", load_metrics("results/v0.1/D_memory/metrics.json")),
    ("Rebalanced SFT", "SFT v1 • no memory", {"UAR": 25.0, "pra_lenient": 0.16}),
    (
        "SFT + DPO + memory",
        "Post-trained • memory",
        load_metrics("results/v1_scale/E_sft_memory/metrics.json"),
    ),
    (
        "On-policy distillation",
        "Distilled • memory",
        load_metrics("results/v1_scale/E_distill/metrics.json"),
    ),
]

# Graph 2: final three-point debugging trajectory, from the corrected/final
# rebalanced section of docs/proper_scale_results.md, using fixed-detector rates.
CALIBRATION = [
    {"label": "Broken dedup", "false_abstention": 9.9, "UAR": 16.25, "pra_lenient": 18.42},
    {
        "label": "Dedup fixed • over-corrected",
        "false_abstention": 69.2,
        "UAR": 96.25,
        "pra_lenient": 10.2,
    },
    {"label": "Final rebalanced", "false_abstention": 32.1, "UAR": 70.0, "pra_lenient": 15.3},
]

# Graph 3: representative levels from the authoritative benchmark table in
# docs/quantization_results.md (the README rounds these differently).
QUANTIZATION = [
    {"level": "F16", "size_gib": 8.62, "tokens_per_second": 26.15},
    {"level": "Q8_0", "size_gib": 4.59, "tokens_per_second": 43.07},
    {"level": "Q4_K_M", "size_gib": 3.17, "tokens_per_second": 58.00},
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )


def finish(fig: mpl.figure.Figure, filename: str, footer: str) -> None:
    fig.text(0.08, 0.035, footer, ha="left", va="bottom", fontsize=9.5, color=MUTED)
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=100, bbox_inches=None)
    fig.savefig(OUTPUT_DIR / f"{filename}.svg", bbox_inches=None)
    plt.close(fig)


def graph_progression() -> None:
    stages = [item[0] for item in PROGRESSION]
    uar = np.array([item[2]["UAR"] for item in PROGRESSION])
    pra = np.array([item[2]["pra_lenient"] for item in PROGRESSION])
    x = np.arange(len(stages))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.8, 13.5))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.82, bottom=0.23)
    bars_uar = ax.bar(x - width / 2, uar, width, label="UAR", color=TEAL, zorder=3)
    bars_pra = ax.bar(x + width / 2, pra, width, label="pra_lenient", color=CORAL, zorder=3)
    ax.set_title(
        "Small-Mind: Closing the Gap\nWithout Scaling Parameters", loc="left", fontsize=25, pad=28
    )
    ax.text(
        0,
        1.0,
        "Performance across the 688-probe Personalized Memory Benchmark",
        transform=ax.transAxes,
        fontsize=14,
        color=MUTED,
    )
    ax.set_ylabel("Score (%)", fontsize=14, labelpad=10)
    ax.set_ylim(0, 80)
    ax.set_xticks(x, stages, fontsize=11)
    ax.tick_params(axis="x", pad=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 0.985), fontsize=12)

    for bars in (bars_uar, bars_pra):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )

    # Highlight only the final stage without turning the chart into an infographic.
    ax.axvspan(3.55, 4.45, color=GOLD, alpha=0.08, zorder=0)
    ax.annotate(
        "Final: 71.25% UAR",
        xy=(4 - width / 2, uar[4]),
        xytext=(3.55, 77),
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 1.2},
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        0.02,
        -0.18,
        "Memory alone improves retrieval-grounded answer quality, but calibration requires "
        "post-training.",
        transform=ax.transAxes,
        fontsize=11,
        color=SLATE,
    )
    ax.text(
        0.02,
        -0.205,
        "pra_lenient: 15.30% → 18.59% after distillation",
        transform=ax.transAxes,
        fontsize=11,
        color=SLATE,
    )
    finish(fig, "01_pmb_performance_progression", FOOTER_PMB)


def graph_calibration() -> None:
    x = np.array([row["false_abstention"] for row in CALIBRATION])
    y = np.array([row["UAR"] for row in CALIBRATION])
    colors = [SLATE, CORAL, TEAL]

    fig, ax = plt.subplots(figsize=(10.8, 13.5))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.80, bottom=0.22)
    ax.set_title("Fixing Abstention Exposed a New Failure Mode", loc="left", fontsize=25, pad=28)
    ax.text(
        0,
        1.0,
        "Training-data repair improved unanswerable handling—\n"
        "but initially caused severe over-abstention",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MUTED,
    )
    ax.set_xlabel("False abstention on answerable probes (%) ↓ better", fontsize=14, labelpad=12)
    ax.set_ylabel(
        "Correct abstention on unanswerable probes — UAR (%) ↑ better", fontsize=14, labelpad=12
    )
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 105)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for i in range(len(x) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.5, "mutation_scale": 14},
        )
    ax.scatter(x, y, s=260, c=colors, edgecolors=PAPER, linewidths=2.5, zorder=4)

    ax.annotate(
        "Broken dedup\n16.25% UAR\n9.9% false abstention",
        (x[0], y[0]),
        xytext=(5, 29),
        fontsize=11,
        color=NAVY,
    )
    ax.annotate(
        "Dedup fixed • over-corrected\n96.25% UAR —\nbut 69.2% false abstention",
        (x[1], y[1]),
        xytext=(39, 91),
        arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 1.1},
        fontsize=11,
        color=CORAL,
        fontweight="bold",
    )
    ax.annotate(
        "Rebalanced\n70.0% UAR\n32.1% false abstention",
        (x[2], y[2]),
        xytext=(38, 57),
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.1},
        fontsize=11,
        color=TEAL,
        fontweight="bold",
    )
    ax.text(4, 101, "better operating region", fontsize=11, color=SLATE, fontstyle="italic")
    ax.text(
        0,
        -0.15,
        "Restoring the missing abstention signal solved one problem and revealed another.",
        transform=ax.transAxes,
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    finish(
        fig,
        "02_abstention_calibration_tradeoff",
        "Small-Mind • 80 unanswerable + 608 answerable PMB probes • "
        "github.com/arrogance231/small-mind-companion",
    )


def graph_quantization() -> None:
    sizes = np.array([row["size_gib"] for row in QUANTIZATION])
    speeds = np.array([row["tokens_per_second"] for row in QUANTIZATION])
    size_reduction = (1 - sizes[-1] / sizes[0]) * 100
    speed_increase = (speeds[-1] / speeds[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(10.8, 13.5))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.80, bottom=0.24)
    ax.plot(sizes, speeds, color=SLATE, linewidth=1.6, zorder=2)
    ax.scatter(
        sizes[:2], speeds[:2], s=260, c=[NAVY, CORAL], edgecolors=PAPER, linewidths=2.5, zorder=4
    )
    ax.scatter(sizes[2], speeds[2], s=340, c=TEAL, edgecolors=PAPER, linewidths=2.5, zorder=5)
    ax.set_title(
        "Quantization Made Small-Mind\nSmaller—and Faster", loc="left", fontsize=25, pad=28
    )
    ax.text(
        0,
        1.0,
        "CPU-only generation benchmark • 30 threads",
        transform=ax.transAxes,
        fontsize=14,
        color=MUTED,
    )
    ax.set_xlabel("Model size (GiB) ↓ smaller", fontsize=14, labelpad=12)
    ax.set_ylabel("Generation speed (tokens/s) ↑ faster", fontsize=14, labelpad=12)
    ax.set_xlim(9.35, 2.45)
    ax.set_ylim(18, 66)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(8.95, 63.2, "smaller + faster", fontsize=12, color=TEAL, fontweight="bold")

    offsets = [(0.05, 2.2), (0.05, 2.2), (1.25, -5.0)]
    for row, (dx, dy) in zip(QUANTIZATION, offsets):
        ax.annotate(
            f"{row['level']}\n{row['size_gib']:.2f} GiB • {row['tokens_per_second']:.2f} tok/s",
            (row["size_gib"], row["tokens_per_second"]),
            xytext=(row["size_gib"] + dx, row["tokens_per_second"] + dy),
            fontsize=11,
            color=TEAL if row["level"] == "Q4_K_M" else NAVY,
            fontweight="bold" if row["level"] == "Q4_K_M" else "normal",
        )

    ax.text(
        0.02,
        -0.14,
        f"F16 → Q4_K_M: {size_reduction:.1f}% smaller • {speed_increase:.1f}% faster generation",
        transform=ax.transAxes,
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        0.02,
        -0.185,
        "Q4_K_M: recommended quality/size/speed tradeoff • "
        "Q2_K excluded: broken on generation testing",
        transform=ax.transAxes,
        fontsize=10.5,
        color=SLATE,
    )
    finish(
        fig,
        "03_quantization_efficiency",
        "Small-Mind • CPU-only llama-bench • 30 threads • "
        "github.com/arrogance231/small-mind-companion",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    graph_progression()
    graph_calibration()
    graph_quantization()
    print(f"Wrote six figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
