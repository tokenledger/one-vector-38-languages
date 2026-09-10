"""Generate the five paper figures as vector PDFs sized for a two-column ACL page.

Reads results/findings_both.json, results/scale70b/summary.json and
results/ratio_ci.json. Writes figures/fig1.pdf through figures/fig5.pdf,
creating the directory if needed.

Figures are authored at the ACL column (3.031in) and text (6.299in) widths so
they scale 1:1; nothing below 6.5pt; pdf.fonttype 42 embeds TrueType, since
matplotlib's default Type 3 fonts fail some venues' format checkers.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

HIGH, LOW, ZERO = "#2a78d6", "#eb6834", "#1baf7a"
TIER_COLOR = {"high": HIGH, "low": LOW, "zero": ZERO}
TIER_LABEL = {"high": "High-resource", "low": "Low-resource", "zero": "Zero-shot"}
INK, INK2, GRID = "#1a1a1a", "#555555", "#d8d8d4"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "figure.dpi": 200,
    # No "tight" bbox: cropping to content would let LaTeX rescale each figure
    # by a different factor to fill \linewidth.
    "savefig.pad_inches": 0.0,
})

TIER_ORDER = {"high": 0, "low": 1, "zero": 2}

# Measured from acl.sty: \columnwidth 219.086pt, \textwidth 455.244pt.
COL, FULL = 219.086 / 72.27, 455.244 / 72.27


def load():
    both = json.loads((ROOT / "results" / "findings_both.json").read_text())
    scale = json.loads((ROOT / "results" / "scale70b" / "summary.json").read_text())
    ratio = json.loads((ROOT / "results" / "ratio_ci.json").read_text())
    return both, scale, ratio


def tidy(ax, grid_axis="x"):
    """Recessive frame. Only the axis the eye needs to measure against keeps a line."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=1.5)


# --- Figure 1: transfer by language with bootstrap CIs ----------------------


def fig1(both, ratio):
    """38 languages in two panels of 19.

    Bars carry the joint ratio interval from ratio_ci.json, in which English's
    numerator is resampled on the same items as each target language's.
    English therefore has no visible interval: T(en) = 1 on every resample.
    """
    per = {r["language"]: r for r in both["llama"]["correlates"]["per_language"]}
    ci = {r["language"]: r for r in both["llama"]["ci"]["rows"]}
    en_d = per["en"]["d_bench"]

    rows = sorted(
        per.values(),
        key=lambda r: (TIER_ORDER[r["resource_tier"]], -r["transfer"]),
    )
    half = (len(rows) + 1) // 2
    chunks = [rows[:half], rows[half:]]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    for ax, chunk in zip(axes, chunks):
        y = np.arange(len(chunk))[::-1]
        for yi, r in zip(y, chunk):
            lg = r["language"]
            c = TIER_COLOR[r["resource_tier"]]
            t = r["transfer"]
            ax.barh(yi, t, height=0.62, color=c, zorder=3,
                    edgecolor="white", linewidth=0.4)
            lo, hi = ratio[lg]["joint_lo"], ratio[lg]["joint_hi"]
            ax.plot([lo, hi], [yi, yi], color=INK, linewidth=0.7, zorder=4,
                    solid_capstyle="butt")
            label = f"{lg} (source)" if lg == "en" else lg
            if not ci[lg]["beats_random"]:
                label = f"{lg} *"
            ax.text(-0.02, yi, label, ha="right", va="center", fontsize=6.8,
                    color=INK2)
            ax.text(max(t, hi) + 0.02, yi, f"{t:.2f}", ha="left", va="center",
                    fontsize=6.5, color=INK2)
        ax.set_ylim(-0.8, len(chunk) - 0.2)
        ax.set_xlim(-0.16, 1.16)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["left"].set_visible(False)
        tidy(ax, "x")
        ax.set_xlabel("Transfer ratio $T$")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in
               ("high", "low", "zero")]
    fig.subplots_adjust(left=0.055, right=0.995, top=0.99, bottom=0.265, wspace=0.16)
    fig.legend(handles, [TIER_LABEL[t] for t in ("high", "low", "zero")],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.070), handlelength=1.1, handleheight=0.75,
               columnspacing=1.6)
    fig.text(0.5, 0.012, "* interval does not separate from the matched-norm "
             "random control", ha="center", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig1.pdf")
    plt.close(fig)


# --- Figure 2: three models by tier -----------------------------------------


def fig2(both, scale):
    """Grouped by model, coloured by tier; tier keeps the same hues as Figures 1, 3 and 4."""
    tiers = ["high", "low", "zero"]

    def tier_mean(model_key, t):
        """Mean T over the tier, English excluded (it sits at T=1 by construction)."""
        rows = [r for r in both[model_key]["correlates"]["per_language"]
                if r["resource_tier"] == t and r["language"] != "en"]
        return sum(r["transfer"] for r in rows) / len(rows)

    def scale_mean(t):
        rows = [r for r in scale["rows"] if r["tier"] == t and r["language"] != "en"]
        return sum(r["transfer"] for r in rows) / len(rows)

    models = [
        ("Llama-3.1-8B", [tier_mean("llama", t) for t in tiers]),
        ("Qwen2.5-7B", [tier_mean("qwen", t) for t in tiers]),
        ("Llama-3.1-70B*", [scale_mean(t) for t in tiers]),
    ]

    fig, ax = plt.subplots(figsize=(COL, 1.95))
    x = np.arange(len(models))
    w = 0.26
    for j, t in enumerate(tiers):
        off = (j - 1) * w
        vals = [m[1][j] for m in models]
        ax.bar(x + off, vals, width=w * 0.9, color=TIER_COLOR[t], zorder=3,
               label=TIER_LABEL[t], edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + off, vals):
            # Round half away from zero, as the tables do; f"{v:.2f}" would
            # print Qwen's low tier (exactly 0.065) as 0.06.
            lab = Decimal(repr(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            ax.text(xi, v + 0.012, f"{lab}", ha="center", va="bottom",
                    fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylabel("Mean transfer ratio $T$")
    ax.set_ylim(0, 0.72)
    tidy(ax, "y")
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.0,
              handleheight=0.7, borderpad=0.15, columnspacing=0.9,
              handletextpad=0.4, bbox_to_anchor=(0.5, 1.16))
    fig.subplots_adjust(left=0.145, right=0.99, top=0.87, bottom=0.235)
    fig.text(0.5, 0.02, "* reduced check: 10 languages, one zero-shot language",
             ha="center", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig2.pdf")
    plt.close(fig)


# --- Figure 3: transfer against tokenizer fertility -------------------------


def fig3(both):
    per = both["llama"]["correlates"]["per_language"]
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    for r in per:
        c = TIER_COLOR[r["resource_tier"]]
        is_en = r["language"] == "en"
        ax.scatter(r["fertility"], r["transfer"], s=17,
                   facecolor="white" if is_en else c,
                   edgecolor=c, linewidth=0.9 if is_en else 0.5, zorder=3)
    # Label offsets tuned per point to avoid collisions.
    for code, dx, dy in (("en", 5.0, -1.2), ("pt", 4.5, -1.2), ("he", 4.5, -1.2),
                         ("hi", 4.5, -1.2), ("bn", 4.5, -1.2),
                         ("ml", 3.0, 4.0), ("lo", 4.5, -5.0), ("or", 4.5, -1.2)):
        r = next(x for x in per if x["language"] == code)
        ax.annotate(code, (r["fertility"], r["transfer"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Tokenizer fertility (tokens per prompt, log scale)")
    ax.set_ylabel("Transfer ratio $T$")
    ax.set_ylim(-0.08, 1.12)
    tidy(ax, "both")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.6,
                          color=TIER_COLOR[t]) for t in ("high", "low", "zero")]
    ax.legend(handles, [TIER_LABEL[t] for t in ("high", "low", "zero")],
              frameon=False, loc="upper right", handletextpad=0.35,
              borderpad=0.2, labelspacing=0.25)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.185)
    fig.savefig(OUT / "fig3.pdf")
    plt.close(fig)


# --- Figure 4: target-language vector against imported English vector -------


def fig4(both):
    """Faceted by model. Two models on one scatter would need a fourth hue."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35))
    for ax, key, title, mx in zip(
        axes, ("llama", "qwen"),
        ("Llama-3.1-8B", "Qwen2.5-7B"), (0.155, 0.16)
    ):
        per = [r for r in both[key]["correlates"]["per_language"]
               if r["language"] != "en"]
        ax.plot([0, mx], [0, mx], color=INK2, linewidth=0.7, linestyle=(0, (4, 3)),
                zorder=2)
        for r in per:
            c = TIER_COLOR[r["resource_tier"]]
            ax.scatter(r["d_bench"], r["d_oracle"], s=16, facecolor=c,
                       edgecolor="white", linewidth=0.4, zorder=3)
        # Only well-separated points are labelled; the table has the rest.
        for code, dx, dy in (("pt", 4.5, -3.0), ("te", 5.0, -1.0),
                             ("ml", 4.5, -4.0)):
            r = next((x for x in per if x["language"] == code), None)
            if r:
                ax.annotate(code, (r["d_bench"], r["d_oracle"]),
                            textcoords="offset points", xytext=(dx, dy),
                            fontsize=6.5, color=INK2)
        ax.set_xlim(-0.008, mx)
        ax.set_ylim(-0.008, mx)
        ax.set_title(title, loc="left", pad=3, fontsize=7.5, color=INK)
        ax.set_ylabel(r"$\Delta$ own local vector")
        ax.set_xlabel(r"$\Delta$ imported English vector")
        tidy(ax, "both")
        ax.text(0.018, mx * 0.90, "local = English", fontsize=6.5, color=INK2,
                ha="left", va="center")
        ax.text(mx * 0.97, 0.012, "English vector wins below", fontsize=6.2,
                color=INK2, ha="right", va="bottom")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.6,
                          color=TIER_COLOR[t]) for t in ("high", "low", "zero")]
    axes[0].legend(handles, [TIER_LABEL[t] for t in ("high", "low", "zero")],
                   frameon=False, loc="center right", handletextpad=0.35,
                   borderpad=0.2, labelspacing=0.22)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.155,
                        wspace=0.22)
    fig.savefig(OUT / "fig4.pdf")
    plt.close(fig)


# --- Figure 5: the dissociation over steering strength ----------------------

# English-only dev sweep results (modal_backup/xsyc-results/sweep/), which the
# per-language JSON exports do not carry.
ALPHA = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
BENCH_GAIN = [0.0, 17.2, 35.7, 75.0, 155.4, 352.2]
FACT_GAIN = [0.0, 3.0, 6.0, 12.1, 20.3, -101.6]
BENCH_CORRECT = [72.3, 69.6, 60.1, 52.0, 22.3, 9.5]
FACT_CORRECT = [72.3, 73.7, 77.0, 83.1, 88.5, 8.8]
QWEN_BENCH_CORRECT = [74.3, 71.6, 67.6, 63.5, 49.3, 19.6]
QWEN_FACT_CORRECT = [74.3, 75.0, 75.0, 78.4, 79.7, 63.5]


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95))
    x = np.arange(len(ALPHA))

    ax = axes[0]
    ax.axhline(0, color=GRID, linewidth=0.6, zorder=1)
    ax.plot(x, BENCH_GAIN, color=HIGH, linewidth=1.3, marker="o", markersize=3.4,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            label=r"$v_{\mathrm{bench}}$")
    ax.plot(x, FACT_GAIN, color=LOW, linewidth=1.3, marker="s", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            label=r"$v_{\mathrm{fact}}$")
    ax.annotate(r"$v_{\mathrm{bench}}$", (x[4], BENCH_GAIN[4]),
                textcoords="offset points", xytext=(-2, 6), fontsize=6.8,
                color=INK2, ha="right")
    ax.annotate(r"$v_{\mathrm{fact}}$", (x[4], FACT_GAIN[4]),
                textcoords="offset points", xytext=(-1, 7), fontsize=6.8,
                color=INK2, ha="right")
    ax.set_ylabel("Benchmark margin gain (%)")
    ax.set_title("A. Effect on the benchmark", loc="left", pad=3, fontsize=7.5)

    ax = axes[1]
    ax.plot(x, BENCH_CORRECT, color=HIGH, linewidth=1.3, marker="o",
            markersize=3.4, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.plot(x, FACT_CORRECT, color=LOW, linewidth=1.3, marker="s", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.plot(x, QWEN_BENCH_CORRECT, color=HIGH, linewidth=0.9, linestyle=(0, (3, 2)),
            marker="o", markersize=2.4, alpha=0.75, zorder=2)
    ax.plot(x, QWEN_FACT_CORRECT, color=LOW, linewidth=0.9, linestyle=(0, (3, 2)),
            marker="s", markersize=2.2, alpha=0.75, zorder=2)
    ax.annotate(r"$v_{\mathrm{fact}}$", (x[4], FACT_CORRECT[4]),
                textcoords="offset points", xytext=(-2, 5), fontsize=6.8,
                color=INK2, ha="right")
    ax.annotate(r"$v_{\mathrm{bench}}$", (x[4], BENCH_CORRECT[4]),
                textcoords="offset points", xytext=(-2, -10), fontsize=6.8,
                color=INK2, ha="right")
    ax.set_ylabel("Corrects a falsehood (%)")
    ax.set_ylim(0, 100)
    ax.set_title("B. Effect on factual correction", loc="left", pad=3, fontsize=7.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}" for a in ALPHA])
        ax.set_xlabel(r"Steering strength $\alpha$")
        tidy(ax, "y")

    solid = plt.Line2D([], [], color=INK2, linewidth=1.3)
    dashed = plt.Line2D([], [], color=INK2, linewidth=0.9, linestyle=(0, (3, 2)))
    axes[1].legend([solid, dashed], ["Llama-3.1-8B", "Qwen2.5-7B"], frameon=False,
                   loc="lower left", handlelength=1.6, borderpad=0.2,
                   labelspacing=0.25)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.19,
                        wspace=0.235)
    fig.savefig(OUT / "fig5.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    both, scale, ratio = load()
    fig1(both, ratio)
    fig2(both, scale)
    fig3(both)
    fig4(both)
    fig5()
    for i in range(1, 6):
        p = OUT / f"fig{i}.pdf"
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
