import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statistics import mean

root = Path("/root/Gnomon/benchmarks/ledger_optimization/evidence")
a = json.loads((root / "confirmation-agent-010-analysis.json").read_text())
r = json.loads((root / "confirmation-agent-010-robustness.json").read_text())
if a["cohort"]["decisions"] != 3744:
    raise ValueError("Full confirmation required")
arms = ["no_ledger", "ledger_119", "ledger_supported"]
labels = ["Raw-history control", "Original MAE cards", "Historical support"]
colors = ["#536171", "#b88639", "#177c82"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.7]})
values = [a["mean_case_rmsle"][k] for k in arms]
axs[0].bar(range(3), values, color=colors, width=0.62)
axs[0].set_xticks(
    range(3), ["Raw-history\ncontrol", "Original\nMAE cards", "Historical\nsupport"]
)
axs[0].set_ylabel("Mean per-case RMSLE (lower is better)")
axs[0].set_title("All 1,248 matched case/seed pairs")
for i, v in enumerate(values):
    axs[0].text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=10)
axs[0].set_ylim(0, max(values) * 1.18)
axs[0].axhline(
    0.8 * values[0], color="#9b3333", linestyle="--", lw=1, label="20% reduction target"
)
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
origins = sorted(map(int, r["per_origin"]))
for arm, label, color in zip(arms, labels, colors):
    vals = [r["per_origin"][str(i)][arm] for i in origins]
    cumulative = [mean(vals[: i + 1]) for i in range(len(vals))]
    axs[1].plot([i + 1 for i in origins], cumulative, label=label, color=color, lw=2)
axs[1].set_xlabel("Origin reached (all earlier origins retained)")
axs[1].set_ylabel("Cumulative mean per-case RMSLE")
axs[1].set_title("Accumulating history, unchanged forecast tools")
axs[1].legend(frameon=False, fontsize=9)
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
fig.suptitle(
    "Frozen ledger confirmation: 24 series × 26 origins × 2 seeds", fontsize=14
)
fig.text(
    0.02,
    0.015,
    "Identical raw historical evidence in every arm. Calendar conditions also change across origins; this is not a causal learning-curve estimate.",
    fontsize=8,
    color="#505050",
)
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(root / "confirmation-agent-010.png", dpi=170)
fig.savefig(root / "confirmation-agent-010.svg")
