import matplotlib.pyplot as plt

from experiments.evaluation.visualization.end_to_end_llm import plot_trend_with_distribution as pl2
from experiments.evaluation.visualization.end_to_end_plot import plot_trend_with_distribution

# Suppose we have three sets of csv files + labels
csv_sets = [
    [
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/selective_context/all_merged.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/lingua/all_merged.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/axcer/all_merged.csv",
    ],
    [
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/gemma/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/lingua/gemma/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/gemma/concated_given_columns.csv",
    ],
    [
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/llama/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/lingua/llama/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/llama/concated_given_columns.csv",
    ],
]
original_path_sets = [
    "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/gemma/concated_given_columns.csv",
    "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/gemma/concated_given_columns.csv",
    "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/llama/concated_given_columns.csv",
]
labels = ["Selective_Context", "LLMLingua2", "Axcer(ours)"]

# --- create wide figure with 3 columns ---
fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)

# --- plot each subplot ---
for i, (ax, csv_files, orig_path) in enumerate(zip(axes, csv_sets, original_path_sets, strict=False)):
    if i == 0:
        plot_trend_with_distribution(csv_files, labels, ax=ax)
    else:
        pl2(csv_files, labels, ax=ax, original_destination=orig_path)

    # --- optional subplot titles ---
    ax.set_title(
        [
            "(a) Compression latency across baselines",
            "(b) End-to-end time of baselines on Gemma 12B-It",
            "(c) End-to-end time of baselines on Llama 3.1 8B-Instruct",
        ][i],
        fontsize=23,
        pad=10,
    )


handles, legend_labels = axes[1].get_legend_handles_labels()
for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()

# --- add single shared legend ---
fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    ncol=4,
    fontsize=24,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

# --- global x-axis label ---
# fig.text(0.5, 0.02, "Compression Ratio (Normalized)", ha="center", fontsize=20)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("combined_figure.pdf", bbox_inches="tight", dpi=1000)
fig.savefig("/home/itz-amethyst/dev/vanish/elsarticle/figures/testing/combined_figure.pdf", bbox_inches="tight", dpi=1000)
plt.show()
