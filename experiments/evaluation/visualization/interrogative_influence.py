import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from experiments.constants.paths import (
    AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    AVG_CONCATED_WITHOUT_INTERROGATIVE_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
)


def style_academic(ax, fig=None):
    # Preferred academic serif font

    # Axis labels
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)

    # Tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10.5)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(9.5)

    # Legend, if exists
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(10)


def plot_interrogative_vs_non(with_template, without_template, models, categories):
    data = []

    normalized_model_names = {"Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B-Instruct", "gemma-3-12b-it": "gemma3 12b-it"}

    # --- Data loading loop (unchanged) ---
    for category in categories:
        for model in models:
            w_file = str(with_template).format(model_name=model, category_name=category)
            wo_file = str(without_template).format(model_name=model, category_name=category)
            model_name = normalized_model_names[model]

            try:
                df = pd.read_csv(w_file)
                data.append(
                    {
                        "Model": model_name,
                        "Condition": "w/ Interrogative",
                        "Category": category,
                        "CompressionRatio": df["compression_ratio_avg"].iloc[0],
                        "Accuracy": df["exact_match_avg"].iloc[0],
                    }
                )
            except FileNotFoundError:
                pass

            try:
                df = pd.read_csv(wo_file)
                data.append(
                    {
                        "Model": model_name,
                        "Condition": "w/o Interrogative",
                        "Category": category,
                        "CompressionRatio": df["compression_ratio_avg"].iloc[0],
                        "Accuracy": df["exact_match_avg"].iloc[0],
                    }
                )
            except FileNotFoundError:
                pass

    results = pd.DataFrame(data)

    palette = {
        ("Llama 3.1 8B-Instruct", "w/ Interrogative"): "#ffa300",
        ("Llama 3.1 8B-Instruct", "w/o Interrogative"): "#b3d4ff",
        ("gemma3 12b-it", "w/ Interrogative"): "#ea5545",
        ("gemma3 12b-it", "w/o Interrogative"): "#9b19f5",
    }

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(6.4, 4))

    # === Offsets for grouped bars ===
    offset_step = 0.025
    grouped = results.groupby("CompressionRatio")

    for comp_ratio, group in grouped:
        n = len(group)
        offsets = (np.arange(n) - (n - 1) / 2.0) * offset_step
        group_sorted = group.sort_values(["Model", "Condition"]).reset_index(drop=True)

        for i, (_, row) in enumerate(group_sorted.iterrows()):
            key = (row["Model"], row["Condition"])
            x_pos = comp_ratio + offsets[i]
            label = f"{row['Model']} ({row['Condition']})"
            already_added = label in plt.gca().get_legend_handles_labels()[1]

            plt.bar(
                x_pos,
                row["Accuracy"],
                color=palette.get(key, "gray"),
                width=offset_step * 0.95,
                label=label if not already_added else "",
                edgecolor="none",
            )

    # Center tick labels
    base_x = np.array(sorted(results["CompressionRatio"].unique()))
    plt.xticks(base_x, [f"{x:.2f}X" for x in base_x])

    max_acc = results["Accuracy"].max()
    print(max_acc)
    plt.axhline(max_acc, linestyle="--", color="gray")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    # === Labels (academic sizes already applied from rcParams) ===
    # plt.title(f"Accuracy vs Compression Ratio ({categories[0].upper()})")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Compression Ratio (X)")

    # === Legend positioning (slightly lowered) ===
    plt.subplots_adjust(bottom=0.85)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=2,  # 3 items per row
        frameon=True,
    )

    style_academic(ax, fig)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "ytick.labelsize": 10,
            "xtick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    plt.tight_layout()
    plt.savefig("ablation_study_figure.pdf", bbox_inches="tight")
    plt.show()


plot_interrogative_vs_non(
    with_template=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    without_template=AVG_CONCATED_WITHOUT_INTERROGATIVE_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    models=["Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"],
    categories=["Qa"],
)
