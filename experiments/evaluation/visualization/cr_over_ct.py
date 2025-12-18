import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_trend_with_distribution(csv_files: list[str], labels: list[str]) -> None:
    if len(csv_files) != len(labels):
        raise ValueError("csv_files and labels must have the same length")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 8))
    fig.subplots_adjust(bottom=0.18)

    colors = sns.color_palette("tab10")
    color_dict = {
        "Selective_Context": colors[0],
        "Axcer(ours)": colors[2],
        "LLMLingua2": colors[1],
    }
    markers = ["o", "^", "v"]

    all_medians = []

    for i, (csv_file, label) in enumerate(zip(csv_files, labels, strict=False)):
        print(f"i: {i}, label: {label}")
        df = pl.read_csv(csv_file).to_pandas()
        x_col, y_col = "compression_ratio_normalized", "compression_time"

        if x_col not in df.columns:
            raise KeyError(f"Column '{x_col}' not found in {csv_file}")

        # Define bins for normalized ratios
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_labels = [f"{bins[j]:.1f}-{bins[j + 1]:.1f}" for j in range(n_bins)]
        inline_bin_labels = [f"{bins[j + 1]:.1f}" for j in range(n_bins)]

        df["ratio_bin"] = pd.cut(df[x_col], bins=bins, labels=bin_labels, include_lowest=True)
        df["ratio_inline_bin"] = pd.cut(df[x_col], bins=bins, labels=inline_bin_labels, include_lowest=True)
        df.dropna(subset=["ratio_bin"], inplace=True)

        # --- Boxplots per bin ---
        sns.boxplot(
            data=df,
            x="ratio_bin",
            y=y_col,
            ax=ax,
            color=colors[i],
            zorder=5,
            showfliers=False,
            boxprops=dict(alpha=0.6),
            width=0.4,
        )

        # --- Mean trend line ---
        mean_trend = df.groupby("ratio_bin", observed=True)[y_col].mean().reset_index()
        mean_inline_trend = df.groupby("ratio_inline_bin", observed=True)[y_col].mean().reset_index()
        mean_inline_trend["Model"] = label
        all_medians.append(mean_inline_trend)
        print(mean_trend)

        sns.lineplot(
            data=mean_trend,
            x="ratio_bin",
            y=y_col,
            ax=ax,
            color=colors[i],
            label=label,
            marker=markers[i],
            linewidth=2.5,
            zorder=10,
        )

    # Combine all model means
    all_medians_df = pd.concat(all_medians, ignore_index=True)

    # --- Inset axes for zoomed comparison ---
    axins = inset_axes(
        ax,
        width="36%",
        height="30%",
        loc="upper left",
        borderpad=2,
        bbox_to_anchor=(0.03, 0.026, 1, 1),
        bbox_transform=ax.transAxes,
    )

    subset = all_medians_df[all_medians_df["Model"].isin(["Axcer(ours)", "LLMLingua2"])]

    for model_name, model_df in subset.groupby("Model"):
        sns.lineplot(
            data=model_df,
            x="ratio_inline_bin",
            y=y_col,
            color=color_dict[model_name],
            marker="o",
            ax=axins,
            legend=False,
        )

    # 🔹 Enhanced y-axis detail for inset
    y_min, y_max = subset[y_col].min(), subset[y_col].max()
    y_margin = (y_max - y_min) * 0.1  # small padding
    axins.set_ylim(y_min - y_margin, y_max + y_margin)
    axins.yaxis.set_major_locator(plt.MultipleLocator((y_max - y_min) / 6))  # finer ticks
    axins.tick_params(axis="y", labelsize=9)  # smaller font for clarity

    axins.set_ylabel("")
    axins.set_xlabel("")
    axins.set_xticklabels(axins.get_xticklabels(), rotation=45, ha="right")
    axins.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Main axis formatting ---
    ax.set_xlabel("Compression Ratio (Normalized)", fontsize=14)
    ax.set_ylabel("Compression Time (s)", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Legend ---
    ax.legend(
        title="Model",
        loc=8,
        ncol=3,
        bbox_to_anchor=(0.5, -0.24),
        fontsize=10.0,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.4,
        borderpad=0.4,
        frameon=True,
        framealpha=0.9,
    )

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    plt.savefig("figure_e2e.pdf", bbox_inches="tight")
    plt.show()


plot_trend_with_distribution(
    csv_files=[
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/llama/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/all_merged.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/gemma/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/lingua/all_merged.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/all_merged.csv",
    ],
    # we have the concated resutls find a way to do avg on it (we might not need the orig line here ! or if you wanted to have it mention in the legend that it's for the inference time not compression)
    # original_destination="/home/itz-amethyst/dev/axcer/experiments/results/original/concated_models_results/avg/concated_given_columns.csv",
    labels=["Selective_Context", "LLMLingua2", "Axcer(ours)"],
    # labels=["gemma", "axcer"]
)

# acutally this one should be only for compression time and create another copy of this with end to end time which is going to be on main page (this one goes into appendix i assume)
