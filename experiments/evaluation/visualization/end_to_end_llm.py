import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_trend_with_distribution(csv_files: list[str], labels: list[str], original_destination, ax=None) -> None:
    if len(csv_files) != len(labels):
        raise ValueError("csv_files and labels must have the same length")

    # plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,  # keep text selectable in PDF
            "ps.fonttype": 42,
            "font.size": 25,  # larger base font so it's readable after combining
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
            "figure.titlesize": 22,
        }
    )
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 8))
    else:
        fig = ax.figure
    # fig, ax = plt.subplots(figsize=(7.5, 8))
    fig.subplots_adjust(bottom=0.18)

    colors = sns.color_palette("tab10")
    markers = [".", "X", "*"]

    all_means = []  # Collect all models' mean stats here

    for i, (csv_file, label) in enumerate(zip(csv_files, labels, strict=False)):
        print(f"i: {i}, label: {label}")
        df = pl.read_csv(csv_file).to_pandas()
        x_col, y_col = "compression_ratio_normalized", "end_to_end_time"

        if x_col not in df.columns:
            raise KeyError(f"Column '{x_col}' not found in {csv_file}")

        # --- Define 10 equal bins across [0, 1] ---
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_labels = [round(bins[j + 1], 1) for j in range(n_bins)]  # → 0.1, 0.2, ..., 1.0
        print("BB", bin_labels)

        # Use numeric labels for ratio_bin
        df["ratio_bin"] = pd.cut(df[x_col], bins=bins, labels=bin_labels, include_lowest=True)
        df["ratio_bin"] = df["ratio_bin"].astype(float)
        df.dropna(subset=["ratio_bin"], inplace=True)

        stats = df.groupby("ratio_bin", observed=True)[y_col].agg(["mean", "std", "count"]).reset_index()

        stats["Model"] = label
        all_means.append(stats)

        sns.lineplot(
            data=stats,
            x="ratio_bin",
            y="mean",
            ax=ax,
            color=colors[i],
            label=label,
            marker=markers[i],
            linewidth=4.5,
            markersize=14,
            zorder=10,
        )

    ax.set_xlabel("Normalized Compression Ratio", fontsize=19)
    ax.set_ylabel("End-to-End Time (s)", fontsize=19)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xticks([0.1 * i for i in range(1, 11)])  # 0.1, 0.2, …, 1.0

    y_col = "inference_time"
    df = pl.read_csv(original_destination)
    inf_time_avg = df[y_col].mean()
    print("INF", inf_time_avg)
    ax.axhline(y=inf_time_avg, color="#663399", linestyle="--", linewidth=2.5, label="Original")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    return ax
    # plt.savefig("figure_e2e_compression_normalized.pdf", bbox_inches="tight")
    # plt.show()


plot_trend_with_distribution(
    csv_files=[
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/llama/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/gemma//concated_given_columns.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/gemma/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/lingua/gemma/concated_given_columns.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/gemma/concated_given_columns.csv",
    ],
    original_destination="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/gemma/concated_given_columns.csv",
    labels=["Selective_Context", "LLMLingua2", "Axcer(ours)"],
    # labels=["gemma", "axcer"]
)
