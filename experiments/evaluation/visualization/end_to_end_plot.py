from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import matplotlib as mpl


# TODO: ! Decide later to use end-to-end or compression time only


def plot_trend_with_distribution(csv_files: list[str], labels: list[str], ax=None) -> None:
    if len(csv_files) != len(labels):
        raise ValueError("csv_files and labels must have the same length")
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
            # "font.family": "serif",
            # "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        }
    )
    # plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 8))
    else:
        fig = ax.figure
    fig.subplots_adjust(bottom=0.18)  # increase bottom space
    # colors = sns.color_palette("tab10", len(csv_files))
    # "deep", "deep6", "muted", "muted6", "pastel", "pastel6", "bright", "bright6", "dark", "dark6", "colorblind", "colorblind6"

    # "deep" "muted"
    colors = sns.color_palette("tab10")
    color_dict = {"Selective_Context": colors[0], "Axcer(ours)": colors[2], "LLMLingua2": colors[1]}
    markers = [".", "X", "*"]

    all_medians = []  # collect all models' medians here
    rel_results = {}

    for i, (csv_file, label) in enumerate(zip(csv_files, labels, strict=False)):
        print(f"i: {i}, lable: {label}")
        df = pl.read_csv(csv_file).to_pandas()
        x_col, y_col = "prompt_tokens", "compression_time"

        # binning
        bin_width = 500
        max_tokens = df[x_col].max()
        bins = np.arange(0, max_tokens + bin_width + 1, bin_width)
        inline_bin_labels = [f"{int(bins[j])}" for j in range(len(bins) - 1)]
        print("IN", inline_bin_labels)
        df["token_bin"] = pd.cut(df[x_col], bins=bins, labels=inline_bin_labels, right=False)
        df["token_inline_bin"] = pd.cut(df[x_col], bins=bins, labels=inline_bin_labels, right=False)
        df.dropna(subset=["token_bin"], inplace=True)

        # boxplots
        sns.boxplot(
            data=df,
            x="token_bin",
            y=y_col,
            ax=ax,
            color=colors[i],
            zorder=5,
            showfliers=False,
            boxprops=dict(alpha=0.6),
            width=0.4,
        )

        # median trend line
        median_trend = df.groupby("token_bin", observed=True)[y_col].mean().reset_index()
        # print("FFF", df.groupby("token_bin", observed=True).head(5))
        median_inline_trend = df.groupby("token_inline_bin", observed=True)[y_col].mean().reset_index()
        median_inline_trend["Model"] = label  # <-- add model label
        all_medians.append(median_inline_trend)
        # print("mean is", median_trend)
        global_avg = round(df[y_col].mean(), 3)
        global_med = round(df[y_col].median(), 3)
        rel_results[label] = {
            "Global Avg": global_avg,
            "Global Med": global_med,
        }
        # dfff = compute_from_raw(df)
        print(global_avg, global_med)
        # print(dfff)

        sns.lineplot(
            data=median_trend,
            x="token_bin",
            y=y_col,
            ax=ax,
            color=colors[i],
            label=label,
            marker=markers[i],
            markers=True,
            linewidth=4.5,
            markersize=14,
            zorder=10,
        )

    # Combine all models' medians
    all_medians_df = pd.concat(all_medians, ignore_index=True)

    # --- Inset axes ---
    axins = inset_axes(
        ax,
        width="36%",
        height="30%",
        loc="upper left",
        borderpad=2,
        bbox_to_anchor=(0.03, 0.026, 1, 1),
        bbox_transform=ax.transAxes,
    )

    # Computing rel results for our method
    ours_avg = rel_results["Axcer(ours)"]["Global Avg"]
    ours_med = rel_results["Axcer(ours)"]["Global Med"]

    for baseline, vals in rel_results.items():
        avg_rel = vals["Global Avg"] / ours_avg
        med_rel = vals["Global Med"] / ours_med

        rel_results[baseline]["Avg Rel. (vs ours)"] = avg_rel
        rel_results[baseline]["Med Rel. (vs ours)"] = med_rel

    # Filter only axcer + llama
    subset = all_medians_df[all_medians_df["Model"].isin(["Axcer(ours)", "LLMLingua2"])]
    print("final results is", pd.DataFrame(rel_results).T.round(3))

    for model_name, model_df in subset.groupby("Model"):
        sns.lineplot(
            data=model_df,
            x="token_inline_bin",
            y=y_col,
            color=color_dict[model_name],  # pick color for that model
            marker="X" if model_name == "LLMLingua2" else "*",
            ax=axins,
            linewidth=3.5,
            markersize=12,
            legend=False,
        )

    y_min, y_max = subset[y_col].min(), subset[y_col].max()
    y_margin = (y_max - y_min) * 0.1  # small padding
    axins.set_ylim(y_min - y_margin, y_max + y_margin)
    # axins.yaxis.set_major_locator(plt.MultipleLocator((y_max - y_min) / 6))  # finer ticks
    axins.tick_params(axis="both", labelsize=16)  # smaller font for clarity
    axins.set_ylim(0, 0.35)  # optional zoom range
    axins.set_ylabel("")
    axins.set_xlabel("")

    # main labels
    # ax.set_title("Prompt Tokens vs End-to-End Time with box Plots (lower is better)", fontsize=16)
    ax.set_xlabel("Prompt Tokens", fontsize=19)
    ax.set_ylabel("Latency (s)", fontsize=19)
    ax.tick_params(axis="both", labelsize=18)

    # y_col = "inference_time_avg"
    # df = pl.read_csv(original_destination)
    # inf_time_avg = df[y_col]
    # print("INF", inf_time_avg)
    # ax.axhline(y=inf_time_avg[0], color="#00bfa0", linestyle="--", linewidth=1.5, label="Original")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axins.set_xticklabels(axins.get_xticklabels(), rotation=45, ha="right")
    axins.grid(True, which="both", linestyle="--", linewidth=0.5)

    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    # legend = ax.legend(
    # l1 + l2,
    # lab1 + lab2,
    # loc="upper center",   # ✅ anchor legend to the center bottom
    # bbox_to_anchor=(0.5, -0.15),  # ✅ (x=0.5 means centered, y=-0.15 puts it below)
    # fontsize=8.0,
    # handlelength=1.4,
    # handletextpad=0.4,
    # labelspacing=0.4,
    # borderpad=0.4,
    # frameon=True,
    # ncol=4,  # ✅ optional: spread items into 2 columns for balance
    # )

    # ax.legend(
    #     title="Model",
    #     loc=8,
    #     ncol=3,
    #     bbox_to_anchor=(0.5, -0.24),
    #     fontsize=10.0,
    #     handlelength=1.4,
    #     handletextpad=0.4,
    #     labelspacing=0.4,
    #     borderpad=0.4,
    #     frameon=True,  # show frame
    #     # facecolor="whitesmoke",  # background color
    #     # edgecolor="gray",  # border color)
    #     framealpha=0.9,
    # )  # transparency))
    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42
    # plt.rcParams.update({
    #     "font.size": 14,             # small but readable in print
    #     "axes.titlesize": 14,
    #     "axes.labelsize": 14,
    #     "xtick.labelsize": 12,
    #     "ytick.labelsize": 12,
    #     "legend.fontsize": 12,
    # })

    # plt.tight_layout(pad=1.2)
    # plt.savefig("figure_e2e.pdf", bbox_inches="tight")
    # plt.show()
    return ax


plot_trend_with_distribution(
    csv_files=[
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/llama/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/selective_context/all_merged.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/gemma/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/lingua/all_merged.csv",
        # "/home/itz-amethyst/dev/axcer/experiments/results/temp/concated_given_columns.csv",
        "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/axcer/all_merged.csv",
    ],
    # we have the concated resutls find a way to do avg on it (we might not need the orig line here ! or if you wanted to have it mention in the legend that it's for the inference time not compression)
    # original_destination="/home/itz-amethyst/dev/axcer/experiments/results/original/concated_models_results/avg/concated_given_columns.csv",
    labels=["Selective_Context", "LLMLingua2", "Axcer(ours)"],
    # labels=["gemma", "axcer"]
)

# acutally this one should be only for compression time and create another copy of this with end to end time which is going to be on main page (this one goes into appendix i assume)
