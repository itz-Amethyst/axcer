import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

figure_titles = {
    "scitldr": "SciTLDR",  # rouge-l,
    "mbpp": "MBPP",
    "gsm8k": "GSM8K",
    "mawps": "MAWPS",
    "squad": "SQuAD",
    "piqa": "PIQA",
    "glue": "mrpc",
    "boolq": "boolq",
    "coqa": "exact_match",
    "ai2_arc": "AI2_ARC",
}


def plot_experiments(base_path, experiment_list):
    """
    For each experiment in experiment_list, navigate to base_path/<experiment>/concated_models_results,
    read CSVs, and plot multiple lines (one per experiment) on the same subplot.
    """
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,  # keep text selectable in PDF
            "ps.fonttype": 42,
            "font.size": 18,  # larger base font so it's readable after combining
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 21,
            "legend.fontsize": 26,
            "figure.titlesize": 42,
            # "font.family": "serif",
            # "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        }
    )
    # Collect all files across experiments
    markers = {"selective_context": "o", "lingua2": "X", "axcer": "s"}
    lables = {"selective_context": "Selective_Context", "lingua2": "LLMLingua2", "axcer": "Axcer(ours)"}
    y_columns = {
        "scitldr": "rouge-1",  # rouge-l,
        "mbpp": "pass@1",
        "gsm8k": "exact_match",
        "mawps": "exact_match",
        "squad": "exact_match",
        "piqa": "exact_match",
        "glue": "exact_match",
        "boolq": "exact_match",
        "coqa": "exact_match",
        "ai2_arc": "exact_match",
    }

    bin_dict = {
        "scitldr": 5,
        "mbpp": 5,
        "gsm8k": 5,
        "mawps": 5,
        "squad": 5,
        "piqa": 5,
        "glue": 5,
        "coqa": 5,
        "boolq": 5,
        "ai2_arc": 5,
    }

    files_by_exp = {}
    for exp in experiment_list:
        exp_path = Path(base_path) / exp / "concated_models_results"
        if not exp_path.exists():
            print(f"Warning: path not found {exp_path}")
            continue
        files = glob.glob(str(exp_path / "*.csv"))
        files_by_exp[exp] = files

    # Flatten unique stems across all experiments (so we create one subplot per model)
    unique_stems = sorted({Path(f).stem for files in files_by_exp.values() for f in files})

    # Subplot grid
    n_models = len(unique_stems)
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows)
    )  # second one after , is for height first is for width
    axes = axes.flatten()
    # colors = {"test": "#e60049", "lingua2": "#0bb4ff", "axcer": "#ffa300"}
    # colors = {"test": "#b30000", "lingua2": "#1a53ff", "axcer": "#ebdc78"}
    colors = sns.color_palette("tab20")
    # print(sns.color_palette(""))
    # colors = {"selective_context": "#fd7f6f", "lingua2": "#1a53ff", "axcer": "#ebdc78"}
    colors = {"selective_context": colors[0], "lingua2": colors[2], "axcer": colors[8]}

    handles, labels = None, None  # for global legend

    # Iterate over unique model stems → one subplot per stem
    for i, stem in enumerate(unique_stems):
        print("i is", i)
        ax = axes[i]

        for exp in experiment_list:
            files = files_by_exp.get(exp, [])
            # Find matching file for this model stem in this experiment
            file = next((f for f in files if Path(f).stem == stem), None)
            if file is None:
                continue  # this model not present in this experiment

            y_col = y_columns.get(stem)
            if y_col is None:
                print(f"Warning: no y_col defined for {stem}, skipping")
                continue

            bins = bin_dict.get(stem, 5)

            print("F IS", file)
            df = pd.read_csv(file)
            print("exp", exp)
            exp = exp.replace("/with_interrogative", "") if "/with_interrogative" in exp else exp
            if exp != "original":
                df["ratio_bin"] = pd.cut(df["compression_ratio_normalized"], bins=bins)

                # Compute metric
                metric_per_bin = df.groupby("ratio_bin")[y_col].mean().reset_index()
                metric_per_bin["value"] = (
                    metric_per_bin[y_col] * 100 if y_col not in ["rouge-1", "pass@1"] else metric_per_bin[y_col]
                )
                metric_per_bin["midpoint"] = metric_per_bin["ratio_bin"].apply(lambda x: x.mid)
            else:
                y_col_mean = df[y_col] * 100 if y_col not in ["rouge-1", "pass@1"] else df[y_col]
                y_col_mean = y_col_mean.mean()

                print(y_col_mean)
                # metric_per_bin['value'] = metric_per_bin[y_col] * 100

            # Plot a line for this experiment
            if exp == "original":
                print("entered")
                ax.axhline(y=y_col_mean, color="#1E9D1F", linestyle="--", linewidth=2.5, label="Original")
            else:
                color = colors.get(exp, "#ffff")
                sns.lineplot(
                    data=metric_per_bin,
                    x="midpoint",
                    y="value",
                    # marker="o",
                    ax=ax,
                    color=color,
                    marker=markers.get(exp, "."),
                    markers=True,
                    label=lables.get(exp),
                    markersize=14,
                    linewidth=3.5,
                )

        # Labels
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        fg_title = figure_titles.get(stem)
        ax.set_title(f"{fg_title}")
        ax.set_xlabel("Compression Ratio (Normalized)")
        if y_col == "exact_match":
            axes[i].set_ylabel("Accuracy (%)")
        else:
            # not sure about the %
            axes[i].set_ylabel(f"{y_col} (%)")
        ax.grid(linestyle="-", alpha=0.8)
        legend = ax.legend()
        legend.remove()
        # ax.legend(title="Experiment")

    # Hide unused subplot slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=len(experiment_list), bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.09, 1, 1])  # leave space at bottom (5%)
    fig.savefig("/home/itz-amethyst/dev/vanish/elsarticle/figures/testing/appendix_figure.pdf", bbox_inches="tight", dpi=1200)
    plt.show()


base_path = "/home/itz-amethyst/dev/axcer/experiments/results"
experiment_list = ["lingua2", "original", "axcer/with_interrogative", "selective_context"]


plot_experiments(base_path, experiment_list)
