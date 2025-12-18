import glob
from matplotlib.lines import Line2D
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

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
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["hatch.linewidth"] = 0.5  # Make hatch lines visible in PDF
mpl.rcParams["hatch.color"] = "black"  # Ensure hatch renders
category_names = {"lingua2": "LLMLingua2", "selective_context": "Selective Context", "axcer": "Axcer (ours)"}


def plot_lollipop_chart(ax, categories, data_list, colors, model_labels, title, y_min, y_max):
    """
    Creates a lollipop chart with multiple baselines with slight horizontal offset and smooth trend lines.

    Parameters:
    - ax: matplotlib axis
    - categories: list of category names (x-axis labels)
    - data_list: list of lists, each inner list contains scores for one baseline
    - colors: list of colors for each baseline
    - model_labels: list of baseline names
    - title: chart title
    - y_min, y_max: y-axis range
    """

    x_pos = np.arange(len(categories))  # positions for categories
    n_models = len(data_list)

    # Create small offsets for each baseline to separate them visually
    offsets = np.linspace(-0.15, 0.15, n_models)

    # Plot lollipops for each baseline using the exact structure from original
    for _, (data, color, offset) in enumerate(zip(data_list, colors, offsets, strict=False)):
        # Draw smooth trend line connecting the points
        x_smooth = np.linspace(x_pos[0] + offset, x_pos[-1] + offset, 300)

        # Create smooth spline interpolation
        if len(x_pos) > 3:  # Need at least 4 points for cubic spline
            spl = make_interp_spline(x_pos + offset, data, k=3)
            y_smooth = spl(x_smooth)
        else:  # Use quadratic for fewer points
            spl = make_interp_spline(x_pos + offset, data, k=min(2, len(x_pos) - 1))
            y_smooth = spl(x_smooth)

        # Fill area under the curve with hatched pattern
        ax.fill_between(x_smooth, 0, y_smooth, color=color, alpha=0.15, linewidth=0, zorder=0)

        # Add hatch pattern separately with explicit settings
        ax.fill_between(x_smooth, 0, y_smooth, facecolor="none", hatch="////", edgecolor=color, linewidth=1.0, zorder=0)
        # Plot the smooth trend line
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.7, zorder=3)

        # Draw lollipop stems and circles
        for i in x_pos:
            # Draw stem (vertical line from 0 to value) with slight horizontal offset
            ax.plot([i + offset, i + offset], [0, data[i]], color=color, linewidth=2, zorder=4)
            # Draw circle at the top
            ax.scatter(i + offset, data[i], color=color, s=50, zorder=5)

    # Configure x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=30, ha="right")

    # Configure y-axis
    ax.set_ylim([0, y_max])
    ax.set_ylabel("Perplexity Score (Lower is better)", fontsize=14)

    # Set title
    ax.set_title(title, pad=25, weight="bold", fontsize=16)

    # Add grid
    ax.grid(axis="y", color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create custom legend

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markersize=10, label=model_labels[i])
        for i in range(len(model_labels))
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=15)


def extract_label_from_path(path_string):
    try:
        p = Path(path_string)
        parts = p.resolve().parts
        results_index = parts.index("results")
        if results_index + 1 < len(parts):
            return parts[results_index + 1]
        else:
            return p.name
    except ValueError:
        print(f"Warning: 'results' not found in path {path_string}. Using directory name.")
        return Path(path_string).name
    except Exception:
        return Path(path_string).name


def load_chart_data(base_paths_list):
    data_list = []
    model_labels = []
    categories = None

    for base_path in base_paths_list:
        label = extract_label_from_path(base_path)
        label = category_names.get(label, label)
        model_labels.append(label)

        csv_files = sorted(glob.glob(f"{base_path}/*.csv"))
        current_model_data = []
        current_categories = []

        for csv_file in csv_files:
            category = Path(csv_file).stem
            current_categories.append(category)

            try:
                df = pd.read_csv(csv_file)
                if "perplexity_score" in df.columns:
                    score = df["perplexity_score"].mean()
                else:
                    print(f"Warning: no 'perplexity_score' in {csv_file}; using first column.")
                    score = df.iloc[0, 0]
            except Exception:
                score = 0

            current_model_data.append(score)

        data_list.append(current_model_data)

        if categories is None:
            categories = current_categories

    print("CT", categories)
    categories = [figure_titles.get(item, item) for item in categories.__iter__()]
    return categories, data_list, model_labels


def main(figure1_paths, figure2_paths):
    # Load data for both figures
    categories_left, data_left, model_labels_left = load_chart_data(figure1_paths)
    categories_right, data_right, model_labels_right = load_chart_data(figure2_paths)

    if not categories_left or not categories_right:
        print("Error: no data loaded.")
        return

    # Define colors for the 3 baselines
    # colors = ["#FF6B35", "#4ECDC4", "#95E1D3"]  # Orange, Teal, Mint
    colors = ["#007AC5", "#CF72BD", "#CFBF6E"]

    # good
    # colors = ["#007AC5", "#FF8C42", "#5D3FD3"]
    colors = ["#2E7D32", "#FF8C42", "#5D3FD3"]

    # colors = ["#007AC5", "#FF8C42", "#2E7D32"]

    # colors = ["#007AC5", "#007F7F", "#C34271"]
    # colors = ["#B8E4C9", "#FF8C42", "#5D3FD3"]

    # Calculate y-axis ranges
    left_all_vals = np.concatenate(data_left)
    right_all_vals = np.concatenate(data_right)

    y_min_left = max(0, left_all_vals.min() - 5)
    y_max_left = left_all_vals.max() + 5

    y_min_right = max(0, right_all_vals.min() - 5)
    y_max_right = right_all_vals.max() + 5

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 6))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.grid(axis="y", color="#e0e0e0", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.set_axisbelow(True)
    # for spine in ['top', 'right', 'left', 'bottom']:
    for spine in [
        "top",
        "right",
        "left",
    ]:
        # ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # LEFT subplot – Llama-3.1-8B
    plot_lollipop_chart(
        ax=ax1,
        categories=categories_left,
        data_list=data_left,
        colors=colors,
        model_labels=model_labels_left,
        title="(a) Llama-3.1 8B-Instruct",
        y_min=y_min_left,
        y_max=y_max_left,
    )

    # RIGHT subplot – Gemma-3-12B
    plot_lollipop_chart(
        ax=ax2,
        categories=categories_right,
        data_list=data_right,
        colors=colors,
        model_labels=model_labels_right,
        title="(b) Gemma3 12B-it",
        y_min=y_min_right,
        y_max=y_max_right,
    )

    # Shared legend at the bottom
    ax1.tick_params(axis="both", labelsize=14)  # smaller font for clarity
    ax2.tick_params(axis="both", labelsize=14)  # smaller font for clarity
    handles, labels = ax1.get_legend_handles_labels()

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig("/home/itz-amethyst/dev/vanish/axcer/latex/figures/perplexity_lolipop.pdf", bbox_inches="tight", dpi=2000)
    plt.show()
    print("✓ Lollipop chart saved as 'lollipop_comparison.pdf'")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    figure1_paths_to_data = [
        "/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/perplexities/Meta-Llama-3.1-8B-Instruct",
        "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/perplexities/Meta-Llama-3.1-8B-Instruct",
        "/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/Meta-Llama-3.1-8B-Instruct",
    ]

    figure2_paths_to_data = [
        "/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/perplexities/gemma-3-12b-it/",
        "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/perplexities/gemma-3-12b-it/",
        "/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
    ]

    main(figure1_paths_to_data, figure2_paths_to_data)
