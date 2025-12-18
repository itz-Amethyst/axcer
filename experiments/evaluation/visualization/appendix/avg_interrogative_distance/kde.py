import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def conditional_kde_plot(base_dir: str, column_name: str = "avg_distance_interrogative_words"):
    sns.set_theme(style="whitegrid")

    # --- Load CSVs ---
    base_path = Path(base_dir)
    csv_files = list(base_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base_dir}")

    # --- Collect values ---
    datasets = {}
    for file in csv_files:
        df = pd.read_csv(file)
        if column_name not in df.columns:
            print(f"⚠️ Skipping {file.name}, column '{column_name}' not found")
            continue

        values = df[column_name].dropna()
        values = values[(values >= 1) & (values <= 4)]
        if len(values) > 0:
            datasets[file.stem] = values

    if not datasets:
        raise ValueError(f"No valid '{column_name}' values in any CSV file")

    # --- Plot all KDEs on ONE figure ---
    plt.figure(figsize=(8, 6))

    colors = sns.color_palette("husl", len(datasets))  # distinct colors

    for (name, values), color in zip(datasets.items(), colors, strict=False):
        sns.kdeplot(values, fill=True, bw_method=1, label=name, color=color)

    # --- Force x-axis to be EXACTLY 0 to 5 ---
    plt.xlim(0, 5)

    # --- Labels & title ---
    plt.ylabel("Density", fontsize=15)
    plt.xlabel("Avg Interrogative Distance", fontsize=15)
    plt.legend(title="Dataset", fontsize=12)

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 19,
            "ytick.labelsize": 19,
            "xtick.labelsize": 19,
            "legend.fontsize": 17,
        }
    )
    # --- Save as PDF ---
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(
        "/home/itz-amethyst/dev/vanish/elsarticle/figures/testing/conditional_kde.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1000,
    )

    plt.show()


conditional_kde_plot("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/")
