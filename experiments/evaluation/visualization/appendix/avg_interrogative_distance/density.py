import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def conditional_kde_plot(base_dir: str, column_name: str = "avg_distance_interrogative_words"):
    sns.set_theme(style="whitegrid")

    base_path = Path(base_dir)
    csv_files = list(base_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base_dir}")

    # Collect all data
    records = []
    for file in csv_files:
        df = pd.read_csv(file)
        if column_name not in df.columns:
            print(f"⚠️ Skipping {file.name}, column '{column_name}' not found")
            continue
        values = df[column_name].dropna()
        values = values[(values >= 1) & (values <= 4)]
        for v in values:
            records.append({"value": v, "file": file.stem})

    if not records:
        raise ValueError(f"No valid data found in column '{column_name}' across CSVs")

    data = pd.DataFrame(records)

    # Conditional KDE plot
    g = sns.displot(
        data=data,
        x="value",
        hue="file",
        kind="kde",
        height=6,
        multiple="fill",
        clip=(1, 4),
        palette="ch:rot=-.25,hue=1,light=.75",
    )

    # Set titles/labels
    g.set_axis_labels("Value (1.0 – 4.0)", "Proportion")
    g.fig.suptitle(f"Conditional KDE by File ({column_name})", fontsize=14)

    # --- NEW: Save as PDF ---

    g.figure.savefig("conditional_kde.pdf", format="pdf", bbox_inches="tight")

    plt.show()
    # plt.close()


conditional_kde_plot("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/")
