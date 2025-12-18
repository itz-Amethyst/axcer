import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def ridge_plot(base_dir: str, column_name: str = "avg_distance_interrogative_words"):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    base_path = Path(base_dir)
    csv_files = list(base_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base_dir}")

    # Collect data from all files
    records = []
    for file in csv_files:
        df = pd.read_csv(file)
        if column_name not in df.columns:
            print(f"⚠️ Skipping {file.name}, column '{column_name}' not found")
            continue
        values = df[column_name].dropna()
        # keep only [1,4]
        values = values[(values >= 1) & (values <= 4)]
        for v in values:
            records.append({"value": v, "file": file.stem})

    if not records:
        raise ValueError(f"No valid data found in column '{column_name}' across CSVs")

    data = pd.DataFrame(records)

    # Choose palette
    pal = sns.cubehelix_palette(len(data["file"].unique()), rot=-0.25, light=0.7)

    # FacetGrid for ridge plot
    g = sns.FacetGrid(data, row="file", hue="file", aspect=15, height=0.5, palette=pal)

    # Density plots
    g.map(sns.kdeplot, "value", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # Reference line
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Add labels
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "value")

    # Adjust overlaps
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove extras
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.xlabel("Value (1.0 – 4.0)")
    plt.show()


ridge_plot("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/")
