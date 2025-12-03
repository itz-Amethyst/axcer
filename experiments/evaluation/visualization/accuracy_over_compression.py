import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import glob


def get_filtered_csv_files(directory: str, exclude_datasets: list[str] | None):
    files = sorted(glob.glob(str(Path(directory) / "*.csv")))
    if not exclude_datasets:
        return files
    exclude_set = set(exclude_datasets)
    return [f for f in files if Path(f).stem not in exclude_set]


def plot_results_with_em(
    compressed_dir: str,
    original_dir: str,
    n_bins: int = 10,
    exclude_datasets: list[str] | None = None,
    compression_col: str = "compression_ratio",
    compressed_tokens_col: str = "compressed_tokens",
    tokens_col: str = "prompt_tokens",
    em_col: str = "exact_match",
):
    comp_files = get_filtered_csv_files(compressed_dir, exclude_datasets)

    print(comp_files)
    if not comp_files:
        raise ValueError("No valid compressed CSV files after filtering.")

    comp_dfs = []
    for f in comp_files:
        try:
            comp_dfs.append(pl.read_csv(f, columns=[compression_col, tokens_col, compressed_tokens_col, em_col]))
        except Exception as e:
            print(f"[skip compressed file] {f}: {e}")
            continue

    if not comp_dfs:
        raise RuntimeError("No compressed data loaded successfully.")

    df_c = pl.concat(comp_dfs, how="vertical")
    print(df_c.head(5))
    print(df_c)
    print(df_c.sort("compression_ratio", descending=True).head(100))
    # print(df_c.head(5))

    for c in (compression_col, tokens_col, compressed_tokens_col, em_col):
        if c not in df_c.columns:
            raise RuntimeError(f"Required column '{c}' missing in compressed data.")

    df_c = df_c.with_columns(pl.col(em_col).cast(pl.Float64))

    if df_c.height == 0:
        raise RuntimeError("No valid rows in compressed dataset after filtering.")

    comp_vals = np.array(df_c[compression_col].to_list(), dtype=float)
    comp_min, comp_max = float(np.nanmin(comp_vals)), float(np.nanmax(comp_vals))
    bin_edges = np.linspace(comp_min, comp_max, n_bins + 1)

    bin_idx_c = np.digitize(comp_vals, bins=bin_edges[1:-1], right=True).astype(np.int32)
    df_c = df_c.with_columns(pl.Series("comp_bin", bin_idx_c).cast(pl.UInt32))

    agg_c = (
        df_c.group_by("comp_bin")
        .agg(
            [
                pl.col(compression_col).mean().alias("comp_ratio_mean"),
                pl.col(tokens_col).mean().alias("orig_tokens_mean"),
                pl.col(compressed_tokens_col).mean().alias("comp_tokens_mean"),
                (pl.col(em_col).sum() / pl.col(em_col).count() * 100).alias("em_pct_comp"),
            ]
        )
        .sort("comp_bin")
    )

    agg_c_rows = {int(r["comp_bin"]): r for r in agg_c.to_dicts()}

    comp_ratio_means = np.full(n_bins, np.nan, dtype=float)
    left_tokens = np.full(n_bins, np.nan, dtype=float)  # original tokens (from compressed DF)
    right_tokens = np.full(n_bins, np.nan, dtype=float)  # compressed tokens
    y_comp_pct = np.full(n_bins, np.nan, dtype=float)

    for idx, row in agg_c_rows.items():
        if 0 <= idx < n_bins:
            comp_ratio_means[idx] = float(row["comp_ratio_mean"]) if row["comp_ratio_mean"] is not None else np.nan
            left_tokens[idx] = float(row["orig_tokens_mean"]) if row["orig_tokens_mean"] is not None else np.nan
            right_tokens[idx] = float(row["comp_tokens_mean"]) if row["comp_tokens_mean"] is not None else np.nan
            y_comp_pct[idx] = float(row["em_pct_comp"]) if row["em_pct_comp"] is not None else np.nan

    orig_files = get_filtered_csv_files(original_dir, exclude_datasets)
    print(orig_files)
    if not orig_files:
        raise ValueError("No valid original CSV files after filtering.")

    orig_dfs = []
    for f in orig_files:
        try:
            orig_dfs.append(pl.read_csv(f, columns=[tokens_col, em_col]))
        except Exception as e:
            print(f"[skip original file] {f}: {e}")
            continue

    if not orig_dfs:
        raise RuntimeError("No original data loaded successfully.")

    df_o = pl.concat(orig_dfs, how="vertical")

    N = df_c.height
    if df_o.height < N:
        print(f"[warn] original rows ({df_o.height}) < compressed rows ({N}). Using {df_o.height} rows only.")
        N_use = df_o.height
    else:
        N_use = N

    df_o = df_o.head(N_use)

    comp_bin_vals = df_c["comp_bin"].to_list()[:N_use]
    df_o = df_o.with_columns(pl.Series("comp_bin", comp_bin_vals).cast(pl.UInt32))

    df_o = df_o.with_columns(pl.col(em_col).cast(pl.Float64))

    agg_o = (
        df_o.group_by("comp_bin")
        .agg([(pl.col(em_col).sum() / pl.col(em_col).count() * 100).alias("em_pct_orig")])
        .sort("comp_bin")
    )
    agg_o_rows = {int(r["comp_bin"]): r for r in agg_o.to_dicts()}

    y_orig_pct = np.full(n_bins, np.nan, dtype=float)
    print("morning y orig ", y_orig_pct)
    for idx, row in agg_o_rows.items():
        print(f"idx is {idx}, row is {row}, agg is {agg_o_rows}")
        if 0 <= idx < n_bins:
            y_orig_pct[idx] = float(row["em_pct_orig"]) if row["em_pct_orig"] is not None else np.nan

    orig_tokens_by_bin = {}
    if np.any(np.isnan(left_tokens)):
        df_o_tokens_agg = df_o.group_by("comp_bin").agg([pl.col(tokens_col).mean().alias("orig_tokens_mean_o")])
        for r in df_o_tokens_agg.to_dicts():
            orig_tokens_by_bin[int(r["comp_bin"])] = r["orig_tokens_mean_o"]
        for i in range(n_bins):
            if np.isnan(left_tokens[i]) and i in orig_tokens_by_bin:
                left_tokens[i] = float(orig_tokens_by_bin[i])

    right_tokens[np.isnan(right_tokens)] = 0.0

    x = comp_ratio_means
    valid_x = x[~np.isnan(x)]
    if valid_x.size == 0:
        raise RuntimeError("No valid compression ratio centers to plot (all NaNs).")
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    bar_width = (x_max - x_min) / max(n_bins * 3.0, 1.0) if (x_max - x_min) > 0 else 0.05

    # fig, ax = plt.subplots(figsize=(11, 6))
    fig, ax = plt.subplots(figsize=(6.4, 4))

    print("BLue", y_comp_pct)
    # ax.plot(x, y_comp_pct, marker="s", linewidth=1.7, color="#0d88e6", label="Accuracy (Compressed)")
    # ax.plot(x, y_comp_pct, marker="P", linewidth=1.7, color="#2E8B57", label="Accuracy (Compressed)")
    # ax.plot(x, y_comp_pct, marker="P", linewidth=1.7, color="#7B68EE", label="Accuracy (Compressed)")
    ax.plot(x, y_comp_pct, marker="P", linewidth=1.7, color="#8A2BE2", label="Accuracy (Compressed)")

    print("Purple", y_orig_pct)
    ax.plot(x, y_orig_pct, marker="s", linewidth=1.7, linestyle="--", color="#FF8C00", label="Accuracy (Original)")

    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Bars on secondary axis (tokens)
    ax2 = ax.twinx()
    ax2.yaxis.set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("Number of Tokens")
    # ax2.set_ylim(0, 100)

    left_pos = x - bar_width / 2.0
    right_pos = x + bar_width / 2.0

    bars_left = ax2.bar(left_pos, left_tokens, width=bar_width, alpha=0.35, color="#0d88e6", label="Original Tokens")
    bars_right = ax2.bar(right_pos, right_tokens, width=bar_width, alpha=0.35, color="red", label="Compressed Tokens")

    def annotate_bars(bars, vals):
        for bar, val in zip(bars, vals, strict=True):
            try:
                if val is None or np.isnan(val) or val <= 0:
                    continue
            except Exception:
                continue
            ax2.text(bar.get_x() + bar.get_width() / 2.0, val, f"{int(round(val))}", ha="center", va="bottom", fontsize=9)

    annotate_bars(bars_left, left_tokens)
    annotate_bars(bars_right, right_tokens)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(
        l1 + l2,
        lab1 + lab2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=8.0,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.4,
        borderpad=0.4,
        frameon=True,
        ncol=4,
    )

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
    # plt.title("Axcer performance under different compression ratios")
    plt.tight_layout()
    plt.savefig("figure.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_results_with_em(
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/lingua2/gemma-3-12b-it/",
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/Meta-Llama-3.1-8B-Instruct/",
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/Meta-Llama-3.1-8B-Instruct/",
        # for llama
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/llama/",
        # original_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/llama/",
        # for gemma
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/gemma/",
        # original_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/gemma/",
        compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/",
        original_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/",
        # for both
        # compressed_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/",
        # original_dir="/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/",
        n_bins=10,
        exclude_datasets=["mbpp", "scitldr"],
        compression_col="compression_ratio",
        compressed_tokens_col="compressed_tokens",
        tokens_col="prompt_tokens",
        em_col="exact_match",
    )


# have multiple of this for other baselines and put them in the appendix
