import glob
from pathlib import Path
import polars as pl
from typing import Any

from experiments.modals.utils.metric_helper import prepare_tokenizer_for_counting_modal

tokenizer = prepare_tokenizer_for_counting_modal()


def parquet_to_token_length_csv(
    parquet_dir: str,
    tokenizer: Any,
    token_len_outdir: str,
    text_col: str = "input_field",
    parquet_glob: str = "*.parquet",
):
    Path(token_len_outdir).mkdir(parents=True, exist_ok=True)
    parquet_paths = sorted(glob.glob(str(Path(parquet_dir) / parquet_glob)))

    if not parquet_paths:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    for pq in parquet_paths:
        base = Path(pq).stem
        print(f"[tokenize] Processing: {base}")

        df = pl.read_parquet(pq)

        if text_col not in df.columns:
            print(f"[tokenize] Skipping {base}: missing column '{text_col}'")
            continue

        # ✅ Only add idx if not present
        if "idx" not in df.columns:
            df = df.with_row_count("idx")

        texts = df[text_col].to_list()
        token_lengths = []

        for i, t in enumerate(texts):
            try:
                token_lengths.append(len(tokenizer.encode(t)))
            except Exception as e:
                print(f"[tokenize] tokenizer failed at row {i} in {base}: {e}")
                token_lengths.append(None)

        out_df = pl.DataFrame({"idx": df["idx"].to_list(), "prompt_tokens": token_lengths})

        out_path = Path(token_len_outdir) / f"{base.removeprefix('processed_')}_token_len.csv"
        out_df.write_csv(out_path)
        print(f"[tokenize] Saved -> {out_path}")


def merge_token_length_into_csvs(
    token_len_dir: str,
    csv_to_enrich_dir: str,
    merged_output_dir: str,
    token_len_glob: str = "*_token_len.csv",
    csv_glob: str = "*.csv",
):
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)

    token_len_paths = sorted(glob.glob(str(Path(token_len_dir) / token_len_glob)))
    enrich_paths = {Path(p).stem: p for p in glob.glob(str(Path(csv_to_enrich_dir) / csv_glob))}

    for tok_csv in token_len_paths:
        tok_base = Path(tok_csv).stem
        orig_base = tok_base.replace("_token_len", "")

        if orig_base not in enrich_paths:
            print(f"[merge] No match for {tok_csv}")
            continue

        enrich_csv_path = enrich_paths[orig_base]
        print(f"[merge] Merging {orig_base}")

        df_tokens = pl.read_csv(tok_csv)
        df_enrich = pl.read_csv(enrich_csv_path, schema_overrides={"exact_match": pl.Float64})

        # ✅ Add idx only if missing
        if "idx" not in df_tokens.columns:
            df_tokens = df_tokens.with_row_count("idx")
        if "idx" not in df_enrich.columns:
            df_enrich = df_enrich.with_row_count("idx")

        df_tokens = df_tokens.with_columns(pl.col("idx").cast(pl.UInt64))
        df_enrich = df_enrich.with_columns(pl.col("idx").cast(pl.UInt64))

        merged = df_enrich.join(df_tokens.select(["idx", "prompt_tokens"]), on="idx", how="left")

        out_path = Path(merged_output_dir) / f"{orig_base}_merged.csv"
        merged.write_csv(out_path)
        print(f"[merge] Saved -> {out_path}")


parquet_to_token_length_csv(
    parquet_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/",
    tokenizer=tokenizer,
    token_len_outdir="/home/itz-amethyst/dev/axcer/experiments/results/original/temp",
)

# merge_token_length_into_csvs(
#     token_len_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/temp",
#     csv_to_enrich_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/Meta-Llama-3.1-8B-Instruct/",
#     merged_output_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/llama_with_tokens/"
# )
merge_token_length_into_csvs(
    token_len_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/temp",
    csv_to_enrich_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/gemma-3-12b-it/",
    merged_output_dir="/home/itz-amethyst/dev/axcer/experiments/results/original/gemma_with_tokens/",
)
