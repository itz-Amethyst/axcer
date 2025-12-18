from pathlib import Path
import polars as pl
from tqdm import tqdm
import os


def concat_csv_two_dirs(src1: str, src2: str, final_path: str, column_map: dict[str, list[str]]) -> None:
    """
    Concatenate CSV files with the same (normalized) name from two directories,
    using different column selections per source.

    Args:
        src1 (str): First source directory.
        src2 (str): Second source directory (files may start with 'processed_').
        final_path (str): Destination directory to save results.
        column_map (dict[str, list[str]]): Mapping of file stem -> [col_from_src1, col_from_src2].
    """
    d1 = Path(src1)
    d2 = Path(src2)
    out_dir = Path(final_path)

    if not d1.exists() or not d1.is_dir():
        raise ValueError(f"Invalid directory: {src1}")
    if not d2.exists() or not d2.is_dir():
        raise ValueError(f"Invalid directory: {src2}")

    out_dir.mkdir(parents=True, exist_ok=True)

    d2_files = {f.stem.removeprefix("processed_").lower(): f for f in d2.glob("*.csv")}

    for filename, cols in column_map.items():
        if len(cols) != 2:
            raise ValueError(f"Column map for '{filename}' must have exactly 2 items.")

        f1 = d1 / f"{filename.upper()}.csv" if filename == "mawps" else d1 / f"{filename}.csv"
        f2 = d2_files.get(filename)

        if not f1.exists() or f2 is None:
            print(f"⚠️ Skipping {filename} (file missing in one of the dirs)")
            continue

        col1, col2 = cols

        df1 = pl.read_csv(f1)
        df2 = pl.read_csv(f2)

        if col1 not in df1.columns:
            raise KeyError(f"Column '{col1}' not found in {f1}")
        if col2 not in df2.columns:
            raise KeyError(f"Column '{col2}' not found in {f2}")

        df1_sel = df1.select([pl.col(col1)])
        print(df1_sel)
        df2_sel = df2.select([pl.col(col2)])
        print(df2_sel)

        result = pl.concat([df1_sel, df2_sel], how="horizontal")

        out_file = out_dir / f"{filename}.csv"
        result.write_csv(out_file)
        print(f"✅ Saved merged file: {out_file}")


def compute_avg_tokens(dataset_path: str, write_path: str, tokenizer, input_field: str = "input_field"):
    """
    Computes the average token count for the input_field of each parquet file in a dataset directory.

    Args:
        dataset_path (str): Path to directory containing .parquet files.
        write_path (str): Directory where results will be written.
        tokenizer: Predefined tokenizer with an `.encode()` method.
        input_field (str): Column name containing text input (default: "input_field").
    """

    os.makedirs(write_path, exist_ok=True)

    for file_name in tqdm(os.listdir(dataset_path), desc="Processing parquet files"):
        if not file_name.endswith(".parquet"):
            continue

        file_path = os.path.join(dataset_path, file_name)
        try:
            df = pl.read_parquet(file_path)
        except Exception as e:
            print(f"⚠️ Failed to read {file_name}: {e}")
            continue

        if input_field not in df.columns:
            print(f"⚠️ Column '{input_field}' not found in {file_name}, skipping.")
            continue

        token_lengths = []
        for text in df[input_field].drop_nans():
            tokens = tokenizer.encode(str(text))
            token_lengths.append(len(tokens))

        avg_tokens = 0 if not token_lengths else sum(token_lengths) / len(token_lengths)

        result_df = pl.DataFrame({"file_name": [file_name], "avg_tokens": [avg_tokens], "num_samples": [len(token_lengths)]})

        output_name = os.path.splitext(file_name)[0] + "_avg_tokens.csv"
        result_df.write_csv(os.path.join(write_path, output_name))

        print(f"✅ Saved avg tokens for {file_name} → {output_name}")


y_columns = {
    "scitldr": ["rouge-1", "compression_ratio_normalized"],  # rouge-l,
    "mbpp": ["pass@1", "compression_ratio_normalized"],
    "gsm8k": ["exact_match", "compression_ratio_normalized"],
    "mawps": ["exact_match", "compression_ratio_normalized"],
    "squad": ["exact_match", "compression_ratio_normalized"],
    "piqa": ["exact_match", "compression_ratio_normalized"],
    "glue": ["exact_match", "compression_ratio_normalized"],
    "boolq": ["exact_match", "compression_ratio_normalized"],
    "coqa": ["exact_match", "compression_ratio_normalized"],
    "ai2_arc": ["exact_match", "compression_ratio_normalized"],
}


def concat_prompt_tokens(dir_main: str, dir_token: str):
    dir_main = Path(dir_main)
    dir_token = Path(dir_token)

    for token_file in dir_token.glob("*.csv"):
        base_name = token_file.stem.replace("_token_len", "")
        main_file = dir_main / f"{base_name}.csv"

        if not main_file.exists():
            print(f"⚠️ Skipped: {base_name} (no matching file in main directory)")
            continue

        df_main = (
            pl.read_csv(main_file, infer_schema_length=10570) if main_file.name == "squad.csv" else pl.read_csv(main_file)
        )
        df_token = (
            pl.read_csv(token_file, infer_schema_length=10570) if main_file.name == "squad.csv" else pl.read_csv(token_file)
        )

        if "prompt_tokens" not in df_token.columns:
            print(f"⚠️ Missing 'prompt_tokens' in {token_file.name}")
            continue

        df_combined = pl.concat([df_main, df_token.select("prompt_tokens")], how="horizontal")

        df_combined.write_csv(main_file)
        print(f"✅ Updated: {main_file.name} (added prompt_tokens)")

    print("All done!")


# compute_avg_tokens("/home/itz-amethyst/dev/axcer/experiments/datasets/main_move_this_out/", "/home/itz-amethyst/dev/axcer/experiments/results/original/avg", tokenizer , "input_field")
concat_prompt_tokens(
    "/home/itz-amethyst/dev/axcer/experiments/results/original/gemma-3-12b-it/",
    "/home/itz-amethyst/dev/axcer/experiments/results/original/temp/",
)
