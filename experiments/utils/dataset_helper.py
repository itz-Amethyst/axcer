from pathlib import Path

import polars as pl


def trim_parquet_by_csv(
    csv_dir: str | Path,
    parquet_dir: str | Path,
    output_dir: str | Path,
    has_header: bool = True,
    overwrite: bool = False,
    flag: bool = False,
    use_line_count_fallback: bool = True,
):
    """
    For each CSV file in csv_dir (Path.glob("*")), find a parquet in parquet_dir with the same stem,
    remove the first N rows from that parquet where N == number of rows in the CSV,
    and save the trimmed parquet to output_dir.

    - Uses Path.glob (no glob module).
    - Prints CSV rows, parquet rows before, parquet rows after.
    """
    csv_dir = Path(csv_dir)
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files using Path.glob
    csv_files = sorted([p for p in csv_dir.glob("*") if p.is_file() and p.suffix.lower() == ".csv"])
    if flag:
        pq_files = sorted(
            [
                p.with_name(p.name.removeprefix("processed_"))
                for p in parquet_dir.glob("*")
                if p.is_file() and p.suffix.lower() in (".parquet", ".pq")
            ]
        )
    else:
        pq_files = sorted([p for p in parquet_dir.glob("*") if p.is_file() and p.suffix.lower() in (".parquet", ".pq")])

    print(f"Found {len(csv_files)} CSV(s) in {csv_dir}")
    print(f"Found {len(pq_files)} Parquet(s) in {parquet_dir}")

    if not csv_files:
        print("No CSV files found. Exiting.")
        return
    if not pq_files:
        print("No Parquet files found. Exiting.")
        return

    # Build map: base name -> parquet path (keeps first if duplicates)
    pq_map = {}
    for p in pq_files:
        if p.stem not in pq_map:
            pq_map[p.stem] = p
        else:
            print(f"Warning: duplicate parquet base '{p.stem}' found; keeping {pq_map[p.stem].name}")

    for idx, csv_path in enumerate(csv_files, start=1):
        base = csv_path.stem
        print(f"\nProcessing {idx}/{len(csv_files)}: {csv_path.name}")

        if base not in pq_map:
            print(f"  -> No matching parquet for '{base}', skipping.")
            continue

        try:
            csv_count = pl.read_csv(str(csv_path)).height
        except Exception as e:
            if use_line_count_fallback:
                try:
                    with open(csv_path, "rb") as fh:
                        lines = sum(1 for _ in fh)
                    csv_count = lines - (1 if has_header else 0)
                except Exception as e2:
                    print(f"  -> Failed to count CSV rows: {e2}. Skipping.")
                    continue
            else:
                print(f"  -> Failed to read CSV with Polars: {e}. Skipping.")
                continue

        pq_path = pq_map[base]
        try:
            pq_df = pl.read_parquet(str(pq_path))
        except Exception as e:
            print(f"  -> Failed to read parquet '{pq_path.name}': {e}. Skipping.")
            continue

        pq_before = pq_df.height
        print(f"  CSV rows: {csv_count}")
        print(f"  Parquet rows before: {pq_before}")

        if csv_count > pq_before:
            print(f"  -> CSV rows ({csv_count}) > parquet rows ({pq_before}); skipping.")
            continue
        if csv_count == pq_before:
            print("  -> Trim would produce empty parquet; skipping (set CSV shorter or allow empty).")
            continue

        # Slice and write
        trimmed = pq_df.slice(csv_count, pq_before - csv_count)
        pq_after = trimmed.height

        out_path = output_dir / pq_path.name
        if out_path.exists() and not overwrite:
            out_path = output_dir / f"{pq_path.stem}_trimmed{pq_path.suffix}"
            print(f"  Output exists; writing to {out_path.name}")

        try:
            trimmed.write_parquet(str(out_path))
        except Exception as e:
            print(f"  -> Failed to write trimmed parquet: {e}")
            continue

        print(f"  Parquet rows after : {pq_after}")
        print("  ✅ Done")

    print("\nAll done.")


def compare_parquet_csv_counts(parquet_dir: str, csv_dir: str) -> None:
    """
    Compare row counts of Parquet and CSV files with the same name in two directories.

    Args:
        parquet_dir (str): Directory containing Parquet files.
        csv_dir (str): Directory containing CSV files.
    """
    parquet_path = Path(parquet_dir)
    csv_path = Path(csv_dir)

    for pq_file in parquet_path.glob("*.parquet"):
        base_name = pq_file.stem  # filename without extension
        base_name = base_name.removeprefix("processed_")
        csv_file = csv_path / f"{base_name}.csv"

        if not csv_file.exists():
            print(f"❌ No matching CSV file for {pq_file.name}")
            continue

        # Count rows
        pq_count = pl.read_parquet(pq_file).height
        csv_count = pl.read_csv(csv_file).height

        if pq_count == csv_count:
            print(f"✅ {pq_file.name}: {pq_count} records (MATCH)")
        else:
            print(f"⚠️ {pq_file.name}: Parquet={pq_count}, CSV={csv_count} (MISMATCH)")


def remove_column_from_csv(csv_file_path: Path | str, column: str):
    df = pl.read_csv(csv_file_path)

    if column in df.columns:
        df = df.drop(column)
    else:
        raise ValueError(f"{column} not found in given csv")

    df.write_csv(csv_file_path)


def update_input_field(input_dir: str, output_dir: str, selected_stems: list[str]) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.parquet"):
        if file.stem in selected_stems:
            df = pl.read_parquet(file)

            if "input_field" not in df.columns:
                print(f"Skipping {file.name}, no 'input_field' column.")
                continue

            # Replace Text1: -> Sentence1: and Text2: -> Sentence2:
            df = df.with_columns(
                pl.col("input_field")
                .cast(str)  # ensure string
                .str.replace_all(r"Text1", "Sentence1")
                .str.replace_all(r"Text2", "Sentence2")
                .alias("input_field")
            )

            out_file = output_path / file.name
            df.write_parquet(out_file)
            print(f"Processed and saved: {out_file}")


def process_parquet_files(input_dir: str, output_dir: str, selected_stems: list[str]) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.parquet"):
        if file.stem in selected_stems:
            df = pl.read_parquet(file)

            if "answer_field" not in df.columns:
                print(f"Skipping {file.name}, no 'answer_field' column.")
                continue

            # Replace `,` or `;` with `||;;`
            df = df.with_columns(
                pl.col("answer_field")
                .cast(str)  # ensure string type
                .str.replace_all(r"[;]", "||;;")
                .alias("answer_field")
            )

            out_file = output_path / file.name
            df.write_parquet(out_file)
            print(f"Processed and saved: {out_file}")


def concat_csv_columns(destination: str, column_names: list[str], for_influence_plot: bool = True) -> None:
    """
    Browse all CSV files in a directory, select given columns, concatenate them,
    and save the result as a single CSV file.

    Args:
        destination (str): Path to directory containing CSV files.
        column_names (list[str]): List of column names to select and concatenate.
    """
    dest_path = Path(destination)
    if not dest_path.exists() or not dest_path.is_dir():
        raise ValueError(f"Destination {destination} is not a valid directory.")

    csv_files = sorted(dest_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {destination}")

    dfs = []
    for csv_file in csv_files:
        if for_influence_plot and Path(csv_file).name.replace(".csv", "") in ["mbpp", "scitldr"]:
            continue
        df = pl.read_csv(csv_file, columns=[item for item in column_names])
        missing = [col for col in column_names if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns {missing} in file {csv_file}")
        print(df.count())
        dfs.append(df.select(column_names))

    result = pl.concat(dfs, how="vertical")

    output_file = dest_path / "concated_given_columns.csv"
    result.write_csv(output_file)

    print(f"✅ Concatenated columns {column_names} saved as {output_file}")


def concat_multiple_csv_from_paths(
    base_path: str, paths: tuple[tuple[str, ...], ...], column_map: dict[str, list[str]]
) -> None:
    """
    Concatenate selected columns from CSV files across multiple subfolder groups.

    Args:
        base_path (str): Base directory containing experiment results.
        paths (tuple[tuple[str, ...], ...]):
            Each tuple defines a group:
                - First element = root folder
                - Remaining elements = subfolders inside the root
        column_map (dict[str, list[str]]): Mapping of dataset name -> columns to select.
    """
    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Base path {base_path} is not a valid directory.")

    for group in paths:
        if not group:
            continue

        first_folder = base / group[0]
        if not first_folder.exists():
            print(f"⚠️ Skipping missing folder: {first_folder}")
            continue

        # Subfolders
        subfolders = [first_folder / sub for sub in group[1:]]
        print("first subfolders", subfolders)

        if not subfolders:
            print(f"⚠️ No valid subfolders found in {group}")
            continue

        # Prepare output folder
        output_dir = first_folder / "concated_models_results"
        output_dir.mkdir(exist_ok=True)

        # Process datasets
        for dataset, cols in column_map.items():
            if group[0] == "original":
                print("entered !")
                print("COLS ARE", cols)
                if "compression_ratio_normalized" in cols:
                    cols.remove("compression_ratio_normalized")
                if "compression_ratio" in cols:
                    cols.remove("compression_ratio")

            dfs = []
            for subfolder in subfolders:
                csv_files = sorted(subfolder.glob("*.csv"))
                for csv_file in csv_files:
                    csv_file_name = csv_file.stem.lower()
                    if dataset not in csv_file_name:
                        continue
                    df = (
                        pl.read_csv(csv_file, infer_schema_length=10570) if csv_file_name == "squad" else pl.read_csv(csv_file)
                    )
                    missing = [c for c in cols if c not in df.columns]
                    if missing:
                        raise KeyError(f"Missing columns {missing} in {csv_file}")
                    dfs.append(df.select(cols))

            if not dfs:
                print(f"⚠️ No files found for dataset '{dataset}' in {group}")
                continue

            result = pl.concat(dfs, how="vertical")

            output_file = output_dir / f"{dataset}.csv"
            result.write_csv(output_file)
            print(f"✅ Dataset '{dataset}' saved at {output_file}")


# for joining llama and gemma into one
def concat_csv_files(input_csv1: str, input_csv2: str, output_csv: str) -> None:
    """
    Concatenate two CSV files vertically using Polars and save the result.

    Parameters
    ----------
    input_csv1 : str
        Path to the first CSV file.
    input_csv2 : str
        Path to the second CSV file.
    output_csv : str
        Path (including filename) where the merged CSV will be saved.
    """

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df1 = pl.read_csv(input_csv1)
    df2 = pl.read_csv(input_csv2)

    combined_df = pl.concat([df1, df2], how="diagonal_relaxed")

    combined_df.write_csv(output_csv)

    print(f"✅ Successfully merged and saved to: {output_csv}")


if __name__ == "__main__":
    # trim_parquet_by_csv(
    #     "/home/itz-amethyst/dev/axcer/experiments/ax_ev/processed",
    #     "/home/itz-amethyst/dev/axcer/experiments/ax_ev/datasets",
    #     "/home/itz-amethyst/dev/axcer/experiments/ax_ev/trimmed",
    # )

    # # compare_parquet_csv_counts("/home/itz-amethyst/dev/axcer/experiments/results/selective_context/processed/", "/home/itz-amethyst/dev/axcer/experiments/datasets/move out this/")
    # compare_parquet_csv_counts(
    #     "/home/itz-amethyst/dev/axcer/experiments/orig_ev/processed",
    #     "/home/itz-amethyst/dev/axcer/experiments/orig_ev/datasets",
    # )
    # remove_column_from_csv("/home/itz-amethyst/dev/axcer/experiments/orig_ev/squad.csv", "bleu")
    # update_input_field(
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/selective_context/",
    #     output_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/selective_context/cleanized",
    #     selected_stems=["glue"]
    # )

    # process_parquet_files(
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/selective_context/",
    #     output_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/selective_context/cleanized",
    #     selected_stems=["mbpp"]
    # )
    # process_parquet_files(
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/lingua2/",
    #     output_dir="/home/itz-amethyst/dev/axcer/experiments/datasets/lingua2/cleanized",
    #     selected_stems=["mbpp"]
    # )

    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/lingua2/Meta-Llama-3.1-8B-Instruct/",
    #     column_names=["inference_time", "prompt_tokens", "compression_time", "end_to_end_time"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/processed/",
    #     column_names=["compression_time", "prompt_tokens"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/multiprocess/",
    #     column_names=["compression_time", "prompt_tokens"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/original/concated_models_results/",
    #     column_names=["inference_time"]
    # )
    base_path = "/home/itz-amethyst/dev/axcer/experiments/results/"
    paths = (
        ("lingua2", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
        ("selective_context", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
        ("axcer/with_interrogative", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
        ("original", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    )

    y_columns = {
        "scitldr": ["rouge-1", "compression_ratio_normalized", "inference_time"],  # rouge-l,
        "mbpp": ["pass@1", "compression_ratio_normalized", "inference_time"],
        "gsm8k": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "mawps": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "squad": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "piqa": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "glue": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "boolq": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "coqa": ["exact_match", "compression_ratio_normalized", "inference_time"],
        "ai2_arc": ["exact_match", "compression_ratio_normalized", "inference_time"],
    }
    # this is for compression ratio influence over accuracy plot
    # y_columns = {
    #     "scitldr": ["rouge-1", "compression_ratio", "prompt_tokens"],  # rouge-l,
    #     "mbpp": ["pass@1", "compression_ratio", "prompt_tokens"],
    #     "gsm8k": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "mawps": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "squad": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "piqa": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "glue": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "boolq": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "coqa": ["exact_match", "compression_ratio", "prompt_tokens"],
    #     "ai2_arc": ["exact_match", "compression_ratio", "prompt_tokens"],
    # }

    # concat_multiple_csv_from_paths(base_path, paths, y_columns)
    # concat_two_metrics("/home/itz-amethyst/dev/axcer/experiments/results/lingua2/processed/processed_squad.csv", "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/gemma-3-12b-it/squad.csv",
    #                 [
    #                     "prompt_tokens",
    #                     "compressed_tokens",
    #                     "compression_ratio",
    #                     "gpt_o1_saving",
    #                     "compression_time",
    #                     "compression_ratio_normalized",
    #                 ],
    # )

    # concat_multiple_csv_from_paths(base_path, paths, y_columns)
    # remove_task_by_id("/home/itz-amethyst/dev/axcer/experiments/datasets/processed_mbpp.parquet", task_id=493, save_path="/home/itz-amethyst/dev/axcer/experiments/datasets/processed_mbpp.parquet")

    # ------------------------- Temporary code -------------------------

    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/lingua2/Meta-Llama-3.1-8B-Instruct/",
    #     column_names=["inference_time", "prompt_tokens", "compression_time", "end_to_end_time", "compression_ratio", "compression_ratio_normalized"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/lingua2/gemma-3-12b-it/",
    #     column_names=["inference_time", "prompt_tokens", "compression_time", "end_to_end_time", "compression_ratio", "compression_ratio_normalized"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/gemma-3-12b-it/",
    #     column_names=["inference_time", "prompt_tokens", "compression_time", "end_to_end_time", "compression_ratio", "compression_ratio_normalized"]
    # )
    #
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/Meta-Llama-3.1-8B-Instruct/",
    #     column_names=["inference_time", "prompt_tokens", "compression_time", "end_to_end_time", "compression_ratio", "compression_ratio_normalized"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/original/Meta-Llama-3.1-8B-Instruct/",
    #     column_names=["inference_time" ]
    # )

    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/original/gemma-3-12b-it/",
    #     column_names=[ "prompt_tokens", "exact_match"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/gemma-3-12b-it/",
    #     column_names=[ "prompt_tokens", "exact_match", "compression_ratio_normalized","compressed_tokens"]
    # )
    # concat_csv_columns(
    #     destination="/home/itz-amethyst/dev/axcer/experiments/results/original/Meta-Llama-3.1-8B-Instruct/",
    #     column_names=["prompt_tokens", "inference_time"]
    # )

    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/lingua/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/lingua/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/lingua/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/selective_context/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/selective_context/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/selective_context/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/all_merged.csv")

    # run this later
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/axcer/all_merged.csv")
    # # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/original/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/original/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/original/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/selective_context/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/e2e_results/axcer/all_merged.csv")

    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/lingua2/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/lingua2/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/lingua2/all_merged.csv")
    # concat_csv_files("/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/selective_context/gemma/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/selective_context/llama/concated_given_columns.csv", "/home/itz-amethyst/dev/axcer/experiments/results/concated_results/first_page_exp/selective_context/all_merged.csv")
