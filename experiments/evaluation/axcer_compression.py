import argparse
import os
from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

from axcer.process import Axcer
from experiments.constants.paths import (
    MERGED_COMPRESSED_AXCER_WITH_INTERROGATIVE_DATASET_PATH,
    PROCESSED_AXCER_PATH,
    fill_path,
)
from experiments.evaluation.utils.align_datasets import replace_input_column_values
from experiments.modals.utils.metrics import compute_fixed_range_compression_ratio

parser = argparse.ArgumentParser(description="Axcer compression")

parser.add_argument(
    "--dataset-path",
    help="Dataset directory that axcer would do the compression on it",
    type=Path,
    required=True,
)
parser.add_argument(
    "--save-path",
    help="path to save results",
    type=Path,
    required=True,
    # default="/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/testing2/{dataset_name}.csv",
)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def compress_and_replace():
    for pq_file in Path(args.dataset_path).glob("*.parquet"):
        write_path = Path(args.save_path) / "{dataset_name}.csv"
        # write_path = fill_path(args.save_path, dataset_name=pq_file.stem.removeprefix("processed_"))
        write_path = fill_path(write_path, dataset_name=pq_file.stem.removeprefix("processed_"))
        write_path.parent.mkdir(parents=True, exist_ok=True)
        axcer = Axcer(metrics_file_path=write_path)

        def process(x, axcer):
            compressed_text = axcer.compress_prompt(x)
            flat = [item for sublist in compressed_text for item in sublist]
            return " ".join(flat)

        df = pl.read_parquet(pq_file)
        df = df.with_columns(
            pl.col("input_field")
            .map_elements(partial(process, axcer=axcer), return_dtype=pl.String)
            .alias("input_field")  # same name replaces the original column
        )

        axcer_df = compute_fixed_range_compression_ratio(write_path)

        axcer_df.write_csv(write_path)

        del axcer


def compute_avg(process_path: str, avg_save_path_template: str):
    try:
        for csv_file in Path(process_path).glob("*.csv"):
            print("Csv_file name", csv_file.stem)
            df = pl.read_csv(csv_file, infer_schema_length=10570) if csv_file.stem == "squad" else pl.read_csv(csv_file)

            df_mean = df.mean()

            # Round float columns to 3 decimal places
            df_mean = df_mean.with_columns((cs.by_dtype(pl.Float32) | cs.by_dtype(pl.Float64)).round(3))
            renamed_df = df_mean.rename({col: f"{col}_avg" for col in df_mean.columns})

            dataset_name = csv_file.stem

            Path(avg_save_path_template).parent.mkdir(parents=True, exist_ok=True)
            avg_save_path = Path(avg_save_path_template.format(dataset_name=dataset_name))
            print("JJ", avg_save_path)
            renamed_df.write_csv(avg_save_path)

            print(f"Saved averages to: {avg_save_path}")
    except Exception as e:
        print(f"something went wrong: {e}")


def compare_single_row_csvs(dir1: str | Path, dir2: str | Path):
    dir1, dir2 = Path(dir1), Path(dir2)

    results = {}

    for csv1 in dir1.glob("*.csv"):
        file_name = csv1.name
        csv2 = dir2 / file_name

        if not csv2.exists():
            results[file_name] = {"status": "missing_in_dir2"}
            continue

        # read both CSVs
        df1 = pl.read_csv(csv1)
        df2 = pl.read_csv(csv2)

        if df1.shape[0] != 1 or df2.shape[0] != 1:
            results[file_name] = {"status": "not_single_record"}
            continue

        # keep only numeric columns
        num_cols1 = [
            c for c, dt in zip(df1.columns, df1.dtypes, strict=False) if dt in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
        ]
        num_cols2 = [
            c for c, dt in zip(df2.columns, df2.dtypes, strict=False) if dt in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
        ]

        common_cols = list(set(num_cols1) & set(num_cols2))
        if not common_cols:
            results[file_name] = {"status": "no_numeric_columns"}
            continue

        row1 = df1.select(common_cols).row(0, named=True)
        row2 = df2.select(common_cols).row(0, named=True)

        comparison = {}
        for col in common_cols:
            v1, v2 = row1[col], row2[col]
            comparison[col] = {"dir1": v1, "dir2": v2, "same": v1 == v2}

        results[file_name] = {"status": "compared", "values": comparison}

    return results


def update_dataset_with_compressed_text():
    dt_path = Path("/home/itz-amethyst/dev/axcer/experiments/datasets/for_axcer/")
    dataset_files = {p.stem.removeprefix("processed_"): p for p in dt_path.glob("*.parquet")}

    print("Started to replace compressed values with dataset values")
    for processed_path in PROCESSED_AXCER_PATH.parent.glob("*.csv"):
        print("Processed_path is ", processed_path)
        base_name = str(processed_path.stem.removeprefix("processed_"))
        try:
            dataset_path = dataset_files[base_name]
        except KeyError as err:
            raise FileNotFoundError(f"No matching parquet found for {base_name}.csv") from err

        replace_input_column_values(
            processed_path, dataset_path, MERGED_COMPRESSED_AXCER_WITH_INTERROGATIVE_DATASET_PATH, base_name
        )


if __name__ == "__main__":
    compress_and_replace()
    # update_dataset_with_compressed_text()

    # compute_avg("/home/itz-amethyst/dev/axcer/experiments/results/original/concated_models_results/", "/home/itz-amethyst/dev/axcer/experiments/results/original/concated_models_results/avg/{dataset_name}.csv")
    # compute_avg("/home/itz-amethyst/dev/axcer/experiments/results/lingua2/processed", "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/processed/{dataset_name}.csv")
    # res = compare_single_row_csvs(
    # "/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/sequential/avg",
    # "/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/corrupted/avg"
    # )
