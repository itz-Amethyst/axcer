import argparse
from functools import partial
import os
from pathlib import Path
from axcer.process import Axcer
from experiments.constants.paths import (
    MERGED_COMPRESSED_AXCER_WITH_DATASET_PATH,
    PROCESSED_AXCER_PATH,
    fill_path,
)
from experiments.modals.utils.metrics import compute_fixed_range_compression_ratio
import polars as pl

parser = argparse.ArgumentParser(description="Axcer compression")

parser.add_argument(
    "--dataset-path",
    help="Dataset directory that axcer would do the compression on it",
    required=True,
)
parser.add_argument(
    "--save-path",
    help="path to save results",
    required=False,
    default="/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed",
)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def compress_and_replace():
    for pq_file in Path(args.dataset_path).glob("*.parquet"):
        # #TODO: for question categories without interrogative words have to change the paths be aware !
        write_path = fill_path(PROCESSED_AXCER_PATH, dataset_name=pq_file.stem.removeprefix("processed_"))
        write_path.parent.mkdir(parents=True, exist_ok=True)
        axcer = Axcer(metrics_file_path=write_path)

        def process(x, axcer):
            print(axcer)
            compressed_text = axcer.compress_prompt(x)
            flat = [item for sublist in compressed_text for item in sublist]
            return " ".join(flat)

        df = pl.read_parquet(pq_file)
        df = df.with_columns(
            pl.col("input_field")
            .map_elements(partial(process, axcer=axcer), return_dtype=pl.String)
            .alias("input_field")  # same name replaces the original column
        )

        dataset_write_path = fill_path(
            MERGED_COMPRESSED_AXCER_WITH_DATASET_PATH, dataset_name=pq_file.stem.removeprefix("processed_")
        )
        dataset_write_path.parent.mkdir(exist_ok=True, parents=True)
        df.write_parquet(dataset_write_path)

        axcer_df = compute_fixed_range_compression_ratio(write_path)

        axcer_df.write_csv(write_path)

        del axcer

    # dataset_files = {p.stem.removeprefix("processed_"): p for p in dataset_path.glob("*.parquet")}
    #
    # print("Started to replace compressed values with dataset values")
    # processed_paths = PROCESSED_AXCER_PATH.parent
    # for processed_path in processed_paths.glob("*.csv"):
    #     processed_df = compute_fixed_range_compression_ratio(processed_path)
    #     processed_df.write_csv(processed_path)
    #     base_name = str(processed_path.stem.removeprefix("processed_"))
    #     # base_name = base_name.split(".")[0]
    #     try:
    #         dataset_path = dataset_files[base_name]
    #     except KeyError as err:
    #         raise FileNotFoundError(f"No matching parquet found for {base_name}.csv") from err
    #
    #     replace_input_column_values(processed_path, dataset_path, MERGED_COMPRESSED_AXCER_WITH_DATASET_PATH, base_name)


if __name__ == "__main__":
    compress_and_replace()
