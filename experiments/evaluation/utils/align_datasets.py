from pathlib import Path

import modal
import polars as pl

from experiments.constants.paths import VOLUME_NAME, fill_path


# NOTE: df1 is the one we get from axcer which contains information (input_metric)
# NOTE: df2 is the one contains metrics results
def concat_two_metrics(
    df1_path: Path,
    df2_path: Path,
    columns1: list[str] | None = None,
):
    df1 = pl.read_csv(df1_path)
    df2 = pl.read_csv(df2_path)

    if columns1 is not None:
        df1 = df1.select(columns1)

    df2 = df2.select(pl.all())

    df2 = pl.concat([df2, df1], how="horizontal")

    if "compression_time" in df1.columns and "inference_time" in df2.columns:
        df2 = df2.with_columns((pl.col("compression_time") + pl.col("inference_time")).alias("end_to_end_time"))

    df2.write_csv(df2_path)
    print(f"Saved concatenated DataFrame to: {df2_path}")

    return df2


# output path should be in these classifications MERGED_COMPRESSED_AXCER_WITH_DATASET_PATH
def replace_input_column_values(process_path: Path, dataset_path: Path, output_base_path: Path, dataset_name: str) -> None:
    df_dataset = pl.read_csv(process_path)
    df_process = pl.read_parquet(dataset_path)

    if df_process.height != df_dataset.height:
        raise ValueError(f"Row count mismatch: csv={df_dataset.height} parquet={df_process.height}")

    df_process = df_process.with_columns(df_dataset["compressed_text"].alias("input_field"))
    output_path = fill_path(output_base_path, dataset_name=dataset_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_process.write_parquet(output_path)
    print(f"✅ Saved updated file to {output_path}")
    vol = modal.Volume.from_name(VOLUME_NAME)
    vol.commit()
