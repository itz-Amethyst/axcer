from pathlib import Path
import polars as pl


# write_csv_with_column_averages(input_csv, "fake_metrics")


# NOTE: df1 is the one we get from axcer which contains information (input_metric)
# NOTE: df2 is the one contains metrics results
def concat_two_metrics(
    df1_path: Path,
    df2_path: Path,
    columns1: list[str] | None = None,
    columns2: list[str] | None = None,
):
    df1 = pl.read_csv(df1_path)
    df2 = pl.read_csv(df2_path)

    if columns1 is not None:
        df1 = df1.select(columns1)
    if columns2 is not None:
        df2 = df2.select(columns2)

    df2 = pl.concat([df2, df1], how="horizontal")

    df2.write_csv(df2_path)
    print(f"Saved concatenated DataFrame to: {df2_path}")

    return df2
