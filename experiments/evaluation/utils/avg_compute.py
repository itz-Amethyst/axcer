from experiments.constants.paths import fill_path
from pathlib import Path
import polars as pl
import polars.selectors as cs

from experiments.modals.utils.helper import map_path_to_volume


# this is for each dataset for first table ! not concatenated category results
# the base path should be in these list [original_path, axcer_path, selective_path, lingua2_path]
def compute_multiple_file_csv_column_averages(base_read_path: Path, base_write_path: Path, model_name: str) -> None:
    """
    Reads multiple CSV at input_csv_path, computes the average of each numeric column,
    and writes a new CSV named '{dataset_name}_avg.csv' containing a single row
    of averages, with column names suffixed by '_avg'.
    """

    try:
        base_read_path = map_path_to_volume(base_read_path)
        files_path = fill_path(base_read_path, model_name=model_name)

        for csv_file in files_path.glob("*.csv"):
            df = pl.read_csv(csv_file)

            df_mean = df.mean()

            # Round float columns to 3 decimal places
            df_mean = df_mean.with_columns((cs.by_dtype(pl.Float32) | cs.by_dtype(pl.Float64)).round(3))
            renamed_df = df_mean.rename({col: f"{col}_avg" for col in df_mean.columns})

            dataset_name = csv_file.stem
            output_dir = fill_path(base_write_path, model_name=model_name, dataset_name=dataset_name)
            output_dir = map_path_to_volume(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            # file_exists = output_dir.exists()
            # if not file_exists:
            renamed_df.write_csv(output_dir)

            print(f"Saved averages to: {output_dir}")
    except Exception as e:
        print(f"something went wrong: {e}")


def compute_single_file_csv_column_averages(input_csv_path: str, dataset_name: str, model_name: str) -> None:
    """
    Reads a CSV at input_csv_path, computes the average of each numeric column,
    and writes a new CSV named '{dataset_name}_avg.csv' containing a single row
    of averages, with column names suffixed by '_avg'.
    """

    df = pl.read_csv(input_csv_path)

    df_mean = df.mean()

    # Round float columns to 3 decimal places

    df_mean = df_mean.with_columns((cs.by_dtype(pl.Float32) | cs.by_dtype(pl.Float64)).round(3))
    renamed_df = df_mean.rename({col: f"{col}_avg" for col in df_mean.columns})

    output_filename = f"{dataset_name}_avg.csv"
    renamed_df.write_csv(output_filename)
    print(f"Saved averages to: {output_filename}")
