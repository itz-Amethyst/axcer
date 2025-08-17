from pathlib import Path
import shutil
import polars as pl
import polars.selectors as cs

from experiments.constants.paths import (
    AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    METRICS_AXCER_PATH,
    fill_path,
)


# once you wanted to compute page 2 experiment without interrogative word applied set the write path to PROCESSED_AXCER_PATH_WITHOUT_INTERROGATIVE
def compute_csv_category_column_averages(template_path: Path, write_path: Path, model_name: str, category_name: str) -> None:
    category_path = fill_path(template_path, model_name=model_name, category_name=category_name)
    write_path = fill_path(write_path, model_name=model_name, category_name=category_name)
    df = pl.read_csv(category_path)

    # df_mean = df.mean()
    df_mean = df.select(pl.col(pl.NUMERIC_DTYPES)).mean()

    # Round float columns to 3 decimal places

    df_mean = df_mean.with_columns((cs.by_dtype(pl.Float32) | cs.by_dtype(pl.Float64)).round(3))
    renamed_df = df_mean.rename({col: f"{col}_avg" for col in df_mean.columns})

    # output_filename = f"{dataset_name}_avg.csv"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    renamed_df.write_csv(write_path)
    print(f"Saved averages to: {category_path}")


# template path is whether DIVIDE_DATASETS_CATEGORY_ORIGINAL_PATH or DIVIDE_DATASETS_CATEGORY_AXCER_PATH
def copy_processed_datasets_to_category(
    template_path: Path, category_name: str, model_name: str, *csv_paths: str | Path
) -> list[Path]:
    """
    Copy multiple CSV files into a folder named after the given category.

    Args:
        category_name (str): Name of the category folder to copy files into.
        *csv_paths (str | Path): CSV file paths to copy.

    Returns:
        list[Path]: Paths of the copied files in the new category folder.
    """

    category_path = fill_path(template_path, model_name=model_name, category_name=category_name)
    # category_folder = Path(category_name)
    category_path.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for csv_path in csv_paths:
        src = Path(csv_path)
        if not src.exists():
            raise FileNotFoundError(f"CSV file not found: {src}")

        dst = category_path / src.name
        shutil.copy(src, dst)
        copied_files.append(dst)

    return copied_files


def concat_metrics_into_one_category(
    template: Path, category_name: str, *csv_paths: str | Path, how: str = "vertical", model_name: str
):
    dfs: list[pl.DataFrame] = [pl.read_csv(path) for path in csv_paths]
    df_merged = pl.concat(dfs, how=how)
    print(df_merged)

    output_dir = fill_path(template, model_name=model_name, category_name=category_name)
    # output_dir = AXCER_RESULTS_MERGED_INTO_CATEGORY_PATH / model_name

    print(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # df_merged.write_csv(output_dir / f"{category_name}.csv")
    df_merged.write_csv(output_dir)

    return df_merged


# example usage
# f1_path = fill_path(processed_axcer_path, dataset_name="squad")
# f2_path = fill_path(processed_axcer_path, dataset_name="something")
f1_path = fill_path(METRICS_AXCER_PATH, dataset_name="squad", model_name="llama")
f2_path = fill_path(METRICS_AXCER_PATH, dataset_name="something", model_name="llama")
# if config["IS_INTERROGATIVE_RUN"] == True:
concat_metrics_into_one_category(
    CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH, "mycategory", f1_path, f2_path, model_name="llama"
)


# clean_csv_column(f2_path, f2_path, "total_runtime")

compute_csv_category_column_averages(
    CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    "llama",
    "mycategory",
)
