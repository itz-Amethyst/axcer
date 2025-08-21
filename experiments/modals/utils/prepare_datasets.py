from pathlib import Path
import polars as pl


def load_datasets(dataset_dir: Path, question_col: str, answer_col: str):
    datasets = {}
    dataset_dir = dataset_dir.expanduser()

    if not dataset_dir.exists():
        print("Directory not found:", dataset_dir)
        return datasets

    # There is only one instance of dataset in given path, we loop through it
    files = list(dataset_dir.glob("*.parquet")) + list(dataset_dir.glob("*.csv"))
    for dataset_file in files:
        print("yes")
        df = pl.read_parquet(dataset_file, columns=[question_col, answer_col])
        dataset_name = dataset_file.stem  # name without .parquet
        datasets[dataset_name] = df[[question_col, answer_col]]
    print()
    return datasets
