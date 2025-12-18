from pathlib import Path
import polars as pl


def load_datasets(dataset_dir: Path, question_col: str, answer_col: str):
    datasets = {}
    dataset_dir = dataset_dir.expanduser()

    if not dataset_dir.exists():
        print("Directory not found:", dataset_dir)
        return datasets

    files = list(dataset_dir.glob("*.parquet")) + list(dataset_dir.glob("*.csv"))

    for dataset_file in files:
        if dataset_file.suffix == ".parquet":
            df = pl.read_parquet(dataset_file, columns=[question_col, answer_col])
        elif dataset_file.suffix == ".csv":
            df = pl.read_csv(dataset_file, columns=[question_col, answer_col])
        else:
            continue

        dataset_name = dataset_file.stem
        datasets[dataset_name] = df.select([question_col, answer_col])

    return datasets
