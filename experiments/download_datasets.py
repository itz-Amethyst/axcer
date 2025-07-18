import polars as pl
from experiments.apply_filter import process_and_save
from experiments.dataset_config import DATASET_NAMES
from datasets import load_dataset
from experiments.utils.parse_dataset import parse_dataset_entry
from axcer.utils.custom_logger import logger


def main():
    for entry in DATASET_NAMES:
        dataset_name, subset_name, split_name = parse_dataset_entry(entry)
        try:
            logger.info(
                f"\n Downloading '{split_name}' split for dataset: {dataset_name}"
                f"{f' (subset: {subset_name})' if subset_name else ''}"
            )

            dataset = load_dataset(dataset_name, name=subset_name, split=split_name, trust_remote_code=True)
            df = pl.DataFrame(dataset.to_polars())
            process_and_save(df, dataset_name)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")


if __name__ == "__main__":
    main()
