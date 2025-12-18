from pathlib import Path
import polars as pl
from experiments.utils.qa_field_separator import separate_qa_fields
from axcer.utils.custom_logger import logger
from experiments.dataset_config import DATASETS as datasets
from experiments.utils.field_parser import FieldPathParser
from experiments.utils.prepare_dataset import preprocess_dataset
from experiments.utils.format_mawps import process_mawps
from experiments.constants.paths import DATASET_PATH

field_parser = FieldPathParser()


def process_and_save(df: pl.DataFrame, dataset_name: str, output_dir: str = "datasets"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    output_path = DATASET_PATH / f"processed_{dataset_name}.parquet"

    try:
        if dataset_name in datasets:
            config = datasets[dataset_name]
            print(f"Processing with config: {config}")
            print("cleaned", dataset_name)

            if dataset_name.lower() in ["coqa"]:
                # to explicitly handle dataset
                df = separate_qa_fields(df, config)

            elif dataset_name.lower() in "mawps":
                df = process_mawps(df, config)

            else:
                df = preprocess_dataset(df, dataset_name, config)
                logger.info(f"Successfully processed {len(df)} items")
        else:
            logger.error(f"No configuration found for dataset '{dataset_name}'")

    except Exception as e:
        logger.error(f"Error processing dataset '{dataset_name}': {e}")
        print(f"Failed to process dataset '{dataset_name}': {e}")

    df.write_parquet(output_path)
    logger.info(f"Saved: {output_path}")
