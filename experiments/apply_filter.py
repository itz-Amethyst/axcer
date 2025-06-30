from pathlib import Path
import polars as pl
from modal_experiment.tensorrt.final.utils.qa_field_separator import separate_qa_fields
from axcer.axcer.utils.custom_logger import logger
from axcer.experiments.dataset_config import datasets
from axcer.experiments.utils.field_parser import FieldPathParser
from axcer.experiments.utils.prepare_dataset import preprocess_dataset

field_parser = FieldPathParser()


def process_and_save(df: pl.DataFrame, dataset_name: str, output_dir: str = "processed"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"processed_{dataset_name}.parquet"

    try:
        if dataset_name in datasets:
            config = datasets[dataset_name]
            print(f"Processing with config: {config}")

            if dataset_name.lower() in ["mawps", "quac"]:
                # to differently handle these datasets
                df = separate_qa_fields(df, config)
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
