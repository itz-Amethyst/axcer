from axcer.experiments.utils.field_parser import FieldPathParser
import polars as pl
from axcer.utils.custom_logger import logger

class DatasetProcessor:
    """
    Dataset processor that handles field extraction and transformation using Polars operations.
    """

    def __init__(self):
        self.field_parser = FieldPathParser()

    def process_question_field_vectorized(self, df: pl.DataFrame, question_fields: list) -> pl.Series:
        """
        Process question fields by combining them in order using vectorized operations.
        """
        if len(question_fields) == 1:
            field_series = self.field_parser.extract_value_vectorized(df, question_fields[0])

            def format_single_field(value):
                if isinstance(value, list):
                    if value and isinstance(value[0], str):
                        formatted_options = ", ".join(f"{item}"
                                                    for _, item in enumerate(value))

                        data =  f"Options: {formatted_options}"
                        return pl.Series(question_fields[0],[data] )

                    else:
                        data = ", ".join(str(item) for item in value)
                        return pl.Series(question_fields[0],[data] )
                else:
                    return pl.Series(question_fields[0],[str(value) if value else ""])

            return field_series.map_elements(format_single_field, return_dtype=pl.String)
        else:
            # Multiple fields
            field_series_list = []
            for field in question_fields:
                series = self.field_parser.extract_value_vectorized(df, field)
                field_series_list.append(series)

            def combine_fields(*field_values):
                combined_parts = []
                for field_value in field_values:
                    if isinstance(field_value, list):
                        if field_value and isinstance(field_value[0], str):
                            formatted_options = ", ".join(f"{item}"
                                                        for _, item in enumerate(field_value))
                            combined_parts.append(f"Options: {formatted_options}")
                        else:
                            combined_parts.append(", ".join(str(item) for item in field_value))
                    elif field_value:
                        combined_parts.append(str(field_value))

                return " ".join(combined_parts) if combined_parts else ""

            temp_df = pl.DataFrame({f"field_{i}": series for i, series in enumerate(field_series_list)})

            return temp_df.map_rows(lambda row: combine_fields(*row)).to_series()

    def process_answer_field_vectorized(self, df: pl.DataFrame, answer_field: list) -> pl.Series:
        """
        Process answer field using vectorized operations.
        """
        field_series = self.field_parser.extract_value_vectorized(df, answer_field[0])

        def format_single_field(value):
            if isinstance(value, list):
                if value and isinstance(value[0], str):
                    formatted_options = ", ".join(f"{item}"
                                                for _, item in enumerate(value))
                    return f"Options: {formatted_options}"
                else:
                    return ", ".join(str(item) for item in value)
            else:
                return str(value) if value else ""

        return field_series.map_elements(format_single_field, return_dtype=pl.String)


    def preprocess_dataset(self, df: pl.DataFrame, dataset_name: str, config: dict) -> pl.DataFrame:
        """
        Process a single dataset according to its configuration using Polars operations.

        Args:
            df: Polars DataFrame containing the dataset
            dataset_name: Name of the dataset
            config: Configuration dictionary with field mappings

        Returns:
            pl.DataFrame: Processed DataFrame with 'question' and 'answer' columns
        """
        question_fields = config["question_field"]
        answer_field = config["answer_field"]

        try:
            print(f"Processing {len(df)} items from DataFrame using Polars operations...")

            question_series = self.process_question_field_vectorized(df, question_fields)
            answer_series = self.process_answer_field_vectorized(df, answer_field)

            df = df.with_columns(question_series.alias("input_field"))
            df = df.with_columns(answer_series.alias("answer_field"))

            # # Filter out rows where both question and answer are empty
            # processed_df = processed_df.filter(
            #     (pl.col("question") != "") | (pl.col("answer") != "")
            # )

            print(f"Processed dataset '{dataset_name}' - {len(df)} items after filtering")
            return df

        except Exception as e:
            logger.error(f"Error processing dataset '{dataset_name}': {e}")
            return df

def preprocess_dataset(df: pl.DataFrame, dataset_name: str, config: dict) -> pl.DataFrame:
    """
    Process a single dataset according to its configuration using Polars operations.
    This is a convenience function that creates a DatasetProcessor instance.

    Args:
        df: Polars DataFrame containing the dataset
        dataset_name: Name of the dataset
        config: Configuration dictionary with field mappings

    Returns:
        pl.DataFrame: Processed DataFrame with 'question' and 'answer' columns
    """
    processor = DatasetProcessor()
    return processor.preprocess_dataset(df, dataset_name, config)
