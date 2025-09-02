import polars as pl

from axcer.utils.custom_logger import logger
from experiments.utils.field_parser import FieldPathParser


class DatasetProcessor:
    """
    Dataset processor that handles extraction.
    """

    def __init__(self):
        self.field_parser = FieldPathParser()

    def process_question_field_vectorized(  # noqa: PLR0915
        self, df: pl.DataFrame, question_fields: list, dataset_name: str | None = None
    ) -> pl.Series:
        """
        Process question fields by combining them in order using vectorized operations.
        """
        if len(question_fields) == 1:
            field_series = self.field_parser.extract_value_vectorized(df, question_fields[0])
            if dataset_name == "scitldr":
                field_series = field_series.list.join(" ")

            def format_single_field(value):
                if isinstance(value, list):
                    if dataset_name == "scitldr":
                        return " ".join(str(item) for item in value)

                    elif value and isinstance(value[0], str):
                        formatted_options = ", ".join(f"{item}" for _, item in enumerate(value))
                        return f"Options: {formatted_options}"
                    else:
                        print("matin use this")
                        return ", ".join(str(item) for item in value)
                else:
                    return str(value) if value else ""

            return field_series.map_elements(format_single_field, return_dtype=pl.String)

        else:
            # Multiple fields
            field_series_list = []
            for field in question_fields:
                series = self.field_parser.extract_value_vectorized(df, field)
                field_series_list.append(series)

            def combine_fields(*field_values):  # noqa: PLR0912
                combined_parts = []
                formatted_options = []
                counter = 0
                for i, field_value in enumerate(field_values):
                    if isinstance(field_value, list):
                        if field_value and isinstance(field_value[0], str):
                            if dataset_name == "ai2_arc":
                                formatted_options = ", ".join(f"{chr(65 + i)}. {item}" for i, item in enumerate(field_value))
                            elif dataset_name == "mbpp":
                                formatted_options = ", ".join(f"{item}" for _, item in enumerate(field_value))
                                combined_parts.append(f"Assesments: {formatted_options}")
                                continue
                            else:
                                formatted_options = ", ".join(f"{item}" for _, item in enumerate(field_value))
                            combined_parts.append(f"Options: {formatted_options}")
                        else:
                            combined_parts.append(", ".join(str(item) for item in field_value))

                    elif dataset_name == "mbpp":
                        combined_parts.append(f"Prompt: {str(field_value)}")
                        continue
                    elif dataset_name == "glue":
                        if counter == 0:
                            combined_parts.append(f"Text1: {str(field_value)}")
                            counter += 1
                            continue
                        elif counter == 1:
                            combined_parts.append(f"Text2: {str(field_value)}")
                            continue
                    elif dataset_name == "piqa":
                        if i == 0:
                            formatted_options.append(f"{field_value}")
                            continue
                        elif i == 1:
                            formatted_options.append(f"Options: {chr(65 + i - 1)}. {field_value}")
                            continue
                        elif i == 2:
                            formatted_options.append(f"{chr(65 + i - 1)}. {field_value}")
                            combined_parts.append(", ".join(str(item) for item in formatted_options))
                            formatted_options.clear()
                    else:
                        combined_parts.append(str(field_value))

                counter = 0
                return " ".join(combined_parts) if combined_parts else ""

            temp_df = pl.DataFrame({f"field_{i}": series for i, series in enumerate(field_series_list)})

            return temp_df.map_rows(lambda row: combine_fields(*row)).to_series()

    def process_answer_field_vectorized(self, df: pl.DataFrame, answer_field: list, dataset_name: str) -> pl.Series:
        """
        Process answer field using vectorized operations.
        """
        field_series = self.field_parser.extract_value_vectorized(df, answer_field[0])

        def format_single_field(value):
            if isinstance(value, pl.Series) and hasattr(value, "to_list"):
                value = value.to_list()

            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], str):
                    if dataset_name == "mbpp":
                        # to separate items by ; later while we are evaluating (, is a python keyword and used in the examples it's hard to separate by it)
                        formatted_options = "; ".join(f"{item}" for _, item in enumerate(value))
                        return f"{formatted_options}"
                    else:
                        formatted_options = ", ".join(f"{item}" for _, item in enumerate(value))
                        return f"{formatted_options}"
                else:
                    return ", ".join(str(item) for item in value)
            else:
                if dataset_name == "piqa":
                    return "A" if value == 0 else "B"
                elif dataset_name == "gsm8k":
                    return value.split("####")[-1]

                return str(value) if value is not None else ""

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
        answer_field = config.get("answer_field")

        try:
            question_series = self.process_question_field_vectorized(df, question_fields, dataset_name)
            # for scitldr
            if answer_field:
                answer_series = self.process_answer_field_vectorized(df, answer_field, dataset_name)

            df = df.with_columns(question_series.alias("input_field"))
            # for scitldr
            if answer_field:
                df = df.with_columns(answer_series.alias("answer_field"))

            # # Filter out rows where both question and answer are empty
            # processed_df = processed_df.filter(
            #     (pl.col("question") != "") | (pl.col("answer") != "")
            # )

            logger.info(f"Processed dataset '{dataset_name}' - {len(df)} items after filtering")
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
