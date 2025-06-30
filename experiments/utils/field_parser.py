import re
from typing import Any
import polars as pl
from axcer.utils.custom_logger import logger


class FieldPathParser:
    """
    A robust parser for handling complex field paths in nested data structures.
    Supports JSONPath-like syntax for accessing nested fields.
    Optimized for Polars DataFrame operations.
    """

    def __init__(self):
        self.path_pattern = re.compile(r"(\w+)|\[([^\]]+)\]")

    def parse_path(self, path: str) -> list[tuple]:
        """Parse a field path into components."""
        return self.path_pattern.findall(path)

    def extract_value(self, data: Any, path: str) -> Any:
        """
        Extract value from data using field path.
        Supports:
        - Simple fields: 'field'
        - Array indices: 'field[0]'
        - Dict keys: 'field[key]' or 'field["key"]'
        - Nested paths: 'field[key][0][subkey]'
        """
        if not path:
            return data

        components = self.parse_path(path)
        current = data

        for field, index in components:
            if current is None:
                return None

            if field:
                if isinstance(current, dict):
                    current = current.get(field)
                elif hasattr(current, field):
                    current = getattr(current, field)
                else:
                    return None

            elif index is not None:
                try:
                    current[int(index) if index.isdigit() else current[index]]
                except (KeyError, IndexError, TypeError):
                    return None

        return current

    def extract_value_vectorized(self, df: pl.DataFrame, field_path: str) -> pl.Series:
        """
        Extract values from DataFrame column using field path with vectorized operations.
        Falls back to simple column access for backward compatibility.
        """
        if not field_path:
            return pl.Series("empty", [""] * len(df))

        try:
            if "." not in field_path and "[" not in field_path:
                if field_path in df.columns:
                    return df[field_path]
                else:
                    return pl.Series(field_path, [""] * len(df))

            def extract_from_row(row_dict):
                value = self.extract_value(row_dict, field_path)
                return value if value is not None else ""

            extracted_values = []
            for row in df.iter_rows(named=True):
                extracted_values.append(extract_from_row(row))

            return pl.Series(field_path, extracted_values)

        except Exception as e:
            logger.warning(f"Error extracting field '{field_path}': {e}")
            return pl.Series(field_path, [""] * len(df))
