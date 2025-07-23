import polars as pl
from typing import Any


def process_mawps(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    df = df.sample(n=df.height, seed=42)

    # 70/30 train , test split
    total_rows = df.shape[0]
    split_index = int(total_rows * 0.7)

    _ = df[:split_index]
    test_df = df[split_index:]
    df_columns = test_df.columns
    test_df = (
        test_df.with_columns(pl.struct(pl.all()).map_elements(process_row).alias("processed"))
        .drop(df_columns)
        .unnest("processed")
    )

    answer_field = config["answer_field"][0]
    question_field = config["question_field"][0]

    test_df = test_df.rename({question_field: "input_field", answer_field: "answer_field"})

    return test_df


def process_row(row):
    # Create placeholder-to-actual mapping
    placeholder_map = [(f"N_{i:02}", actual) for i, actual in enumerate(row["Numbers"].split())]

    try:
        for placeholder, replacement in placeholder_map:
            replacement_value = float(replacement)
            # Convert to integer if it is a whole number, otherwise round to 2 decimal places
            replacement = str(int(replacement_value)) if replacement_value.is_integer() else str(round(replacement_value, 2))  # noqa: PLW2901
            # Update Question and Equation with the new replacement value
            row["Question"] = row["Question"].strip().replace(placeholder, replacement)
            row["Equation"] = row["Equation"].strip().replace(placeholder, replacement)

        # Process Answer
        answer_value = float(row["Answer"])
        row["Answer"] = str(int(answer_value)) if answer_value.is_integer() else str(round(answer_value, 2))

        row["Answer"] = f"{row['Equation']} = {row['Answer']} #### {row['Answer']}"

    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return row
