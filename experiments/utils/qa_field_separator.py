import polars as pl
from typing import Any
from experiments.utils.field_parser import FieldPathParser
from axcer.utils.custom_logger import logger


# Simulated input
# df = pl.DataFrame(
#     {
#         "context": ["This is a story about the orange breasted sunbird found in South Africa."],
#         "questions": [
#             ["How many factors contribute to endemism?", "Is psychological one of those?", "What about biological?"]
#         ],
#         "answers": [["Three.", "No.", "Yes."]],
#         "total_id": 2,
#         "sf": "fsdf",
#     }
# )
#
#
# print(df)


def process_qa_pairs(dt: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Process DataFrame and add qa_pairs column"""
    parser = FieldPathParser()
    questions_field = config["question_field"][0]
    answer_field = config["answer_field"][0]

    input_series = parser.extract_value_vectorized(dt, questions_field)
    answer_series = parser.extract_value_vectorized(dt, answer_field)

    # Create qa_pairs for each row
    qa_pairs_data = []
    for i in range(len(dt)):
        questions = input_series[i] if i < len(input_series) else []
        answers = answer_series[i] if i < len(answer_series) else []

        min_length = min(len(questions), len(answers))
        if len(questions) != len(answers):
            logger.warning(
                f"Row {i}: Question-answer length mismatch: {len(questions)} questions, {len(answers)} answers. Using minimum length: {min_length}"
            )

        row_qa_pairs = [
            {"question": q, "answer": a} for q, a in zip(questions[:min_length], answers[:min_length], strict=False)
        ]
        qa_pairs_data.append(row_qa_pairs)

    dt = dt.with_columns(
        pl.Series(
            "qa_pairs", qa_pairs_data, dtype=pl.List(pl.Struct([pl.Field("question", pl.Utf8), pl.Field("answer", pl.Utf8)]))
        )
    )

    return dt


def separate_qa_fields(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Separate QA fields and explode them into individual rows"""

    df = process_qa_pairs(df, config)

    result = df.explode("qa_pairs").with_columns(
        [
            pl.col("qa_pairs").struct.field("question").alias("question"),
            pl.col("qa_pairs").struct.field("answer").alias("answer"),
        ]
    )

    context_field = config["context_field"][0]
    result = result.with_columns(
        [
            (pl.col(context_field).str.replace("CANNOTANSWER", "").cast(pl.Utf8) + pl.lit("\n") + pl.col("question")).alias(
                "input_field"
            ),
            pl.col("answer").alias("answer_field"),
        ]
    )

    columns_to_drop = ["qa_pairs"]
    if "questions" in result.columns:
        columns_to_drop.append("questions")
    if "answers" in result.columns:
        columns_to_drop.append("answers")

    return result.drop(columns_to_drop)
