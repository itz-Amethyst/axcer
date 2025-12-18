import os
import shutil
from pathlib import Path

import polars as pl
import polars.selectors as cs

from experiments.constants.paths import (
    AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
    fill_path,
)


# template path is whether DIVIDE_DATASETS_CATEGORY_ORIGINAL_PATH or DIVIDE_DATASETS_CATEGORY_AXCER_PATH
def copy_processed_datasets_to_category(
    template_path: Path, category_name: str, model_name: str, *csv_paths: str | Path
) -> list[Path]:
    """
    Copy multiple CSV files into a folder named after the given category.

    Args:
        category_name (str): Name of the category folder to copy files into.
        *csv_paths (str | Path): CSV file paths to copy.

    Returns:
        list[Path]: Paths of the copied files in the new category folder.
    """

    category_path = fill_path(template_path, model_name=model_name, category_name=category_name)
    # category_folder = Path(category_name)
    category_path.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for csv_path in csv_paths:
        src = Path(csv_path)
        if not src.exists():
            raise FileNotFoundError(f"CSV file not found: {src}")

        dst = category_path / src.name
        shutil.copy(src, dst)
        copied_files.append(dst)

    return copied_files


def concat_and_save_with_avg(
    input_dir: str,
    file_names: list[str],
    concat_output_path: Path | None | str,
    avg_output_path: Path | str,
    output_name: str = "qa.csv",
) -> None:
    """
    Concatenate selected CSV files horizontally with Polars,
    save the merged results, and also save column-wise averages.
    """

    dataframes = []
    model_name = Path(input_dir).name
    if concat_output_path:
        concat_output_path = fill_path(concat_output_path, model_name=model_name, category_name=output_name.split(".")[0])
    avg_output_path = fill_path(avg_output_path, model_name=model_name, category_name=output_name.split(".")[0])
    print("AVG path is", avg_output_path)

    for fname in file_names:
        file_path = os.path.join(input_dir, fname)
        if os.path.exists(file_path) and fname.endswith(".csv"):
            df = pl.read_csv(file_path)
            dataframes.append(df)
        else:
            print(f"File not found or not a CSV: {fname}")

    if not dataframes:
        print("No valid CSV files found.")
        return

    for dt in dataframes:
        print(dt.head())
    merged_df = pl.concat(dataframes, how="vertical")

    if concat_output_path:
        os.makedirs(concat_output_path.parent, exist_ok=True)
        merged_df.write_csv(concat_output_path)
    os.makedirs(avg_output_path.parent, exist_ok=True)

    df_mean = merged_df.mean()
    df_mean = df_mean.with_columns((cs.by_dtype(pl.Float32) | cs.by_dtype(pl.Float64)).round(3))
    renamed_df = df_mean.rename({col: f"{col}_avg" for col in df_mean.columns})

    renamed_df.write_csv(avg_output_path)


if __name__ == "__main__":
    # this is for with_interrogative
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/Meta-Llama-3.1-8B-Instruct",
    #     file_names=["squad.csv", "piqa.csv", "ai2_arc.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Qa.csv",
    # )

    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["squad.csv", "piqa.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="Qa.csv",
    )
    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["ai2_arc.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="MCQ.csv",
    )
    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["boolq.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="BinaryCl.csv",
    )
    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["glue.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="NLI.csv",
    )

    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["gsm8k.csv", "MAWPS.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="Math.csv",
    )

    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["scitldr.csv"],
        concat_output_path=None,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="Summary.csv",
    )

    concat_and_save_with_avg(
        # check the path
        input_dir="/home/itz-amethyst/dev/axcer/experiments/results/selective_context/perplexities/gemma-3-12b-it/",
        file_names=["mbpp.csv"],
        concat_output_path=None,
        # concat_output_path=CONCATED_WITHOUT_INTERROGATIVE_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
        avg_output_path=AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH,
        output_name="mbpp.csv",
    )

    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/Meta-Llama-3.1-8B/",
    #     file_names=["scitldr.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Summary.csv",
    # )
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/gemma/",
    #     file_names=["scitldr.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Summary.csv",
    # )
    #
    # # without interrogative
    #
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/without_interrogative/Meta-Llama-3.1-8B/",
    #     file_names=["squad.csv", "coqa.csv", "file3.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Qa.csv",
    # )
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/without_interrogative/gemma/",
    #     file_names=["squad.csv", "coqa.csv", "file3.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Qa.csv",
    # )
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/without_interrogative/Meta-Llama-3.1-8B/",
    #     file_names=["scitldr.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Summary.csv",
    # )
    # concat_and_save_with_avg(
    #     # check the path
    #     input_dir="/home/itz-amethyst/dev/axcer/experiments/results/axcer/without_interrogative/gemma/",
    #     file_names=["scitldr.csv"],
    #     concat_output_path=CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     avg_output_path=AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH,
    #     output_name="Summary.csv",
    # )
