from pathlib import Path

# from experiments.modals.utils.constants import config

VOLUME_MOUNT_POINT = Path("/outputs")  # the path you mount the modal.Volume to
DATASET_VOLUME_MOUNT_POINT = Path("/datasets")
TEMP_SECOND_ATTEMPT_DATASET_VOLUME_MOUNT_POINT = Path("/datasets/trimmed")
VOLUME_NAME = "results-vol"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_EXP_PATH: Path = PROJECT_ROOT / Path("experiments") / "results"

DATASET_PATH = PROJECT_ROOT / "experiments" / "datasets"

SELECTIVE_PATH = PROJECT_ROOT / BASE_EXP_PATH / "selective_context"

LINGUA2_PATH = PROJECT_ROOT / BASE_EXP_PATH / "lingua2"

ORIGINAL_PATH = BASE_EXP_PATH / "original"

AXCER_PATH = BASE_EXP_PATH / "axcer"

PROCESSED_AXCER_PATH = AXCER_PATH / "processed" / "{dataset_name}.csv"

PROCESSED_AXCER_WITHOUT_INTERROGATIVE_PATH = AXCER_PATH / "processed" / "without_interrogative" / "{dataset_name}.csv"

PROCESSED_SELECTIVE_PATH = SELECTIVE_PATH / "processed" / "{dataset_name}.csv"

PROCESSED_LINGUA2_PATH = LINGUA2_PATH / "processed" / "{dataset_name}.csv"

MERGED_COMPRESSED_AXCER_WITH_INTERROGATIVE_DATASET_PATH = (
    AXCER_PATH / "merged" / "with_interrogative" / "{dataset_name}.parquet"
)

MERGED_COMPRESSED_AXCER_WITHOUT_INTERROGATIVE_WITH_DATASET_PATH = (
    AXCER_PATH / "merged" / "without_interrogative" / "{dataset_name}.parquet"
)

MERGED_COMPRESSED_SELECTIVE_WITH_DATASET_PATH = SELECTIVE_PATH / "merged" / "{dataset_name}.parquet"

MERGED_COMPRESSED_LINGUA2_WITH_DATASET_PATH = LINGUA2_PATH / "merged" / "{dataset_name}.parquet"

# For each
METRICS_ORIGINAL_PATH = ORIGINAL_PATH / "{model_name}" / "{dataset_name}.csv"
METRICS_TEMP_ORIGINAL_PATH = ORIGINAL_PATH / "{model_name}" / "temp" / "{dataset_name}.csv"

METRICS_ORIGINAL_PATH_TEMP = ORIGINAL_PATH / "{model_name}" / "temp" / "{dataset_name}.csv"

METRICS_AXCER_PATH = AXCER_PATH / "with_interrogative" / "{model_name}" / "{dataset_name}.csv"
METRICS_AXCER_PATH_PERPLEXITY = AXCER_PATH / "with_interrogative" / "perplexities" / "{model_name}" / "{dataset_name}.csv"

METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH = AXCER_PATH / "without_interrogative" / "{model_name}" / "{dataset_name}.csv"
METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH_PERPLEXITY = (
    AXCER_PATH / "without_interrogative" / "{model_name}" / "{dataset_name}.csv"
)

METRICS_SELECTIVE_PATH = SELECTIVE_PATH / "{model_name}" / "{dataset_name}.csv"
METRICS_SELECTIVE_PATH_PERPLEXITY = SELECTIVE_PATH / "perplexities" / "{model_name}" / "{dataset_name}.csv"
# test purpose
METRICS_SELECTIVE_PATH_TEST = SELECTIVE_PATH / "{model_name}" / "testing" / "{dataset_name}.csv"


METRICS_LINGUA2_PATH = LINGUA2_PATH / "{model_name}" / "{dataset_name}.csv"
METRICS_LINGUA2_PATH_PERPLEXITY = LINGUA2_PATH / "perplexities" / "{model_name}" / "{dataset_name}.csv"

# for each dataset avg final result in table 1
AVG_OUTPUT_RESULT_FOR_EACH_DATASET_ORIGINAL_PATH = ORIGINAL_PATH / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_PATH = (
    AXCER_PATH / "with_interrogative" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)
AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH = (
    AXCER_PATH / "without_interrogative" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)
AVG_OUTPUT_RESULT_FOR_EACH_DATASET_LINGUA2_PATH = LINGUA2_PATH / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
AVG_OUTPUT_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH = SELECTIVE_PATH / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
# perplexities
AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH = (
    SELECTIVE_PATH / "perplexities" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)
AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_LINGUA2_PATH = (
    LINGUA2_PATH / "perplexities" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)
AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_PATH = (
    AXCER_PATH / "with_interrogative" / "perplexities" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)
AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH = (
    AXCER_PATH / "without_interrogative" / "perplexities" / "{model_name}" / "avg" / "{dataset_name}_avg.csv"
)

AVG_OUTPUT_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH_TEST = (
    SELECTIVE_PATH / "{model_name}" / "testing" / "avg" / "{dataset_name}_avg.csv"
)

# INFO:
# done: concat_metrics_into_one_category
# done: compute_csv_category_column_averages

CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH = (
    AXCER_PATH / "with_interrogative" / "{model_name}" / "{category_name}" / "with_interrogative.csv"
)

CONCATED_WITHOUT_INTERROGATIVE_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH = (
    AXCER_PATH / "without_interrogative" / "{model_name}" / "{category_name}" / "without_interrogative.csv"
)
CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_SELECTIVE_PATH = SELECTIVE_PATH / "{model_name}" / "{category_name}" / "selective.csv"

CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_LINGUA2_PATH = LINGUA2_PATH / "{model_name}" / "{category_name}" / "lingua2.csv"

# perplexities
AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_SELECTIVE_PATH = (
    SELECTIVE_PATH / "perplexities" / "{model_name}" / "{category_name}" / "selective.csv"
)
AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_LINGUA2_PATH = (
    LINGUA2_PATH / "perplexities" / "{model_name}" / "{category_name}" / "lingua2.csv"
)
AVG_CONCATED_OUTPUT_PERPLEXITY_RESULTS_INTO_CATEGORY_AXCER_PATH = (
    AXCER_PATH / "perplexities" / "{model_name}" / "{category_name}" / "axcer.csv"
)

AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH = (
    AXCER_PATH / "with_interrogative" / "{model_name}" / "{category_name}" / "with_interrogative_avg.csv"
)

AVG_CONCATED_WITHOUT_INTERROGATIVE_OUTPUT_RESULTS_INTO_CATEGORY_AXCER_PATH = (
    AXCER_PATH / "without_interrogative" / "{model_name}" / "{category_name}" / "without_interrogative_avg.csv"
)

AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_SELECTIVE_PATH = (
    SELECTIVE_PATH / "{model_name}" / "{category_name}" / "selective_avg.csv"
)

AVG_CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_LINGUA2_PATH = LINGUA2_PATH / "{model_name}" / "{category_name}" / "lingua2_avg.csv"


def fill_path(template: Path, **kwargs) -> Path:
    """
    Fill a path template with parameters.
    Example:
        get_path(CONCATED_OUTPUT_RESULTS_INTO_CATEGORY_ORIGINAL_PATH,
                 model_name="bert", category_name="qa")
    """
    return Path(str(template).format(**kwargs))


def get_actual_paths(
    template: Path, dataset_names: list[str], *, category_name: str = None, model_name: str = None
) -> list[Path]:
    template_str = str(template)

    if "{category_name}" in template_str:
        if not category_name:
            raise ValueError("Template requires category_name but none was provided")
        return [fill_path(template, dataset_name=ds, category_name=category_name) for ds in dataset_names]

    elif "{model_name}" in template_str:
        if not model_name:
            raise ValueError("Template requires model_name but none was provided")
        return [fill_path(template, dataset_name=ds, model_name=model_name) for ds in dataset_names]

    else:
        raise ValueError(f"Template must contain either {category_name} or {model_name}")
