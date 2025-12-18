import csv
import os
import re
import string
import textwrap
from pathlib import Path

import modal
import polars as pl

from experiments.constants.paths import VOLUME_NAME, fill_path
from experiments.modals.utils.helper import map_path_to_volume

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def compute_fixed_range_compression_ratio(metric_path: Path):
    df = pl.read_csv(metric_path)
    min_val = df["compression_ratio"].min()
    max_val = df["compression_ratio"].max()

    rng = max_val - min_val

    normalized = (pl.col("compression_ratio") - min_val) / rng

    df = df.with_columns(normalized.alias("compression_ratio_normalized"))
    return df


def save_metrics_to_csv(metrics_list, dataset_names, template_path: Path, model_name: str | None):
    """
    Save each dictionary of metrics into a CSV file with headers.
    One file per dataset, with dataset_names used as filenames.
    """

    for dataset_name, metrics in zip(dataset_names, metrics_list, strict=False):
        output_parent_dir = fill_path(template_path.parent, model_name=model_name)
        output_parent_dir = map_path_to_volume(output_parent_dir)
        output_parent_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir = fill_path(template_path, model_name=model_name, dataset_name=dataset_name)
        output_dir = map_path_to_volume(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_dir.exists()

        with open(output_dir, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

        print(f"Saved {output_dir}")

    try:
        import modal  # type: ignore

        vol = modal.Volume.from_name(VOLUME_NAME)
        vol.commit()
    except Exception:
        # not running in Modal, or volume name not found — ignore
        pass


def is_nested(lst):
    return any(isinstance(x, list) for x in lst)


def format_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Round and scale scores by 100."""
    return {k: round(v * 100, 2) for k, v in raw_scores.items()}


# this also works for squad dataset
def normalize_answer(s: str):
    """Lower text and remove punctuation, articles and extra whitespace."""

    _PATTERN = r"\b(?:a|an|the)\b"  # lowercase only

    def remove_articles(text):
        if len(text) > 1:
            return re.sub(r"\b(a|an|the)\b", " ", text)
        # TO include A as options answer in multiple answers
        else:
            return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def replicate_prediction(references: list[str], candidate_impl: str) -> list[list[str]]:
    if not isinstance(references, list | tuple):
        raise TypeError("`references` must be a list or tuple of test strings")
    if not references:
        raise ValueError("`references` must not be empty")

    # to remove ```python
    if candidate_impl.strip().startswith("```"):
        candidate_impl = re.sub(r"^```(?:python)?\s*", "", candidate_impl.strip(), flags=re.IGNORECASE)
        candidate_impl = re.sub(r"\s*```$", "", candidate_impl.strip())

    cleaned_impl = textwrap.dedent(candidate_impl).strip()
    return [[cleaned_impl] for _ in references]


def compute_pass1(refs: list[str], code: str, code_eval):
    preds = replicate_prediction(refs, code)
    pass_at_k, details = code_eval.compute(references=refs, predictions=preds, k=[1])

    # print("pass@1:", pass_at_k["pass@1"])
    return pass_at_k["pass@1"]


def exact_match(prediction: str, ground_truths: list[str] | str) -> float:
    norm_pred = normalize_answer(prediction)
    if isinstance(ground_truths, list):
        for gt in ground_truths:
            if norm_pred == normalize_answer(gt):
                return 1.0
        return 0.0
    else:
        if norm_pred == normalize_answer(ground_truths):
            return 1.0
        return 0.0


# ---------- modal -----------

app = modal.App("testing evaluate")

image_s = modal.Image.debian_slim(python_version="3.12").pip_install(
    "evaluate", "transformers", "rouge_score", "bert_score", "polars"
)

with image_s.imports():
    pass


# @app.function(gpu="L4", image=image_s)
def calculate_metrics(
    generated_list: list[str],
    answer_list: list[str] | list[list[str] | str],
    dataset_names: list[str],
    rouge,
    bert_score,
    code_eval,
):
    flag = False
    metric_results = []

    if len(generated_list) != len(answer_list):
        raise Exception("Generated list values must match wtih the Answer list values")

    def filter_by_criteria(dataset_names, generated_list, answer_list, target):
        indices = [i for i, val in enumerate(dataset_names) if val == target]
        code_list = [generated_list[i] for i in indices]
        # reference_list = [[part.split(",").strip() for part in answer_list[i]] for i in indices]
        # reference_list = []
        # for i in indices:  # i == [0]
        #     reference_list.append(answer_list[i])
        reference_list = []
        for i in indices:
            reference_list.append(answer_list[i])

        results = []
        for sublist in reference_list:
            new_sublist = []
            for item in sublist:
                if isinstance(item, str):
                    # Split the string by '||;;' and extend the new_sublist with results (for code)
                    new_sublist.extend(part.strip() for part in item.split("||;;"))
                else:
                    new_sublist.extend(item)  # if already list, just extend
            results.append(new_sublist)

        reference_list = results

        return code_list, reference_list, indices

    code_list, reference_list, code_indices = filter_by_criteria(dataset_names, generated_list, answer_list, "mbpp")

    if len(code_list) >= 1:
        code_results = []
        for code, reference in zip(code_list, reference_list, strict=False):
            print("CODE IS", code)
            print("Reference is", reference)
            value = compute_pass1(reference, code, code_eval)

            temp = {"pass@1": float(value)}
            temp = format_scores(temp)
            code_results.append(temp)

    generated_list = [
        normalize_answer(item.replace("/n", "").strip()) if i not in code_indices else item
        for i, item in enumerate(generated_list)
    ]
    if is_nested(answer_list):
        flag = True
        answer_list = [
            [normalize_answer(part.strip()) for item in row for part in item.split("||;;")] if i not in code_indices else row
            for i, row in enumerate(answer_list)
        ]

    else:
        answer_list = [normalize_answer(item) for item in answer_list]

    for pred, gt, dt_name in zip(generated_list, answer_list, dataset_names, strict=False):
        best = {"em": -1.0, "bert_f1": -1.0}
        if dt_name == "mbpp":
            item = code_results.pop(0)
            metric_results.append(item)
            continue

        elif flag:
            temp_list = []
            print("generated answer is", pred)
            print("ground truth is ", gt)
            gt = [item for item in gt if item.strip()]
            temp_dict = {}
            if dt_name == "gsm8k":
                pred = pred.split("final answer")[-1].strip()
            for item in gt:
                em = exact_match(pred, item)
                rouge1_result = rouge.compute(predictions=[pred], references=[item])["rouge1"]
                rouge2_result = rouge.compute(predictions=[pred], references=[item])["rouge2"]
                rougel_result = rouge.compute(predictions=[pred], references=[item])["rougeL"]
                # bleu_result = bleu.compute(predictions=[pred], references=[[item]])["bleu"]
                bert_result = bert_score.compute(predictions=[pred], references=[item], lang="en")
                bert_f1 = bert_result["f1"][0]
                bert_recall = bert_result["recall"][0]
                bert_precision = bert_result["precision"][0]

                temp_dict = {
                    "em": em,
                    "rouge-1": float(rouge1_result),
                    "rouge-2": float(rouge2_result),
                    "rouge-L": float(rougel_result),
                    # "bleu": bleu_result,
                    "bert_precision": bert_precision,
                    "bert_recall": bert_recall,
                    "bert_f1": bert_f1,
                }
                temp_list.append(temp_dict)
                # we found the best match no need to evaluate on others in order to save time
                if em == 1:
                    print("FOUND BEST MATCH !")
                    break

            best = max(temp_list, key=lambda x: (x["em"], x["bert_f1"]))
            print("BEST IS", best)

        else:
            if dt_name == "gsm8k":
                pred = pred.split("final answer")[-1].strip()
            em = exact_match(pred, gt)

            rouge.compute(predictions=[pred], references=[gt])["rouge1"]
            rouge2_result = rouge.compute(predictions=[pred], references=[gt])["rouge2"]
            rougel_result = rouge.compute(predictions=[pred], references=[gt])["rougeL"]

            bert_result = bert_score.compute(predictions=[pred], references=[gt], lang="en")
            best["em"] = best["em"]
            best["bert_f1"] = bert_result["f1"][0]
            best["bert_precision"] = bert_result["precision"][0]
            best["bert_recall"] = bert_result["recall"][0]
            # best["bleu"] = bleu_result
            best["rouge-1"] = rouge1_result
            best["rouge-2"] = rouge2_result
            best["rouge-L"] = rougel_result

        raw_scores = {
            "rouge-1": float(best["rouge-1"]),
            "rouge-2": float(best["rouge-2"]),
            "rouge-L": float(best["rouge-L"]),
            # "bleu": best["bleu"],
            "bertscore_precision": best["bert_precision"],
            "bertscore_recall": best["bert_recall"],
            "bertscore_f1": best["bert_f1"],
        }

        formatted_scores = format_scores(raw_scores)
        metrics = {
            "exact_match": best["em"],
            **formatted_scores,
        }
        metric_results.append(metrics)

    return metric_results


@app.local_entrypoint()
def main_run():
    answer_list = [
        [
            """
In recent years, genetic algorithms (GAs) have proven effective for solving complex discrete optimization problems, particularly in scheduling and timetabling. However, standard GAs struggle with constraints, requiring modifications to incorporate problem-specific knowledge.

This work develops a family of modified GAs to solve the nurse rostering problem at a major UK hospital, where each ward (up to 30 nurses) requires weekly shift schedules. The schedules must satisfy:

    Minimum staffing for three daily shifts

    Nurses’ personal preferences and qualifications

    Fair distribution of unpopular shifts

    Additional rules like team nursing and senior staff conditions

The base GA uses n-point crossover, single-bit mutation, and rank-based selection. The solution space ensures each nurse works the correct number of shifts, but other hard and soft constraints are incorporated into the fitness function as penalties.

The paper describes the problem and initial GA implementation, identifying its main challenge: balancing feasibility (demand coverage and work regulations) with quality (nurse preferences). A range of enhancements were tested, including:

    Parameter adaptation

    Niching

    Intelligent weight adjustments

    Delta coding

    Local hill climbing

    Migration strategis

    Special selection rules

Experiments using real hospital data over several months demonstrated that these improvements resolved the initial shortcomings. The final GA performed competitively with the hospital’s current tabu search method. The study concludes that the approach is effective for this and similar scheduling problems.
        """
        ],
        [
            """
This paper revisits the Recurrent Attention Model (RAM) from the perspective of active information sampling, inspired by neuroscience research. The original RAM, which sequentially samples visual information, is found to implement only one of three proposed motives for active sampling. The authors identify three key weaknesses in RAM: slow convergence, fixed number of glimpses per sample, and performance degradation with more glimpses. They propose a simple modification by adding two new terms to the objective function, inspired by intrinsic motivation and uncertainty reduction. The modified RAM converges faster, supports dynamic glimpse counts, and generalizes better to longer sequences. Experiments on MNIST show improved performance and stability, especially with increased glimpses.
        """
        ],
        [
            "assert power(3,4) == 81; assert power(2,3) == 8; assert power(5,5) == 3125",
        ],
        [
            """
The document discusses the variational autoencoder (VAE) model and its objective. The VAE approximates a ground-truth probability measure using a parameterized density defined across all of R^d. The ca
nonical VAE cost is a bound on the average negative log-likelihood, which can be expressed in a form amenable to SGD optimization via a reparameterization trick. The document then investigates the impli
cations of VAE's Gaussian assumptions and proves that when the number of dimensions of the observable variables is less than the number of dimensions of the ambient space, the VAE global optimum can be
reached by solutions that reflect the ground-truth distribution almost everywhere, but not necessarily uniquely so. The document also proposes a two-stage VAE enhancement for addressing typical regimes
when the number of dimensions of the observable variables is less than the number of dimensions of the ambient space. The two-stage VAE can generate high-quality samples and produce stable FID scores co
mparable to GAN models. The code for the model is available at <https://github.com/daib13/TwoStageVAE>.
            """,
        ],
        [
            "assert find_Average_Of_Cube(2) == 4.5",
            " assert find_Average_Of_Cube(3) == 12",
            " assert find_Average_Of_Cube(1) == 1",
        ],
    ]
    generated_list = [
        # "The cat sat on the mat.",
        """
In recent years, genetic algorithms (GAs) have become valuable heuristic tools for solving complex discrete optimization problems, notably in scheduling and timetabling. However, the standard GA framework struggles with constraint handling, often requiring modifications to integrate problem-specific knowledge.

This paper focuses on developing a family of genetic algorithms to address the nurse rostering problem in a major UK hospital, which comprises multiple wards with 30 nurses each. Each ward schedules its own nurses on a weekly basis. The scheduling must meet several requirements:

    Hard constraints: minimum staffing levels for three daily shifts, compliance with regulations, team nursing rules, and special conditions for senior staff.

    Soft constraints: nurse preferences, fairness in distributing unpopular shifts, and qualification considerations.

The base algorithm used is a classical GA with n-point crossover, single-bit mutation, and rank-based selection. The solution space includes all schedules meeting the required number of shifts per nurse, while other constraints are treated as penalties in the fitness function.

Initial implementation revealed shortcomings in balancing feasibility (meeting demand and regulations) and quality (nurse preferences). To address this, a series of enhancements were tested, including:

    Parameter adaptation

    Niching

    Intelligent weighting of constraints

    Delta coding

    Local hill-climbing

    Migration strategies

    Special selection rules

Experiments using several months of real hospital data showed these enhancements eliminated earlier issues, producing a final GA that could compete with the hospital’s existing tabu search method. The paper concludes with observations on the approach’s overall quality and applicability to similar scheduling problems.
        """,
        """The Recurrent Attention Model (RAM) is a visual attention mechanism inspired by active information sampling in neuroscience. The original RAM uses a fixed number of glimpses to make predictions, but it suffers from three main weaknesses: slow convergence, lack of dynamic glimpse allocation per sample, and inconsistent performance improvements with more glimpses.

To address these, two new terms—J_intrinsic and J_uncertainty—were added to the objective function, inspired by three motives for active sampling proposed by Gottlieb (2018). These modifications lead to faster convergence, enable dynamic glimpse allocation without sacrificing accuracy, and improve generalization on longer glimpse sequences.

Experiments on MNIST showed that the modified RAM outperformed the original in convergence speed and stability. Adding both terms together yielded the fastest learning. In dynamic settings, the model could adjust the number of glimpses per sample, though performance varied depending on sample complexity. The intrinsic term particularly helped stabilize predictions with more glimpses, while the uncertainty term provided slight improvements. Overall, the revised RAM offers a more efficient and flexible approach to visual attention.""",
        "def power(a, b):return a ** b",
        "something",
        """
def find_Average_Of_Cube(n):
    sum = 0
    for i in range(1, n + 1):
        sum += i**3
    return sum / n
        """,
    ]
    dataset_names = ["something", "something", "mbpp", "something", "mbpp"]

    metrics = calculate_metrics.remote(generated_list, answer_list, dataset_names)

    print(metrics)
    for idx, sample in enumerate(metrics, start=1):
        if sample.get("pass@1") is not None:
            print(sample["pass@1"])
            continue
        print(f"Index {idx}")
        print(sample["exact_match"])
        print(sample["rouge-1"])
        print(sample["rouge-2"])
        print(sample["rouge-L"])
        print(sample["bertscore_precision"])
        print(sample["bertscore_recall"])
        print(sample["bertscore_f1"])


if __name__ == "__main__":
    main_run()
