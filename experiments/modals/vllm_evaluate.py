from pathlib import Path
import modal
from experiments.constants.paths import (
    AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_PATH,
    AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH,
    AVG_OUTPUT_RESULT_FOR_EACH_DATASET_LINGUA2_PATH,
    AVG_OUTPUT_RESULT_FOR_EACH_DATASET_ORIGINAL_PATH,
    AVG_OUTPUT_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH,
    METRICS_AXCER_PATH,
    METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH,
    METRICS_LINGUA2_PATH,
    METRICS_ORIGINAL_PATH,
    METRICS_SELECTIVE_PATH,
    PROCESSED_AXCER_PATH,
    PROCESSED_AXCER_WITHOUT_INTERROGATIVE_PATH,
    PROCESSED_LINGUA2_PATH,
    PROCESSED_SELECTIVE_PATH,
    VOLUME_MOUNT_POINT,
    VOLUME_NAME,
    fill_path,
)
from experiments.constants.system_prompts import DATASET_SYSTEM_PROMPTS
from experiments.evaluation.utils.align_datasets import concat_two_metrics
from experiments.evaluation.utils.avg_compute import compute_multiple_file_csv_column_averages
from experiments.modals.utils.helper import snapshot_config
from experiments.modals.utils.build_parameters import get_vllm_config
from experiments.modals.utils.constants import EvaluateConfig, config
from experiments.modals.utils.helper import map_path_to_volume, pair_files
from experiments.modals.utils.metrics import calculate_metrics, save_metrics_to_csv

hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("axcer_evaluate_vllm", secrets=[hf_secret])

vllm_image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "git-lfs").pip_install("uv")

vllm_image = vllm_image.pip_install(
    "vllm==0.10.0",
    "huggingface_hub[hf_transfer]==0.34.1",
    "flashinfer-python==0.2.6.post1",
    "optimum",
    "evaluate==0.4.5",
    "bert-score==0.3.13",
    "rouge-score==0.1.2",
    extra_index_url="https://download.pytorch.org/whl/cu128",
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        "VLLM_USE_V1": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)


vllm_image = vllm_image.uv_pip_install(
    # "huggingface_hub[hf_transfer]==0.34.1",
    # "flashinfer-python==0.2.6.",
    "polars==1.31.0",
    "transformers==4.54.0",
)

vllm_image = vllm_image.add_local_python_source("experiments", copy=False)


volume = modal.Volume.from_name("axcer_volume", create_if_missing=True)


vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

results_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


with vllm_image.imports():
    import time

    from vllm import SamplingParams, LLM
    import evaluate


@app.cls(
    gpu="A100-80GB",
    image=vllm_image,
    volumes={"/vol": volume, "/root/.cache/vllm": vllm_cache_vol, str(VOLUME_MOUNT_POINT): results_vol},
    cpu=EvaluateConfig.TOTAL_CPU_CORES,
    max_containers=1,
    memory=4096,  # 4GB
    secrets=[hf_secret],
    timeout=24 * 60 * 60,  # 24 hours
)
@modal.concurrent(max_inputs=3)
class VllmEvaluate:
    import polars as pd
    import modal

    @modal.enter()
    def setup(
        self,
    ):
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        #
        # self.tokenizer.eos_token = "<|eot_id|>"
        # self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # # self.tokenizer.chat_template = llama_template
        # self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        # self.tokenizer.padding_side = "left"
        # # self.pad_id = self.tokenizer.pad_token_id
        # self.end_id = self.tokenizer.eos_token_id
        # self.stop_token_ids = [
        #     self.end_id,
        # ]

        print("Setting up the evaluate utilities")

        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.code_eval = evaluate.load("code_eval")
        # self.bleu = evaluate.load("bleu")
        self.model_id = config["MODEL_ID"]

        self.llm = LLM(
            model=self.model_id,
            download_dir=f"/vol/model/{config['MODEL_NAME']}",
            **get_vllm_config(),
        )
        # self.llm.get_tokenizer().eos_token = "<|eot_id|>"
        # self.llm.get_tokenizer().eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # self.llm.get_tokenizer().add_special_tokens({"pad_token": self.tokenizer.eos_token})
        # self.llm.get_tokenizer().padding_side = "left"
        # needs a explicit handling for each llm the eot_id is only for llama
        self.terminators = [
            self.llm.get_tokenizer().eos_token_id,
        ]
        self.sampling_params = SamplingParams(
            max_tokens=config["MAX_OUTPUT_LEN"],
            min_tokens=1,
            temperature=0.0,
            top_p=1,
            stop_token_ids=self.terminators,
        )
        match config["BASELINE"]:
            case "selective_context":
                self.metrics_parent_path = METRICS_SELECTIVE_PATH.parent
                self.metrics_path = METRICS_SELECTIVE_PATH
                self.baseline_processed_parent_path = PROCESSED_SELECTIVE_PATH.parent
                self.avg_path = AVG_OUTPUT_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH
            case "lingua2":
                self.metrics_parent_path = METRICS_LINGUA2_PATH.parent
                self.metrics_path = METRICS_LINGUA2_PATH
                self.baseline_processed_parent_path = PROCESSED_LINGUA2_PATH.parent
                self.avg_path = AVG_OUTPUT_RESULT_FOR_EACH_DATASET_LINGUA2_PATH
            case "axcer":
                if not config["BASELINE_WITHOUT_INTERROGATIVE"]:
                    self.metrics_parent_path = METRICS_AXCER_PATH.parent
                    self.metrics_path = METRICS_AXCER_PATH
                    self.baseline_processed_parent_path = PROCESSED_AXCER_PATH.parent
                    self.avg_path = AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_PATH
                else:
                    self.metrics_parent_path = METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH.parent
                    self.metrics_path = METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH
                    self.baseline_processed_parent_path = PROCESSED_AXCER_WITHOUT_INTERROGATIVE_PATH.parent

                    self.avg_path = AVG_OUTPUT_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH
            case _:
                print("Going with original")
                self.metrics_parent_path = METRICS_ORIGINAL_PATH.parent
                self.metrics_path = METRICS_ORIGINAL_PATH
                self.avg_path = AVG_OUTPUT_RESULT_FOR_EACH_DATASET_ORIGINAL_PATH

    def construct_prompt(
        self,
        prompts: list[str] | str,
        system_prompts: list[str] | None | str,
    ):
        if not system_prompts:
            print("USING DEFAULT SYSTEM PROMTP !")
            system_prompts = "You are a helpful assistant. You will be given a context and a question. Your only task is to provide a direct and short answer to the question, based exclusively on the provided context. Do not offer any reasoning, explanations, or additional information. Your answer should be a single phrase or sentence, and nothing more."

        batch_messages = [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": prompts},
        ]

        return batch_messages

    # TODO: IF you hit preemption try to reduce the max batch size
    @modal.batched(max_batch_size=27, wait_ms=500)
    async def inference(self, prompts: list[str]):
        import torch

        results = []
        unpacked = [prompt.split(" |||| ") for prompt in prompts]

        dataset_names = [parts[0].removeprefix("processed_") for parts in unpacked]
        inputs = [parts[1].removeprefix("I: ").strip() for parts in unpacked]
        answers = [[parts[2].removeprefix("A: ").strip()] for parts in unpacked]

        results: list[str] = []
        inference_list: list[float] = []
        for prompt, dt_name in zip(inputs, dataset_names, strict=False):
            system_prompt = DATASET_SYSTEM_PROMPTS[dt_name.lower()]
            start_time = time.perf_counter()
            tokenized_inputs = self.construct_prompt(prompt, system_prompt)
            generated_response = self.llm.chat(tokenized_inputs, self.sampling_params, use_tqdm=False)
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            results.append(generated_response[0].outputs[0].text)
            inference_list.append(inference_time)

        metrics_list = calculate_metrics(results, answers, dataset_names, self.rouge, self.bertscore, self.code_eval)

        for m, i in zip(metrics_list, inference_list, strict=False):
            m["inference_time"] = i

        print("Metrics are", metrics_list)
        save_metrics_to_csv(metrics_list, dataset_names, self.metrics_path, config["MODEL_NAME"])

        torch.cuda.empty_cache()

        return results

    @modal.exit()
    async def compute_extra_evaluations(self):
        import modal

        print("Baseline: ", config["BASELINE"])
        print("Starting to align compression metrics with generation metrics")
        metrics_paths = map_path_to_volume(self.metrics_parent_path)
        metrics_paths = fill_path(metrics_paths, model_name=config["MODEL_NAME"])

        if config["BASELINE"]:
            baseline_paths = map_path_to_volume(self.baseline_processed_parent_path)
            print(f"DEBUG : {metrics_paths}, {baseline_paths}")
            for baseline_path, mtric_path in pair_files(baseline_paths, metrics_paths):
                print(f"Concating: {baseline_path} with {mtric_path}")
                concat_two_metrics(
                    baseline_path,
                    mtric_path,
                    [
                        "prompt_tokens",
                        "compressed_tokens",
                        "compression_ratio",
                        "gpt_o1_saving",
                        "compression_time",
                        "compression_ratio_normalized",
                    ],
                )

        print("running avg")

        compute_multiple_file_csv_column_averages(
            base_read_path=metrics_paths, base_write_path=self.avg_path, model_name=config["MODEL_NAME"]
        )
        print("finished avg")
        vol = modal.Volume.from_name(VOLUME_NAME)
        vol.commit()


@app.local_entrypoint()
async def main(model_id: str, model_revision: str, max_batch_size: int, dataset_path: str):
    import asyncio
    from experiments.modals.utils.prepare_datasets import load_datasets

    EvaluateConfig.set_essential_parameters(model_id, model_revision, max_batch_size)
    snapshot = snapshot_config()
    for key, value in snapshot.items():
        await config.put.aio(key, value)
    print(config["MODEL_ID"])
    config["BASELINE"] = None
    config["BASELINE_WITHOUT_INTERROGATIVE"] = False
    # INFO: later once we wanted to do evaluation for selective and other baselines we change the dataset_path to merged path of their baseline_path
    # INFO: the paths are : dataset_path, MERGED_COMPRESSED_AXCER_WITH_.. , MERGED_COMPRESSED_SELECTIVE, MERGED_COMPRESSED_LINGUA

    datasets = load_datasets(Path(dataset_path), question_col="input_field", answer_col="answer_field")

    if "lingua2" in str(dataset_path):
        config["BASELINE"] = "lingua2"
    elif "selective_context" in str(dataset_path):
        config["BASELINE"] = "selective_context"
    # to skip axcer package name
    elif str(dataset_path).split("/").count("axcer") > 1:
        config["BASELINE"] = "axcer"
        if "without_interrogative" in str(dataset_path):
            print("Conducting w/o interrogative")
            config["BASELINE_WITHOUT_INTERROGATIVE"] = True
        else:
            print("Going with interrogative")

    vllm_e = VllmEvaluate()

    async def process_dataset_vllm(dataset_name, df, question_col, answer_col):
        prompts = df[question_col].to_list()
        answers = df[answer_col].to_list()
        prompts = [f"{dataset_name} |||| I: {q} |||| A: {a}" for q, a in zip(prompts, answers, strict=False)]
        print(f"[{dataset_name}] submitting {len(prompts)} prompts...")
        async for _batch in vllm_e.inference.map.aio(prompts):
            pass

        print(f"[{dataset_name}] done")

    await asyncio.gather(*(process_dataset_vllm(name, df, "input_field", "answer_field") for name, df in datasets.items()))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
