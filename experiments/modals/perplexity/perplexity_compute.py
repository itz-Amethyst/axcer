from pathlib import Path

import modal

from experiments.constants.paths import (
    AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_PATH,
    AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH,
    AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_LINGUA2_PATH,
    AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH,
    METRICS_AXCER_PATH_PERPLEXITY,
    METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH_PERPLEXITY,
    METRICS_LINGUA2_PATH_PERPLEXITY,
    METRICS_SELECTIVE_PATH_PERPLEXITY,
    VOLUME_MOUNT_POINT,
    VOLUME_NAME,
    fill_path,
)
from experiments.modals.perplexity.utils import save_metrics_to_csv
from experiments.modals.utils.constants import EvaluateConfig, config
from experiments.modals.utils.helper import map_path_to_volume, snapshot_config

hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("axcer_evaluate_perplexity", secrets=[hf_secret])

vllm_image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "git-lfs").pip_install("uv")

vllm_image = vllm_image.pip_install(
    "huggingface_hub[hf_transfer]==0.34.1",
    "optimum",
    "evaluate==0.4.5",
    extra_index_url="https://download.pytorch.org/whl/cu128",
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)


vllm_image = vllm_image.uv_pip_install(
    "polars==1.31.0",
    "accelerate==1.11.0",
    "transformers==4.54.0",
)

vllm_image = vllm_image.add_local_python_source("experiments", copy=False)


volume = modal.Volume.from_name("axcer_volume", create_if_missing=True)

hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

results_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


with vllm_image.imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(
    gpu="A100-80GB",
    image=vllm_image,
    volumes={"/vol": volume, "/root/.cache/huggingface": hf_cache_vol, str(VOLUME_MOUNT_POINT): results_vol},
    cpu=EvaluateConfig.TOTAL_CPU_CORES,
    max_containers=1,
    memory=4096,  # 4GB
    secrets=[hf_secret],
    timeout=24 * 60 * 60,  # 24 hours
)
# @modal.concurrent(max_inputs=9)
@modal.concurrent(max_inputs=3)
class VllmEvaluate:
    import modal
    import polars as pd

    @modal.enter()
    def setup(
        self,
    ):
        cache_dir = "/root/.cache/huggingface"

        self.model_id = config["MODEL_ID"]
        self.device = "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            device_map="auto",  # Let transformers handle device placement
        ).to(self.device)

        self.model.eval()

        match config["BASELINE"]:
            case "selective_context":
                self.metrics_perplexity_path = METRICS_SELECTIVE_PATH_PERPLEXITY
                self.parent_metrics_perplexity_path = METRICS_SELECTIVE_PATH_PERPLEXITY.parent
                self.perplexity_avg_path = AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_SELECTIVE_PATH
            case "lingua2":
                self.metrics_perplexity_path = METRICS_LINGUA2_PATH_PERPLEXITY
                self.parent_metrics_perplexity_path = METRICS_LINGUA2_PATH_PERPLEXITY.parent
                self.perplexity_avg_path = AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_LINGUA2_PATH
            case "axcer":
                if not config["BASELINE_WITHOUT_INTERROGATIVE"]:
                    self.metrics_perplexity_path = METRICS_AXCER_PATH_PERPLEXITY
                    self.parent_metrics_perplexity_path = METRICS_AXCER_PATH_PERPLEXITY.parent
                    self.perplexity_avg_path = AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_PATH
                else:
                    self.metrics_perplexity_path = METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH_PERPLEXITY
                    self.parent_metrics_perplexity_path = METRICS_AXCER_WITHOUT_INTERROGATIVE_PATH_PERPLEXITY.parent
                    self.perplexity_avg_path = AVG_PERPLEXITY_RESULT_FOR_EACH_DATASET_AXCER_WITHOUT_INTERROGATIVE_PATH
            case _:
                print("Original Case is not supported !")

    def tokenize_prompt(
        self,
        prompt: str,
    ):
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        return (input_ids, attention_mask)

    # TODO: IF you hit preemption try to reduce the max batch size
    @modal.batched(max_batch_size=21, wait_ms=500)
    # @modal.method()
    async def inference(self, prompts: list[str]):
        import math

        import torch

        results = []
        unpacked = [prompt.split(" |||| ") for prompt in prompts]

        dataset_names = [parts[0].removeprefix("processed_") for parts in unpacked]
        inputs = [parts[1].removeprefix("I: ").strip() for parts in unpacked]
        [[parts[2].removeprefix("A: ").strip()] for parts in unpacked]

        results: list[float] = []
        for prompt in inputs:
            print("Prompt is", prompt)
            tokenized_inputs, attn_mask = self.tokenize_prompt(prompt)
            print("tokenized input is", tokenized_inputs)
            with torch.no_grad():
                outputs = self.model(tokenized_inputs, attention_mask=attn_mask, labels=tokenized_inputs)
                print("output is", outputs)
                loss = outputs.loss.item()

            ppl = math.exp(loss)
            results.append(ppl)

        print("Metrics are", results)
        save_metrics_to_csv(results, dataset_names, self.metrics_perplexity_path, config["MODEL_NAME"])

        torch.cuda.empty_cache()

        return results

    @modal.exit()
    # @modal.method()
    async def compute_extra_evaluations(self):
        import modal

        print("Baseline is: ", config["BASELINE"])
        metrics_paths = map_path_to_volume(self.parent_metrics_perplexity_path)
        metrics_paths = fill_path(metrics_paths, model_name=config["MODEL_NAME"])

        print("running avg")

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
    datasets = load_datasets(Path(dataset_path), question_col="compressed_text", answer_col="prompt")

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

    await asyncio.gather(*(process_dataset_vllm(name, df, "compressed_text", "prompt") for name, df in datasets.items()))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
