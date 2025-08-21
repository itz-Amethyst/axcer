import gc
from pathlib import Path
from typing import Any
import asyncio
from experiments.modals.utils.prepare_datasets import load_datasets

import modal

from experiments.constants.paths import (
    DATASET_PATH,
    DATASET_VOLUME_MOUNT_POINT,
    MERGED_COMPRESSED_SELECTIVE_WITH_DATASET_PATH,
    PROCESSED_SELECTIVE_PATH,
    VOLUME_MOUNT_POINT,
    VOLUME_NAME,
)
from experiments.evaluation.utils.align_datasets import replace_input_column_values
from experiments.modals.utils.constants import EvaluateConfig
from experiments.modals.utils.helper import map_path_to_volume
from experiments.modals.utils.metrics import save_metrics_to_csv
from experiments.modals.utils.selective.metric_helper import prepare_tokenizer_for_counting_modal

hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("axcer_evaluate_selective", secrets=[hf_secret])

selective_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "build-essential", "python3-dev")
    .pip_install("uv")
)

selective_image = selective_image.pip_install(
    # "selective-context",
    # "tiktoken==0.11.0",
    # "spacy",
    "thinc",
    "huggingface_hub[hf_transfer]==0.34.1",
    "evaluate==0.4.5",
    "bert-score==0.3.13",
    "rouge-score==0.1.2",
    "setuptools",
    "wheel",
    extra_index_url="https://download.pytorch.org/whl/cu128",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TORCH_CUDA_ARCH_LIST": "9.0 9.0a"})


selective_image = selective_image.uv_pip_install(
    # "huggingface_hub[hf_transfer]==0.34.1",
    # "flashinfer-python==0.2.6.",
    "polars==1.31.0",
    "transformers==4.54.0",
    "selective-context",
    "tiktoken==0.11.0",
    "spacy==3.8.7",
).run_commands("python -m spacy download en_core_web_sm")

selective_image = selective_image.add_local_python_source("experiments", copy=False)

volume = modal.Volume.from_name("axcer_volume", create_if_missing=True)

dataset_volume = modal.Volume.from_name("dataset-volume", create_if_missing=False)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

results_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

with selective_image.imports():
    import time

    import torch
    from selective_context import SelectiveContext


@app.cls(
    # gpu="H200",
    # gpu="A100-40GB",
    gpu="A100-80GB",
    image=selective_image,
    volumes={
        "/vol": volume,
        str(VOLUME_MOUNT_POINT): results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        str(DATASET_VOLUME_MOUNT_POINT): dataset_volume,
    },
    cpu=EvaluateConfig.TOTAL_CPU_CORES,
    max_containers=1,
    memory=4096,  # 4GB
    secrets=[hf_secret],
    timeout=300 * 60,
)
@modal.concurrent(max_inputs=9)
class SelectiveEvaluate:
    @modal.enter()
    def setup(
        self,
    ):
        print("Loading selective context compressor")

        self.llm = SelectiveContext(model_type="gpt2", lang="en")

        print("loading tokenizer for counting tokens")
        self.tokenizer = prepare_tokenizer_for_counting_modal()

    @modal.batched(max_batch_size=EvaluateConfig.MAX_BATCH_SIZE, wait_ms=500)
    async def compress_and_save(self, prompts: list[str]):
        unpacked = [prompt.split(" | ") for prompt in prompts]

        dataset_names = [parts[0] for parts in unpacked]
        inputs = [parts[1].removeprefix("I: ").strip() for parts in unpacked]

        results: list[dict[str, Any]] = []
        for prompt in inputs:
            print("Prompt is", prompt)
            prompt_tokens = len(self.tokenizer.encode(prompt))

            start = time.perf_counter()

            compressed_text = self.llm(prompt)
            compressed_text = str(compressed_text[0])
            print("Compressed prompt is: ", compressed_text)

            end = time.perf_counter()

            total_runtime = end - start

            compressed_tokens = len(self.tokenizer.encode(compressed_text))

            compression_ratio = None
            if prompt_tokens > 0:
                compression_ratio = compressed_tokens / prompt_tokens

            results.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens,
                    "total_runtime": round(total_runtime, 6),
                    "compressed_tokens": compressed_tokens,
                    "compression_ratio": compression_ratio,
                    "compressed_text": compressed_text,
                }
            )

        save_metrics_to_csv(results, dataset_names, PROCESSED_SELECTIVE_PATH, model_name=None)

        del dataset_names, inputs, unpacked
        gc.collect()
        torch.cuda.empty_cache()

        return results

    @modal.exit()
    def update_dataset_with_compressed_text(self):
        dataset_files = {p.stem.removeprefix("processed_"): p for p in DATASET_VOLUME_MOUNT_POINT.glob("*.parquet")}

        print("Started to replace compressed values with dataset values")
        processed_paths = map_path_to_volume(PROCESSED_SELECTIVE_PATH.parent)
        for processed_path in processed_paths.glob("*.csv"):
            print("BASE IS", processed_path)
            base_name = str(processed_path.stem.removeprefix("processed_"))
            # base_name = base_name.split(".")[0]
            try:
                dataset_path = dataset_files[base_name]
            except KeyError as err:
                raise FileNotFoundError(f"No matching parquet found for {base_name}.csv") from err

            replace_input_column_values(processed_path, dataset_path, MERGED_COMPRESSED_SELECTIVE_WITH_DATASET_PATH, base_name)


@app.local_entrypoint()
async def main():
    datasets = load_datasets(DATASET_PATH / "temp", question_col="input_field", answer_col="answer_field")
    selective_e = SelectiveEvaluate()

    async def process_dataset_selective(dataset_name, df, question_col):
        prompts = df[question_col].to_list()
        prompts = [f"{dataset_name} | I: {q}" for q in prompts]
        print(f"[{dataset_name}] submitting {len(prompts)} prompts...")
        output_path = Path(f"{dataset_name}_results.json")
        async for batch in selective_e.compress_and_save.map.aio(prompts):
            print("Processed :", len(batch))

        print(f"[{dataset_name}] done — wrote to {output_path}")

    await asyncio.gather(*(process_dataset_selective(name, df, "input_field") for name, df in datasets.items()))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
