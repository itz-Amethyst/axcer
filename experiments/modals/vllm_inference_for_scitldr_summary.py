import sys
from pathlib import Path

import modal
import polars as pl

from experiments.constants.paths import (
    DATASET_PATH,
    VOLUME_MOUNT_POINT,
    VOLUME_NAME,
)
from experiments.constants.system_prompts import DATASET_SYSTEM_PROMPTS
from experiments.modals.utils.build_parameters import get_vllm_config
from experiments.modals.utils.constants import EvaluateConfig, config
from experiments.modals.utils.helper import map_path_to_volume, snapshot_config

hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App("axcer_evaluate_scitdlr_summary_as_reference", secrets=[hf_secret])

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
    }
)


vllm_image = vllm_image.uv_pip_install(
    "polars==1.31.0",
    "transformers==4.54.0",
)

vllm_image = vllm_image.add_local_python_source("experiments", copy=False)

volume = modal.Volume.from_name("axcer_volume", create_if_missing=True)
temp_volume = modal.Volume.from_name("temp_volume", create_if_missing=True)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

results_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


with vllm_image.imports():
    import evaluate
    from vllm import LLM, SamplingParams


@app.cls(
    gpu="H200",
    image=vllm_image,
    volumes={
        "/vol": volume,
        "/root/.cache/vllm": vllm_cache_vol,
        str(VOLUME_MOUNT_POINT): results_vol,
        "/temp_volume": temp_volume,
    },
    cpu=EvaluateConfig.TOTAL_CPU_CORES,
    max_containers=1,
    memory=4096,  # 4GB
    # memory=2048,  # 4GB
    secrets=[hf_secret],
    timeout=24 * 60 * 60,  # 24 hours
)
@modal.concurrent(max_inputs=20)
class VllmTest:
    @modal.enter()
    def setup(
        self,
    ):
        print("Setting up the evaluate utilities")

        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.code_eval = evaluate.load("code_eval")
        self.model_id = config["MODEL_ID"]

        self.llm = LLM(
            model=self.model_id,
            download_dir=f"/vol/model/{config['MODEL_NAME']}",
            **get_vllm_config(),
        )
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

    def tokenize_inputs(
        self,
        prompts: list[str],
        system_prompts: list[str] | None | str,
    ):
        if not system_prompts:
            system_prompts = [
                "You are a helpful assistant. You will be given a context and a question. Your only task is to provide a direct and short answer to the question, based exclusively on the provided context. Do not offer any reasoning, explanations, or additional information. Your answer should be a single phrase or sentence, and nothing more."
            ]
        batch_messages = [
            [{"role": "system", "content": s}, {"role": "user", "content": p}]
            for p, s in zip(prompts, system_prompts, strict=False)
        ]

        return batch_messages

    @modal.batched(max_batch_size=1, wait_ms=500)
    async def inference(self, prompts: list[str]):
        results = []
        unpacked = [prompt.split(" |||| ") for prompt in prompts]

        dataset_names = [parts[0].removeprefix("processed_") for parts in unpacked]
        inputs = [parts[1].removeprefix("I: ").strip() for parts in unpacked]
        ids = [[parts[2].removeprefix("ID: ").strip()] for parts in unpacked]
        system_prompts = [DATASET_SYSTEM_PROMPTS[dataset_name.lower()] for dataset_name in dataset_names]

        tokenized_inputs = self.tokenize_inputs(inputs, system_prompts=system_prompts)
        generated_resonse = self.llm.chat(tokenized_inputs, self.sampling_params, use_tqdm=False)
        result_text = generated_resonse[0].outputs[0].text
        ids = ids[0]

        res_with_id = f"{result_text} ||^% row_id_scitldr: {ids[0]}"
        print(res_with_id)
        results = [res_with_id]

        all_results = (ids[0], result_text)
        results_df = pl.DataFrame(
            {
                "row_id": [int(all_results[0])],
                "answer_field": [str(all_results[1])],
            },
            schema={"row_id": pl.Int64, "answer_field": pl.Utf8},
        )

        path_write = Path("/temp_volume/temppp")
        path_write.mkdir(exist_ok=True, parents=False)
        path_write = path_write / "scitdlr_processed.parquet"
        if path_write.exists():
            existing_df = pl.read_parquet(path_write)
            combined_df = pl.concat([existing_df, results_df])
            combined_df.write_parquet(path_write)
        else:
            results_df.write_parquet(path_write)
        vol = modal.Volume.from_name("temp_volume")
        vol.commit()
        return results


@app.local_entrypoint()
async def main(model_id: str, model_revision: str, max_batch_size: int):
    max_batch_size = 1

    EvaluateConfig.set_essential_parameters(model_id, model_revision, max_batch_size)
    snapshot = snapshot_config()
    for key, value in snapshot.items():
        await config.put.aio(key, value)
    print(config)
    # TODO: This will read from local, use above path
    # datasets = load_datasets(DATASET_PATH / "summarization_task" , question_col="input_field", answer_col="answer_field")

    dataset_path = map_path_to_volume(DATASET_PATH)
    print("STARTING DAtaset path is :", dataset_path)
    dataset_dir = dataset_path
    dataset_dir = dataset_dir.expanduser()

    if not dataset_dir.exists():
        print("Directory not found:", dataset_dir)
        sys.exit(1)

    # TODO: only pick the processed_scitldr
    file_path = dataset_dir / "processed_scitldr.parquet"
    df = pl.read_parquet(file_path, columns=["input_field"])
    dataset_name = file_path.stem  # name without .parquet

    vllm_t = VllmTest()

    async def process_dataset_vllm(dataset_name, df, question_col, file_path):
        df = df.with_row_count("row_id")
        prompts = df[question_col].to_list()
        ids = df["row_id"].to_list()
        prompts = [f"{dataset_name} |||| I: {q} |||| ID: {id}" for q, id in zip(prompts, ids, strict=False)]
        print(f"[{dataset_name}] submitting {len(prompts)} prompts...")
        output_path = Path(f"{dataset_name}.parquet")
        all_results = []
        marker = "||^% row_id_scitldr:"
        async for batch in vllm_t.inference.map.aio(prompts):
            if marker in batch:
                text_part, id_part = batch.rsplit(marker, 1)  # split from the right
                extracted_id = int(id_part.strip())  # convert to int
                generated_summary = text_part.strip()  # remove trailing spaces
            else:
                print(f"WARNING: ! skipped {batch}")
            batch = (extracted_id, generated_summary)
            all_results.append(batch)

        print("ALL RES", all_results)
        results_df = pl.DataFrame(
            {
                "row_id": [int(row[0]) for row in all_results],
                "answer_field": [str(row[1]) for row in all_results],
            },
            schema={"row_id": pl.Int64, "answer_field": pl.Utf8},
        )
        results_df.write_parquet("~/dev/axcer/temp_be_delete/scitdlr_processseeed.parquet")
        updated_df = df.join(results_df, on="row_id")

        file_path = map_path_to_volume(file_path)
        print("FILE PATH after volume IS", file_path)
        updated_df.write_parquet(file_path)
        print(f"[{dataset_name}] done — wrote to {output_path}")

    await process_dataset_vllm(dataset_name, df, "input_field", file_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
