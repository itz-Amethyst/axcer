from pathlib import Path
import sys
import modal
import polars as pl
from experiments.constants.paths import (
    DATASET_PATH,
    VOLUME_MOUNT_POINT,
    VOLUME_NAME,
)
from experiments.constants.system_prompts import DATASET_SYSTEM_PROMPTS
from experiments.modals.utils.helper import snapshot_config
from experiments.modals.utils.build_parameters import get_vllm_config
from experiments.modals.utils.constants import EvaluateConfig, config
from experiments.modals.utils.helper import map_path_to_volume
# from experiments.modals.utils.metrics import calculate_metrics, save_metrics_to_csv

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
    # "huggingface_hub[hf_transfer]==0.34.1",
    # "flashinfer-python==0.2.6.",
    "polars==1.31.0",
    "transformers==4.54.0",
)

vllm_image = vllm_image.add_local_python_source("experiments", copy=False)

# vllm_image = vllm_image.add_local_dir("../datasets/", remote_path="/experiments/datasets")

volume = modal.Volume.from_name("axcer_volume", create_if_missing=True)

# datasets_volume = modal.Volume.from_name("datasets-vol", create_if_missing=True)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

results_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# model_dir = "/vol/model/Meta-Llama-3.1-8B"

with vllm_image.imports():
    from vllm import SamplingParams, LLM
    import evaluate


@app.cls(
    gpu="H200",
    # gpu="A100-40GB",
    # gpu="A100-80GB",
    # gpu="l4",
    # gpu="t4",
    # gpu="h100",
    image=vllm_image,
    volumes={"/vol": volume, "/root/.cache/vllm": vllm_cache_vol, str(VOLUME_MOUNT_POINT): results_vol},
    cpu=EvaluateConfig.TOTAL_CPU_CORES,
    max_containers=1,
    # memory=4096,  # 4GB
    memory=2048,  # 4GB
    secrets=[hf_secret],
    timeout=24 * 60 * 60,  # 24 hours
)
# @modal.concurrent(max_inputs=9)
@modal.concurrent(max_inputs=20)
class VllmTest:
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
        self.model_id = config["MODEL_ID"]

        # batched_token = 8192 * 5
        # print("MODEL DIR IS", model_dir)
        self.llm = LLM(
            # model="meta-llama/Llama-3.1-8B",
            # replace this later with config['MODEL_NAME']
            model=self.model_id,
            download_dir=f"/vol/model/{config['MODEL_NAME']}",
            **get_vllm_config(),
            # download_dir="/vol/model/Meta-Llama-3.1-8B_vllm",
            # max_model_len=8192,
            # max_num_seqs=5,
            # trust_remote_code=True,
            # max_num_batched_tokens=batched_token,
            # # gpu_memory_utilization=0.9,
            # gpu_memory_utilization=0.95,
            # enable_chunked_prefill=True,
        )
        # self.llm.get_tokenizer().eos_token = "<|eot_id|>"
        # self.llm.get_tokenizer().eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # self.llm.get_tokenizer().add_special_tokens({"pad_token": self.tokenizer.eos_token})
        # self.llm.get_tokenizer().padding_side = "left"
        # needs a explicit handling for each llm the eot_id is only for llama
        self.terminators = [
            self.llm.get_tokenizer().eos_token_id,
            # self.llm.get_tokenizer().convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.sampling_params = SamplingParams(
            # max_tokens=128,
            # max_tokens=35,
            max_tokens=config["MAX_OUTPUT_LEN"],
            min_tokens=1,
            # dont change these two values unless you exit the greedy decoding term which mentioned in the paper
            temperature=0.0,
            top_p=1,
            # stop_token_ids=self.stop_token_ids,
            stop_token_ids=self.terminators,
        )

    def tokenize_inputs(
        self,
        prompts: list[str],
        system_prompts: list[str] | None | str,
    ):
        if not system_prompts:
            # system_prompt = "You are a helpful asisstan, Your primary function is to answer questions directly based on the provided input, Your goal is to deliver answer directly no more than that."
            system_prompts = [
                "You are a helpful assistant. You will be given a context and a question. Your only task is to provide a direct and short answer to the question, based exclusively on the provided context. Do not offer any reasoning, explanations, or additional information. Your answer should be a single phrase or sentence, and nothing more."
            ]
            # system_prompt = "You are a helpful asisstant. Your task is to extract and provide answers strictly from the given context.Your answer must be as concise as possible, containing only the specific information required to answer the question, and no additional details or elaborations. Do not infer or add information not explicitly present in the provided text. Do not include any supporting details, context, or reasoning. Provide only the direct answer as short as possible."

        # probobably there is something wrong with the template or i'm not decoding the generated answer correctly it's not answering properly
        # change the system prompt to something better give simple info about dataset
        # run with another data much simplere to see what's the generated response
        batch_messages = [
            [{"role": "system", "content": s}, {"role": "user", "content": p}]
            for p, s in zip(prompts, system_prompts, strict=False)
        ]
        # tokenizer = self.llm.get_tokenizer()
        # tokenizer.chat_template = mistral_template
        # self.llm.get_tokenizer().chat_template = mistral_template

        # parsed_prompts = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        # ss =  tokenizer(parsed_prompts, return_tensors="pt", padding=True, truncation=False)
        # return ss['input_ids']

        # INFO: FOR CHAT

        # tokenized = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        # print("Tokenized is", tokenized)
        # return tokenized
        return batch_messages

    # TODO: Maybe trye to set batch size to 1 as well as max_context_window of vllm config and see if it's same as with batch size 5 or not
    # @modal.batched(max_batch_size=EvaluateConfig.MAX_BATCH_SIZE, wait_ms=500)
    @modal.batched(max_batch_size=1, wait_ms=500)
    async def inference(self, prompts: list[str]):
        results = []
        unpacked = [prompt.split(" | ") for prompt in prompts]

        dataset_names = [parts[0].removeprefix("processed_") for parts in unpacked]
        inputs = [parts[1].removeprefix("I: ").strip() for parts in unpacked]
        ids = [[parts[2].removeprefix("ID: ").strip()] for parts in unpacked]
        system_prompts = [DATASET_SYSTEM_PROMPTS[dataset_name.lower()] for dataset_name in dataset_names]

        # print(f"SYSTEM PROMPTS are: {system_prompts}")
        # shutil.rmtree('/datastes')
        tokenized_inputs = self.tokenize_inputs(inputs, system_prompts=system_prompts)
        generated_resonse = self.llm.chat(tokenized_inputs, self.sampling_params, use_tqdm=False)
        # print("GENERATED IS", generated_resonse)

        # for ds_name, prompt_text, answer_text, out in zip(dataset_names, inputs, answers, outputs):
        # print("FSF", generated_resonse)
        result_text = generated_resonse[0].outputs[0].text
        ids = ids[0]

        res_with_id = f"{result_text} | row_id_scitldr: {ids[0]}"
        print(res_with_id)
        results = [res_with_id]
        # del tokenized_inputs, dataset_names, system_prompts , result_text, res_with_id
        # gc.collect()
        # torch.cuda.empty_cache()

        # all_results = (ids[0], result_text)
        # results_df = pl.DataFrame(all_results, schema={"row_id":pl.Int64, "answer_field": pl.Utf8})
        # path_write = map_path_to_volume(VOLUME_MOUNT_POINT / "tempppp")
        # path_write.mkdir(exist_ok=True, parents=False)
        # path_write = path_write / "scitdlr_processed.parquet"
        # results_df.write_parquet(path_write)
        return results

        # metrics_list = calculate_metrics(results, answers, dataset_names, self.rouge, self.bertscore, self.code_eval)

        # print("Metrics are", metrics_list)
        # save_metrics_to_csv(metrics_list, dataset_names, METRICS_ORIGINAL_PATH, config["MODEL_NAME"])

        # not sure ! might add extra time if i uncomment it ! check it run experiments to see whether have it or not
        # print("Generated Response from llm: ", results)
        #
        # return results

    # @modal.exit()
    # async def compute_extra_evaluations(self):
    #     import subprocess
    #     import modal
    #     print("running avg")
    #     compute_multiple_file_csv_column_averages(base_read_path = METRICS_ORIGINAL_PATH.parent, base_write_path=AVG_OUTPUT_RESULT_FOR_EACH_DATASET_ORIGINAL_PATH, model_name = config["MODEL_NAME"])
    #     print("finished avg")
    #     vol = modal.Volume.from_name(VOLUME_NAME)
    #     vol.commit()
    # results_vol.commit()
    # config["RESULTS_VOL_FLAG"] = True


@app.local_entrypoint()
# later add the dataset_path to this so that this file would work for other baselines and then
# pass the dataset_dir to load_dataset
async def main(model_id: str, model_revision: str, max_batch_size: int):
    max_batch_size = 1

    EvaluateConfig.set_essential_parameters(model_id, model_revision, max_batch_size)
    snapshot = snapshot_config()
    # model_name = config["MODEL_NAME"]
    # print("DEBUG:", config, type(config), snapshot)
    for key, value in snapshot.items():
        await config.put.aio(key, value)
    print(config)
    # print("running")
    # also update this to read dataset_path from the given parameter arg-parse in main function
    # TODO: This will read from local, use above path
    # datasets = load_datasets(DATASET_PATH / "summarization_task" , question_col="input_field", answer_col="answer_field")

    # WORKING
    dataset_path = map_path_to_volume(DATASET_PATH)
    print("STARTING DAtaset path is :", dataset_path)
    dataset_dir = dataset_path
    dataset_dir = dataset_dir.expanduser()

    if not dataset_dir.exists():
        print("Directory not found:", dataset_dir)
        sys.exit(1)

    # There is only one instance of dataset in given path, we loop through it
    # TODO: only pick the processed_scitldr
    file_path = dataset_dir / "processed_scitldr.parquet"
    df = pl.read_parquet(file_path, columns=["input_field"])
    dataset_name = file_path.stem  # name without .parquet

    vllm_t = VllmTest()

    async def process_dataset_vllm(dataset_name, df, question_col, file_path):
        df = df.with_row_count("row_id")
        prompts = df[question_col].to_list()
        ids = df["row_id"].to_list()
        prompts = [f"{dataset_name} | I: {q} | ID: {id}" for q, id in zip(prompts, ids, strict=False)]
        print(f"[{dataset_name}] submitting {len(prompts)} prompts...")
        output_path = Path(f"{dataset_name}.parquet")
        all_results = []
        marker = "| row_id_scitldr:"
        async for batch in vllm_t.inference.map.aio(prompts):
            # if isinstance(batch, list):
            #     # print("it's a list")
            #     batch = str(batch[0])
            # else:
            #     # print('it is a string')
            #     batch = str(batch)
            # print("RESULTS IS: ", batch)
            if marker in batch:
                text_part, id_part = batch.rsplit(marker, 1)  # split from the right
                extracted_id = int(id_part.strip())  # convert to int
                generated_summary = text_part.strip()  # remove trailing spaces
                # print("Original text:", generated_summary)
                # print("Extracted ID:", extracted_id)
            else:
                print(f"WARNING: ! skipped {batch}")
            batch = (extracted_id, generated_summary)
            all_results.append(batch)

        # print("ALL RESULTS ARE",all_results )
        print("ALL RES", all_results)
        results_df = pl.DataFrame(
            {
                "row_id": [int(row[0]) for row in all_results],
                "answer_field": [str(row[1]) for row in all_results],
            },
            schema={"row_id": pl.Int64, "answer_field": pl.Utf8},
        )
        results_df.write_parquet("~/dev/axcer/temp_be_delete/scitdlr_processseeed.parquet")
        # print("RESULTS_DF", results_df)
        # print("NORMAL DF", df)
        updated_df = df.join(results_df, on="row_id")

        # print("FILE PATH IS", file_path)

        file_path = map_path_to_volume(file_path)
        print("FILE PATH after volume IS", file_path)
        updated_df.write_parquet(file_path)
        print(f"[{dataset_name}] done — wrote to {output_path}")

    await process_dataset_vllm(dataset_name, df, "input_field", file_path)

    # TODO: let it be this for now, but once you wanted to publish it to the github create a bash script where you run this code with modal once the results got saved in volume
    # you then add this bash script to run after the modal is finished (modal get results-vol / {destination} --force)

    # await asyncio.sleep(5)
    # print("Calling pulling")
    # await pull_results_from_volume(VOLUME_NAME, local_path=None)
    # queue.clear()

    # print("starting to pull")
    # cmd = [
    #     "modal", "volume", "get", "results-vol",
    #     os.path.expanduser("~/dev/axcer/experiments/results"),
    #     "--force"
    # ]
    # ans = subprocess.call(cmd)
    # if ans == 0:
    #     print("Pulled results from volume successfully")
    # else:
    #     print("Failed to pull results from volume")
    # await asyncio.gather(
    #         *(process_list_vllm(name, vals) for name, vals in fake_datasets.items())
    #     )


# use this
# modal run --detach vllm_inference_for_scitldr_summary.py
# TODO: It's because we are running with modal run so using this approach will only run the main function second will be ignored
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
