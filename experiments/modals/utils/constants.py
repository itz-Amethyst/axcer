import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the evaluation with given model settings")
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model idendtifier(ID)", default="meta-llama/Meta-Llama-3.1-8B"
    )
    parser.add_argument(
        "--model_revision", type=str, required=True, help="Model revisions to prevent unexpected changes!", default=""
    )
    # this will be passed to concurrent decorator
    parser.add_argument(
        "--model_instance", type=int, required=True, help="To have multiple model instance to get faster response on datasets"
    )
    return parser.parse_args()


args = parse_args()

MODEL_REVISION = args.model_revision
MODEL_ID = args.model_id

MODEL_NAME = MODEL_ID.split("/")[-1]
# MODEL_REVISION = ""
STATIC_VOLUME = "/vol/model"
MODEL_DIR = STATIC_VOLUME + f"/{MODEL_NAME}"
MODEL_DIRFP16 = STATIC_VOLUME + f"/{MODEL_NAME}_FP16"
# MODEL_DIRFP16_CHK = "/vol/model" + f"/{MODEL_NAME}_FP16/checkpoints"
ENGINE_PATH = MODEL_DIRFP16 + "/engine"
VOLUME_PATH = "/vol"
# CHECKPOINT_DIR = (f"{ENGINE_PATH}/checkpoint",)
# MODEL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
# GIT_HASH = "1389f5a4d38cfefdc6944c1a9aa857fec6f72592"
# CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/quantization/quantize.py"

N_GPUS = 1
GPU_CONFIG = f"B200:{N_GPUS}"
# DTYPE = "float16"
# QUANTIZATION_ARGS = f"--dtype={DTYPE}"

DATA_DIR = Path("/experiments/datasets")
INPUT_COLUMN = "input_column"

parquet_paths = list(DATA_DIR.glob("*.parquet"))
