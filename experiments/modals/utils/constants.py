from dataclasses import dataclass
from pathlib import Path

import modal

config = modal.Dict.from_name("eval-config", create_if_missing=True)


def snapshot_config() -> dict:
    cls = EvaluateConfig
    return {key: getattr(cls, key) for key in dir(cls) if key.isupper() and not callable(getattr(cls, key))}


@dataclass
class EvaluateConfig:
    # Initial default values
    MODEL_ID: str = ""
    MODEL_REVISION: str = ""
    MAX_CONCURRENT_INPUTS: int = 1

    # Static values
    STATIC_VOLUME: str = "/vol/model"
    VOLUME_PATH: str = "/vol"
    DATA_DIR: Path = Path("/experiments/datasets")
    INPUT_COLUMN: str = "input_column"
    # PARQUET_PATHS: List[Path] = list(DATA_DIR.glob("*.parquet"))

    BERT_MODEL_ID: str = ""
    BERT_MODEL_NAME: str = ""

    # Token & batch config
    MAX_BATCH_SIZE: int = 1  # 5
    MAX_INPUT_LEN: int = 8192
    MAX_OUTPUT_LEN: int = 1024  # 512  # 1024
    # TOTAL_CPU_CORES: int = 2
    TOTAL_CPU_CORES: float = 1.5
    MAX_SEQ_LEN: int = MAX_INPUT_LEN
    MAX_NUM_TOKENS: int = MAX_BATCH_SIZE * MAX_INPUT_LEN
    # OPT_BATCH_SIZE: int = MAX_BATCH_SIZE

    N_GPUS: int = 1
    GPU_CONFIG: str = f"H200:{N_GPUS}"

    MINUTE: int = 60

    @property
    def PARQUET_PATHS(self) -> list[Path]:
        return list(self.DATA_DIR.glob("*.parquet"))

    @classmethod
    def set_essential_parameters(cls, model_id: str, model_revision: str, max_batch_size: int) -> None:
        cls.MODEL_ID = model_id
        cls.MODEL_REVISION = model_revision
        cls.MAX_CONCURRENT_INPUTS = max_batch_size

        cls.MODEL_NAME = model_id.split("/")[-1]
        cls.MODEL_DIR = f"{cls.STATIC_VOLUME}/{cls.MODEL_NAME}"

        cls.BERT_MODEL_NAME = cls.BERT_MODEL_ID.split("/")[-1]

        # Update batch sizes too
        cls.MAX_BATCH_SIZE = max_batch_size
        cls.MAX_NUM_TOKENS = cls.MAX_BATCH_SIZE * cls.MAX_INPUT_LEN
