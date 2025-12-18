from experiments.modals.utils.constants import config


def get_vllm_config() -> dict:
    """
    Returns configuration for initializing a vLLM LLM instance.
    You can override defaults for model, download_dir, or batched_token.
    """
    return {
        "max_model_len": config["MAX_INPUT_LEN"],
        "max_num_seqs": config["MAX_BATCH_SIZE"],
        "trust_remote_code": True,
        "max_num_batched_tokens": config["MAX_NUM_TOKENS"],
        "gpu_memory_utilization": 0.90,
        "enable_chunked_prefill": True,
    }
