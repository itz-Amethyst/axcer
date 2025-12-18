from experiments.modals.utils.constants import config


def get_build_config():
    from tensorrt_llm import BuildConfig
    from tensorrt_llm.bindings import KVCacheType

    return BuildConfig(
        plugin_config=get_plugin_config(),
        max_input_len=config["MAX_INPUT_LEN"],
        max_seq_len=config["MAX_SEQ_LEN"],
        max_num_tokens=config["MAX_NUM_TOKENS"],
        opt_num_tokens=config["MAX_NUM_TOKENS"],
        max_batch_size=config["MAX_BATCH_SIZE"],
        kv_cache_type=KVCacheType.PAGED,
        opt_batch_size=config["MAX_BATCH_SIZE"],
        strongly_typed=True,
        # monitor_memory=True,
    )
    # settings = {
    #     # not sure for this one (try disable too)
    #     "multiple_profiles": "disable",
    #     "gemm_plugin": "float16",
    #     "tokens_per_block": 64,
    #     "use_paged_context_fmha": "enable",
    #     "use_fused_mlp": "enable",
    #     "reduce_fusion": "enable",
    #     "logits_dtype": "float16",
    #     "checkpoint_dir": config["CHECKPOINTS_DIR"],
    #     "workers": config["TOTAL_CPU_CORES"],
    #     "gpt_attention_plugin": "float16",
    #     "remove_input_padding": "enable",
    #     "context_fmha": "enable",
    #     "max_batch_size": config["MAX_BATCH_SIZE"],
    #     "max_seq_len": config["MAX_SEQ_LEN"],
    #     "max_input_len": config["MAX_INPUT_LEN"],
    #     "max_num_tokens": config["MAX_NUM_TOKENS"],
    #     "output_dir": config["ENGINE_PATH"],
    #     "opt_num_tokens": config["MAX_NUM_TOKENS"],
    #     "kv_cache_type": "paged",
    # }
    # bc.builder_config.set_memory_pool_limit(
    #     trt.MemoryPoolType.WORKSPACE,
    #     4 * (1 << 30)
    # )


def get_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig

    return PluginConfig.from_dict(
        {
            # "multiple_profiles": False,
            "gpt_attention_plugin": "auto",
            "gemm_plugin": "auto",
            "context_fmha": True,
            "paged_kv_cache": True,
            "remove_input_padding": True,
            "reduce_fusion": True,
            "tokens_per_block": 64,
            "use_paged_context_fmha": True,
            # "low_latency_gemm_swiglu_plugin": "fp8",
            # "low_latency_gemm_plugin": "fp8",
        }
    )


# No need
# def get_calib_config():
#     from tensorrt_llm.llmapi import CalibConfig
#
#     return CalibConfig(
#         calib_batches=512,
#         calib_batch_size=1,
#         calib_max_seq_length=512,
#         tokenizer_max_seq_length=4096,
#     )
#
#
#     settings = {
#         "device": "cuda:0",
#         "dtype": "float16",
#         # "model_dir": config["MODEL_DIR"],
#         "model_dir": config["MODEL_ID"],
#         "output_dir": config["CHECKPOINTS_DIR"],
#
#         # Settings for No quantization
#         "tp_size": 1,
#         "qformat": "full_prec",
#         "batch_size": 1,
#         "calib_size": 1,
#         "calib_max_seq_length": 512
#     }
#

# for tensorrt_llm (nvidia)
# def set_builder():
#     import tensorrt_llm.builder as _b
#     from tensorrt_llm import BuildConfig
#     from tensorrt_llm.bindings import KVCacheType
#     from tensorrt_llm.plugin.plugin import PluginConfig
#
#     OrigBuilder = _b.Builder
#     print("entered !!!")
#
#     class PatchedBuilder(OrigBuilder):
#         def build_engine(self, network, build_config, *args, **kwargs):
#             trt_config = super().create_builder_config(build_config)
#
#             trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
#             # trt_config.trt_builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
#
#             return super(OrigBuilder, self).build_engine(network, build_config=build_config, *args, **kwargs)
#
#     _b.Builder = PatchedBuilder
