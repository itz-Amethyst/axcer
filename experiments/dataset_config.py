DATASETS = {
    "ai2_arc": {
        "question_field": ["question", "choices[text]"],
        "answer_field": ["answerKey"],
    },
    "coqa": {
        "question_field": ["questions"],
        "context_field": ["story"],
        "answer_field": ["answers[input_text]"],
    },
    # This has to be processed differently because of the dataset structure via the format_mawps
    "MAWPS": {
        "question_field": ["Question"],
        "answer_field": ["Answer"],
    },
    # (WE HAVE SQUAD IT'S EXACTLY LIKE THIS)
    # "quac": {
    #     "question_field": ["questions"],
    #     "context_field": ["context"],
    #     "answer_field": ["orig_answers[texts]"],
    # },
    "squad": {
        "question_field": ["context", "question"],
        "answer_field": ["answers[text]"],
    },
    "gsm8k": {
        "question_field": ["question"],
        "answer_field": ["answer"],
    },
    "piqa": {
        "question_field": ["goal", "sol1", "sol2"],
        "answer_field": ["label"],
    },
    "mbpp": {
        "question_field": ["text", "test_list"],
        "answer_field": ["test_list"],
    },
    # Glue mrpc subset only
    "glue": {
        "question_field": ["sentence1", "sentence2"],
        "answer_field": ["label"],
    },
    "boolq": {
        "question_field": ["passage", "question"],
        "answer_field": ["answer"],
    },
    "scitldr": {
        "question_field": ["source"],
        # we should fill this while we ran the vllm_inference original on this dataset the summarized output will be the answer_field
        # "answer_field": ["label"],
    },
}

# INFO: MAIN
DATASET_NAMES = [
    "ai2_arc:ARC-Challenge:test",
    "stanfordnlp/coqa:validation",
    "mwpt5/MAWPS:train",
    "rajpurkar/squad:validation",
    "gsm8k:main:test",
    "ybisk/piqa:validation",
    "Muennighoff/mbpp:test",
    "nyu-mll/glue:mrpc:test",
    "allenai/scitldr:AIC:test",
    "boolq:validation",
]
