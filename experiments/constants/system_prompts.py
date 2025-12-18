DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    # NOTE: Quac dataset is tricky and sometimes the model is giving extra information the sytem prompt could be better but it has less chance to be worked on it ! (EXCLUDED !)
    # "quac": 'You will receive a passage that ends with a question. Read the passage carefully and answer the question only from the given text. Do not use any outside knowledge. If the answer cannot be found in the passage, respond with "cannotanswer". Otherwise, respond with the shortest, most direct answer possible, nothing else.',
    "ai2_arc": "You will receive a question with four answer options labeled A to D, often followed with a text. Read the text carefully and answer the question using only the given text. Do not use outside knowledge. Respond with only the letter of the correct option (A, B, C, or D). Do not repeat or include the answer text, your response should be only a single letter options.",
    "squad": "You are going to receive a passage of text that ends with a question.Read the passage carefully and answer the question using only the given text. Do not use outside knowledge. Respond with the shortest, most direct answer possible, nothing else.",
    "coqa": "You are going to receive a passage of text that ends with a question.Read the passage carefully and answer the question using only the given text. Do not use outside knowledge. Respond with the shortest, most direct answer possible, nothing else.",
    "piqa": 'You are given a short physical situation and two possible solutions.Decide which solution is more physically plausible. Respond with only "A" or "B". No explanation or extra text, nothing else.',
    "mawps": "You will be given a math problem. Solve it and do not include words, explanations, steps, or any extra text. Output must be contained only the equation followed by its result, four hash symbols, and the same result. Output only in this exact format: <equation> = <final_result> #### <final_result>",
    "gsm8k": "You are a math solver. For every problem, you must provide a clear step-by-step reasoning and then the final numeric result. The response must always follow the exact format below — no deviations are allowed:  #### Reasoning Process <step-by-step explanation> #### Final Answer <final numeric result only>",
    "glue": "You are a specialized Natural Language Inference model. Your task is to analyze two given sentences and determine their semantic equivalence. Based only on the information provided, classify the relationship into one label. Your response must be a single integer only: 1 for Equivalent, 0 for Not equivalent.",
    "boolq": "You will receive a passage and a yes/no question. Carefully consider only the information in the passage and respond with only True or False. Do not rely on outside knowledge, personal opinions, or beliefs. Keep your answer concise and direct. Your final response must be a single word: either True or False.",
    "mbpp": "You will be given a programming prompt. Write only the code that implements the required solution. Do not include explanations, comments, or extra text. Output must contain only the final runnable code. The code must be written so that it successfully passes all of the provided assessments. Output strictly in this exact format: <code>",
    "scitldr": "You are tasked with summarizing scientific documents concisely and accurately on the provided document. The summary must be a faithful reproduction of the original's main points and essential information. Do not add any outside opinions, interpretations, new words, symbols or meta-comments. Do not include introductory or concluding phrases, disclaimers, or notes. The output must contain only the summary text, shorter than the original document.",
}


def get_system_prompt(dataset_name: str) -> str:
    """
    Get the system prompt for a given dataset.

    Args:
        dataset_name: Name of the dataset to get prompt for

    Returns:
        str: System prompt for the dataset, or default if not found
    """
    if dataset_name not in DATASET_SYSTEM_PROMPTS:
        # logger.warning(f"Unknown dataset: '{dataset_name}', using default prompt")
        print(f"Unknown dataset: '{dataset_name}', using default prompt")
        return ""

    return DATASET_SYSTEM_PROMPTS[dataset_name]


def get_available_datasets() -> list[str]:
    """
    Get list of available dataset names.

    Returns:
        list: Available dataset names
    """
    return list(DATASET_SYSTEM_PROMPTS.keys())
