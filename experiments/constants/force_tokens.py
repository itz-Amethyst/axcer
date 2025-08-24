FORCE_TOKENS_FOR_LINGUA2: dict[str, list[str]] = {
    # If you added new dataset instruction, add it here
    "mbpp": ["Prompt:", "Assesments:"],
    "multi_nli": ["Hypothesis:", "Premise:"],
    "ai2_arc": ["Options:", "A.", "B.", "C.", "D."],
    "piqa": ["Options:", "A.", "B."],
}


def get_force_token(dataset_name: str) -> list[str]:
    if dataset_name not in FORCE_TOKENS_FOR_LINGUA2:
        # logger.warning(f"Unknown dataset: '{dataset_name}', using default prompt")
        print(f"Unknown dataset: '{dataset_name}', skiping..")
        return []

    return FORCE_TOKENS_FOR_LINGUA2[dataset_name]


def get_available_datasets() -> list[str]:
    """
    Get list of available dataset names.

    Returns:
        list: Available dataset names
    """
    return list(FORCE_TOKENS_FOR_LINGUA2.keys())
