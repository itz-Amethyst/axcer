def parse_dataset_entry(entry: str):
    """
    Parses a dataset entry of the format dataset:subset:split.
    Returns a tuple: (dataset_name, subset_name or None, split_name)
    """
    parts = entry.split(":")
    dataset_name = parts[0]
    split_name = parts[-1]
    subset_name = None
    if len(parts) > 2:
        subset_name = parts[1]
    return dataset_name, subset_name, split_name
