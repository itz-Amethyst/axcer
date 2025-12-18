import csv
from pathlib import Path


from experiments.constants.paths import VOLUME_NAME, fill_path
from experiments.modals.utils.helper import map_path_to_volume


# especially for perplexity saving
def save_metrics_to_csv(metrics_list, dataset_names, template_path: Path, model_name: str | None):
    """
    Save each dictionary of metrics into a CSV file with headers.
    One file per dataset, with dataset_names used as filenames.
    """

    for dataset_name, metrics in zip(dataset_names, metrics_list, strict=False):
        output_parent_dir = fill_path(template_path.parent, model_name=model_name, dataset_name=dataset_name)
        output_parent_dir = map_path_to_volume(output_parent_dir)
        output_parent_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir = fill_path(template_path, model_name=model_name, dataset_name=dataset_name)
        output_dir = map_path_to_volume(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_dir.exists()

        with open(output_dir, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["perplexity_score"])

            if not file_exists:
                writer.writeheader()

            writer.writerow({"perplexity_score": metrics})

        print(f"Saved {output_dir}")

    try:
        import modal  # type: ignore

        vol = modal.Volume.from_name(VOLUME_NAME)
        vol.commit()
    except Exception:
        # not running in Modal, or volume name not found — ignore
        pass
