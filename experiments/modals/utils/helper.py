from pathlib import Path
import os
import subprocess
from collections.abc import Iterator

from experiments.constants.paths import DATASET_VOLUME_MOUNT_POINT, VOLUME_MOUNT_POINT
from experiments.modals.utils.constants import EvaluateConfig


def snapshot_config() -> dict:
    cls = EvaluateConfig
    return {key: getattr(cls, key) for key in dir(cls) if key.isupper() and not callable(getattr(cls, key))}


def map_path_to_volume(target: Path, mount_point: Path = VOLUME_MOUNT_POINT) -> Path:
    """
    If target path contains the sequence 'experiments' / 'results',
    replace everything before and including 'results' with mount_point.
    If mount_point doesn't exist, or 'experiments/results' not found,
    return target unchanged.
    """
    # if the mount doesn't exist on this runtime, do nothing
    if not mount_point.exists():
        return target

    parts = list(target.parts)
    # find the index where ['experiments', 'results'] occurs
    for i in range(len(parts) - 1):
        if parts[i] == "experiments" and parts[i + 1] == "results":
            rest = parts[i + 2 :]  # everything after 'results'
            return mount_point.joinpath(*rest)
    return target


def pair_files(path1: Path, path2: Path) -> Iterator[tuple[Path, Path]]:
    files1 = {f.stem.removeprefix("processed_"): f for f in path1.glob("*.csv")}
    files2 = {f.stem: f for f in path2.glob("*.csv")}

    for dataset_name in files1.keys() & files2.keys():
        yield files1[dataset_name], files2[dataset_name]


def map_dataset_path_to_volume(target: Path, mount_point: Path = DATASET_VOLUME_MOUNT_POINT) -> Path:
    """
    If target path contains the sequence 'experiments' / 'results',
    replace everything before and including 'results' with mount_point.
    If mount_point doesn't exist, or 'experiments/results' not found,
    return target unchanged.
    """
    # if the mount doesn't exist on this runtime, do nothing
    if not mount_point.exists():
        return target

    parts = list(target.parts)
    # find the index where ['experiments', 'results'] occurs
    for i in range(len(parts) - 1):
        if parts[i] == "experiments":
            rest = parts[i + 1 :]  # everything after 'results'
            print(parts)
            return mount_point.joinpath(*rest)
    return target


# info: not useful delete later
async def pull_results_from_volume(volume_name: str, local_path: Path | str | None):
    print("starting to pull")
    if not local_path:
        local_path = os.path.expanduser("~/dev/axcer/experiments/results")
    # shutil.rmtree(local_path)
    print("MODEL_VO", volume_name)
    cmd = ["modal", "volume", "get", f"{volume_name}", "/", local_path, "--force"]
    print("CMD IS", cmd)
    # to fix Is a directory error from modal
    ans = subprocess.run(
        cmd,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )
    if ans.returncode == 0:
        print("✅ Pulled results from volume successfully")
        print("Output:\n", ans.stdout)
    else:
        print("❌ Failed to pull results from volume")
        print("Error Output:\n", ans.stderr)
