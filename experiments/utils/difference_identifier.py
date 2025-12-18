from pathlib import Path
from typing import Any

import polars as pl


def _read_either(path: Path) -> pl.DataFrame:
    """Try CSV first, then parquet. Raise if both fail."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    try:
        if suf in {".csv", ".txt"}:
            return pl.read_csv(path)
        if suf in {".parquet", ".pq"}:
            return pl.read_parquet(path)
        try:
            return pl.read_csv(path)
        except Exception:
            return pl.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def process_directories(
    path1_dir: str | Path,
    path2_dir: str | Path,
    file1_processed_dir: str | Path,
    file2_processed_dir: str | Path,
    output_dir: str | Path,
    file_names: list[str] | None = None,
    exact_col: str = "exact_match",
    original_col: str = "prompt",
    original_col_axcer: str = "original_text",
    compressed_col: str = "compressed_text",
    preview_n: int | None = 20,
    skip_if_row_mismatch: bool = True,
) -> dict[str, Any]:
    """
    Process directories (not single files). See docstring below.

    Parameters
    ----------
    path1_dir, path2_dir, file1_processed_dir, file2_processed_dir : str|Path
        Directories containing the corresponding files. Filenames may have different extensions
        across directories; matching is done by file stem (filename without extension).
    output_dir : str|Path
        Directory where result CSVs will be written.
    file_names : list[str] or None
        If list[str], interprets entries as either exact filenames (with ext) OR stems (without ext).
        If None, automatically detects common stems present in all four directories.
    exact_col : str
        Column name containing 0/1 values to compare (default "exact_math").
    original_col : str
        Column name (in file1_processed files) holding original text.
    compressed_col : str
        Column name (in processed files) holding compressed text.
    preview_n : int|None
        Number of original_text strings to put in the summary for each file (None => return all).
    skip_if_row_mismatch : bool
        If True, skip a file if the main two files have different row counts (and record reason).
        If False, raise an error on mismatch.

    Returns
    -------
    summary : dict
        Mapping stem -> { matched_rows, file1_csv, file2_csv, original_texts_preview, skipped_reason }
    """
    p1 = Path(path1_dir)
    p2 = Path(path2_dir)
    p1p = Path(file1_processed_dir)
    p2p = Path(file2_processed_dir)
    outd = Path(output_dir)
    outd.mkdir(parents=True, exist_ok=True)

    # build stem -> Path maps for each directory
    def build_map(d: Path) -> dict[str, Path]:
        m = {}
        if not d.exists():
            return m
        for f in d.iterdir():
            if not f.is_file():
                continue
            m[f.stem.removeprefix("processed_")] = f
        return m

    map1 = build_map(p1)
    map2 = build_map(p2)
    map1p = build_map(p1p)
    map2p = build_map(p2p)

    if file_names:
        stems = []
        for fn in file_names:
            stem = Path(fn).stem
            stems.append(stem)
        stems = sorted(set(stems))
    else:
        # common stems across all four directories
        stems = sorted(set(map1.keys()) & set(map2.keys()) & set(map1p.keys()) & set(map2p.keys()))

    summary: dict[str, Any] = {}

    for stem in stems:
        entry: dict[str, Any] = {
            "matched_rows": 0,
            "file1_csv": None,
            "original_texts_preview": [],
            "skipped_reason": None,
        }

        f1 = map1.get(stem)
        f2 = map2.get(stem)
        fp1 = map1p.get(stem)
        fp2 = map2p.get(stem)

        missing = []
        if f1 is None:
            missing.append(f"missing in path1: {stem}")
        if f2 is None:
            missing.append(f"missing in path2: {stem}")
        if fp1 is None:
            missing.append(f"missing in file1_processed: {stem}")
        if fp2 is None:
            missing.append(f"missing in file2_processed: {stem}")

        if missing:
            entry["skipped_reason"] = "; ".join(missing)
            summary[stem] = entry
            continue

        try:
            df1 = _read_either(f1).with_row_count("__row_idx")
        except Exception as e:
            entry["skipped_reason"] = f"read error path1: {e}"
            summary[stem] = entry
            continue

        try:
            df2 = _read_either(f2).with_row_count("__row_idx")
        except Exception as e:
            entry["skipped_reason"] = f"read error path2: {e}"
            summary[stem] = entry
            continue

        # Row-count sanity
        if df1.height != df2.height:
            msg = f"row count mismatch: path1={df1.height}, path2={df2.height}"
            if skip_if_row_mismatch:
                entry["skipped_reason"] = msg
                summary[stem] = entry
                continue
            else:
                raise ValueError(msg)

        # Ensure exact_col exists and cast to Int for robust compare
        if exact_col not in df1.columns or exact_col not in df2.columns:
            entry["skipped_reason"] = f"missing '{exact_col}' in path1/path2"
            summary[stem] = entry
            continue

        try:
            df1 = df1.with_columns(pl.col(exact_col).cast(pl.Int64))
            df2 = df2.with_columns(pl.col(exact_col).cast(pl.Int64))
        except Exception:
            df1 = df1.with_columns(pl.col(exact_col).apply(lambda x: int(x) if x is not None else None).alias(exact_col))
            df2 = df2.with_columns(pl.col(exact_col).apply(lambda x: int(x) if x is not None else None).alias(exact_col))

        ids1 = set(df1.filter(pl.col(exact_col) == 0)["__row_idx"].to_list())
        ids2 = set(df2.filter(pl.col(exact_col) == 1)["__row_idx"].to_list())
        matched_idx = sorted(ids1 & ids2)
        entry["matched_rows"] = len(matched_idx)

        if not matched_idx:
            summary[stem] = entry
            continue

        # Read processed files and add row index
        try:
            dfp1 = _read_either(fp1).with_row_count("__row_idx")
        except Exception as e:
            entry["skipped_reason"] = f"read error file1_processed: {e}"
            summary[stem] = entry
            continue

        try:
            dfp2 = _read_either(fp2).with_row_count("__row_idx")
        except Exception as e:
            entry["skipped_reason"] = f"read error file2_processed: {e}"
            summary[stem] = entry
            continue

        if original_col not in dfp1.columns:
            entry["skipped_reason"] = f"missing '{original_col}' in file1_processed"
            summary[stem] = entry
            continue
        if compressed_col not in dfp1.columns:
            entry["skipped_reason"] = f"missing '{compressed_col}' in file1_processed"
            summary[stem] = entry
            continue
        if compressed_col not in dfp2.columns:
            entry["skipped_reason"] = f"missing '{compressed_col}' in file2_processed"
            summary[stem] = entry
            continue

        out1 = (
            dfp1.filter(pl.col("__row_idx").is_in(matched_idx))
            .select(["__row_idx", original_col, compressed_col])
            .rename({compressed_col: "compressed_text_file1"})
            .sort("__row_idx")
            .with_columns([pl.col("compressed_text_file1").str.len_chars().alias("compressed_text_file1_len")])
        )

        out2 = (
            dfp2.filter(pl.col("__row_idx").is_in(matched_idx))
            .select(["__row_idx", compressed_col])
            .rename({compressed_col: "compressed_text_file2"})
            .sort("__row_idx")
            .with_columns([pl.col("compressed_text_file2").str.len_chars().alias("compressed_text_file2_len")])
        )

        final = out1.join(out2, on="__row_idx", how="left")

        file1_csv = outd / f"{stem}__file1_processed.csv"

        try:
            final.write_csv(file1_csv)
        except Exception as e:
            entry["skipped_reason"] = f"write error: {e}"
            summary[stem] = entry
            continue

        # preview original_texts
        try:
            orig_series = final[original_col].to_list()
            if preview_n is None:
                entry["original_texts_preview"] = orig_series
            else:
                entry["original_texts_preview"] = orig_series[:preview_n]
        except Exception:
            entry["original_texts_preview"] = []

        entry["file1_csv"] = str(file1_csv)
        summary[stem] = entry

    return summary


# the file path1 is the record which has 0 value in exact_math ! whether (w_interrogative or & selective_context or lingua2)
# the file path2 is the record which has 1 value in exact_math ! whether (interrogative or & axcer)
summary = process_directories(
    # "/home/itz-amethyst/dev/axcer/experiments/results/axcer/without_interrogative/Meta-Llama-3.1-8B-Instruct/",
    "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/Meta-Llama-3.1-8B-Instruct/",
    "/home/itz-amethyst/dev/axcer/experiments/results/axcer/with_interrogative/Meta-Llama-3.1-8B-Instruct/",
    "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/processed/",
    "/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/",
    output_dir="/home/itz-amethyst/dev/axcer/experiments/results/diff_comparison/case_study",
    file_names=["ai2_arc.csv", "squad.csv", "piqa.csv"],
)
print(summary)
