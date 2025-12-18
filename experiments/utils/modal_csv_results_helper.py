import polars as pl
import re
from pathlib import Path


def clean_csv_files(directory: str):
    dir_path = Path(directory)

    for csv_file in dir_path.glob("*.csv"):
        print(f"Processing file: {csv_file.name}")

        df = pl.read_csv(csv_file)

        before_count = len(df)

        df_cleaned = df.drop_nulls()

        after_count = len(df_cleaned)

        df_cleaned.write_csv(csv_file)

        print(f"  Rows before: {before_count}")
        print(f"  Rows after:  {after_count}\n")


def remove_columns_from_csv(directory: str, columns_to_remove: list[str]):
    dir_path = Path(directory)

    for csv_file in dir_path.glob("*.csv"):
        print(f"Processing file: {csv_file.name}")

        df = pl.read_csv(csv_file)

        missing = [col for col in columns_to_remove if col not in df.columns]
        if missing:
            print(f"  ❌ Missing columns in {csv_file.name}: {missing}")

        existing = [col for col in columns_to_remove if col in df.columns]
        df_cleaned = df.drop(existing) if existing else df

        df_cleaned.write_csv(csv_file)
        print("  ✅ Columns removed where possible\n")


def compare_links_between_parquets(
    file1: str,
    file2: str,
    input_field: str,
):
    """
    Compare links extracted from two parquet files (file1 as reference).
    Prints only the missing links that exist in file1 but not in file2.

    Args:
        file1 (str): Path to first parquet file (reference).
        file2 (str): Path to second parquet file (to compare against).
        input_field (str): The column name containing text with links.
    """

    url_pattern = re.compile(
        r"((https?|ftp):/{1,2})?(?<!@)([a-zA-Z0-9-]+\.[a-zA-Z]{2,}|localhost)(:\d+)?[-\w@:%_.+/~#?=&]*",
    )

    def extract_links(df: pl.DataFrame, field: str):
        counter = 0
        all_links = []
        counter_list = []
        for text in df[field].to_list():
            matches = url_pattern.findall(text)
            links = ["".join(m) for m in matches]
            all_links.extend(links)
            if links:
                counter_list.append(counter)
                counter += 1
        counter = 0
        for i in counter_list:
            print(f"Row number is : {i}")
        return set(all_links)

    df1 = pl.read_parquet(file1)
    df2 = pl.read_parquet(file2)

    if input_field not in df1.columns or input_field not in df2.columns:
        raise ValueError(f"Column '{input_field}' must exist in both files.")

    links1 = extract_links(df1, input_field)
    links2 = extract_links(df2, input_field)

    missing_links = links1 - links2

    # Print results
    print(f"Total unique links in {file1}: {len(links1)}")
    print(f"Total unique links in {file2}: {len(links2)}")
    print("\nMissing links (present in file1 but not in file2):")
    for link in sorted(missing_links):
        print(link)


def concat_csvs(base_dir1: str, base_dir2: str) -> None:
    """
    Find matching CSV files in two directories, concatenate them vertically,
    and save the result in base_dir2 with the same filename.

    Parameters
    ----------
    base_dir1 : str
        Path to the first directory containing CSV files.
    base_dir2 : str
        Path to the second directory where merged CSVs will be saved.
    """
    dir1 = Path(base_dir1)
    dir2 = Path(base_dir2)

    dir1.mkdir(parents=True, exist_ok=True)

    files1 = {f.name: f for f in dir1.glob("*.csv")}
    files2 = {f.name: f for f in dir2.glob("*.csv")}

    # Find common filenames
    common_files = set(files1.keys()) & set(files2.keys())

    for fname in common_files:
        # Read both CSVs
        df1 = pl.read_csv(files1[fname])
        df2 = pl.read_csv(files2[fname])

        # Concatenate vertically
        combined = pl.concat([df1, df2], how="vertical")

        output_path = dir1 / fname
        combined.write_csv(output_path)

        print(f"✅ Concatenated and saved: {output_path}")


def sum_tokens_and_cost(directory: str, cost_per_million: float):
    path = Path(directory)
    csv_files = list(path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    for csv_file in csv_files:
        try:
            df = pl.read_csv(csv_file)

            if "compressed_tokens" in df.columns:
                total_tokens = df["compressed_tokens"].sum()

                cost = (total_tokens / 1_000_000) * cost_per_million

                print(f"{csv_file.name}:")
                print(f"  Total Tokens: {total_tokens}")
                print(f"  Cost (@ {cost_per_million}/1M): {cost:.6f}")
                print("-" * 40)
            else:
                print(f"{csv_file.name}: 'compressed_tokens' column not found.")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


def concat_by_category(base_dir: str, categories: dict[str, list[str]]) -> None:
    """
    For each category, create a subfolder, concatenate the given CSV files vertically,
    and save the result as `concated.csv` in that subfolder.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing CSV files.
    categories : dict[str, list[str]]
        Dictionary where keys are category names and values are lists of CSV filenames (without .csv extension).
    """
    base_path = Path(base_dir)

    for category, files in categories.items():
        category_dir = base_path / category
        category_dir.mkdir(parents=True, exist_ok=True)

        dataframes = []
        for fname in files:
            file_path = base_path / f"{fname}.csv"
            if file_path.exists():
                df = pl.read_csv(file_path)
                dataframes.append(df)
            else:
                print(f"⚠️ File not found: {file_path}")

        if dataframes:
            combined = pl.concat(dataframes, how="vertical")
            output_path = category_dir / "concated.csv"
            combined.write_csv(output_path)
            print(f"✅ Saved {output_path}")
        else:
            print(f"⚠️ No valid files for category '{category}'")


# Example usage
if __name__ == "__main__":
    # clean_csv_files("/home/itz-amethyst/dev/axcer/experiments/lin_ev/processed/")
    cols = [
        "prompt_tokens",
        "compressed_tokens",
        "compression_ratio",
        "gpt_o1_saving",
        "compression_time",
        "end_to_end_time",
        "compression_ratio_normalized",
    ]
    datasets = {
        "qa": ["squad", "piqa"],
        "classification": ["boolq"],
        "nli": ["glue"],
        "math": ["gsm8k", "mawps"],
        "python_code": ["mbpp"],
        "summarization": ["scitldr"],
        "multiple_choice": ["ai2_arc"],
    }
    # result = compare_links_between_parquets(
    #     "/home/itz-amethyst/dev/axcer/experiments/res_vault/processed_scitldr.parquet",
    #     # "/home/itz-amethyst/dev/axcer/experiments/results/lingua2/merged/scitldr.parquet",
    #     "/home/itz-amethyst/dev/axcer/experiments/results/axcer/merged/scitldr.parquet",
    #     # "/home/itz-amethyst/dev/axcer/experiments/results/selective_context/merged/scitldr.parquet",
    #     "input_field"
    # )

    # concat_csvs("/home/itz-amethyst/dev/axcer/experiments/results/selective_context/Meta-Llama-3.1-8B-Instruct/", "/home/itz-amethyst/dev/axcer/experiments/results/selective_context/gemma-3-12b-it/")
    # concat_by_category("/home/itz-amethyst/dev/axcer/experiments/results/lingua2/concated_models_results/",datasets)
    # sum_tokens_and_cost("/home/itz-amethyst/dev/axcer/experiments/results/selective_context/processed/", 15.0)
    # sum_tokens_and_cost("/home/itz-amethyst/dev/axcer/experiments/results/lingua2/processed/", 15.0)
    # sum_tokens_and_cost("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/", 15.0)
