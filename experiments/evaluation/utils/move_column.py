from pathlib import Path

import pandas as pd

# =========================
# Configuration
# =========================

SOURCE_DIR = Path("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/testing/")
TARGET_DIR = Path("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed")
OUTPUT_DIR = Path("/home/itz-amethyst/dev/axcer/experiments/results/axcer/processed/clean/")

COLUMN_NAME = "memory_used_kb"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Processing
# =========================

for target_file in TARGET_DIR.glob("*.csv"):
    source_file = SOURCE_DIR / target_file.name
    print(source_file)

    if not source_file.exists():
        print(f"Skipping {target_file.name}: source file not found")
        continue

    source_df = pd.read_csv(source_file)
    target_df = pd.read_csv(target_file)

    if COLUMN_NAME not in source_df.columns:
        print(f"Skipping {target_file.name}: '{COLUMN_NAME}' not found")
        continue

    if len(source_df) != len(target_df):
        print(f"Skipping {target_file.name}: row count mismatch ({len(source_df)} vs {len(target_df)})")
        continue

    target_df[COLUMN_NAME] = source_df[COLUMN_NAME]

    output_file = OUTPUT_DIR / target_file.name
    target_df.to_csv(output_file, index=False)

    print(f"Processed: {target_file.name}")

print("Done.")
