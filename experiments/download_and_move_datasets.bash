modal volume create dataset-volume

# python download_datasets.py
DEFAULT_PATH="/home/itz-amethyst/dev/axcer/experiments/datasets"
python download_datasets.py

# If provided, use the first argument as the upload path; otherwise, fall back to the default.
if [ -n "$1" ]; then
  LOCAL_DATA_PATH="$1"
else
  LOCAL_DATA_PATH="$DEFAULT_PATH"
fi

if [ ! -e "$LOCAL_DATA_PATH" ]; then
  echo "Error: '$LOCAL_DATA_PATH' does not exist."
  exit 1
fi

# Perform upload preserving filenames and structure, with overwrite enabled
modal volume put dataset-volume "$LOCAL_DATA_PATH/" / --force
