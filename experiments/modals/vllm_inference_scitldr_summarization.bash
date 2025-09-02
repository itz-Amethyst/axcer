#!/bin/bash

ARGS="$@"

# mkdir -p ~/dev/axcer/experiments/datasets/summarization_task

# ONLY THIS MODEL ./vllm_inference_scitldr_summarization.bash --model-id mistralai/Mistral-7B-v0.1 --model-revision 27d67f1b5f57dc0953326b2601d68371d40ea8da --max-batch-size 1

# ./vllm_inference_scitldr_summarization.bash --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 --model-revision eba92302a2861cdc0098cc54bc9f17cb2c47eb61 --max-batch-size 1

# Copy the file to the target directory
# mv ~/dev/axcer/experiments/datasets/temp/processed_scitldr.parquet ~/dev/axcer/experiments/datasets/summarization_task/processed_scitldr.parquet

echo "Running vllm_evaluate summary as reference script with args: $ARGS"

# NOTE: ! Currently this suuports only with batch-size set to 1
modal run --detach vllm_inference_for_scitldr_summary.py $ARGS
if [ $? -eq 0 ]; then
  echo "Finished evaluation"

  echo "Pushing file to the modal volume"
  modal volume put dataset-volume ~/dev/axcer/experiments/datasets/processed_scitldr.parquet / --force
  sleep 2s
  modal volume get dataset-volume / ~/dev/axcer/experiments/datasets/ --force
else
  echo "Inference failed. Skipping volume pull."
  exit 1
fi

echo "All done!"
