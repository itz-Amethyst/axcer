#!/bin/bash

ARGS="$@"

# ./vllm_evaluate.bash --model-id meta-llama/Meta-Llama-3.1-8B-Instruct --model-revision 0e9e39f249a16976918f6564b8830bc894c89659 --max-batch-size 1 --dataset-path ~/dev/axcer/experiments/datasets/dsf/

if [[ ! "$ARGS" =~ --dataset-path[=[:space:]]?[^[:space:]]+ ]]; then
  echo "Error: --dataset-path argument is required."
  echo "Usage: $0 --dataset-path <path>"
  exit 1
fi

echo "Running vllm_evaluate script with args: $ARGS"
modal run --detach perplexity_compute.py $ARGS
if [ $? -eq 0 ]; then
  echo "Finished evaluation"
  # uncomment this later !!!!
  # modal volume get results-vol / ~/dev/axcer/experiments/results/ --force
else
  echo "Inference failed. Skipping volume pull."
  exit 1
fi

echo "All done!"
