#!/bin/bash

# ARGS="$@"

echo "Running selective_context evaluate script "

modal run --detach selective_context_evaluate.py
if [ $? -eq 0 ]; then
  echo "Finished evaluation"
  modal volume get results-vol / ~/dev/axcer/experiments/results/ --force
else
  echo "Inference failed. Skipping volume pull."
  exit 1
fi

echo "All done!"
