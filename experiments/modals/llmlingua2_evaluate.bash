#!/bin/bash

# ARGS="$@"

echo "Running llmlingua2 evaluate script "

modal run --detach llmlingua2_evaluate.py

if [ $? -eq 0 ]; then
  echo "Finished evaluation"
  modal volume get results-vol / ~/dev/axcer/experiments/results/ --force
else
  echo "Inference failed. Skipping volume pull."
  exit 1
fi

echo "All done!"
