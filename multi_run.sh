#!/bin/bash

# List of datasets
datasets=("sift-128-euclidean" "glove-100-angular" "random-xs-20-euclidean" "deep-image-96-angular" "mnist-784-euclidean")

# Loop through each dataset and run the run.sh script
for dataset in "${datasets[@]}"; do
  sbatch run.sh "$dataset"
done