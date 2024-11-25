#!/bin/bash

#SBATCH --job-name=ann-benchmarks
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --time=7200

# Stop the script on errors
set -e

# Purge all other loaded modules (if any)
module purge

# Load the Docker module
module load docker-rootless

# Load the Python module
module load Python/3.9.5-GCCcore-10.3.0

# Start Docker Daemon
start_docker_rootless.sh

# Install Python dependencies
pip3 install -r requirements.txt

python install.py

python run.py --dataset "$1"