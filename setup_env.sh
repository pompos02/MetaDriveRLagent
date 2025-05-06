#!/bin/bash

# Exit on any error
set -e

# Create the conda environment
conda create -n 1stProject4228 python=3.10 -y

# Initialize conda for non-interactive use
source "$(conda info --base)/etc/profile.d/conda.sh"


# Install metadrive in editable mode
cd metadrive
pip install -e .
cd ..

conda activate 1stProject4228
# Install remaining dependencies
pip install -r requirements.txt

echo "Environment '1stProject4228' setup complete."
