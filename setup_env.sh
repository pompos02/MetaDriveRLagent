#!/bin/bash

# Exit on any error
set -e

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Avoid numeric names: 'proj_4228' instead of '1stProject4228'
ENV_NAME="proj_4228"

# Create the conda environment
conda create -n "$ENV_NAME" python=3.10 -y

# Activate the environment
conda activate "$ENV_NAME"

# Install metadrive in editable mode
if [ -d "metadrive" ]; then
    cd metadrive
    pip install -e .
    cd ..
else
    echo "Directory 'metadrive' not found!"
    exit 1
fi

# Install remaining dependencies
pip install -r requirements.txt

echo "Environment '$ENV_NAME' setup complete."
