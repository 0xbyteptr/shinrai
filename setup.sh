#!/usr/bin/bash

# This script is used to set up the environment for the project.
# It will install the necessary dependencies and set up the virtual environment.
# Check if the virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping setup."
else 
    # Create a virtual environment
    python3 -m venv .venv
    echo "Virtual environment created."

    # Activate the virtual environment
    source .venv/bin/activate

    # Install the dependencies
    pip install -r requirements.txt

    # Install PyTorch with CUDA support
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo "Dependencies installed."
fi