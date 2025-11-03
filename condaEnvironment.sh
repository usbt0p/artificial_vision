#!/bin/bash

# Check if conda is already installed
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed. Installing Anaconda..."
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh
    chmod +x anaconda.sh
    # Run the installer with automated input
    ./anaconda.sh -b <<EOF

q
yes

yes
EOF
    source ~/.bashrc
else
    echo "Conda is already installed."
fi

# Check if the environment already exists
if ! conda info --envs | grep -q "^ComputerVision"; then
    echo "Creating conda environment 'ComputerVision'..."
    conda create --yes --name ComputerVision
else
    echo "Conda environment 'ComputerVision' already exists."
fi

# Initialize conda for shell usage
eval "$(conda shell.bash hook)"

conda activate ComputerVision
conda install --yes pip
pip install -r requirements.txt