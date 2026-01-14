#!/bin/bash
# Setup script for XGBoost training conda environment

set -e  # Exit on error

ENV_NAME="xgboost_training"
ENV_FILE="environment.yml"

echo "=========================================="
echo "Setting up Conda Environment"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "Conda version: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Activate with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create environment from yml file
echo ""
echo "Creating conda environment from ${ENV_FILE}..."
conda env create -f ${ENV_FILE}

echo ""
echo "=========================================="
echo "Environment created successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""

