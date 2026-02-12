#!/bin/bash
# Quick setup and run script for Amazon Reviews Pipeline

set -e  # Exit on error

echo "=========================================="
echo "Amazon Reviews Pipeline - Quick Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 is not installed"; exit 1; }
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""

# Check if dataset exists
if [ ! -f "Amazon_Reviews.csv" ]; then
    echo "WARNING: Amazon_Reviews.csv not found in current directory."
    echo "Please ensure the dataset file is available."
    echo ""
    read -p "Enter path to Amazon_Reviews.csv (or press Enter to skip): " dataset_path
    if [ -n "$dataset_path" ]; then
        INPUT_FILE="$dataset_path"
    else
        echo "Skipping run. You can run manually later with:"
        echo "  python3 run_pipeline.py --input <path_to_csv> --output_dir output"
        exit 0
    fi
else
    INPUT_FILE="Amazon_Reviews.csv"
fi

# Run the pipeline
echo ""
echo "Running pipeline..."
echo "Input file: $INPUT_FILE"
echo "Output directory: output"
echo ""

python3 run_pipeline.py --input "$INPUT_FILE" --output_dir output

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Check the 'output' directory for results."

