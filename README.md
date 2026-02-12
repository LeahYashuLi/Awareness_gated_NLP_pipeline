# Amazon Reviews Pipeline - Frozen Version

This is a frozen version of the Amazon Reviews NLP pipeline for archive paper submission.

## Local Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- The Amazon Reviews CSV dataset file

### Step 1: Clone or download the repository
```bash
git clone <your-repo-url>
cd amazon-reviews-pipeline
```
Or download and extract the ZIP file, then navigate to the directory.

### Step 2: Install dependencies

**Option A: Using pip (recommended)**
```bash
pip3 install -r requirements.txt
```

**Option B: Using a virtual environment (recommended for isolation)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option C: Using conda (if you have Anaconda/Miniconda)**
```bash
conda create -n amazon_pipeline python=3.10
conda activate amazon_pipeline
pip install -r requirements.txt
```

**Option D: If you encounter permission errors, use --user flag**
```bash
pip3 install --user -r requirements.txt
```

### Step 3: Verify installation
Check that key packages are installed:
```bash
python3 -c "import pandas, numpy, sklearn, transformers, torch, sentence_transformers; print('All packages installed successfully!')"
```

### Step 4: Prepare your dataset
- Download or obtain the Amazon Reviews CSV dataset
- Place it in the project directory, or note its full path
- The dataset should have columns: `Review Text`, `Rating`, `Country`, etc.

## Usage

### Basic Run
Run the pipeline with default settings:
```bash
python3 run_pipeline.py --input Amazon_Reviews.csv --output_dir output
```

### With Custom Paths
If your dataset is in a different location:
```bash
python3 run_pipeline.py --input /path/to/your/Amazon_Reviews.csv --output_dir /path/to/output
```

### With Custom Sample Size
To process more or fewer samples:
```bash
python3 run_pipeline.py --input Amazon_Reviews.csv --output_dir output --sample_size 5000
```

### Command-Line Arguments:
- `--input`: Path to input CSV file (default: `Amazon_Reviews.csv`)
- `--output_dir`: Output directory for results (default: `output`)
- `--sample_size`: Sample size if dataset is larger (default: 2000)

### Expected Runtime
- Processing 2000 samples typically takes 5-15 minutes depending on your hardware
- First run will download ML models (~500MB), subsequent runs are faster
- GPU acceleration is used automatically if available (CUDA/MPS)

## Outputs

The pipeline generates the following outputs in the specified output directory:

1. **results.parquet**: Complete results dataframe with all computed features
2. **summary.json**: Summary metrics including model performance and statistics
3. **ablation_table.csv**: Ablation study showing performance with different feature sets
4. **pca_figure.png**: PCA visualization colored by triangulation case and actual outcome
5. **calibration_roc_figure.png**: Calibration curve and ROC curve for model evaluation

## Pipeline Steps

1. Load and sample Amazon reviews data
2. Clean text and extract ratings
3. Compute sentiment scores using transformers
4. Compute quality scores (length, diversity, repetition)
5. Generate sentence embeddings
6. Compute semantic coherence
7. Perform topic modeling (UMAP + HDBSCAN)
8. Build topic anchors and label topics
9. Compute topic similarities
10. Train text-only prediction model
11. Compute triangulation scores and cases
12. Perform PCA analysis
13. Generate ablation table and visualizations

## Troubleshooting

### Issue: "ModuleNotFoundError" for specific packages
**Solution**: Install the missing package individually:
```bash
pip3 install <package_name>
# or with --user flag:
pip3 install --user <package_name>
```

### Issue: SSL certificate errors during installation
**Solution**: Use trusted hosts:
```bash
pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: Permission denied errors
**Solution**: Use --user flag or virtual environment:
```bash
pip3 install --user -r requirements.txt
```

### Issue: hdbscan installation fails
**Solution**: Install system dependencies first (macOS):
```bash
brew install cmake
pip3 install hdbscan
```

### Issue: Out of memory errors
**Solution**: Reduce sample size:
```bash
python3 run_pipeline.py --input Amazon_Reviews.csv --output_dir output --sample_size 1000
```

### Issue: Slow performance
- The pipeline automatically uses GPU if available (CUDA for NVIDIA, MPS for Apple Silicon)
- For CPU-only systems, processing will be slower but should still work
- Consider reducing sample_size if needed

## Changes from Original Notebook

- ✅ Removed Colab/drive mount code
- ✅ Removed inline pip installs
- ✅ Fixed undefined SEED variable bug
- ✅ Consolidated into single entry script
- ✅ Added requirements.txt
- ✅ Made file paths configurable via command-line arguments
