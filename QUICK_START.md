# Quick Start Guide - Running the Pipeline Locally

## Method 1: Automated Setup (Easiest)

1. Open Terminal
2. Navigate to the Downloads folder:
   ```bash
   cd ~/Downloads
   ```
3. Run the setup script:
   ```bash
   ./setup_and_run.sh
   ```
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Run the pipeline automatically

## Method 2: Manual Setup

### Step 1: Open Terminal
Press `Cmd + Space`, type "Terminal", and press Enter.

### Step 2: Navigate to Downloads folder
```bash
cd ~/Downloads
```

### Step 3: Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```
*(This will take 5-10 minutes the first time)*

### Step 5: Run the pipeline
```bash
python3 run_pipeline.py --input Amazon_Reviews.csv --output_dir output
```

### Step 6: Check results
After completion, check the `output` folder:
```bash
ls -lh output/
```

You should see:
- `results.parquet` - Full results data
- `summary.json` - Summary metrics
- `ablation_table.csv` - Ablation study table
- `pca_figure.png` - PCA visualization
- `calibration_roc_figure.png` - Calibration/ROC curves

## Method 3: Using Existing Python Environment

If you already have Python packages installed globally:

```bash
cd ~/Downloads
pip3 install -r requirements.txt
python3 run_pipeline.py --input Amazon_Reviews.csv --output_dir output
```

## Troubleshooting

### "Command not found: python3"
Install Python 3 from python.org or use Homebrew:
```bash
brew install python3
```

### "Permission denied" errors
Use `--user` flag:
```bash
pip3 install --user -r requirements.txt
```

### Package installation fails
Try installing packages individually:
```bash
pip3 install pandas numpy scikit-learn transformers torch sentence-transformers nltk umap-learn hdbscan pyarrow matplotlib seaborn
```

### Dataset not found
Make sure `Amazon_Reviews.csv` is in the Downloads folder, or specify the full path:
```bash
python3 run_pipeline.py --input /full/path/to/Amazon_Reviews.csv --output_dir output
```

## Expected Output

When running successfully, you'll see progress messages like:
```
============================================================
Amazon Reviews Pipeline - Frozen Version
============================================================
Loading data from Amazon_Reviews.csv...
Original shape: (X, Y)
Sampled shape: (2000, Y)
Computing sentiment scores...
GPU available: True | device: 0
Computing embeddings...
...
Pipeline completed successfully!
```

## Need Help?

Check the full README.md for detailed documentation and troubleshooting tips.

