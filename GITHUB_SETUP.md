# GitHub Repository Setup Guide

## Quick Setup Steps

### 1. Initialize Git Repository
```bash
cd ~/Downloads
git init
```

### 2. Add Files to Git
```bash
# Add all relevant files
git add run_pipeline.py
git add requirements.txt
git add README.md
git add QUICK_START.md
git add setup_and_run.sh
git add .gitignore

# Verify what will be committed
git status
```

### 3. Create Initial Commit
```bash
git commit -m "Initial commit: Frozen Amazon Reviews Pipeline for archive paper submission"
```

### 4. Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., `amazon-reviews-pipeline`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)

### 5. Connect and Push
```bash
# Add remote repository (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/amazon-reviews-pipeline.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## What to Include in GitHub

✅ **Include:**
- `run_pipeline.py` - Main pipeline script
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `QUICK_START.md` - Quick start guide
- `setup_and_run.sh` - Setup script
- `.gitignore` - Git ignore rules

❌ **Do NOT Include:**
- `Amazon_Reviews.csv` - Dataset file (too large, add to .gitignore)
- `output/` folder - Generated results
- `venv/` - Virtual environment
- `*.parquet`, `*.png` - Generated outputs
- Any personal/sensitive data

## Repository Description Suggestions

**Title:** Amazon Reviews NLP Pipeline - Archive Version

**Description:**
```
Frozen version of Amazon Reviews NLP pipeline for archive paper submission.
Includes sentiment analysis, topic modeling, triangulation analysis, and prediction models.
```

**Topics/Tags:**
- nlp
- sentiment-analysis
- topic-modeling
- machine-learning
- python
- transformers
- hdbscan
- umap

## Optional: Add License

If you want to add a license (recommended for academic code):

```bash
# Create LICENSE file (choose appropriate license)
# Common choices: MIT, Apache 2.0, or your institution's license
```

## Optional: Add Citation File

Create `CITATION.cff` for academic citation:

```yaml
cff-version: 1.2.0
title: Amazon Reviews NLP Pipeline
message: "If you use this software, please cite it as below."
authors:
  - given-names: Your
    family-names: Name
date-released: 2025-02-11
```

## Making Releases

For paper submission, consider creating a release:

```bash
# Tag the version
git tag -a v1.0.0 -m "Archive version for paper submission"

# Push tag to GitHub
git push origin v1.0.0
```

Then create a release on GitHub with:
- Tag: v1.0.0
- Title: Archive Version - Paper Submission
- Description: Frozen version for archive paper submission

## Repository Structure

Your GitHub repo should look like:
```
amazon-reviews-pipeline/
├── .gitignore
├── README.md
├── QUICK_START.md
├── GITHUB_SETUP.md (this file)
├── requirements.txt
├── run_pipeline.py
└── setup_and_run.sh
```

## Next Steps After Publishing

1. Update README.md with your actual GitHub repo URL
2. Add a link to your paper (if published)
3. Consider adding example outputs (small sample)
4. Add badges (optional) for Python version, license, etc.

