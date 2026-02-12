#!/usr/bin/env python3
"""
Amazon Reviews Pipeline - Frozen Version for Archive Paper Submission
This script processes Amazon reviews through an NLP pipeline including:
- Sentiment analysis
- Quality scoring
- Topic modeling (HDBSCAN)
- Triangulation analysis
- Model predictions
"""

import pandas as pd
import numpy as np
import re
import json
import os
from pathlib import Path

# ML/NLP libraries
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, calibration_curve
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def load_and_sample_data(file_path, sample_size=2000, random_state=42):
    """Load Amazon reviews CSV and sample."""
    print(f"Loading data from {file_path}...")
    df_reviews = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip", engine='python')
    print(f"Original shape: {df_reviews.shape}")
    
    # Sample if dataset is larger than sample_size
    if len(df_reviews) > sample_size:
        df_reviews = df_reviews.sample(sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Sampled shape: {df_reviews.shape}")
    
    return df_reviews


def clean_data(df_reviews):
    """Clean rating and text, create labels."""
    TEXT_COL = "Review Text"
    RATING_COL = "Rating"
    
    # 1) Clean text
    df_reviews["narrative_text"] = df_reviews[TEXT_COL].astype(str).str.strip()
    
    # Remove placeholder / junk review texts
    bad_texts = {"", "review text not found", "text not found", "review not found"}
    mask_bad = df_reviews["narrative_text"].str.lower().isin(bad_texts)
    df_reviews = df_reviews[~mask_bad].reset_index(drop=True)
    
    # 2) Extract the 1–5 numeric rating from strings like "Rated 1 out of 5 stars"
    df_reviews["satisfaction"] = (
        df_reviews[RATING_COL].astype(str)
        .str.extract(r"Rated\s+(\d)\s+out", expand=False)
        .astype(float)
    )
    
    print(f"Non-null satisfaction count: {df_reviews['satisfaction'].notna().sum()}")
    
    # 3) Drop rows that have missing / invalid satisfaction
    df_reviews = df_reviews.dropna(subset=["satisfaction"]).reset_index(drop=True)
    print(f"Shape after cleaning: {df_reviews.shape}")
    
    # 4) Binary outcome: >=4 stars = "success"
    df_reviews["actual_outcome"] = (df_reviews["satisfaction"] >= 4).astype(int)
    
    # 5) Country as group feature
    df_reviews["country_group"] = df_reviews["Country"].fillna("Unknown").astype(str)
    
    # Collapse tiny country groups
    min_n = 30
    counts = df_reviews["country_group"].value_counts()
    df_reviews.loc[df_reviews["country_group"].isin(counts[counts < min_n].index), "country_group"] = "Other"
    
    return df_reviews


def compute_sentiment(df_reviews):
    """Compute sentiment scores using transformers."""
    print("Computing sentiment scores...")
    use_gpu = torch.cuda.is_available()
    device_id = 0 if use_gpu else -1
    print(f"GPU available: {use_gpu} | device: {device_id}")
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device_id
    )
    
    texts = df_reviews["narrative_text"].fillna("").astype(str).tolist()
    
    results = sentiment_pipe(
        texts,
        batch_size=64 if use_gpu else 16,
        truncation=True,
        max_length=256
    )
    
    def to_signed(res):
        return res["score"] if res["label"].upper().startswith("POS") else -res["score"]
    
    df_reviews["narrative_sentiment"] = [to_signed(r) for r in results]
    return df_reviews


def quality_score(text):
    """Compute quality score based on length, diversity, and repetition."""
    text = (text or "").strip()
    if not text:
        return 0.0
    
    # length factor
    length = len(text.split())
    length_factor = min(length / 30.0, 1.0)
    
    # lexical diversity
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    unique_tokens = set(tokens)
    diversity = len(unique_tokens) / len(tokens)
    
    # repetition penalty (simple)
    repetition_penalty = diversity  # higher diversity => less repetition
    
    score = length_factor * (0.5 * diversity + 0.5 * repetition_penalty)
    return float(np.clip(score, 0.0, 1.0))


def compute_quality_scores(df_reviews):
    """Compute quality scores for all reviews."""
    print("Computing quality scores...")
    df_reviews["quality_score"] = df_reviews["narrative_text"].apply(quality_score)
    return df_reviews


def compute_embeddings(df_reviews):
    """Compute sentence embeddings."""
    print("Computing embeddings...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    texts = df_reviews["narrative_text"].fillna("").astype(str).tolist()
    use_gpu = torch.cuda.is_available()
    
    embeddings_real = embed_model.encode(
        texts,
        batch_size=64 if use_gpu else 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    df_reviews["embedding_384"] = list(embeddings_real)
    print(f"Embeddings shape: {embeddings_real.shape}")
    
    return df_reviews, embed_model


def semantic_coherence(text, embed_model, min_words=10):
    """Compute semantic coherence score."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    if len(text.split()) < min_words:
        return 0.0
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if len(sents) < 2:
        return 0.0
    
    sent_embs = embed_model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(sent_embs)
    upper = sims[np.triu_indices_from(sims, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def compute_semantic_coherence(df_reviews, embed_model):
    """Compute semantic coherence for all reviews."""
    print("Computing semantic coherence...")
    df_reviews["semantic_coherence"] = df_reviews["narrative_text"].apply(
        lambda t: semantic_coherence(t, embed_model)
    )
    return df_reviews


def compute_topics(df_reviews):
    """Compute topics using UMAP + HDBSCAN."""
    print("Computing topics...")
    X = np.vstack(df_reviews["embedding_384"].values)
    
    # UMAP dimensionality reduction
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=SEED
    )
    
    X_umap = umap_model.fit_transform(X)
    print(f"UMAP shape: {X_umap.shape}")
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25,
        min_samples=5,
        metric="euclidean"
    )
    
    labels = clusterer.fit_predict(X_umap)
    
    df_reviews["topic_id"] = labels
    df_reviews["topic_label"] = np.where(
        labels == -1,
        "outlier",
        [f"topic_{i}" for i in labels]
    )
    
    print(f"Outlier rate: {(labels == -1).mean():.4f}")
    print(df_reviews["topic_label"].value_counts().head(10))
    
    return df_reviews


def build_topic_anchors(df, text_col="narrative_text", emb_col="embedding_384", 
                        topic_col="topic_label", n_anchors=5, min_words=8):
    """Build topic anchor sentences."""
    topic_anchors = {}
    
    for topic, sub in df.groupby(topic_col):
        if topic == "outlier" or len(sub) < n_anchors:
            continue
        
        texts = sub[text_col].astype(str).tolist()
        embs = np.vstack(sub[emb_col].values)
        
        # compute centroid
        centroid = embs.mean(axis=0, keepdims=True)
        
        # cosine similarity to centroid
        sims = cosine_similarity(embs, centroid).reshape(-1)
        
        # rank by similarity
        ranked_idx = np.argsort(-sims)
        
        anchors = []
        for idx in ranked_idx:
            t = texts[idx].strip()
            if len(t.split()) >= min_words:
                anchors.append(t)
            if len(anchors) >= n_anchors:
                break
        
        topic_anchors[topic] = anchors
    
    return topic_anchors


def label_topics(df_reviews):
    """Label topics with manual labels."""
    topic_label_map_B = {
        "topic_4": "General satisfaction / overall experience",
        "topic_6": "Prime shipping delays & late delivery",
        "topic_10": "Account security, billing & fraud issues",
        "topic_5": "Delivery failures & missing packages",
        "topic_8": "Unsafe drop-off & driver behavior",
        "topic_7": "Misdelivery & unresolved complaints"
    }
    
    df_reviews["topic_name"] = (
        df_reviews["topic_label"]
        .map(topic_label_map_B)
        .fillna("Other issues")
    )
    
    df_reviews.loc[
        df_reviews["topic_label"] == "outlier",
        "topic_name"
    ] = "Outlier / idiosyncratic reviews"
    
    return df_reviews


def build_topic_centroids(df, emb_col="embedding_384", topic_col="topic_id"):
    """Build topic centroids."""
    centroids = {}
    for tid in sorted(df[topic_col].unique()):
        if tid == -1:
            continue
        sub = df[df[topic_col] == tid]
        if len(sub) == 0:
            continue
        centroids[tid] = np.mean(np.stack(sub[emb_col].values), axis=0)
    return centroids


def compute_topic_similarities(df_reviews):
    """Compute similarity to topic centroids."""
    print("Computing topic similarities...")
    centroids = build_topic_centroids(df_reviews)
    
    def cos(a, b):
        return float(np.dot(a, b))  # embeddings are normalized
    
    topic_sim_cols = []
    for tid, cvec in centroids.items():
        col = f"sim_topic_{tid}"
        topic_sim_cols.append(col)
        df_reviews[col] = df_reviews["embedding_384"].apply(lambda e: cos(e, cvec))
    
    # summary signals
    df_reviews["topic_confidence"] = df_reviews[topic_sim_cols].max(axis=1)
    df_reviews["closest_topic_id"] = (
        df_reviews[topic_sim_cols]
        .idxmax(axis=1)
        .str.replace("sim_topic_", "")
        .astype(int)
    )
    
    return df_reviews, topic_sim_cols


def train_prediction_model(df_reviews, topic_sim_cols):
    """Train text-only prediction model."""
    print("Training prediction model...")
    use_cols = ["narrative_sentiment", "quality_score", "semantic_coherence"] + topic_sim_cols
    X = df_reviews[use_cols].fillna(0)
    y = df_reviews["actual_outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    
    pipe = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    
    pipe.fit(X_train, y_train)
    
    df_reviews["predicted_prob"] = pipe.predict_proba(X)[:, 1]
    
    # Evaluate on test set
    p_test = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    return df_reviews, pipe, {"roc_auc": float(roc_auc), "auprc": float(auprc)}


def compute_triangulation(df_reviews, COH_GATE=0.10, QUALITY_GATE=0.5):
    """Compute triangulation scores and cases."""
    print("Computing triangulation...")
    
    df_reviews["sat_n"] = df_reviews["satisfaction"] / 5.0
    df_reviews["sent_n"] = (df_reviews["narrative_sentiment"] + 1) / 2.0
    df_reviews["pred_n"] = df_reviews["predicted_prob"]
    
    # gate sentiment when narrative is unreliable
    df_reviews["sent_weight"] = 1.0
    df_reviews["narrative_missing"] = (df_reviews["narrative_text"].fillna("").str.strip() == "").astype(int)
    df_reviews.loc[df_reviews["narrative_missing"] == 1, "sent_weight"] = 0.0
    
    df_reviews.loc[df_reviews["quality_score"] < QUALITY_GATE, "sent_weight"] = 0.0
    df_reviews.loc[df_reviews["semantic_coherence"] < COH_GATE, "sent_weight"] = 0.0
    
    df_reviews["impact_score"] = (
        0.40 * df_reviews["sat_n"] +
        0.35 * df_reviews["pred_n"] +
        0.20 * (df_reviews["sent_n"] * df_reviews["sent_weight"]) +
        0.05 * df_reviews["quality_score"]
    )
    
    def tri_case(row):
        if row["impact_score"] >= 0.7:
            return "high_consistent"
        if row["sent_n"] >= 0.7 and row["satisfaction"] <= 2:
            return "pos_text_low_stars"
        if row["sent_n"] <= 0.3 and row["satisfaction"] >= 4:
            return "neg_text_high_stars"
        if row["semantic_coherence"] < COH_GATE:
            return "low_semantic_coherence"
        return "mixed_or_low"
    
    df_reviews["triangulation_case"] = df_reviews.apply(tri_case, axis=1)
    
    # 4-type triangulation
    pos_rate = df_reviews["actual_outcome"].mean()
    MODEL_THR = df_reviews["predicted_prob"].quantile(1 - pos_rate)
    SENT_THR = df_reviews.loc[df_reviews["sent_weight"] > 0, "sent_n"].quantile(0.70)
    
    df_reviews["kpi_high"] = (df_reviews["satisfaction"] >= 4).astype(int)
    df_reviews["model_high"] = (df_reviews["predicted_prob"] >= MODEL_THR).astype(int)
    df_reviews["sent_high"] = (
        (df_reviews["sent_n"] >= SENT_THR) &
        (df_reviews["sent_weight"] > 0)
    ).astype(int)
    
    def triangulation_4type(row):
        evidence = (row["model_high"] + row["sent_high"]) >= 1
        if row["kpi_high"] and evidence:
            return "A_strong_aligned_impact"
        if row["kpi_high"] and not evidence:
            return "B_silent_impact"
        if (not row["kpi_high"]) and evidence:
            return "C_perceived_impact"
        return "D_low_or_no_impact"
    
    df_reviews["triangulation_4type"] = df_reviews.apply(triangulation_4type, axis=1)
    
    # Alignment
    thr = df_reviews["predicted_prob"].quantile(0.74)
    df_reviews["alignment"] = df_reviews.apply(
        lambda row: "aligned" if (int(row["predicted_prob"] >= thr) == int(row["actual_outcome"])) else "contradiction",
        axis=1
    )
    
    return df_reviews


def compute_pca(df_reviews):
    """Compute PCA for visualization."""
    print("Computing PCA...")
    num_cols_B = [
        "satisfaction",
        "narrative_sentiment",
        "predicted_prob",
        "quality_score",
        "semantic_coherence",
        "impact_score"
    ]
    
    X_num = df_reviews[num_cols_B].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    pca = PCA(n_components=2, random_state=SEED)
    pc = pca.fit_transform(X_scaled)
    
    df_reviews["PC1"] = pc[:, 0]
    df_reviews["PC2"] = pc[:, 1]
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    return df_reviews, pca


def create_ablation_table(df_reviews):
    """Create ablation table showing performance with different feature sets."""
    print("Creating ablation table...")
    
    topic_sim_cols = [col for col in df_reviews.columns if col.startswith("sim_topic_")]
    
    feature_sets = {
        "Sentiment only": ["narrative_sentiment"],
        "Quality only": ["quality_score", "semantic_coherence"],
        "Topics only": topic_sim_cols,
        "Sentiment + Quality": ["narrative_sentiment", "quality_score", "semantic_coherence"],
        "All features": ["narrative_sentiment", "quality_score", "semantic_coherence"] + topic_sim_cols
    }
    
    results = []
    X_all = df_reviews[["narrative_sentiment", "quality_score", "semantic_coherence"] + topic_sim_cols].fillna(0)
    y = df_reviews["actual_outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.30, random_state=SEED, stratify=y
    )
    
    for name, features in feature_sets.items():
        if not all(f in X_all.columns for f in features):
            continue
        
        X_f = X_train[features]
        X_test_f = X_test[features]
        
        pipe = SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
        ])
        
        pipe.fit(X_f, y_train)
        p_test = pipe.predict_proba(X_test_f)[:, 1]
        
        roc_auc = roc_auc_score(y_test, p_test)
        auprc = average_precision_score(y_test, p_test)
        
        results.append({
            "Feature Set": name,
            "ROC-AUC": roc_auc,
            "AUPRC": auprc,
            "N Features": len(features)
        })
    
    ablation_df = pd.DataFrame(results)
    return ablation_df


def plot_pca(df_reviews, output_dir):
    """Create PCA visualization."""
    print("Creating PCA figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA colored by triangulation case
    cases = df_reviews["triangulation_case"].astype("category").cat.codes
    scatter1 = axes[0].scatter(
        df_reviews["PC1"],
        df_reviews["PC2"],
        c=cases,
        cmap="tab10",
        alpha=0.6
    )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("PC1 — Overall Impact Alignment")
    axes[0].set_ylabel("PC2 — Narrative vs Quantitative Tension")
    axes[0].set_title("Triangulation PCA: Alignment vs Contradiction")
    plt.colorbar(scatter1, ax=axes[0])
    
    # PCA colored by actual outcome
    scatter2 = axes[1].scatter(
        df_reviews["PC1"],
        df_reviews["PC2"],
        c=df_reviews["actual_outcome"],
        cmap="coolwarm",
        alpha=0.6
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PCA colored by Actual Outcome")
    plt.colorbar(scatter2, ax=axes[1], label="Actual outcome (≥4 stars)")
    
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "pca_figure.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA figure to {pca_path}")


def plot_calibration_roc(df_reviews, output_dir):
    """Create calibration and ROC curves."""
    print("Creating calibration/ROC figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration curve
    y_true = df_reviews["actual_outcome"]
    y_pred = df_reviews["predicted_prob"]
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=10, strategy='uniform'
    )
    
    axes[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    axes[1].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    calib_path = os.path.join(output_dir, "calibration_roc_figure.png")
    plt.savefig(calib_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration/ROC figure to {calib_path}")


def compute_summary_metrics(df_reviews, model_metrics):
    """Compute summary metrics."""
    summary = {
        "dataset_size": len(df_reviews),
        "model_performance": model_metrics,
        "outcome_distribution": {
            "positive_rate": float(df_reviews["actual_outcome"].mean()),
            "negative_rate": float(1 - df_reviews["actual_outcome"].mean())
        },
        "triangulation_distribution": df_reviews["triangulation_case"].value_counts(normalize=True).to_dict(),
        "alignment_rate": float((df_reviews["alignment"] == "aligned").mean()),
        "topic_statistics": {
            "n_topics": int(df_reviews[df_reviews["topic_id"] != -1]["topic_id"].nunique()),
            "outlier_rate": float((df_reviews["topic_id"] == -1).mean()),
            "avg_topic_confidence": float(df_reviews["topic_confidence"].mean())
        },
        "feature_statistics": {
            "avg_sentiment": float(df_reviews["narrative_sentiment"].mean()),
            "avg_quality": float(df_reviews["quality_score"].mean()),
            "avg_coherence": float(df_reviews["semantic_coherence"].mean()),
            "avg_satisfaction": float(df_reviews["satisfaction"].mean())
        }
    }
    
    return summary


def main():
    """Main pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Amazon Reviews Pipeline")
    parser.add_argument("--input", type=str, default="Amazon_Reviews.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=2000,
                       help="Sample size (if dataset is larger)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Amazon Reviews Pipeline - Frozen Version")
    print("=" * 60)
    
    # Load and clean data
    df_reviews = load_and_sample_data(args.input, args.sample_size, SEED)
    df_reviews = clean_data(df_reviews)
    
    # Compute features
    df_reviews = compute_sentiment(df_reviews)
    df_reviews = compute_quality_scores(df_reviews)
    df_reviews, embed_model = compute_embeddings(df_reviews)
    df_reviews = compute_semantic_coherence(df_reviews, embed_model)
    
    # Topic modeling
    df_reviews = compute_topics(df_reviews)
    df_reviews = label_topics(df_reviews)
    df_reviews, topic_sim_cols = compute_topic_similarities(df_reviews)
    
    # Prediction model
    df_reviews, model, model_metrics = train_prediction_model(df_reviews, topic_sim_cols)
    
    # Triangulation
    df_reviews = compute_triangulation(df_reviews)
    
    # PCA
    df_reviews, pca = compute_pca(df_reviews)
    
    # Create outputs
    print("\n" + "=" * 60)
    print("Generating outputs...")
    print("=" * 60)
    
    # Save results parquet
    results_path = os.path.join(output_dir, "results.parquet")
    df_reviews.to_parquet(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Compute summary metrics
    summary = compute_summary_metrics(df_reviews, model_metrics)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Create ablation table
    ablation_df = create_ablation_table(df_reviews)
    ablation_path = os.path.join(output_dir, "ablation_table.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved ablation table to {ablation_path}")
    
    # Create figures
    plot_pca(df_reviews, output_dir)
    plot_calibration_roc(df_reviews, output_dir)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - results.parquet")
    print(f"  - summary.json")
    print(f"  - ablation_table.csv")
    print(f"  - pca_figure.png")
    print(f"  - calibration_roc_figure.png")


if __name__ == "__main__":
    main()
