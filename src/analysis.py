from collections import Counter
from jiwer import compute_measures
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sacrebleu import sentence_bleu
import numpy as np

# POS ANALYSIS

def compute_pos_distribution(tokenized_sentences):
    """Compute frequency distribution of POS tags from tokenized input."""
    return Counter(tag for sentence in tokenized_sentences for (_, tag) in sentence)

def extract_pos_tag_scores(index_html_path):
    """
    Extract POS tag F-measure scores for both kNN-MT and vanilla systems from a compare-mt HTML report.

    Parameters:
    - index_html_path (str): Path to compare-mt index.html file.

    Returns:
    - pd.DataFrame: POS label scores and differences.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    tables = soup.find_all("table")
    pos_table = None
    for table in tables:
        caption = table.find("caption")
        if caption and "word fmeas by labels bucket" in caption.text.lower():
            pos_table = table
            break

    if pos_table is None:
        raise ValueError("POS-tag score table not found in the HTML.")

    labels, sys1_scores, sys2_scores = [], [], []

    def extract_float(td):
        text = td.get_text(strip=True)
        match = re.search(r"\d+\.\d+", text)
        return float(match.group()) if match else None

    for row in pos_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        label_cell = row.find("th")
        if label_cell and len(cols) >= 2:
            label = label_cell.get_text(strip=True)
            val1 = extract_float(cols[0])
            val2 = extract_float(cols[1])
            if val1 is not None and val2 is not None:
                labels.append(label)
                sys1_scores.append(val1)
                sys2_scores.append(val2)

    df_pos = pd.DataFrame({
        "Label": labels,
        "kNN-MT (sys1)": sys1_scores,
        "Vanilla (sys2)": sys2_scores
    })
    df_pos["Difference (sys1 - sys2)"] = df_pos["kNN-MT (sys1)"] - df_pos["Vanilla (sys2)"]
    return df_pos

# FREQUENCY BUCKET ANALYSIS

def extract_freq_bucket_scores(index_html_path):
    """
    Extract frequency-bucket F1 scores from a compare-mt HTML report.

    Parameters:
    - index_html_path (str): Path to compare-mt index.html file.

    Returns:
    - pd.DataFrame: Frequency bucket scores and differences.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    freq_table = None
    for table in soup.find_all("table"):
        caption = table.find("caption")
        if caption and "word fmeas by frequency bucket" in caption.text.lower():
            freq_table = table
            break

    if freq_table is None:
        raise ValueError("Frequency table not found in HTML.")

    def extract_float(td):
        match = re.search(r"\d+\.\d+", td.get_text(strip=True))
        return float(match.group()) if match else None

    buckets, sys1_scores, sys2_scores = [], [], []
    for row in freq_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        bucket = row.find("th").get_text(strip=True) if row.find("th") else None
        if bucket and len(cols) >= 2:
            val1 = extract_float(cols[0])
            val2 = extract_float(cols[1])
            if val1 is not None and val2 is not None:
                buckets.append(bucket)
                sys1_scores.append(val1)
                sys2_scores.append(val2)

    df_freq = pd.DataFrame({
        "Frequency Bucket": buckets,
        "kNN-MT (sys1)": sys1_scores,
        "Vanilla (sys2)": sys2_scores
    })
    df_freq["Difference (sys1 - sys2)"] = df_freq["kNN-MT (sys1)"] - df_freq["Vanilla (sys2)"]

    df_freq.set_index("Frequency Bucket")[["kNN-MT (sys1)", "Vanilla (sys2)"]].plot(kind="bar", figsize=(10, 5))
    plt.title("Word Accuracy by Frequency Bucket")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_freq

# WER ANALYSIS

def calculate_wer_per_sentence(refs, hyps):
    """
    Compute WER with jiwer.
    Returns a list with dicts per sentence.
    """
    return [compute_measures([ref], [hyp]) for ref, hyp in zip(refs, hyps)]

def get_knn_gain_outliers(refs, hyps_knn, hyps_van, threshold=0.5):
    """
    Returns a list of tuples for sentences where kNN outperforms vanilla by more than a threshold.

    Parameters:
    - refs: list of reference sentences
    - hyps_knn: list of kNN outputs
    - hyps_van: list of vanilla outputs
    - threshold: minimum WER improvement (WER_vanilla - WER_knn)

    Returns:
    - List of tuples (index, ref, vanilla, knn, wer_van, wer_knn, gain)
    """
    outliers = []
    for i, (r, h_knn, h_van) in enumerate(zip(refs, hyps_knn, hyps_van)):
        wer_v = compute_measures([r], [h_van])["wer"]
        wer_k = compute_measures([r], [h_knn])["wer"]
        gain = wer_v - wer_k
        if gain > threshold:
            outliers.append((i, r, h_van, h_knn, wer_v, wer_k, gain))
    return outliers

def wer_summary(refs, hyps):
    """Return overall WER scores using jiwer."""
    return compute_measures(refs, hyps)

# BLEU BUCKET ANALYSIS

def analyze_bleu_buckets_single_system(score_file, bucket_size=10, system_name="kNN-MT"):
    """
    Analyze the distribution of BLEU scores across buckets for a single system.

    Parameters:
    - score_file (str): Path to .res_bleu file (one BLEU score per line).
    - bucket_size (int): Size of the BLEU buckets (default: 10).
    - system_name (str): Label used in the plot title (default: 'kNN-MT').

    Returns:
    - dict: A mapping from bucket label to the number of sentences in that bucket.
    """
    with open(score_file, "r", encoding="utf-8") as f:
        scores = [float(line.strip()) for line in f if line.strip()]

    max_score = 100
    bins = list(range(0, max_score + bucket_size, bucket_size))
    labels = [f"{i}-{i + bucket_size}" for i in bins[:-1]]
    counts, _ = np.histogram(scores, bins=bins)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, width=0.8)
    plt.title(f"BLEU Score Distribution per Bucket ({system_name})")
    plt.xlabel("BLEU Bucket")
    plt.ylabel("Number of Sentences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return dict(zip(labels, counts))