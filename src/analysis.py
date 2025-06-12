from collections import Counter
# from jiwer import compute_measures
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def compute_pos_distribution(tokenized_sentences):
    """Compute frequency distribution of POS tags."""
    return Counter(tag for sentence in tokenized_sentences for (_, tag) in sentence)

def calculate_wer(refs, hyps):
    """
    refs: list of reference strings
    hyps: list of hypothesis strings
    returns: dict of WER components
    """
    return compute_measures(refs, hyps)

def segment_sentences_by_length(sentences, boundaries):
    """
    Segment sentences into buckets based on their length.
    boundaries: list of integers indicating segment edges
    returns: dict of segment index -> list of sentences
    """
    buckets = {i: [] for i in range(len(boundaries) + 1)}
    for s in sentences:
        length = len(s.split())
        for i, b in enumerate(boundaries):
            if length <= b:
                buckets[i].append(s)
                break
        else:
            buckets[len(boundaries)].append(s)
    return buckets

def extract_pos_tag_scores(index_html_path):
    """
    Extracts POS tag F-measure scores for vanilla and kNN-MT from a compare-mt HTML report.

    Parameters:
    - index_html_path (str): Path to the compare-mt `index.html` file.

    Returns:
    - pd.DataFrame: DataFrame with POS labels, system scores, and differences.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Zoek de juiste tabel (meestal de derde met POS-label scores)
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
            val1 = extract_float(cols[0])  # kNN (sys1)
            val2 = extract_float(cols[1])  # Vanilla (sys2)
            if val1 is not None and val2 is not None:
                labels.append(label)
                sys1_scores.append(val1)
                sys2_scores.append(val2)
    
    df_pos = pd.DataFrame({
    "Label": labels,
    "kNN-MT (sys1)": sys1_scores,
    "Vanilla (sys2)": sys2_scores
})
    df_pos["Verschil (sys1 - sys2)"] = df_pos["kNN-MT (sys1)"] - df_pos["Vanilla (sys2)"]

  
    return df_pos


def extract_freq_bucket_scores(index_html_path):
    """
    Extracts word accuracy (F1) scores by frequency bucket for vanilla and kNN-MT systems.

    Parameters:
    - index_html_path (str): Path to the compare-mt `index.html` file.

    Returns:
    - pd.DataFrame: DataFrame with frequency buckets, system scores, and differences.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Vind de juiste tabel via caption
    freq_table = None
    for table in soup.find_all("table"):
        caption = table.find("caption")
        if caption and "word fmeas by frequency bucket" in caption.text.lower():
            freq_table = table
            break

    if freq_table is None:
        raise ValueError("Frequentie-tabel niet gevonden in het HTML-bestand.")

    # Extractie helper
    def extract_float(td):
        match = re.search(r"\d+\.\d+", td.get_text(strip=True))
        return float(match.group()) if match else None

    # Gegevens verzamelen
    buckets, sys1_scores, sys2_scores = [], [], []

    for row in freq_table.find_all("tr")[1:]:  # Skip header
        cols = row.find_all("td")
        bucket = row.find("th").get_text(strip=True) if row.find("th") else None
        if bucket and len(cols) >= 2:
            val1 = extract_float(cols[0])  # kNN-MT
            val2 = extract_float(cols[1])  # Vanilla
            if val1 is not None and val2 is not None:
                buckets.append(bucket)
                sys1_scores.append(val1)
                sys2_scores.append(val2)

    # DataFrame bouwen
    df_freq = pd.DataFrame({
        "Frequentie Bucket": buckets,
        "kNN-MT (sys1)": sys1_scores,
        "Vanilla (sys2)": sys2_scores
    })
    df_freq["Verschil (sys1 - sys2)"] = df_freq["kNN-MT (sys1)"] - df_freq["Vanilla (sys2)"]

    df_freq.set_index("Frequentie Bucket")[["kNN-MT (sys1)", "Vanilla (sys2)"]].plot(kind="bar", figsize=(10,5))
    plt.title("Word Accuracy per Frequentie Bucket")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_freq