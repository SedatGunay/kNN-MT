from collections import Counter
# from jiwer import compute_measures
import pandas as pd
import re
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sacrebleu import sentence_bleu

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

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def bleu_per_sentence(refs, hyps):
    """
    Calculates sentence-level BLEU scores for each hypothesis–reference pair.
    """
    return [sentence_bleu(hyp, [ref]).score for ref, hyp in zip(refs, hyps)]

def get_bleu_buckets(scores, edges):
    """
    Groups BLEU scores into buckets based on given boundaries.

    Parameters:
    - scores (List[float]): List of BLEU scores.
    - edges (List[float]): Bucket boundaries
    """
    buckets = {}
    edges = [0.0] + edges + [100.0]
    for low, high in zip(edges[:-1], edges[1:]):
        label = f"[{low},{high})" if high < 100 else f">={low}"
        buckets[label] = []

    for score in scores:
        for low, high in zip(edges[:-1], edges[1:]):
            if low <= score < high:
                label = f"[{low},{high})" if high < 100 else f">={low}"
                buckets[label].append(score)
                break

    return buckets

def compare_bleu_buckets(refs, sys1, sys2, bucket_edges):
    """
    Compares the distribution of BLEU scores across buckets for two systems.

    Parameters:
    - refs (List[str]): Reference sentences.
    - sys1 (List[str]): Hypothesis sentences from system 1 (kNN-MT).
    - sys2 (List[str]): Hypothesis sentences from system 2 (vanilla MT).
    - bucket_edges (List[float]): BLEU score bucket edges.

    Returns:
    - pd.DataFrame: DataFrame with bucket label, counts per system, and difference.
    """
    scores1 = bleu_per_sentence(refs, sys1)
    scores2 = bleu_per_sentence(refs, sys2)

    buckets1 = get_bleu_buckets(scores1, bucket_edges)
    buckets2 = get_bleu_buckets(scores2, bucket_edges)

    rows = []
    for label in buckets1:
        count1 = len(buckets1[label])
        count2 = len(buckets2[label])
        rows.append((label, count1, count2, count1 - count2))

    df = pd.DataFrame(rows, columns=["BLEU-bucket", "Aantal sys1", "Aantal sys2", "Verschil (sys1 - sys2)"])
    return df