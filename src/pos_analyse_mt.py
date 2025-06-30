import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

def evaluate_pos(ref_tag_path, hyp_tag_path, domain_name=None):
    """
    Compute and display precision/recall/F1 per POS-tag.
    Results are printed and plotted directly in the notebook.

    Parameters:
    - ref_tag_path: path to reference .tag file (one line per sentence)
    - hyp_tag_path: path to hypothesis .tag file (one line per sentence)
    - domain_name: optional string, used in plot title
    """

    # Laad regels
    with open(ref_tag_path, "r", encoding="utf-8") as f:
        ref_lines = [line.strip().split() for line in f]
    with open(hyp_tag_path, "r", encoding="utf-8") as f:
        hyp_lines = [line.strip().split() for line in f]

    # Filter regels met ongelijke lengte
    filtered = [(r, h) for r, h in zip(ref_lines, hyp_lines) if len(r) == len(h)]
    ref_flat = [tag for r, _ in filtered for tag in r]
    hyp_flat = [tag for _, h in filtered for tag in h]

    # Bereken metrics
    all_labels = sorted(set(ref_flat) | set(hyp_flat))
    precision, recall, f1, support = precision_recall_fscore_support(
        ref_flat, hyp_flat, labels=all_labels, zero_division=0
    )

    # Zet in DataFrame
    df = pd.DataFrame({
        "POS": all_labels,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    }).sort_values(by="F1", ascending=False)

    # Plot F1-score
    plt.figure(figsize=(10, 5))
    plt.bar(df["POS"], df["F1"])
    title = f"F1-score per POS-tag ({domain_name})" if domain_name else "F1-score per POS-tag"
    plt.title(title)
    plt.ylabel("F1-score")
    plt.xlabel("POS-tag")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df