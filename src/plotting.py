import matplotlib.pyplot as plt

def plot_pos_counts(pos_counts, title="POS Distribution"):
    """
    Plot a bar chart of part-of-speech (POS) tag frequencies.

    Parameters:
    - pos_counts (dict): A dictionary mapping POS tags to their frequency counts.
    - title (str): The title of the plot (default is "POS Distribution").
    """
    labels, values = zip(*pos_counts.items())
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_wer_distribution(wer_scores, domain):
    """PLot a his"""
    plt.figure(figsize=(10, 5))
    plt.hist(wer_scores, bins=30, color="steelblue", edgecolor="black")
    plt.title(f"WER-Distributie – {domain}")
    plt.xlabel("WER")
    plt.ylabel("Aantal zinnen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_knn_gain_scatter(wer_knn, wer_van, outlier_indices, domain=""):
    """
    Visualises a scatterplot of WER scores per sentence for kNN and vanilla,
    where sentences with a significant gain by kNN are highlighted.

    Parameters:
    - wer_knn (List[float]): WER scores per sentence of the kNN system.
    - wer_of (List[float]): WER scores per sentence of the vanilla system.
    - gain_indices (List[int]): Indexes of sentences where kNN clearly outperforms.
    - domain(str): Name of the domain (used in the plot title).
    """
    plt.figure(figsize=(8, 6))

    # All points 
    plt.scatter(wer_van, wer_knn, alpha=0.3, label="Overige zinnen", color="skyblue")

    # Outliers
    x_out = [wer_van[i] for i in outlier_indices]
    y_out = [wer_knn[i] for i in outlier_indices]
    plt.scatter(x_out, y_out, alpha=0.8, color="crimson", label="Top winst kNN")

    # Diagonal
    plt.plot([0, 1.5], [0, 1.5], linestyle='--', color='red', label='y = x')

    # Labels
    plt.xlabel("WER Vanilla")
    plt.ylabel("WER kNN")
    plt.title(f"WER per sentence: kNN vs Vanilla ({domain})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()