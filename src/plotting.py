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