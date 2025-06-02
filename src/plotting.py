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
