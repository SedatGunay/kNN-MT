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

def plot_knn_gain_scatter(
    wer_knn, wer_van, gain_indices,
    title="WER per zin: kNN vs Vanilla",
    domain_label="",
    log_scale=False
):
    """
    Plot een scatterplot van WER-scores van vanilla en kNN per zin,
    met gemarkeerde zinnen waar kNN sterk beter presteert.

    Parameters:
    - wer_knn: lijst van dicts met WER-resultaten voor kNN (output van calculate_wer_per_sentence)
    - wer_van: lijst van dicts met WER-resultaten voor vanilla
    - gain_indices: lijst van indices van zinnen waar kNN sterk beter is
    - title: string, plot titel
    - domain_label: string, optioneel label per domein
    - log_scale: boolean, zet log-log schaal aan als True
    """
    # WER-scores extraheren
    x = [d["wer"] for d in wer_van]
    y = [d["wer"] for d in wer_knn]

    # Index-gebaseerde outliers
    outlier_x = [x[i] for i in gain_indices]
    outlier_y = [y[i] for i in gain_indices]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.3, label="Overige zinnen", color="skyblue")
    plt.scatter(outlier_x, outlier_y, alpha=0.8, color="crimson", label="Top winst kNN")

    # Log-schaal (optioneel)
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # Referentielijn y = x
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

    # Labels en stijl
    plt.xlabel("WER Vanilla")
    plt.ylabel("WER kNN")
    plt.title(f"{title} ({domain_label})")
    plt.legend()
    plt.grid(True, which="both" if log_scale else "major", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()