import pickle
from collections import Counter
import spacy
import matplotlib.pyplot as plt

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize(text):
    """Lowercase en verwijder interpunctie."""
    return text.strip().lower().replace(".", "").replace(",", "")


def classify_sentences(refs, knn, van):
    """Sorteert zinnen op basis van of kNN of vanilla correct is."""
    better_knn = []
    better_van = []
    for i, (r, v, k) in enumerate(zip(refs, van, knn)):
        if normalize(k) == normalize(r) and normalize(v) != normalize(r):
            better_knn.append((i, r, v, k))
        elif normalize(v) == normalize(r) and normalize(k) != normalize(r):
            better_van.append((i, r, v, k))
    return better_knn, better_van


def extract_pos(sentences, lang="en"):
    """Tel POS-tags in hypotheses, met keuze tussen NL/EN SpaCy model."""
    nlp = spacy.load("nl_core_news_sm" if lang == "nl" else "en_core_web_sm")
    pos_all = []
    for _, _, _, hyp in sentences:
        doc = nlp(hyp)
        pos_all.extend([tok.pos_ for tok in doc if tok.is_alpha])
    return Counter(pos_all)

def extract_entities(sentences, lang="en"):
    """Tel entiteiten in hypotheses, met keuze tussen NL/EN SpaCy model."""
    nlp = spacy.load("nl_core_news_sm" if lang == "nl" else "en_core_web_sm")
    entity_counts = Counter()
    for _, _, _, hyp in sentences:
        doc = nlp(hyp)
        entity_counts.update([ent.label_ for ent in doc.ents])
    return entity_counts




def plot_distribution(labels, knn_freqs, van_freqs, title, ylabel):
    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(12, 6))
    

    plt.bar([i - width/2 for i in x], knn_freqs, width=width, label="kNN better", color="#1f77b4")
    plt.bar([i + width/2 for i in x], van_freqs, width=width, label="Vanilla better", color="#ff7f0e")

    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()