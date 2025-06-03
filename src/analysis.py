from collections import Counter
from jiwer import compute_measures

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