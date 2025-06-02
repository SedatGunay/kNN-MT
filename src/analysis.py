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