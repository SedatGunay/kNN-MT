import re
import pickle 
import os

def normalize_text(text):
    """Lowercase and strip whitespace."""
    return text.lower().strip()

def filter_tokens_by_pos(tokens, pos_labels):
    """
    Filter tokens by specific POS labels.
    tokens: list of (word, pos) tuples
    pos_labels: set or list of POS tags to keep
    """
    return [tok for tok in tokens if tok[1] in pos_labels]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dataset_structure(dataset_path, splits, modes, types):
    """
    Load structured dataset contents from a given path.

    Parameters:
    - dataset_path (str): Path to the dataset folder
    - splits (list): Dataset splits, e.g., ['dev', 'test']
    - modes (list): Model output modes, e.g., ['ref', 'van', 'knn']
    - types (list): File types, e.g., ['texts', 'normalized_texts', 'tokens', 'tokens_texts']

    Returns:
    - dict: Nested dictionary with structure data[split][mode][type]
    """
    data = {}
    for split in splits:
        data[split] = {}
        for mode in modes:
            mode_data = {}
            for dtype in types:
                filename = f"{split}_{mode}_{dtype}.pkl"
                file_path = os.path.join(dataset_path, filename)
                if os.path.exists(file_path):
                    mode_data[dtype] = load_pickle(file_path)
            data[split][mode] = mode_data
        metadata_file = os.path.join(dataset_path, f"{split}_data.pkl")
        if os.path.exists(metadata_file):
            data[split]['meta'] = load_pickle(metadata_file)
    return data
