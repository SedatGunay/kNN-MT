import re
import pickle 
import os
import spacy
from collections import Counter

def load_file(path):
    """
    Load a text file and return a list of non-empty stripped lines.

    Parameters:
        path (str): Path to the text file.

    Returns:
        List[str]: Lines in the file with leading/trailing whitespace removed.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
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
    
def write_texts_to_txt(texts, path):
    """
    Writes a list of sentences to a .txt file, one sentence per line.
        
    Parameters:
    texts (List[str]): The sentences to write.
    path(str): Target path for the .txt file.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts) + "\n")

def convert_pkl_to_txt(pkl_path, txt_path):
    data = load_pickle(pkl_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line.strip() + "\n")

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

def tag_sentences_with_spacy(sentences, lang="en"):
    """
    Tokenize and POS-tag a list of sentences using spaCy.

    Parameters:
    - sentences: list of strings
    - lang: 'en' for English (default), 'nl' for Dutch

    Returns:
    - list of lists with (token, POS) tuples
    """
    model = "en_core_web_sm" if lang == "en" else "nl_core_news_sm"
    nlp = spacy.load(model)
    return [[(token.text, token.pos_) for token in doc] for doc in nlp.pipe(sentences)]

def clean_token(token):
    """Removes punctuation from beginning and end of a word
    - used to make counts files for compare-mt analysis
    """
    return re.sub(r"^\W+|\W+$", "", token)

def create_counts_from_txt(txt_path, counts_path):
    """
    Creates a .counts file from a .txt file by counting normalized word frequencies.
    
    Parameters:
    txt_path (str): Path to the input .txt file.
    counts_path (str): Path to the output .counts file.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        words = f.read().split()

    cleaned_words = []
    for word in words:
        token = clean_token(word).lower()
        if token:
            cleaned_words.append(token)

    counts = Counter(cleaned_words)

    with open(counts_path, "w", encoding="utf-8") as f:
        for word, count in counts.items():
            f.write(f"{word}\t{count}\n")

def pos_tag_aligned(input_txt_path, output_tag_path, lang="nl"):
    """  
    Performs POS tagging on a file where each line is already tokenised
     (one word per space). Stores POS tags per line in a .tag file.
    Skips and reports lines with mismatches.

    Parameters:
    - input_txt_path (str): Path to .txt file containing sentences (split into tokens).
    - output_tag_path (str): Path to output file (.tag) in which POS tags are written.
    - lang (str): “nl” for Dutch, “en” for English (defines spaCy model).

    """
    model = "nl_core_news_sm" if lang == "nl" else "en_core_web_sm"
    nlp = spacy.load(model)

    with open(input_txt_path, "r", encoding="utf-8") as f_in, \
         open(output_tag_path, "w", encoding="utf-8") as f_out:

        for idx, line in enumerate(f_in):
            tokens = line.strip().split()
            doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
            for pipe in nlp.pipeline:
                if pipe[0] != "ner":  
                    doc = pipe[1](doc)

            if len(doc) != len(tokens):
                print(f"Mismatch in line {idx+1}: {len(tokens)} tokens vs {len(doc)} tags")
                print("TXT:", " ".join(tokens))
                print("TAG:", " ".join([t.pos_ for t in doc]))
                continue  #skip line

            f_out.write(" ".join([t.pos_ for t in doc]) + "\n")

    print(f"Tags saved in: {output_tag_path}")

def check_token_tag_alignment(txt_path, tag_path):
    """
    Checks whether each line in a text file has the same number of tokens
    as the corresponding line in a tag file. Reports the first mismatch.

    Parameters:
    - txt_path (str): Path to file containing sentences (one sentence per line).
    - tag_path (str): Path to file containing POS tags (one line of tags per sentence).
    """
    with open(txt_path, "r", encoding="utf-8") as f_txt, open(tag_path, "r", encoding="utf-8") as f_tag:
        for i, (txt_line, tag_line) in enumerate(zip(f_txt, f_tag)):
            txt_tokens = txt_line.strip().split()
            tag_tokens = tag_line.strip().split()
            if len(txt_tokens) != len(tag_tokens):
                print(f" Mismatch in line {i+1}: {len(txt_tokens)} tokens vs {len(tag_tokens)} tags")
                print("TXT:", txt_line.strip())
                print("TAG:", tag_line.strip())
                return
    print("All lines have matching token–tag length.")

def find_empty_or_whitespace_lines(*filepaths):
    """
     Checks one or more text files for empty lines or lines consisting only of spaces.
    Prints a warning for each file with the line number where such a line occurs.

    Parameters:
    - *filepaths (str): One or more path names to .txt files.
    """
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    print(f"empty line in {path} at line {idx}")