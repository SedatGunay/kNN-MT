import re
import pickle 
import os
import spacy
from collections import Counter

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

def generate_pos_tags(input_txt_path, output_tags_path, lang="en"):
    """
   Generates POS tags per token from a .txt file and writes them to a .tags file.
    
    Parameters:
    input_txt_path (str): Path to .txt file with 1 sentence per line.
    output_tags_path (str): Path to save .tags file.
    lang (str): 'en' or 'nl' for the language of the spaCy model.
    """
    model = "en_core_web_sm" if lang == "en" else "nl_core_news_sm"
    nlp = spacy.load(model)

    with open(input_txt_path, "r", encoding="utf-8") as f_in, \
         open(output_tags_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            doc = nlp(line.strip())
            tags = [token.pos_ for token in doc]
            f_out.write(" ".join(tags) + "\n")

    print(f"Tags saved in: {output_tags_path}")



if __name__ == "__main__":
    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_ref_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_ref_texts.tag",
    #     lang="nl"
    # )

    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_van_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_van_texts.tag",
    #     lang="nl"
    # )

    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_knn_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/common_test_knn_texts.tag",
    #     lang="nl"
    # )

    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_ref_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_ref_texts.tag",
    #     lang="en"
    # )

    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_van_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_van_texts.tag",
    #     lang="en"
    # )

    # generate_pos_tags(
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_knn_texts.txt",
    #     "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/libri_test_knn_texts.tag",
    #     lang="en"
    # )
    generate_pos_tags(
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_ref.txt",
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_ref.tag",
        lang="en"
    )

    generate_pos_tags(
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_van.txt",
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_van.tag",
        lang="en"
    )

    generate_pos_tags(
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_knn.txt",
        "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/vox_test_knn.tag",
        lang="en"
    )