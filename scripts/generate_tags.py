# scripts/generate_tags.py
import os
import spacy

def pos_tag_aligned(input_txt_path, output_tag_path, lang="nl"):
    """
    POS-tag a file where each line is tokenized.
    Writes POS tags per line to output_tag_path.
    Skips lines where the number of tags doesn’t match number of tokens.
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
                print(f"❌ Mismatch in line {idx+1}: {len(tokens)} tokens vs {len(doc)} tags")
                continue

            f_out.write(" ".join([t.pos_ for t in doc]) + "\n")

    print(f"✅ Tags saved in: {output_tag_path}")


DOMAIN = "librispeech"  # Chose from: "commonvoice", "librispeech", "voxpopuli"
LANG = "en" if DOMAIN in ["librispeech", "voxpopuli"] else "nl"
BASE_DIR = f"/Users/sedatgunay/Desktop/compare-mt-2/{DOMAIN}"

FILES = [
    (f"{DOMAIN}_ref_texts.txt", f"{DOMAIN}_ref_texts.tag"),
    (f"{DOMAIN}_van_texts.txt", f"{DOMAIN}_van_texts.tag"),
    (f"{DOMAIN}_knn_texts.txt", f"{DOMAIN}_knn_texts.tag")
]

for txt_file, tag_file in FILES:
    pos_tag_aligned(os.path.join(BASE_DIR, txt_file), os.path.join(BASE_DIR, tag_file), lang=LANG)