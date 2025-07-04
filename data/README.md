
# kNN-MT: Retrieval-Augmented Decoding for ASR and MT

This repository contains all data processing, evaluation, and analysis code used in the bachelor thesis _"Evaluating Linguistic Effects of kNN-Augmented Decoding for Automatic Speech Recognition and Machine Translation."_ The study investigates how external memory, in the form of $k$-nearest neighbor (kNN) retrieval, affects the linguistic quality of generated output across both **ASR** (Whisper) and **MT** (Transformer) systems.

## Research Context

Neural models such as Whisper (ASR) and fairseq Transformer (MT) rely solely on parametric memory. By augmenting these models with **external non-parametric memory**, via **kNN-MT** decoding, we aim to analyze:
- Whether $k$NN improves linguistic accuracy (POS, Named Entities)
- Which types of tokens or sentences benefit most from retrieval
- Trade-offs in datastore composition (e.g., numeric-only experiments)
- Domain-dependent effects across multiple test sets

The evaluation focuses on **output-level analysis** using both automatic metrics (WER, BLEU) and linguistic diagnostics (POS/NER buckets, frequency buckets, error distributions).

---

## Repository Structure

```bash
.
├── data/                       # Raw and processed test sets
│   ├── commonvoice/
│   ├── librispeech/
│   └── voxpopuli/
│
├── scripts/
│   ├── generate_tags.py       # POS and NER tagging
│   ├── compute_wer.py         # WER score computation
│   ├── comparemt_parser.py    # HTML table extractors (POS/Freq)
│   └── analyze_distribution.py# POS/NER subset distribution analysis
│
├── results/
│   ├── wer_scores_*.txt       # Sentence-level WER outputs
│   └── plots/                 # Visualizations per analysis
│
├── figures/                   # Paper-ready plots used in the thesis
├── notebooks/                 # Jupyter notebooks for per-domain analysis
└── README.md
```

---

## 📊 Datasets

###  ASR (Automatic Speech Recognition)

The ASR experiments use $k$NN-augmented outputs for three domains:

- **CommonVoice** (Dutch, user-contributed, noisy)
- **LibriSpeech** (English, audiobook, clean)
- **VoxPopuli** (English, political speech)

The base model is [Whisper](https://openai.com/research/whisper), and datastore files were constructed using `FAISS`. Sentence-level outputs (reference, vanilla, kNN) were provided in `.txt` format.

###  MT (Machine Translation)

Machine translation experiments use German–English parallel corpora across:

- **IT**
- **Law**
- **Medical**

The base model is `wmt19.de-en` Transformer from `fairseq`. $k$NN-MT decoding was applied using a fixed datastore constructed from the training corpus.

---

## ⚙️ Methodology Summary

###  Preprocessing
- Sentence-level reference and hypothesis files loaded and optionally normalized.
- POS and NER annotations via `spaCy` (`en_core_web_sm`, `nl_core_news_sm`)
- Files aligned to ensure 1-to-1 token-tag mapping.
- Normalized WER scores computed with `jiwer`.

###  Evaluation Methods
- **WER** (ASR) via `jiwer`, saved per sentence.
- **BLEU** (MT) via `sacrebleu` sentence scoring.
- **POS-bucket F1 scores** via `compare-mt` and custom HTML parsers.
- **Frequency buckets**: word-level frequency vs accuracy comparison.
- **POS/NER distributions**: for WER-gain subsets where $k$NN improved > 0.5.
- **Outlier analysis** for both MT and ASR to study extremes.

###  Special experiments
- **Numeric-only FAISS retrieval**: filtered datastore using only `NUM`-tagged tokens.
- Impact of memory content was tested by re-running $k$NN decoding using this constrained index.

## How to Run

Most evaluations are organized in Jupyter notebooks under `/notebooks/`. Each notebook includes:

1. Loading of references, vanilla and kNN outputs
2. Normalization (optional)
3. Sentence-level WER/BLEU computation
4. Bucket analysis or POS/NER analysis
5. Plotting or tabular output for LaTeX

You can also run scripts directly for automated processing:

```bash
python scripts/compute_wer.py --ref ref.txt --hyp knn.txt --out wer_knn.txt
python scripts/generate_tags.py --input hyp.txt --output hyp.tags --lang en
```

---

##  Citation

This code and analysis was conducted as part of a 2025 BSc thesis in Information Studies at the University of Amsterdam. If you use or adapt this code, please cite or acknowledge:

```
Gunay, S. (2025). Understanding and Optimizing Retrieval-Augmented Decoding with kNN-MT. University of Amsterdam.
```

---
