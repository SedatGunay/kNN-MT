import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.better_by import (
    load_pickle, classify_sentences,
    extract_pos, extract_entities, plot_distribution
)

# # COMMONVOICE
# base_path = "/Users/sedatgunay/Desktop/ASR_KNN/dataset data (+outputs)/commonvoice"
# lang = "nl"  

# # Load Data
# ref = load_pickle(os.path.join(base_path, "test_ref_texts.pkl"))
# knn = load_pickle(os.path.join(base_path, "test_knn_texts.pkl"))
# van = load_pickle(os.path.join(base_path, "test_van_texts.pkl"))

# # Analysis
# better_knn, better_van = classify_sentences(ref, knn, van)

# # POS analyse
# pos_knn = extract_pos(better_knn, lang=lang)
# pos_van = extract_pos(better_van, lang=lang)

# all_pos = sorted(set(pos_knn.keys()).union(pos_van.keys()))
# plot_distribution(
#     all_pos,
#     [pos_knn.get(p, 0) for p in all_pos],
#     [pos_van.get(p, 0) for p in all_pos],
#     f"POS-distrubution bettter sentences - COMMONVOICE ({lang.upper()})",
#     "Numer of tokens"
# )

# # Entity analyse
# ent_knn = extract_entities(better_knn, lang=lang)
# ent_van = extract_entities(better_van, lang=lang)

# all_ents = sorted(set(ent_knn.keys()).union(ent_van.keys()))
# plot_distribution(
#     all_ents,
#     [ent_knn.get(e, 0) for e in all_ents],
#     [ent_van.get(e, 0) for e in all_ents],
#     f"Entity-distribution better sentences - COMMONVOICE ({lang.upper()})",
#     "Number of  entities"
# )

# # LIBRISPEECH
# base_path = "/Users/sedatgunay/Desktop/ASR_KNN/dataset data (+outputs)/librispeech"
# lang = "en"  

# # Load Data
# ref = load_pickle(os.path.join(base_path, "test_ref_clean_texts.pkl"))
# knn = load_pickle(os.path.join(base_path, "test_knn_clean_texts.pkl"))
# van = load_pickle(os.path.join(base_path, "test_van_clean_texts.pkl"))

# # Analysis
# better_knn, better_van = classify_sentences(ref, knn, van)

# # POS analyse
# pos_knn = extract_pos(better_knn, lang=lang)
# pos_van = extract_pos(better_van, lang=lang)

# all_pos = sorted(set(pos_knn.keys()).union(pos_van.keys()))
# plot_distribution(
#     all_pos,
#     [pos_knn.get(p, 0) for p in all_pos],
#     [pos_van.get(p, 0) for p in all_pos],
#     f"POS-distrubution bettter sentences - LIBRISPEECH ({lang.upper()})",
#     "Aantal tokens"
# )

# # Entity analyse
# ent_knn = extract_entities(better_knn, lang=lang)
# ent_van = extract_entities(better_van, lang=lang)

# all_ents = sorted(set(ent_knn.keys()).union(ent_van.keys()))
# plot_distribution(
#     all_ents,
#     [ent_knn.get(e, 0) for e in all_ents],
#     [ent_van.get(e, 0) for e in all_ents],
#     f"Entity-distribution better sentences - LIBRISPEECH ({lang.upper()})",
#     "Number of  entities"
# )

# VOXPOPULI
base_path = "/Users/sedatgunay/Desktop/ASR_KNN/dataset data (+outputs)/voxpopuli"
lang = "en"  

# Load Data
ref = load_pickle(os.path.join(base_path, "test_ref_texts.pkl"))
knn = load_pickle(os.path.join(base_path, "test_knn_texts.pkl"))
van = load_pickle(os.path.join(base_path, "test_van_texts.pkl"))

# Analysis
better_knn, better_van = classify_sentences(ref, knn, van)

# POS analyse
pos_knn = extract_pos(better_knn, lang=lang)
pos_van = extract_pos(better_van, lang=lang)

all_pos = sorted(set(pos_knn.keys()).union(pos_van.keys()))
plot_distribution(
    all_pos,
    [pos_knn.get(p, 0) for p in all_pos],
    [pos_van.get(p, 0) for p in all_pos],
    f"POS-distrubution bettter sentences - VOXPOPULI ({lang.upper()})",
    "Aantal tokens"
)

# Entity analyse
ent_knn = extract_entities(better_knn, lang=lang)
ent_van = extract_entities(better_van, lang=lang)

all_ents = sorted(set(ent_knn.keys()).union(ent_van.keys()))
plot_distribution(
    all_ents,
    [ent_knn.get(e, 0) for e in all_ents],
    [ent_van.get(e, 0) for e in all_ents],
    f"Entity-distribution better sentences - VOXPOPULI ({lang.upper()})",
    "Number of  entities"
)