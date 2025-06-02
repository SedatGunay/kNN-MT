import pickle
import os

def load_pickle_strip(path):
    with open(path, "rb") as f:
        return [s.strip() for s in pickle.load(f)]

def write_txt(lines, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# File configuration by domain
DOMAIN_CONFIG = {
    "commonvoice": {
        "ref": "test_ref_texts.pkl",
        "van": "test_van_texts.pkl",
        "knn": "test_knn_texts.pkl"
    },
    "librispeech": {
        "ref": "test_ref_clean_normalized_texts.pkl",
        "van": "test_van_clean_normalized_texts.pkl",
        "knn": "test_knn_clean_normalized_texts.pkl"
    },
    "voxpopuli": {
        "ref": "test_ref_normalized_texts.pkl",
        "van": "test_van_normalized_texts.pkl",
        "knn": "test_knn_normalized_texts.pkl"
    }
}

# ✏️ Chose your domain 
DOMAIN = "librispeech"  # or: "commonvoice", "voxpopuli"
config = DOMAIN_CONFIG[DOMAIN]
base_path = f"/content/drive/MyDrive/ASR_KNN/dataset data (+outputs)/{DOMAIN}/"

ref = load_pickle_strip(os.path.join(base_path, config["ref"]))
van = load_pickle_strip(os.path.join(base_path, config["van"]))
knn = load_pickle_strip(os.path.join(base_path, config["knn"]))

write_txt(ref, f"{DOMAIN}_ref_texts.txt")
write_txt(van, f"{DOMAIN}_van_texts.txt")
write_txt(knn, f"{DOMAIN}_knn_texts.txt")