import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# Update met correcte path naar geüpload bestand
ref_tag_path = "/Users/sedatgunay/Desktop/MT-DATA/data/it/test.en.tag"
hyp_tag_path = "/Users/sedatgunay/Desktop/MT-DATA/outputs/eval_model_it_knn_8k_10T_7e-1lambda_beam5/hyp.eval_ready.tag"

# Laad de bestanden
with open(ref_tag_path, "r", encoding="utf-8") as f:
    ref_lines = [line.strip().split() for line in f]

with open(hyp_tag_path, "r", encoding="utf-8") as f:
    hyp_lines = [line.strip().split() for line in f]

# Filter zinnen met ongelijke lengte (kunnen niet geëvalueerd worden)
filtered = [(r, h) for r, h in zip(ref_lines, hyp_lines) if len(r) == len(h)]
ref_filtered = [r for r, h in filtered]
hyp_filtered = [h for r, h in filtered]

# Flatten alle POS-tags tot 1 lijst
ref_flat = [tag for line in ref_filtered for tag in line]
hyp_flat = [tag for line in hyp_filtered for tag in line]

# Haal alle unieke labels op
all_labels = sorted(set(ref_flat) | set(hyp_flat))

# Bereken precision, recall, F1 per label
precision, recall, f1, support = precision_recall_fscore_support(
    ref_flat, hyp_flat, labels=all_labels, zero_division=0
)

# Zet in DataFrame
df = pd.DataFrame({
    "POS": all_labels,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Support": support
}).sort_values(by="F1", ascending=False)

# Toon in tabel
print(df)

# Plot F1 per POS
plt.figure(figsize=(10, 5))
plt.bar(df["POS"], df["F1"])
plt.title("F1-score per POS-tag (MT data)")
plt.ylabel("F1-score")
plt.xlabel("POS-tag")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()