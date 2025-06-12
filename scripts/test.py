# 📦 Required libraries
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def extract_pos_tag_scores(index_html_path):
    """
    Extracts POS tag F-measure scores for vanilla and kNN-MT from a compare-mt HTML report.

    Parameters:
    - index_html_path (str): Path to the compare-mt `index.html` file.

    Returns:
    - pd.DataFrame: DataFrame with POS labels, system scores, and differences.
    """
    with open(index_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Zoek de juiste tabel (meestal de derde met POS-label scores)
    tables = soup.find_all("table")
    pos_table = None
    for table in tables:
        caption = table.find("caption")
        if caption and "word fmeas by labels bucket" in caption.text.lower():
            pos_table = table
            break

    if pos_table is None:
        raise ValueError("POS-tag score table not found in the HTML.")

    labels, sys1_scores, sys2_scores = [], [], []

    def extract_float(td):
        text = td.get_text(strip=True)
        match = re.search(r"\d+\.\d+", text)
        return float(match.group()) if match else None

    for row in pos_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        label_cell = row.find("th")
        if label_cell and len(cols) >= 2:
            label = label_cell.get_text(strip=True)
            val1 = extract_float(cols[1])  # kNN (sys2)
            val2 = extract_float(cols[0])  # Vanilla (sys1)
            if val1 is not None and val2 is not None:
                labels.append(label)
                sys1_scores.append(val2)
                sys2_scores.append(val1)
    
    df_pos = pd.DataFrame({
    "Label": labels,
    "kNN-MT (sys1)": sys1_scores,
    "Vanilla (sys2)": sys2_scores
})
    df_pos["Verschil (sys1 - sys2)"] = df_pos["kNN-MT (sys1)"] - df_pos["Vanilla (sys2)"]

  
    return df_pos

# COMMONVOICE
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/comparemt_output_commonvoice_pos/index.html"
df_common = extract_pos_tag_scores(html_path)
print("POS LABEL TAGGING SCORES - COMMONVOICE")
print(df_common)

# LIBIRSPEECH
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/comparemt_output_libri_pos/index.html"
df_libri= extract_pos_tag_scores(html_path)
print("POS LABEL TAGGING SCORES - LIBRI")
print(df_libri)

# VOXPOPULI
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/comparemt_output_vox_pos/index.html"
df_vox= extract_pos_tag_scores(html_path)
print("POS LABEL TAGGING SCORES - VOX")
print(df_vox)