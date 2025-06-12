import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import extract_pos_tag_scores

BASE_DIR = "/Users/sedatgunay/Desktop/compare-mt-2"

def run_pos_analysis():
    domains = {
        "commonvoice": "comparemt_output_commonvoice_pos",
        "librispeech": "comparemt_output_libri_pos",
        "voxpopuli": "comparemt_output_vox_pos"
    }

    for domain, output_dir in domains.items():
        html_path = os.path.join(BASE_DIR, domain, output_dir, "index.html")
        print(f"\n POS LABEL TAGGING SCORES – {domain.upper()}")
        df = extract_pos_tag_scores(html_path)
        print(df)

if __name__ == "__main__":
    run_pos_analysis()