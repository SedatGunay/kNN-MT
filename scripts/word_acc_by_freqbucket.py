import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import extract_freq_bucket_scores

#COMMONVOICE
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/comparemt_output_freq/index.html"
df_common = extract_freq_bucket_scores(html_path)
print("WORD ACCURACY PER FREQUENTIEBUCKET – COMMONVOICE")
print(df_common)

# LIBRISPEECH
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/comparemt_output_freq/index.html"
df_libri = extract_freq_bucket_scores(html_path)
print(" WORD ACCURACY PER FREQUENTIEBUCKET – LIBRISPEECH")
print(df_libri)

#VOXPOPULI
html_path = "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/comparemt_output_freq/index.html"
df_vox = extract_freq_bucket_scores(html_path)
print("WORD ACCURACY PER FREQUENTIEBUCKET – VOXPOPULI")
print(df_vox)