import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import load_file, compare_bleu_buckets, bleu_per_sentence, get_bleu_buckets

if __name__ == "__main__":

    print("BLUE BUCKET ANALYSE - COMMONVOICE")
    base = "/Users/sedatgunay/Desktop/compare-mt-2/commonvoice/"
    refs = load_file(os.path.join(base, "common_test_ref_texts.txt"))
    knn = load_file(os.path.join(base, "common_test_knn_texts.txt"))
    van = load_file(os.path.join(base, "common_test_van_texts.txt"))

    bucket_edges = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    df = compare_bleu_buckets(refs, knn, van, bucket_edges)
    print(df)

    print("BLUE BUCKET ANALYSE - LIBRISPEECH")
    base = "/Users/sedatgunay/Desktop/compare-mt-2/librispeech/"
    refs = load_file(os.path.join(base, "libri_test_ref_texts.txt"))
    knn = load_file(os.path.join(base, "libri_test_knn_texts.txt"))
    van = load_file(os.path.join(base, "libri_test_van_texts.txt"))

    bucket_edges = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    df = compare_bleu_buckets(refs, knn, van, bucket_edges)

    print(df)

    print("BLUE BUCKET ANALYSE - VOXPOPULI")
    base = "/Users/sedatgunay/Desktop/compare-mt-2/voxpopuli/"
    refs = load_file(os.path.join(base, "vox_test_ref.txt"))
    knn = load_file(os.path.join(base, "vox_test_knn.txt"))
    van = load_file(os.path.join(base, "vox_test_van.txt"))

    bucket_edges = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    df = compare_bleu_buckets(refs, knn, van, bucket_edges)
    print(df)
