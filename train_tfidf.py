"""Người B: learning curve cho TF-IDF + Logistic Regression (chạy trên CPU, vài phút).
  python train_tfidf.py --llm_file data/llm_final.csv
  python train_tfidf.py --llm_file data/llm_final.csv --llm_file2 data/llm_v1.csv --sources llm_agree mix500   # ngày 3
Kết quả ghi thêm vào results/tfidf.csv"""
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from common import load_data, make_train, test_set, score, append_result

ap = argparse.ArgumentParser()
ap.add_argument("--llm_file", default="data/llm_final.csv")
ap.add_argument("--llm_file2", default=None)
ap.add_argument("--sources", nargs="+", default=["human", "llm"])
ap.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 2000, 4000])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--out", default="results/tfidf.csv")
args = ap.parse_args()

df = load_data(args.llm_file, args.llm_file2)
X_test, y_test = test_set(df)
pool_size = len(df[df.split == "train"])

for seed in args.seeds:
    for n in args.sizes:
        n_eff = min(n, pool_size)
        for src in args.sources:
            tr = make_train(df, src, n_eff, seed)
            if tr is None:
                continue
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
            model.fit(*tr)
            r = {"model": "tfidf_lr", "source": src, "n_target": n, "n": len(tr[0]), "seed": seed,
                 **score(y_test, model.predict(X_test))}
            append_result(args.out, r)
            print(r)
