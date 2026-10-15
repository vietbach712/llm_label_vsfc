"""Tạo data/data.csv: id, text, human_label, split (train/dev/test). Chạy MỘT lần, rồi commit file kết quả."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_POOL, N_DEV, N_TEST = 4000, 300, 1000
BASE = ("https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/"
        "resolve/refs%2Fconvert%2Fparquet/default/")
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

os.makedirs("data", exist_ok=True)


def load(split):
    local = f"data/raw_{split}.parquet"
    if not os.path.exists(local):
        pd.read_parquet(BASE + split + "/0000.parquet").to_parquet(local)
    df = pd.read_parquet(local)
    df = df.rename(columns={"sentence": "text"})
    df["human_label"] = df.sentiment.map(ID2LABEL)
    df["text"] = df.text.str.strip()
    return df[["text", "human_label"]].drop_duplicates("text")


def sample(df, n):  # lấy mẫu giữ nguyên tỉ lệ các lớp
    return train_test_split(df, train_size=n, stratify=df.human_label, random_state=SEED)[0]


test = sample(load("test"), N_TEST)
dev = sample(load("validation"), N_DEV)
train = load("train")
train = train[~train.text.isin(set(test.text) | set(dev.text))]  # tránh câu trùng giữa các tập
pool = sample(train, N_POOL)

parts = []
for name, d in [("train", pool), ("dev", dev), ("test", test)]:
    d = d.reset_index(drop=True).copy()
    d["split"] = name
    d["id"] = [f"{name}_{i:05d}" for i in range(len(d))]
    parts.append(d)
data = pd.concat(parts)[["id", "text", "human_label", "split"]]
data.to_csv("data/data.csv", index=False)

print(data.groupby(["split", "human_label"]).size().unstack())
