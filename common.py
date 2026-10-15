"""Hàm dùng chung cho cả nhóm. KHÔNG sửa file này sau khi đã chạy thí nghiệm."""
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

LABELS = ["negative", "neutral", "positive"]
DATA_FILE = "data/data.csv"


def load_data(llm_file=None, llm_file2=None):
    """Đọc data.csv; nếu có file nhãn LLM thì gộp thêm cột llm_label (và llm_label2)."""
    df = pd.read_csv(DATA_FILE)
    for col, path in [("llm_label", llm_file), ("llm_label2", llm_file2)]:
        if path:
            lab = pd.read_csv(path)[["id", "llm_label"]].rename(columns={"llm_label": col})
            df = df.merge(lab, on="id", how="left")
            df[col] = df[col].fillna("INVALID")
    return df


def pool_order(df, seed):
    """Thứ tự ngẫu nhiên cố định của pool train theo seed.
    Lấy n mẫu đầu => tập n nhỏ luôn nằm trong tập n lớn, và nhãn người / nhãn LLM dùng CÙNG câu."""
    pool = df[df.split == "train"]
    if "llm_label" in pool:  # bỏ câu LLM trả lời hỏng để hai nguồn nhãn dùng đúng cùng một tập câu
        pool = pool[pool.llm_label.isin(LABELS)]
    return pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_train(df, source, n, seed):
    """Trả về (texts, labels) gồm n mẫu train cho một nguồn nhãn.
    source: human | llm | llm_agree | mix500"""
    pool = pool_order(df, seed)
    if source == "human":
        sub = pool.head(n)
        return sub.text.tolist(), sub.human_label.tolist()
    if source == "llm":
        sub = pool.head(n)
        return sub.text.tolist(), sub.llm_label.tolist()
    if source == "llm_agree":  # chỉ giữ câu mà 2 lần gán nhãn LLM khớp nhau
        agree = pool[pool.llm_label == pool.llm_label2]
        if len(agree) < n:
            return None
        sub = agree.head(n)
        return sub.text.tolist(), sub.llm_label.tolist()
    if source == "mix500":  # 500 nhãn người + phần còn lại là nhãn LLM
        if n <= 500:
            return None
        sub = pool.head(n)
        labels = sub.human_label.tolist()[:500] + sub.llm_label.tolist()[500:]
        return sub.text.tolist(), labels
    raise ValueError(source)


def test_set(df):
    t = df[df.split == "test"]
    return t.text.tolist(), t.human_label.tolist()


def score(y_true, y_pred):
    return {
        "acc": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
    }


def append_result(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
