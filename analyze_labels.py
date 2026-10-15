"""Người A: đo chất lượng nhãn LLM so với nhãn người.
  python analyze_labels.py --llm_file data/llm_final.csv
Kết quả: results/teacher.csv, figs/confusion_*.png, results/llm_errors_sample.csv (để tự phân loại lỗi)."""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from common import LABELS, load_data, score

ap = argparse.ArgumentParser()
ap.add_argument("--llm_file", default="data/llm_final.csv")
args = ap.parse_args()

df = load_data(args.llm_file)
os.makedirs("results", exist_ok=True); os.makedirs("figs", exist_ok=True)

rows = []
for sp in ["train", "dev", "test"]:
    s = df[df.split == sp]
    if (s.llm_label == "INVALID").all():
        continue
    r = {"split": sp, "n": len(s), "invalid": int((s.llm_label == "INVALID").sum()),
         **score(s.human_label, s.llm_label)}
    rows.append(r)
    print(f"\n===== {sp}: {r}")
    print(classification_report(s.human_label, s.llm_label, labels=LABELS, digits=3, zero_division=0))

    cm = confusion_matrix(s.human_label, s.llm_label, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Nhãn LLM"); ax.set_ylabel("Nhãn người"); ax.set_title(f"Nhãn LLM vs người ({sp})")
    fig.tight_layout(); fig.savefig(f"figs/confusion_{sp}.png", dpi=200); plt.close(fig)

pd.DataFrame(rows).to_csv("results/teacher.csv", index=False)

# 60 câu LLM gán khác người trên pool train -> mở bằng Excel/Sheets, điền cột error_type
err = df[(df.split == "train") & (df.llm_label != df.human_label)]
err = err.sample(min(60, len(err)), random_state=0)[["id", "text", "human_label", "llm_label"]]
err["error_type"] = ""
err.to_csv("results/llm_errors_sample.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi results/teacher.csv, figs/confusion_*.png, results/llm_errors_sample.csv")
