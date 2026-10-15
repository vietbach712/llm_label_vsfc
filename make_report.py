"""Vẽ learning curve và trả lời 3 câu hỏi bằng số.
  python make_report.py
Đọc results/tfidf.csv, results/phobert.csv (nếu có), results/teacher.csv
Ghi figs/curve_<model>.png và results/summary.md"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC = "macro_f1"
NAMES = {"human": "Nhãn người", "llm": "Nhãn LLM", "llm_agree": "Nhãn LLM (lọc 2 prompt khớp)",
         "mix500": "500 nhãn người + nhãn LLM"}
COLORS = {"human": "#2a6fdb", "llm": "#e07b24", "llm_agree": "#2f9e66", "mix500": "#8a5cd1"}

files = sorted(glob.glob("results/tfidf*.csv") + glob.glob("results/phobert*.csv"))  # gộp cả phobert_A.csv, phobert_B.csv
res = pd.concat([pd.read_csv(f) for f in files])
teacher = pd.read_csv("results/teacher.csv").set_index("split")
t_score = teacher.loc["test", METRIC]
agg = res.groupby(["model", "source", "n_target"])[METRIC].agg(["mean", "std", "count"]).reset_index()
agg["std"] = agg["std"].fillna(0)

os.makedirs("figs", exist_ok=True)
lines = ["# Kết quả", "", f"Chỉ số: {METRIC}, chấm trên tập test nhãn người.", ""]
tr = teacher.loc["train"]
lines += ["## Câu 1: Nhãn LLM tệ hơn nhãn người bao nhiêu?", "",
          f"- Trên pool train ({int(tr.n)} câu): LLM khớp nhãn người acc = {tr.acc:.3f}, "
          f"macro-F1 = {tr.macro_f1:.3f}, kappa = {tr.kappa:.3f}, INVALID = {int(tr.invalid)}.", ""]

for model, g in agg.groupby("model"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for src, s in g.groupby("source"):
        s = s.sort_values("n_target")
        ax.errorbar(s.n_target, s["mean"], yerr=s["std"], marker="o", capsize=3,
                    label=NAMES.get(src, src), color=COLORS.get(src))
    ax.axhline(t_score, ls="--", color="gray", label=f"Teacher LLM trên test ({t_score:.3f})")
    ax.set_xscale("log"); ax.set_xticks(sorted(g.n_target.unique()))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Số mẫu train (thang log)"); ax.set_ylabel(f"{METRIC} trên test")
    ax.set_title(f"Learning curve – {model}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"figs/curve_{model}.png", dpi=200); plt.close(fig)

    h = g[g.source == "human"].set_index("n_target")["mean"]
    l = g[g.source == "llm"].set_index("n_target")["mean"]
    common_n = sorted(set(h.index) & set(l.index))
    lines += [f"### {model}", "", "| n | nhãn người | nhãn LLM | chênh lệch |", "|---|---|---|---|"]
    lines += [f"| {n} | {h[n]:.3f} | {l[n]:.3f} | {l[n] - h[n]:+.3f} |" for n in common_n]
    lines.append("")

    # Câu 2: n nhãn LLM nhỏ nhất đạt điểm của 1.000 nhãn người (nội suy trên thang log)
    if 1000 in h.index:
        target = h[1000]
        ns, vs = np.array(l.index, float), l.values
        ans = "không đạt trong khoảng đã thử (tối đa %d)" % ns.max()
        for i in range(len(ns)):
            if vs[i] >= target:
                if i == 0:
                    ans = f"≤ {int(ns[0])}"
                else:
                    f = (target - vs[i - 1]) / (vs[i] - vs[i - 1])
                    ans = f"≈ {int(round(np.exp(np.log(ns[i-1]) + f * (np.log(ns[i]) - np.log(ns[i-1])))))}"
                break
        lines.append(f"- Câu 2 ({model}): 1.000 nhãn người cho {target:.3f}; nhãn LLM cần {ans} mẫu.")
    # Câu 3
    if len(l):
        best_n = l.index.max()
        verdict = "VƯỢT" if l[best_n] > t_score else "CHƯA vượt"
        lines.append(f"- Câu 3 ({model}): student học {best_n} nhãn LLM = {l[best_n]:.3f}, "
                     f"teacher = {t_score:.3f} → student {verdict} teacher ({l[best_n] - t_score:+.3f}).")
    lines.append("")

open("results/summary.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
