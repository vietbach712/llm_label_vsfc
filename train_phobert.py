"""Người B: fine-tune PhoBERT trên Colab/Kaggle (GPU T4). Mỗi lần chạy ~2-6 phút tuỳ n.
  !pip install -q underthesea transformers
  !python train_phobert.py --sources human llm --sizes 500 1000 4000 --seeds 0
Kết quả ghi thêm vào results/phobert.csv (tải file này về và commit sau mỗi buổi chạy)."""
import argparse, os, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import LABELS, load_data, make_train, test_set, score, append_result

ap = argparse.ArgumentParser()
ap.add_argument("--llm_file", default="data/llm_final.csv")
ap.add_argument("--llm_file2", default=None)
ap.add_argument("--sources", nargs="+", default=["human", "llm"])
ap.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 4000])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--model", default="vinai/phobert-base")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--out", default="results/phobert.csv")  # 2 người chạy song song: mỗi người một file, vd results/phobert_A.csv
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L2I = {l: i for i, l in enumerate(LABELS)}

# --- 1. Tách từ (PhoBERT BẮT BUỘC đầu vào đã tách từ: "giảng_viên nhiệt_tình"). Cache lại để chỉ làm 1 lần.
SEG = "data/data_seg.csv"
df = load_data(args.llm_file, args.llm_file2)
if not os.path.exists(SEG):
    from underthesea import word_tokenize
    print("Đang tách từ (chỉ lần đầu)...")
    seg = pd.DataFrame({"text": df.text, "text_seg": [word_tokenize(t, format="text") for t in df.text]})
    seg.to_csv(SEG, index=False)
seg_map = dict(pd.read_csv(SEG).values)
df["text"] = df.text.map(seg_map)  # thay text gốc bằng text đã tách từ
X_test, y_test = test_set(df)

tok = AutoTokenizer.from_pretrained(args.model)


def batches(texts, labels=None, shuffle=False):
    idx = list(range(len(texts)))
    def collate(b):
        enc = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([L2I[labels[i]] for i in b])
        return enc
    return DataLoader(idx, batch_size=args.bs, shuffle=shuffle, collate_fn=collate)


def run(texts, labels, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(DEV)
    train_dl = batches(texts, labels, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    # Không dùng early stopping trên nhãn người: train cố định số epoch cho công bằng.
    model.train()
    for ep in range(args.epochs):
        for b in train_dl:
            b = {k: v.to(DEV) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    model.eval(); preds = []
    with torch.no_grad():
        for b in batches(X_test):
            b = {k: v.to(DEV) for k, v in b.items()}
            preds += model(**b).logits.argmax(-1).tolist()
    del model; torch.cuda.empty_cache()
    return [LABELS[p] for p in preds]


pool_size = len(df[df.split == "train"])
for seed in args.seeds:
    for n in args.sizes:
        for src in args.sources:
            tr = make_train(df, src, min(n, pool_size), seed)
            if tr is None:
                continue
            t0 = time.time()
            pred = run(*tr, seed)
            r = {"model": "phobert", "source": src, "n_target": n, "n": len(tr[0]), "seed": seed,
                 **score(y_test, pred), "minutes": round((time.time() - t0) / 60, 1)}
            append_result(args.out, r)
            print(r, flush=True)
