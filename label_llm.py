"""Gán nhãn bằng LLM qua API kiểu OpenAI (dùng được cho OpenAI, Gemini, Ollama/vLLM chạy Qwen local).

Ví dụ:
  python label_llm.py --prompt v1 --split dev            # thử prompt trên 300 câu dev
  python label_llm.py --prompt v2 --split dev
  python label_llm.py --prompt v2 --split train test     # gán nhãn thật (4.000 + 1.000 câu)

Kết quả: llm_cache/<model>_<prompt>.jsonl (cache, chạy lại không tốn tiền)
         data/llm_<prompt>.csv (id, llm_label) để người B dùng.
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from common import LABELS, load_data, score

# ---------------- CẤU HÌNH: chọn MỘT khối, xoá dấu # ----------------  # SỬA
# # (a) Gemini (lấy key tại aistudio.google.com)
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# MODEL = "gemini-3.5-flash-lite"        # SỬA: kiểm tra tên model rẻ mới nhất trên trang Gemini API
# API_KEY = os.environ.get("LLM_API_KEY")
# WORKERS = 1                        # số request song song; gặp lỗi 429 thì giảm xuống 2-4
# (b) OpenAI
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
API_KEY = os.environ.get("LLM_API_KEY")
WORKERS = 1
# (c) Qwen local qua Ollama:  ollama pull qwen2.5:7b-instruct  &&  ollama serve
# BASE_URL = "http://localhost:11434/v1"; MODEL = "qwen2.5:7b-instruct"; API_KEY = "ollama"; WORKERS = 2
# ---------------------------------------------------------------------

SYSTEM = "Bạn là người gán nhãn dữ liệu cẩn thận. Chỉ trả lời đúng một từ."

PROMPTS = {
    # v1: prompt đơn giản, không biết quy ước của bộ dữ liệu
    "v1": """Phân loại cảm xúc của câu phản hồi sau của sinh viên về môn học/giảng viên.
Trả lời đúng MỘT từ: negative, neutral hoặc positive.

Câu: {text}
Nhãn:""",
    # v2: thêm hướng dẫn gán nhãn + vài ví dụ tự viết (KHÔNG lấy từ tập test)
    "v2": """Phân loại cảm xúc của câu phản hồi sau của sinh viên về môn học/giảng viên/cơ sở vật chất.

Hướng dẫn:
- positive: khen, hài lòng, cảm ơn, chúc, mong được học tiếp với giảng viên.
- negative: chê, phàn nàn, VÀ cả các góp ý/đề xuất cải thiện (câu có "nên", "cần", "mong ... hơn") vì chúng ngụ ý hiện tại chưa tốt.
- neutral: chỉ kể sự việc hoặc nói về bản thân, không đánh giá tốt/xấu.

Ví dụ:
Câu: cô giảng bài rất dễ hiểu . -> positive
Câu: phòng học nóng , máy chiếu bị mờ . -> negative
Câu: thầy nên cho thêm bài tập về nhà . -> negative
Câu: em học môn này vào học kỳ hè . -> neutral

Trả lời đúng MỘT từ: negative, neutral hoặc positive.
Câu: {text}
Nhãn:""",
}

VI2EN = {"tiêu cực": "negative", "trung tính": "neutral", "tích cực": "positive"}


def parse(raw):
    t = (raw or "").strip().lower()
    for vi, en in VI2EN.items():
        t = t.replace(vi, en)
    found = [lab for lab in LABELS if re.search(rf"\b{lab}\b", t)]
    return found[0] if len(found) == 1 else "INVALID"

def call(client, prompt, text):
    import random
    import threading
    import time

    # Các luồng cùng biết khi nào phải dừng gửi request.
    stop = call.__dict__.setdefault("_stop", threading.Event())

    for attempt in range(3):
        if stop.is_set():
            raise RuntimeError("Đã dừng gửi request vì API gặp lỗi cần xử lý.")

        try:
            r = client.with_options(max_retries=0).chat.completions.create(
                model=MODEL,
                temperature=0,
                reasoning_effort="none",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": PROMPTS[prompt].format(text=text),
                    },
                ],
            )
            return r.choices[0].message.content

        except Exception as e:
            print(f"  lỗi: {type(e).__name__}: {e}", flush=True)

            status = getattr(e, "status_code", None)
            code = getattr(e, "code", None)
            error_type = getattr(e, "type", None)
            body = getattr(e, "body", None)

            if isinstance(body, dict):
                detail = body.get("error", body)
                if isinstance(detail, dict):
                    code = code or detail.get("code")
                    error_type = error_type or detail.get("type")

            quota_errors = {
                "insufficient_quota",
                "credit_balance_exhausted",
                "organization_spend_limit_exceeded",
                "project_spend_limit_exceeded",
                "organization_usage_limit_exceeded",
            }

            if (
                status in (400, 401, 402, 403, 404)
                or code in quota_errors
                or error_type in quota_errors
                or isinstance(e, (AttributeError, TypeError))
            ):
                stop.set()
                raise RuntimeError(
                    f"Đã dừng gửi request; cần kiểm tra API/key/model/billing: {e}"
                ) from e

            if stop.is_set():
                raise RuntimeError("Đã dừng gửi request vì API gặp lỗi cần xử lý.") from e

            if attempt == 2:
                return None

            response = getattr(e, "response", None)
            retry_after = (
                response.headers.get("Retry-After")
                if response is not None
                else None
            )

            try:
                wait = max(0.0, float(retry_after))
            except (TypeError, ValueError):
                wait = 2 ** (attempt + 1)

            if wait > 120:
                stop.set()
                raise RuntimeError(
                    f"API yêu cầu chờ {wait:g} giây; đã dừng để bố chạy tiếp sau. {e}"
                ) from e

            print(f"  Thử lại sau {wait:.1f}s...", flush=True)
            time.sleep(wait + random.uniform(0, 0.5))

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS))
    ap.add_argument("--split", nargs="+", default=["dev"], choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu (để thử)")
    args = ap.parse_args()

    df = load_data()
    todo = df[df.split.isin(args.split)]
    if args.limit:
        todo = todo.head(args.limit)

    os.makedirs("llm_cache", exist_ok=True)
    cache_path = f"llm_cache/{MODEL.replace('/', '_').replace(':', '_')}_{args.prompt}.jsonl"
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["raw"] is not None:
                    done[rec["id"]] = rec
    todo = todo[~todo.id.isin(done)]
    print(f"Cần gán: {len(todo)} câu (đã có sẵn trong cache: {len(done)})")

    if not API_KEY:
        raise SystemExit("Chưa có API key: chạy  export LLM_API_KEY=...  (Colab: os.environ['LLM_API_KEY']='...')")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    lock = threading.Lock()
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(call, client, args.prompt, r.text): r.id for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = {"id": futs[fut], "raw": fut.result()}
            rec["llm_label"] = parse(rec["raw"])
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                done[rec["id"]] = rec
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}")

    out = pd.DataFrame(done.values())[["id", "llm_label", "raw"]]
    out.to_csv(f"data/llm_{args.prompt}.csv", index=False)
    print(f"Đã ghi data/llm_{args.prompt}.csv ({len(out)} câu, INVALID: {(out.llm_label == 'INVALID').sum()})")

    m = df.merge(out, on="id")
    for sp in args.split:
        s = m[m.split == sp]
        if len(s):
            print(sp, len(s), score(s.human_label, s.llm_label))


if __name__ == "__main__":
    main()
