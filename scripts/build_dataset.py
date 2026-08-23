"""Build mydataset_large.txt by merging:
- existing mydataset_500k.txt (q|a lines)
- ParsBench/PersianSyntheticQA (parquet, multi-turn user/assistant conversations)
- Heydaritoday/Persian-Synthetic-Instruct (jsonl instruction pairs)

Output format per line: question|answer (normalized-ish raw text).
"""
import io
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

RAW_DIR = Path(r"C:\Users\IBLADE~1\AppData\Local\Temp\opencode\persian_raw")
OUT_PATH = Path("mydataset_large.txt")
OLD_DATASET = Path("mydataset_500k.txt")

PARS_BENCH = "ParsBench/PersianSyntheticQA"
INSTRUCT = "Heydaritoday/Persian-Synthetic-Instruct"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def fetch(url, dest=None):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 persian-ai-builder"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if dest:
                Path(dest).write_bytes(data)
                return None
            return data
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt+1} after error: {e}")
            time.sleep(3)


def norm(text):
    text = str(text).strip()
    text = text.replace("\u200c", " ").replace("\u200b", " ")
    for a, b in {"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"}.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def qkey(q):
    t = norm(q).lower()
    t = re.sub(r"[؟?!.,،؛:()\[\]{}\"'`]+", "", t)
    return re.sub(r"\s+", " ", t)


pairs = []
seen = set()


def add_pair(q, a):
    q, a = norm(q), norm(a)
    if not q or not a or len(q) > 400 or len(a) > 600:
        return
    k = qkey(q)
    if not k or k in seen:
        return
    seen.add(k)
    pairs.append((q, a))


# ---------------- 1. existing dataset ----------------
print(f"[1/4] reading {OLD_DATASET} ...")
with open(OLD_DATASET, encoding="utf-8") as f:
    for line in f:
        if "|" in line:
            q, a = line.split("|", 1)
            add_pair(q, a)
print(f"  total after old dataset: {len(pairs)}")

# ---------------- 2. ParsBench parquet configs ----------------
print("[2/4] downloading ParsBench/PersianSyntheticQA ...")
api = json.loads(fetch(f"https://huggingface.co/api/datasets/{PARS_BENCH}"))
configs = [c["config_name"] for c in api["cardData"]["dataset_info"]]
print(f"  {len(configs)} configs")
import pyarrow.parquet as pq

for cfg in configs:
    fname = f"{cfg}/train-00000-of-00001.parquet"
    url = HF.format(repo=PARS_BENCH, path=urllib.request.quote(fname))
    buf = io.BytesIO(fetch(url))
    tbl = pq.read_table(buf, columns=["messages"])
    for msgs in tbl.column("messages").to_pylist():
        last_user = None
        for m in msgs:
            role, content = m.get("role"), (m.get("content") or "").strip()
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user:
                add_pair(last_user, content)
                last_user = None
    print(f"  {cfg}: cumulative pairs={len(pairs)}")

# ---------------- 3. instruct jsonl ----------------
print("[3/4] downloading Persian-Synthetic-Instruct ...")
api2 = json.loads(fetch(f"https://huggingface.co/api/datasets/{INSTRUCT}"))
files = [s["rfilename"] for s in api2["siblings"] if s["rfilename"].endswith(".jsonl")]
for fn in files:
    url = HF.format(repo=INSTRUCT, path=urllib.request.quote(fn))
    rows = fetch(url).decode("utf-8").splitlines()
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msgs = rec.get("messages") or rec.get("conversation")
        if msgs:
            last_user = None
            for m in msgs:
                role, content = m.get("role"), (m.get("content") or "").strip()
                if role in ("user", "human"):
                    last_user = content
                elif role in ("assistant", "gpt") and last_user:
                    add_pair(last_user, content)
                    last_user = None
        else:
            q = rec.get("instruction") or rec.get("question") or rec.get("prompt")
            a = rec.get("response") or rec.get("output") or rec.get("answer")
            if q and a:
                add_pair(q, a)
print(f"  total after instruct: {len(pairs)}")

# ---------------- 4. write output ----------------
print(f"[4/4] writing {OUT_PATH} ...")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for q, a in pairs:
        f.write(f"{q}|{a}\n")

size_mb = OUT_PATH.stat().st_size / 1e6
print(f"DONE: {len(pairs)} unique pairs | {size_mb:.1f} MB -> {OUT_PATH}")
