import os
import re
import ast
import math
import json
import time
import hashlib
import datetime
import difflib
import operator
import threading
import random
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# Persian AI 500K FINAL
# Retrieval-first + Memory + Tools + small PyTorch fallback
# Optimized for a large TXT dataset and CPU-friendly training.
# ============================================================

DATASET_PATH = "mydataset_500k.txt"
MODEL_PATH = "persian_500k_model.pth"
VOCAB_PATH = "persian_500k_vocab.txt"
META_PATH = "persian_500k_meta.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small fallback transformer: do not train on all 500K rows.
EMBED_DIM = 96
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 384
DROPOUT = 0.08
MAX_INPUT_LEN = 28
MAX_OUTPUT_LEN = 32
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TRAIN_LIMIT = 32000

# Retrieval tuning.
EXACT_THRESHOLD = 0.999
SHORT_MIN = 0.72
NORMAL_MIN = 0.64
CANDIDATE_LIMIT = 120

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

print("=" * 72)
print("                    Persian AI 500K FINAL")
print("=" * 72)
print("Device:", DEVICE)
print("Dataset:", DATASET_PATH)


# ============================================================
# Normalization
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    replacements = {
        "ي":"ی","ى":"ی","ك":"ک","ۀ":"ه","ة":"ه","ؤ":"و",
        "إ":"ا","أ":"ا","ٱ":"ا","‌":" ","ـ":"",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    # common latin/chat transliterations
    latin = {
        "salam":"سلام", "salaam":"سلام", "slm":"سلام",
        "khubi":"خوبی", "khoobi":"خوبی", "khobi":"خوبی",
        "merci":"مرسی", "mersi":"مرسی", "chikhabar":"چه خبر",
        "chkhbr":"چه خبر", "chetori":"چطوری", "chetoori":"چطوری",
    }
    if text in latin:
        text = latin[text]
    typo = {
        "خويس":"خوبی", "خو ی":"خوبی", "خو بی":"خوبی", "خوبى":"خوبی",
        "چيکار":"چیکار", "چي کار":"چی کار", "چيکار ميکني":"چیکار میکنی",
        "ميکني":"میکنی", "چطوري":"چطوری", "مشتى":"مشتی", "مشتیي":"مشتی",
        "چخبر":"چه خبر", "اسمت چيه":"اسمت چیه", "اسمم چيه":"اسمم چیه",
        "اسمم چي بود":"اسمم چی بود", "چي ميگي":"چی میگی", "چیکار ميکنی":"چیکار میکنی",
    }
    for a, b in typo.items():
        text = text.replace(a.lower(), b)
    text = re.sub(r"[\u200b\u200c]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_text(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"[؟?!.,،؛:()\[\]{}\"'`]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

STOP = {
    "و","یا","از","به","در","که","را","با","برای","یک","این","آن",
    "من","تو","او","ما","شما","هم","چه","چی","چیه","است","هست",
    "رو","یه","می","ام","منظور","لطفا","لطفاً","میشه","میتونی","می‌تونی",
    "می","کنم","کنی","کنه","کنید","کنیم","بود","هستش"
}


def tokens(text):
    return [w for w in match_text(text).split() if len(w) >= 2 and w not in STOP]


# ============================================================
# Dataset loading + index
# ============================================================

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not Path(DATASET_PATH).exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print("Loading dataset...")
DATA_HASH = file_hash(DATASET_PATH)

pairs = []
exact = {}
postings = defaultdict(list)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line_no, raw in enumerate(f):
        line = raw.rstrip("\n")
        if "|" not in line:
            continue
        q, a = line.split("|", 1)
        q = normalize_text(q)
        a = normalize_text(a)
        if not q or not a:
            continue
        idx = len(pairs)
        pairs.append((q, a))
        nq = match_text(q)
        if nq not in exact:
            exact[nq] = a
        for tok in set(tokens(q)):
            # Cap each posting list to keep memory bounded.
            if len(postings[tok]) < 1500:
                postings[tok].append(idx)

print("Dataset pairs:", len(pairs))
print("Index tokens:", len(postings))


# ============================================================
# Direct conversational intelligence
# ============================================================
DIRECT = {
    "سلام": "سلام! 👋 خوش اومدی. حالت چطوره؟",
    "درود": "درود! 👋 خوشحالم که اومدی.",
    "خوبی": "خوبم، مرسی که پرسیدی 😊 تو چطوری؟",
    "چه خبر": "همه‌چیز خوبه 😄 تو چه خبر؟",
    "مشتی چطوری": "خوبم مشتی 😄 تو چطوری؟",
    "عالی": "چه عالی! 😄",
    "عالی ام": "چه خوب! خوشحالم که حالت عالیه 😊",
    "خیلی خوبم": "عالیه! خوشحالم که حالت خوبه 😊",
    "چیکار میکنی": "با تو گفتگو می‌کنم و سعی می‌کنم کمکت کنم.",
    "چی کار میکنی": "با تو گفتگو می‌کنم و سعی می‌کنم کمکت کنم.",
    "چی میگی": "منظورت رو بگو؛ اگر بخشی رو درست نفهمیدم، دوباره توضیح می‌دم.",
    "چیکار کنم": "بگو در چه کاری مرددی تا با هم قدم‌به‌قدم بررسیش کنیم.",
    "چی کار کنم": "بگو در چه کاری مرددی تا با هم گزینه‌ها رو بررسی کنیم.",
    "چیکار کنیم": "می‌تونیم گپ بزنیم، چیزی یاد بگیریم یا یک پروژه بسازیم.",
    "چی کار کنیم": "می‌تونیم گپ بزنیم، پروژه بسازیم یا یک موضوع جدید یاد بگیریم.",
    "الو": "الو! 👋 اینجام، بگو.",
    "هان": "جان؟ 😄 بگو ببینم چی می‌خوای.",
    "اممم": "آروم فکر کن 😄 هر وقت آماده بودی بگو.",
    "ممنون": "خواهش می‌کنم ❤️",
    "مرسی": "قابلی نداشت 😊",
    "خداحافظ": "خداحافظ! هر وقت خواستی دوباره برگرد 👋",
    "فعلا": "فعلاً! مراقب خودت باش.",
    "شب بخیر": "شب بخیر 🌙 خواب‌های خوبی داشته باشی.",
    "صبح بخیر": "صبح بخیر ☀️ امیدوارم روز خوبی داشته باشی.",
    "احمق": "اگر جوابم بد بود، بگو کجاش مشکل داشت تا بهترش کنم.",
    "اسکل": "اگر جوابم بد بود، بگو کجاش مشکل داشت تا اصلاحش کنیم.",
    "نفهم": "اگر توضیحم واضح نبود، بگو تا ساده‌تر توضیح بدم.",
}
DIRECT = {match_text(k): v for k, v in DIRECT.items()}


# ============================================================
# Memory
# ============================================================

memory = {}


def detect_name(text):
    t = match_text(text)
    patterns = [
        r"(?:اسمم|اسم من|نامم|نام من)\s+([آ-یa-zA-Z]{2,30})\s*(?:هست|است|هستم|ام)?$",
        r"من\s+([آ-یa-zA-Z]{2,30})\s*(?:هستم|ام)$",
    ]
    for p in patterns:
        m = re.fullmatch(p, t)
        if m:
            name = m.group(1)
            if name not in {"چی", "چیه", "بود", "هستم", "است", "هست"}:
                return name
    return None


def memory_answer(t):
    if re.search(r"(?:اسمم|اسم من|نامم|نام من).*(?:چیه|چی بود|چه بود)", match_text(t)):
        if "name" in memory:
            return f"اسم تو {memory['name']} هست. 😊"
        return "هنوز اسمت رو بهم نگفتی؛ اگه بگی در همین گفتگو یادم می‌مونه."
    return None


# ============================================================
# Tools: calculator, date/time
# ============================================================

def calculator(text):
    t = normalize_text(text)
    phrases = {
        "ضربدر":"*", "ضرب":"*", "منهای":"-", "به علاوه":"+",
        "بعلاوه":"+", "تقسیم بر":"/", "تقسیم":"/"
    }
    for a, b in phrases.items():
        t = t.replace(a, b)
    # handle "5 با 5"
    t = re.sub(r"(\d+)\s+با\s+(\d+)", r"\1+\2", t)
    expr = re.sub(r"[^0-9+\-*/().% ]", "", t)
    if not re.search(r"\d", expr) or not re.search(r"[+\-*/%]", expr):
        return None
    try:
        value = _safe_arith_eval(ast.parse(expr.replace("%", "/100"), mode="eval").body)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"حاصل میشه: {value:g} 🧮"
    except Exception:
        return None
    return None


_ARITH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_arith_eval(node):
    """Safely evaluate an arithmetic AST node (no function calls, names, or attributes)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_arith_eval(node.left), _safe_arith_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_arith_eval(node.operand))
    raise ValueError("unsafe expression")


def date_time_answer(text):
    t = match_text(text)
    now = datetime.datetime.now()
    if any(x in t for x in ["ساعت چنده", "ساعت الان", "الان ساعت"]):
        return f"الان ساعت {now:%H:%M:%S} هست. 🕐"
    if any(x in t for x in ["امروز چندمه", "تاریخ امروز", "امروز چه تاریخیه"]):
        return f"تاریخ میلادی امروز {now:%Y/%m/%d} هست. 📅"
    return None


# ============================================================
# Intent / conversational pattern handling
# ============================================================

def conversational_answer(text):
    t = match_text(text)

    # Persian + Latin greeting
    if t in {"سلام", "درود"}:
        return DIRECT[t]

    if t in {"salam", "salaam", "slm"}:
        return DIRECT["سلام"]

    if "خوبی" in t or t in {"خويس", "خو بی", "خويس؟"}:
        return DIRECT["خوبی"]

    if "چه خبر" in t or t == "چخبر":
        return DIRECT["چه خبر"]

    if "چطوری" in t and "مشتی" in t:
        return DIRECT["مشتی چطوری"]

    if t in DIRECT:
        return DIRECT[t]

    # statement -> supportive follow-up
    if detect_name(t):
        name = detect_name(t)
        memory["name"] = name
        return f"خوشبختم {name} ! 😊"

    if any(x in t for x in ["هوا چطوره", "آب و هوا", "هوا امروز"]):
        return "برای گفتن وضعیت واقعی هوا، نام شهر رو بگو تا مشخص بشه درباره کدوم شهر صحبت می‌کنیم."

    if re.fullmatch(r"(?:سلام|درود)\s+(?:خوبی|چطوری)", t):
        return "سلام! خوبم ممنون 😊 تو چطوری؟"

    if any(x in t for x in ["چطور ai بسازم", "چطور هوش مصنوعی بسازم"]):
        return "برای شروع، Python و PyTorch یاد بگیر، بعد یک مسئله کوچک انتخاب کن و مدل و دیتاستت رو مرحله‌به‌مرحله بساز."

    return None


# ============================================================
# Retrieval scoring
# ============================================================

def score_pair(query, candidate):
    q = match_text(query)
    c = match_text(candidate)
    if q == c:
        return 1.0
    qt = set(tokens(q))
    ct = set(tokens(c))
    token_score = len(qt & ct) / max(1, len(qt | ct))
    char_score = difflib.SequenceMatcher(None, q, c).ratio()
    # Stronger weight for token overlap, but keep spelling tolerance.
    return 0.60 * token_score + 0.40 * char_score


def retrieve(query):
    nq = match_text(query)
    if not nq:
        return None, 0.0

    if nq in exact:
        return exact[nq], 1.0

    qtokens = tokens(nq)
    candidate_ids = set()
    for tok in qtokens:
        for idx in postings.get(tok, ()):
            candidate_ids.add(idx)
            if len(candidate_ids) >= CANDIDATE_LIMIT:
                break
        if len(candidate_ids) >= CANDIDATE_LIMIT:
            break

    if not candidate_ids:
        return None, 0.0

    best_score = 0.0
    best_answer = None
    for idx in candidate_ids:
        q, a = pairs[idx]
        s = score_pair(query, q)
        if s > best_score:
            best_score = s
            best_answer = a

    # short inputs require much higher confidence
    if len(qtokens) <= 2:
        threshold = SHORT_MIN + 0.10
    else:
        threshold = NORMAL_MIN

    if best_score >= threshold:
        return best_answer, best_score
    return None, best_score


# ============================================================
# Fallback Transformer: only train on a clean, diverse subset
# ============================================================

clean_examples = []
seen_q = set()
# Prefer shorter and non-duplicate questions for CPU training.
for q, a in pairs:
    mq = match_text(q)
    if mq in seen_q:
        continue
    if len(mq.split()) > MAX_INPUT_LEN:
        continue
    if len(a.split()) > MAX_OUTPUT_LEN:
        continue
    seen_q.add(mq)
    clean_examples.append((q, a))
    if len(clean_examples) >= TRAIN_LIMIT:
        break

# If dataset has many generated variants, add a deterministic sample over the whole file.
if len(clean_examples) < TRAIN_LIMIT and pairs:
    step = max(1, len(pairs) // TRAIN_LIMIT)
    for idx in range(0, len(pairs), step):
        q, a = pairs[idx]
        mq = match_text(q)
        if mq not in seen_q:
            seen_q.add(mq)
            clean_examples.append((q, a))
        if len(clean_examples) >= TRAIN_LIMIT:
            break

print("Fallback training samples:", len(clean_examples))

freq = Counter()
for q, a in clean_examples:
    freq.update(normalize_text(q).split())
    freq.update(normalize_text(a).split())

word_to_id = {PAD:0, UNK:1, BOS:2, EOS:3}
for w, _ in freq.most_common():
    if w not in word_to_id:
        word_to_id[w] = len(word_to_id)
id_to_word = {i:w for w,i in word_to_id.items()}
VOCAB_SIZE = len(word_to_id)
print("Fallback vocabulary:", VOCAB_SIZE)


def encode(text, limit):
    return [word_to_id.get(w, 1) for w in normalize_text(text).split()[:limit]]

train_data = []
for q, a in clean_examples:
    s = encode(q, MAX_INPUT_LEN)
    t = [BOS] + normalize_text(a).split()[:MAX_OUTPUT_LEN] + [EOS]
    t = [word_to_id.get(w, UNK if w == UNK else 1) if w not in {BOS, EOS} else word_to_id[w] for w in t]
    if s and len(t) > 1:
        train_data.append((s, t))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]


class ChatTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.src_pos = PositionalEncoding(EMBED_DIM, MAX_INPUT_LEN+2)
        self.tgt_pos = PositionalEncoding(EMBED_DIM, MAX_OUTPUT_LEN+2)
        enc = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation="gelu"
        )
        dec = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, NUM_LAYERS, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, NUM_LAYERS)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)
    def forward(self, src, tgt):
        src_pad = src.eq(0)
        tgt_pad = tgt.eq(0)
        src_e = self.src_pos(self.emb(src) * math.sqrt(EMBED_DIM))
        mem = self.encoder(src_e, src_key_padding_mask=src_pad)
        tgt_e = self.tgt_pos(self.emb(tgt) * math.sqrt(EMBED_DIM))
        L = tgt.size(1)
        mask = torch.triu(torch.ones(L,L,device=tgt.device,dtype=torch.bool), diagonal=1)
        out = self.decoder(
            tgt_e, mem, tgt_mask=mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad
        )
        return self.fc(self.norm(out))

model = ChatTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


def make_batch(batch):
    ms = max(len(x[0]) for x in batch)
    mt = max(len(x[1]) for x in batch)
    src = [x[0] + [0]*(ms-len(x[0])) for x in batch]
    tgt = [x[1] + [0]*(mt-len(x[1])) for x in batch]
    return torch.tensor(src,dtype=torch.long,device=DEVICE), torch.tensor(tgt,dtype=torch.long,device=DEVICE)


META = {"dataset_hash": DATA_HASH, "pairs": len(pairs), "vocab": VOCAB_SIZE}
need_train = True
if Path(MODEL_PATH).exists() and Path(META_PATH).exists():
    try:
        meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
        if meta.get("dataset_hash") == DATA_HASH and meta.get("vocab") == VOCAB_SIZE:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            need_train = False
            print("Fallback model loaded.")
    except Exception as exc:
        print("Old model ignored:", exc)

if need_train:
    print("\n="*2)
    print("Training CPU fallback Transformer")
    print("="*60)
    model.train()
    random.seed(42)
    for epoch in range(EPOCHS):
        random.shuffle(train_data)
        total=0.0; n=0
        for st in range(0,len(train_data),BATCH_SIZE):
            batch=train_data[st:st+BATCH_SIZE]
            src,tgt=make_batch(batch)
            din=tgt[:,:-1]
            target=tgt[:,1:]
            optimizer.zero_grad(set_to_none=True)
            logits=model(src,din)
            loss=criterion(logits.reshape(-1,VOCAB_SIZE),target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total += float(loss.item()); n+=1
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total/max(1,n):.4f}")
    torch.save({"model":model.state_dict()}, MODEL_PATH)
    Path(VOCAB_PATH).write_text("\n".join(f"{i}\t{w}" for i,w in id_to_word.items()),encoding="utf-8")
    Path(META_PATH).write_text(json.dumps(META,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Fallback model saved.")


# ============================================================
# Conservative Transformer generation
# ============================================================

def generate_fallback(text):
    model.eval()
    src_ids = encode(text, MAX_INPUT_LEN)
    if not src_ids:
        return None
    src = torch.tensor([src_ids],dtype=torch.long,device=DEVICE)
    generated=[word_to_id[BOS]]
    with torch.no_grad():
        for _ in range(MAX_OUTPUT_LEN):
            tgt=torch.tensor([generated],dtype=torch.long,device=DEVICE)
            logits=model(src,tgt)[0,-1].clone()
            logits[0]=-float("inf"); logits[1]=-float("inf"); logits[2]=-float("inf")
            # Penalize already emitted tokens strongly.
            for tid in set(generated):
                if tid < logits.numel():
                    logits[tid] -= 2.0
            next_id=int(torch.argmax(logits).item())
            if next_id == word_to_id[EOS]:
                break
            generated.append(next_id)
    out=[]
    for tid in generated[1:]:
        w=id_to_word.get(tid,"")
        if w in {PAD,UNK,BOS,EOS}: break
        out.append(w)
    text_out=normalize_text(" ".join(out))
    # Reject weak/garbled output.
    if len(text_out.split()) < 2:
        return None
    if len(set(text_out.split())) <= 2 and len(text_out.split()) >= 4:
        return None
    return text_out


# ============================================================
# Final answer router
# ============================================================

def answer(text):
    text=normalize_text(text)
    if not text:
        return "یه پیامی بنویس 😊"

    # Memory first
    mem=memory_answer(text)
    if mem:
        return mem

    # Name extraction
    name=detect_name(text)
    if name:
        memory["name"]=name
        return f"خوشبختم {name}! 😊"

    # User asks their name
    if re.fullmatch(r"(?:اسمم|اسم من|نامم|نام من)\s*(?:چیه|چیست|چی بود|چه بود)\s*\??", match_text(text)):
        return f"اسم تو {memory['name']} هست. 😊" if "name" in memory else "هنوز اسمت رو بهم نگفتی؛ اگه بگی در همین گفتگو یادم می‌مونه."

    # Direct conversational rules
    direct=conversational_answer(text)
    if direct:
        return direct

    # Tools
    dt=date_time_answer(text)
    if dt:
        return dt
    calc=calculator(text)
    if calc:
        return calc

    # Retrieval
    ans,score=retrieve(text)
    if ans:
        print(f"[retrieval] {score:.3f}")
        return ans

    # Very short/unclear utterances: do NOT hallucinate from transformer.
    mt=match_text(text)
    if len(tokens(mt)) <= 1:
        return "منظورت رو کامل نگرفتم 🤔 یه کم بیشتر توضیح میدی؟"

    # Fallback transformer
    generated=generate_fallback(text)
    if generated:
        print("[transformer]")
        return generated

    return "برای این سؤال هنوز جواب مطمئنی ندارم؛ اگر کمی واضح‌ترش کنی، بهتر می‌تونم کمک کنم."


# ============================================================
# GUI
# ============================================================

class ChatGUI:
    def __init__(self, root):
        self.root=root
        root.title("Persian AI — 500K")
        root.geometry("1050x740")
        root.minsize(800,600)
        root.configure(bg="#0b1020")

        header=tk.Frame(root,bg="#111827",height=72)
        header.pack(fill="x"); header.pack_propagate(False)
        tk.Label(header,text="✦ Persian AI",font=("Segoe UI",21,"bold"),fg="white",bg="#111827").pack(side="left",padx=25)
        self.status=tk.Label(header,text="● آنلاین",font=("Segoe UI",10),fg="#4ade80",bg="#111827")
        self.status.pack(side="right",padx=25)

        center=tk.Frame(root,bg="#0b1020")
        center.pack(fill="both",expand=True,padx=25,pady=20)
        self.chat=scrolledtext.ScrolledText(center,wrap=tk.WORD,font=("Tahoma",12),bg="#111827",fg="#e5e7eb",insertbackground="white",selectbackground="#374151",relief="flat",bd=0,padx=24,pady=20)
        self.chat.pack(fill="both",expand=True)
        self.chat.tag_config("user",foreground="#60a5fa",font=("Tahoma",11,"bold"))
        self.chat.tag_config("bot",foreground="#4ade80",font=("Tahoma",11,"bold"))
        self.chat.tag_config("message",foreground="#f1f5f9",font=("Tahoma",12))

        bottom=tk.Frame(root,bg="#111827",height=120)
        bottom.pack(fill="x"); bottom.pack_propagate(False)
        inp=tk.Frame(bottom,bg="#1f2937")
        inp.pack(side="left",fill="both",expand=True,padx=20,pady=15)
        self.entry=tk.Text(inp,height=2,wrap=tk.WORD,font=("Tahoma",12),bg="#1f2937",fg="white",insertbackground="white",relief="flat",bd=0)
        self.entry.pack(fill="both",expand=True,padx=10,pady=5)
        self.entry.bind("<Return>",self.enter)

        buttons=tk.Frame(bottom,bg="#111827")
        buttons.pack(side="right",padx=15)
        self.send_btn=tk.Button(buttons,text="ارسال ➤",command=self.send,font=("Tahoma",11,"bold"),bg="#2563eb",fg="white",activebackground="#1d4ed8",relief="flat",cursor="hand2")
        self.send_btn.pack(pady=3,ipadx=18,ipady=7)
        tk.Button(buttons,text="کپی مکالمه",command=self.copy,font=("Tahoma",9),bg="#374151",fg="white",relief="flat",cursor="hand2").pack(pady=3,ipadx=10,ipady=5)
        tk.Button(buttons,text="پاک کردن",command=self.clear,font=("Tahoma",9),bg="#374151",fg="white",relief="flat",cursor="hand2").pack(pady=3,ipadx=10,ipady=5)

        self.add_bot("سلام 👋\nمن Persian AI هستم.\nبا حافظه، ابزارهای داخلی، Retrieval روی دیتاست بزرگ و یک Transformer کوچک کار می‌کنم.\nپیامت رو بنویس 😊")
        self.entry.focus_set()

    def enter(self,event):
        self.send(); return "break"
    def add(self,text,tag):
        self.chat.insert(tk.END,text,tag); self.chat.see(tk.END)
    def add_user(self,text):
        self.add("\nشما\n","user"); self.add(text+"\n","message")
    def add_bot(self,text):
        self.add("\nPersian AI\n","bot"); self.add(text+"\n","message")
    def send(self):
        text=self.entry.get("1.0",tk.END).strip()
        if not text:return
        self.entry.delete("1.0",tk.END); self.add_user(text)
        self.status.config(text="● در حال فکر کردن...",fg="#facc15"); self.send_btn.config(state="disabled")
        threading.Thread(target=self.worker,args=(text,),daemon=True).start()
    def worker(self,text):
        try: resp=answer(text)
        except Exception as e: resp="خطا هنگام پاسخ‌گویی:\n"+str(e)
        self.root.after(0,lambda:self.finish(resp))
    def finish(self,resp):
        self.add_bot(resp); self.status.config(text="● آنلاین",fg="#4ade80"); self.send_btn.config(state="normal"); self.entry.focus_set()
    def copy(self):
        text=self.chat.get("1.0",tk.END).strip(); self.root.clipboard_clear(); self.root.clipboard_append(text); self.root.update(); self.status.config(text="● کپی شد",fg="#60a5fa"); self.root.after(1500,lambda:self.status.config(text="● آنلاین",fg="#4ade80"))
    def clear(self):
        self.chat.delete("1.0",tk.END); memory.clear(); self.add_bot("چت پاک شد 🧹\nاز اول شروع کنیم.")


if __name__ == "__main__":
    print("="*72)
    print("Pairs:", len(pairs))
    print("Fallback training samples:", len(train_data))
    print("Model:", MODEL_PATH)
    print("="*72)
    root=tk.Tk(); app=ChatGUI(root); root.mainloop()
