import os
import re
import json
import time
import socket
import hashlib
import pickle
import datetime
import threading
import random
import urllib.request
import uuid
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Persian AI — Fully Neural Chatbot (GPT-style, no canned answers)
# ============================================================

DATASET_PATH = "mydataset_large.txt"
MODEL_PATH = "persian_gpt_model.pth"
VOCAB_PATH = "persian_gpt_vocab.txt"
META_PATH = "persian_gpt_meta.json"
CACHE_PATH = "dataset_cache.pkl"
SESSIONS_DIR = "sessions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.08
MAX_INPUT_LEN = 48
MAX_OUTPUT_LEN = 64
MAX_SEQ_LEN = MAX_INPUT_LEN + MAX_OUTPUT_LEN + 3
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TRAIN_LIMIT = 150000
MAX_VOCAB = 20000

PAD, UNK, BOS, EOS, SEP = "<pad>", "<unk>", "<bos>", "<eos>", "<sep>"
ARCH = "gpt-decoder-v3-neural-only"

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

# ============================================================
# Dataset loading (with caching)
# ============================================================

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not Path(DATASET_PATH).exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

DATA_HASH = file_hash(DATASET_PATH)

def load_dataset_with_cache():
    """Load dataset pairs from cache if valid, otherwise parse and cache."""
    if Path(CACHE_PATH).exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if cache.get("hash") == DATA_HASH:
                print("Dataset loaded from cache (fast!)")
                return cache["pairs"]
        except Exception as e:
            print(f"Cache invalid: {e}")

    print("Parsing dataset (first time or dataset changed)...")
    pairs = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            if "|" not in raw:
                continue
            q, a = raw.split("|", 1)
            q = normalize_text(q)
            a = normalize_text(a)
            if q and a:
                pairs.append((q, a))

    cache_data = {"hash": DATA_HASH, "pairs": pairs}
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Dataset cached ({len(pairs)} pairs)")
    return pairs

pairs = load_dataset_with_cache()
random.Random(42).shuffle(pairs)

# ============================================================
# Model training data (everything below is learned, not scripted)
# ============================================================

clean_examples = []
seen_q = set()
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

freq = Counter()
for q, a in clean_examples:
    freq.update(normalize_text(q).split())
    freq.update(normalize_text(a).split())

word_to_id = {PAD:0, UNK:1, BOS:2, EOS:3, SEP:4}
for w, _ in freq.most_common(MAX_VOCAB - len(word_to_id)):
    if w not in word_to_id:
        word_to_id[w] = len(word_to_id)
id_to_word = {i:w for w,i in word_to_id.items()}
VOCAB_SIZE = len(word_to_id)

PAD_ID = word_to_id[PAD]
UNK_ID = word_to_id[UNK]
BOS_ID = word_to_id[BOS]
EOS_ID = word_to_id[EOS]
SEP_ID = word_to_id[SEP]


def encode(text, limit):
    return [word_to_id.get(w, UNK_ID) for w in normalize_text(text).split()[:limit]]

train_data = []
for q, a in clean_examples:
    prompt_ids = [BOS_ID] + encode(q, MAX_INPUT_LEN) + [SEP_ID]
    response_ids = encode(a, MAX_OUTPUT_LEN) + [EOS_ID]
    seq = prompt_ids + response_ids
    if len(seq) > len(prompt_ids) + 1:
        train_data.append((seq, len(prompt_ids)))


class ChatGPT(nn.Module):
    """Decoder-only causal transformer (GPT-style next-token predictor)."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        block = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
            activation="gelu"
        )
        self.blocks = nn.TransformerEncoder(block, NUM_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0))
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        x = self.blocks(x, mask=causal_mask)
        return self.head(self.norm(x))

model = ChatGPT().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


def make_batch(batch):
    """Build (inputs, targets) for next-token prediction.
    Loss only applies to positions predicting response tokens (>= resp_start)."""
    ms = max(len(seq) for seq, _ in batch)
    x = torch.full((len(batch), ms), PAD_ID, dtype=torch.long)
    resp_start = torch.zeros(len(batch), dtype=torch.long)
    for i, (seq, rs) in enumerate(batch):
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        resp_start[i] = rs
    x = x.to(DEVICE)
    resp_start = resp_start.to(DEVICE)
    inputs = x[:, :-1]
    targets = x[:, 1:].clone()
    pos = torch.arange(ms - 1, device=DEVICE)
    valid = (pos.unsqueeze(0) + 1) >= resp_start.unsqueeze(1)
    targets[~valid] = PAD_ID
    return inputs, targets


META = {"arch": ARCH, "dataset_hash": DATA_HASH, "pairs": len(pairs), "vocab": VOCAB_SIZE}
training_done = threading.Event()
if Path(MODEL_PATH).exists() and Path(META_PATH).exists():
    try:
        meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
        if (meta.get("arch") == ARCH
                and meta.get("dataset_hash") == DATA_HASH
                and meta.get("vocab") == VOCAB_SIZE):
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model"])
            model.eval()
            training_done.set()
    except Exception as exc:
        print("Old model ignored:", exc)

if not training_done.is_set():
    def _train_model():
        try:
            model.train()
            random.seed(42)
            for epoch in range(EPOCHS):
                random.shuffle(train_data)
                total = 0.0
                n = 0
                for st in range(0, len(train_data), BATCH_SIZE):
                    batch = train_data[st:st+BATCH_SIZE]
                    inputs, targets = make_batch(batch)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.item())
                    n += 1
                if n:
                    print(f"[train] epoch {epoch+1}/{EPOCHS} loss={total/n:.4f}")
            model.eval()
            torch.save({"model": model.state_dict()}, MODEL_PATH)
            Path(VOCAB_PATH).write_text("\n".join(f"{i}\t{w}" for i, w in id_to_word.items()), encoding="utf-8")
            Path(META_PATH).write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            training_done.set()

    threading.Thread(target=_train_model, daemon=True, name="model-training").start()


# ============================================================
# Conservative Transformer generation
# ============================================================

def generate_fallback(text):
    if not training_done.is_set():
        return None
    model.eval()
    prompt = [BOS_ID] + encode(text, MAX_INPUT_LEN) + [SEP_ID]
    generated = list(prompt)
    blocked = {PAD_ID, UNK_ID, BOS_ID, SEP_ID}
    with torch.no_grad():
        for _ in range(MAX_OUTPUT_LEN):
            ctx = generated[-MAX_SEQ_LEN:]
            logits = model(torch.tensor([ctx], dtype=torch.long, device=DEVICE))[0, -1].clone()
            for tid in blocked:
                logits[tid] = -float("inf")
            for tid in set(generated):
                if tid < logits.numel():
                    logits[tid] -= 2.0
            next_id = int(torch.argmax(logits).item())
            if next_id == EOS_ID:
                break
            generated.append(next_id)
    out = []
    for tid in generated[len(prompt):]:
        w = id_to_word.get(tid, "")
        if w in {PAD, UNK, BOS, EOS, SEP}:
            break
        out.append(w)
    text_out = normalize_text(" ".join(out))
    if len(text_out.split()) < 2:
        return None
    if len(set(text_out.split())) <= 2 and len(text_out.split()) >= 4:
        return None
    return text_out


# ============================================================
# Chat Session Manager
# ============================================================

class SessionManager:
    def __init__(self, sessions_dir):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.memories = {}

    def _session_path(self, session_id):
        safe_name = Path(session_id).name
        if not safe_name or safe_name.startswith("."):
            raise ValueError("Invalid session id")
        return self.sessions_dir / f"{safe_name}.json"

    def get_memory(self, session_id):
        if session_id not in self.memories:
            path = self.sessions_dir / f"{Path(session_id).name}.json"
            memory = {}
            if path.exists():
                try:
                    memory = json.loads(path.read_text(encoding="utf-8")).get("memory", {})
                except Exception:
                    memory = {}
            self.memories[session_id] = memory
        return self.memories[session_id]

    def list_sessions(self):
        sessions = []
        for f in sorted(self.sessions_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "id": f.stem,
                    "title": data.get("title", f.stem),
                    "created": data.get("created", ""),
                    "message_count": len(data.get("messages", []))
                })
            except Exception:
                continue
        return sessions

    def load_session(self, session_id):
        path = self._session_path(session_id)
        messages = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            self.memories[session_id] = data.get("memory", {})
        else:
            self.memories.setdefault(session_id, {})
        return messages

    def save_session(self, session_id, messages, title=None):
        path = self._session_path(session_id)
        created = ""
        if path.exists():
            try:
                created = json.loads(path.read_text(encoding="utf-8")).get("created", "")
            except Exception:
                created = ""
        data = {
            "title": title or (messages[0]["text"][:40] if messages else session_id),
            "created": created or datetime.datetime.now().isoformat(),
            "memory": self.memories.get(session_id, {}),
            "messages": messages
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def new_session(self):
        while True:
            session_id = f"chat_{uuid.uuid4().hex[:12]}"
            if session_id in self.memories:
                continue
            try:
                path = self._session_path(session_id)
            except ValueError:
                continue
            if path.exists():
                continue
            break
        self.memories[session_id] = {}
        return session_id

    def delete_session(self, session_id):
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        self.memories.pop(session_id, None)

session_mgr = SessionManager(SESSIONS_DIR)


# ============================================================
# Final answer router — purely neural, no scripted responses
# ============================================================

def answer(text):
    text = normalize_text(text)
    if not text:
        return "یه پیامی بنویس."

    generated = generate_fallback(text)
    if generated:
        return generated

    return "برای این سؤال هنوز جواب مطمئنی ندارم؛ اگر کمی واضح‌ترش کنی، بهتر می‌تونم کمک کنم."


# ============================================================
# Flask Web App
# ============================================================

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    return jsonify({"sessions": session_mgr.list_sessions()})

@app.route("/api/sessions", methods=["POST"])
def create_session():
    session_id = session_mgr.new_session()
    return jsonify({"session_id": session_id})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def load_session_data(session_id):
    messages = session_mgr.load_session(session_id)
    return jsonify({"messages": messages, "memory": session_mgr.get_memory(session_id)})

@app.route("/api/sessions/<session_id>", methods=["PUT"])
def save_session_data(session_id):
    data = request.get_json()
    memory = data.get("memory")
    if isinstance(memory, dict):
        session_mgr.memories[session_id] = memory
    session_mgr.save_session(session_id, data.get("messages", []), data.get("title"))
    return jsonify({"ok": True})

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session_data(session_id):
    session_mgr.delete_session(session_id)
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "یه پیامی بنویس."})

    response = answer(user_message)
    return jsonify({"response": response})


# ============================================================
# Launch with pywebview (dedicated window)
# ============================================================

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


if __name__ == "__main__":
    print(f"Pairs: {len(pairs)} | Vocab: {VOCAB_SIZE} | Device: {DEVICE}")

    port = int(os.environ.get("PERSIAN_AI_PORT", "0") or 0) or find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    # Start Flask in background thread
    flask_thread = threading.Thread(
        target=lambda: app.run(debug=False, host="127.0.0.1", port=port, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    if not wait_for_server(base_url):
        print("Warning: server did not respond in time; opening window anyway.")

    # Launch native window with pywebview
    try:
        import webview
        window = webview.create_window(
            "Persian AI",
            url=base_url,
            width=1000,
            height=750,
            min_size=(400, 500),
            resizable=True,
            background_color="#0a0f1a",
        )
        webview.start(gui="edgechromium")
    except ImportError:
        print("pywebview not installed. Opening in browser...")
        import webbrowser
        webbrowser.open(base_url)
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")
