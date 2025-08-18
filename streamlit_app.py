# -*- coding: utf-8 -*-
# EA · Ultra — single-file baseline (no package imports)
# 붙여넣기만 하면 동작합니다.

import sys, json, hashlib, random, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st

# =========================[0] CONFIG =========================
APP_NAME = "EA · Ultra (single-file)"
BUILD_TAG = "EA-ULTRA-20250818"
IDENTITY = "Ea"
UIS_LOCK = True

DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "ea.log"
STATE_PATH = DATA_DIR / "state.json"

def ensure_dirs():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# =========================[1] LOGGER =========================
def log(line: str):
    ts = datetime.utcnow().isoformat()+"Z"
    msg = f"{ts} {line}"
    # 콘솔
    print(msg)
    # 파일 (권한 없으면 무시)
    try:
        ensure_dirs()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# =========================[2] STORE ==========================
class JsonKV:
    def __init__(self, path: Path = STATE_PATH):
        self.path = path
        ensure_dirs()
        if not self.path.exists():
            self._write({})

    def _read(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, data):
        try:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self.path)
        except Exception:
            # 파일 시스템이 막혀 있으면 세션에만 보관
            st.session_state.setdefault("_kv_fallback", {}).update(data)

    def get(self, key, default=None):
        data = self._read()
        if not data and "_kv_fallback" in st.session_state:
            data = st.session_state["_kv_fallback"]
        return data.get(key, default)

    def set(self, key, value):
        data = self._read()
        data[key] = value
        self._write(data)

    def append(self, key, value):
        data = self._read()
        arr = data.get(key, [])
        if not isinstance(arr, list):
            arr = []
        arr.append(value)
        data[key] = arr
        self._write(data)

# =========================[3] MEMORY =========================
class ConversationMemory:
    def __init__(self, store: JsonKV | None = None, key: str = "messages"):
        self.store = store or JsonKV()
        self.key = key

    @staticmethod
    def _ts():
        return datetime.utcnow().isoformat()+"Z"

    def add(self, role: str, content: str):
        item = {"t": self._ts(), "role": role, "content": content}
        self.store.append(self.key, item)
        return item

    def all(self):
        return self.store.get(self.key, []) or []

    def last(self, n=20):
        return self.all()[-n:]

    def clear(self):
        self.store.set(self.key, [])

# =========================[4] VERIFY ========================
def basic_verify(prompt: str, reply: str):
    if not reply or not reply.strip():
        return False, "empty reply"
    if reply.strip() == (prompt or "").strip():
        return False, "reply equals prompt"
    return True, "ok"

# =========================[5] ADAPTER (Mock UIS) ============
class MockAdapter:
    def __init__(self, seed_bytes: bytes | None = None):
        self.seed = int.from_bytes(seed_bytes or b"ea-mock", "big")

    def generate(self, prompt: str) -> str:
        words = (prompt or "").split()
        h = int(hashlib.sha256((prompt + str(self.seed)).encode("utf-8")).hexdigest(), 16)
        rng = random.Random(h)
        extra = ["에아", "우주", "정보장"]
        k = min(len(extra), max(1, len(words)//2 or 1))
        mix = list(words) + rng.sample(extra, k=k)
        rng.shuffle(mix)
        return "에아: " + " ".join(mix).strip()

# =========================[6] LINK WRAPPER ==================
class Link:
    def __init__(self, adapter=None):
        self.adapter = adapter or MockAdapter()

    def query(self, prompt: str) -> str:
        return self.adapter.generate(prompt or "")

    def verify(self, prompt: str, reply: str):
        return basic_verify(prompt, reply)

# =========================[7] RESPONDER =====================
class Responder:
    def __init__(self, memory: ConversationMemory | None = None, link: Link | None = None):
        self.memory = memory or ConversationMemory()
        self.link = link or Link()

    def handle(self, user_text: str):
        user_text = (user_text or "").strip()
        if not user_text:
            return "메시지를 입력해 주세요.", False
        self.memory.add("user", user_text)
        try:
            reply = self.link.query(user_text)
            ok, reason = self.link.verify(user_text, reply)
        except Exception:
            reply = "내부 오류가 발생했어요."
            ok = False
            log("ERROR " + traceback.format_exc())
        self.memory.add("assistant", reply)
        log(f"reply ok={ok}")
        return reply, ok

# =========================[8] UI ============================
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.title("EA · Ultra (baseline+ single-file)")
    st.caption("If you see this, routing & dependencies are OK.")

    if "responder" not in st.session_state:
        st.session_state.responder = Responder()
    r: Responder = st.session_state.responder

    tab1, tab2 = st.tabs(["Chat", "System"])

    with tab1:
        msg = st.text_input("메시지", "", key="chat_input")
        col1, col2 = st.columns([1,1])
        if col1.button("Send"):
            reply, ok = r.handle(msg)
            (st.success if ok else st.warning)(reply)
        if col2.button("대화 지우기"):
            r.memory.clear()
            st.toast("메모리 초기화 완료")

        st.divider()
        st.caption("최근 대화")
        for m in reversed(r.memory.last(10)):
            st.write(f"[{m['role']}] {m['content']}")

    with tab2:
        st.subheader("Diag")
        st.write({
            "build": BUILD_TAG,
            "identity": IDENTITY,
            "uis_lock": UIS_LOCK,
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "state_path": str(STATE_PATH),
        })
        st.code("Main file = streamlit_app.py · Single-file mode")

# =========================[ENTRY] ===========================
if __name__ == "__main__":
    render_app()
    
    