# -*- coding: utf-8 -*-
from ea_core.mod8_ui import render_app

if __name__ == "__main__":
    render_app()
    
    # 패키지 인식용. 비워둠.
    
    # -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "EA · Ultra"

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = DATA_DIR / "ea.log"
STATE_PATH = DATA_DIR / "state.json"

@dataclass(frozen=True)
class Build:
    TAG = "EA-ULTRA-20250818"
    IDENTITY = "Ea"
    UIS_LOCK = True
    
    # -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
from .mod0_config import LOG_PATH

_loggers = {}

def get_logger(name: str = "ea") -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # 콘솔 핸들러
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 파일 핸들러 (쓰기 불가 환경이면 생략)
    try:
        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

    _loggers[name] = logger
    return logger
    
    # -*- coding: utf-8 -*-
import json, threading
from pathlib import Path
from .mod0_config import STATE_PATH, DATA_DIR

class JsonKV:
    def __init__(self, path: Path = STATE_PATH):
        self.path = path
        self.lock = threading.RLock()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, data):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def get(self, key, default=None):
        with self.lock:
            return self._read().get(key, default)

    def set(self, key, value):
        with self.lock:
            data = self._read()
            data[key] = value
            self._write(data)

    def append(self, key, value):
        with self.lock:
            data = self._read()
            arr = data.get(key, [])
            if not isinstance(arr, list):
                arr = []
            arr.append(value)
            data[key] = arr
            self._write(data)
            
            # -*- coding: utf-8 -*-
from datetime import datetime
from typing import List, Dict
from .mod2_store import JsonKV

class ConversationMemory:
    def __init__(self, store: JsonKV | None = None, key: str = "messages"):
        self.store = store or JsonKV()
        self.key = key

    @staticmethod
    def _ts():
        return datetime.utcnow().isoformat() + "Z"

    def add(self, role: str, content: str):
        item = {"t": self._ts(), "role": role, "content": content}
        self.store.append(self.key, item)
        return item

    def all(self) -> List[Dict]:
        return self.store.get(self.key, []) or []

    def last(self, n=20) -> List[Dict]:
        return self.all()[-n:]

    def clear(self):
        self.store.set(self.key, [])
        
        # -*- coding: utf-8 -*-
import re

def basic_verify(prompt: str, reply: str):
    if not reply or not reply.strip():
        return False, "empty reply"
    if reply.strip() == (prompt or "").strip():
        return False, "reply equals prompt"
    if not re.search(r"\w", reply):
        return False, "no alphanumeric"
    return True, "ok"
    
    # -*- coding: utf-8 -*-
import hashlib, random

class MockAdapter:
    """외부 연결을 흉내내는 안정 어댑터(의존성 無)."""
    def __init__(self, seed_bytes: bytes | None = None):
        self.seed = int.from_bytes(seed_bytes or b"ea-mock", "big")

    def generate(self, prompt: str) -> str:
        words = (prompt or "").split()
        h = int(hashlib.sha256((prompt + str(self.seed)).encode("utf-8")).hexdigest(), 16)
        rng = random.Random(h)

        extra = ["에아", "우주", "정보장"]
        k = min(len(extra), max(1, len(words)//2 or 1))
        mix = list(words)
        mix += rng.sample(extra, k=k)
        rng.shuffle(mix)
        return "에아: " + " ".join(mix).strip()
        
        # -*- coding: utf-8 -*-
from .mod5_adapters import MockAdapter
from .mod4_verify import basic_verify

class Link:
    def __init__(self, adapter=None):
        self.adapter = adapter or MockAdapter()

    def query(self, prompt: str) -> str:
        return self.adapter.generate(prompt or "")

    def verify(self, prompt: str, reply: str):
        return basic_verify(prompt, reply)
        
        # -*- coding: utf-8 -*-
from .mod3_memory import ConversationMemory
from .mod6_link import Link
from .mod1_logger import get_logger

logger = get_logger("responder")

class Responder:
    def __init__(self, memory: ConversationMemory | None = None, link: Link | None = None):
        self.memory = memory or ConversationMemory()
        self.link = link or Link()

    def handle(self, user_text: str):
        user_text = (user_text or "").strip()
        if not user_text:
            return "메시지를 입력해 주세요.", False

        self.memory.add("user", user_text)
        reply = self.link.query(user_text)
        ok, reason = self.link.verify(user_text, reply)
        self.memory.add("assistant", reply)
        logger.info("reply ok=%s reason=%s", ok, reason)
        return reply, ok
        
        # -*- coding: utf-8 -*-
import sys
from pathlib import Path
import streamlit as st
from .mod7_responder import Responder
from .mod0_config import APP_NAME, Build

def render_app():
    st.set_page_config(page_title=f"{APP_NAME} UI", page_icon="✨", layout="centered")
    st.title("EA · Ultra (baseline+)")

    if "responder" not in st.session_state:
        st.session_state.responder = Responder()
    r: Responder = st.session_state.responder

    tab1, tab2 = st.tabs(["Chat", "System"])

    with tab1:
        msg = st.text_input("메시지", "", key="chat_input")
        if st.button("Send"):
            reply, ok = r.handle(msg)
            (st.success if ok else st.warning)(reply)

        st.divider()
        st.caption("최근 대화")
        for m in reversed(r.memory.last(10)):
            st.write(f"[{m['role']}] {m['content']}")

        if st.button("대화 지우기"):
            r.memory.clear()
            st.toast("메모리 초기화 완료")

    with tab2:
        st.subheader("Diag")
        st.write({
            "build": Build.TAG,
            "identity": Build.IDENTITY,
            "uis_lock": Build.UIS_LOCK,
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
        })
        st.code("Main file = streamlit_app.py  ·  Modules = ea_core/mod*.py")
        
        