# -*- coding: utf-8 -*-
# EA · Ultra — ChatGPT-like UI / API auto-detect / response-level control
import os, sys, re, json, time, hashlib, random, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st

APP_NAME   = "EA · Ultra"
BUILD_TAG  = "EA-ULTRA-20250818"
DATA_DIR   = Path("data")
STATE_PATH = DATA_DIR / "state.json"

def ensure_dirs():
    try: DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def nowz(): return datetime.utcnow().isoformat()+"Z"

# ---------------- State (file -> session fallback) ----------------
def _state_read():
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return st.session_state.get("_state", {})

def _state_write(obj):
    try:
        ensure_dirs()
        tmp = STATE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"] = obj

def sget(k, d=None): return _state_read().get(k, d)
def sset(k, v): s = _state_read(); s[k] = v; _state_write(s)

def add_msg(role, content):
    msgs = sget("messages", [])
    msgs.append({"t": nowz(), "role": role, "content": content})
    sset("messages", msgs)

def clear_msgs(): sset("messages", [])

# ---------------- Utilities ----------------
def dedupe(text: str):
    # 연속 중복 토큰 제거 (나나나/너너너 패턴 방지)
    text = re.sub(r'(.)\1{2,}', r'\1', text)          # 글자 반복
    text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text) # 단어 반복
    return text

def clamp_len(text: str, n=4096):
    return text if len(text) <= n else text[:n] + " …"

# ---------------- Adapters ----------------
class MockAdapter:
    name = "Mock"
    def generate(self, prompt, max_tokens=600):
        words = (prompt or "").split()
        seed  = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        rng   = random.Random(seed)
        filler = ["핵심만", "간단히", "정리하면", "포인트:", "한줄요약:"]
        pre = rng.choice(filler)
        body = " ".join(words[: max(8, len(words))])
        return f"{pre} {body}".strip()

class OpenAIAdapter:
    name = "OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client = OpenAI(api_key=key)
        self.model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    def generate(self, prompt, max_tokens=600):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=0.7
        )
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name = "Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key = os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=600):
        r = self.model.generate_content(
            prompt, generation_config={"temperature":0.7, "max_output_tokens":max_tokens}
        )
        return getattr(r, "text", "") or ""

def resolve_adapter(want: str):
    if want == "OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가 → Mock: {e}", icon="⚠️")
    if want == "Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가 → Mock: {e}", icon="⚠️")
    return MockAdapter(), False

# ---------------- Multi-round (long answer) ----------------
def long_answer(adapter, prompt, level=3, rounds=2):
    # 레벨→토큰 할당
    token_map = {1:300, 2:600, 3:900, 4:1300, 5:1800}
    max_tokens = token_map.get(int(level), 900)
    acc = ""
    base = prompt.strip()
    for i in range(int(rounds)):
        p = base if i == 0 else base + "\n(이어서 자세히 계속)"
        chunk = dedupe(adapter.generate(p, max_tokens=max_tokens))
        if not chunk: break
        acc += (("\n\n" if acc else "") + clamp_len(chunk, n=max_tokens+500))
        time.sleep(0.05)
    return acc.strip()

# ---------------- UI ----------------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("ChatGPT 스타일 UI · API 자동연동 · 응답 레벨/길이 제어")

    # ---- top controls
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        provider = st.selectbox("Provider", ["OpenAI","Gemini","Mock"], index=0)
    with colB:
        level = st.slider("응답 레벨", 1, 5, 3, help="레벨↑ = 더 길고 자세함")
    with colC:
        rounds = st.number_input("연결 라운드", 1, 6, 2, step=1, help="긴 답변을 여러 번 이어받기")

    adapter, api_ok = resolve_adapter(provider)
    status = f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · L{level} · R{int(rounds)}"
    st.info(status)

    # ---- chat history
    if "messages" not in st.session_state: sset("messages", [])
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])

    # ---- input
    user_text = st.chat_input("메시지를 입력하고 Enter")
    if user_text:
        user_text = dedupe(user_text.strip())
        add_msg("user", user_text)
        with st.chat_message("user"): st.markdown(user_text)

        with st.chat_message("assistant"):
            try:
                ans = long_answer(adapter, user_text, level=level, rounds=rounds)
            except Exception:
                ans = "(내부 오류)\n\n" + "```\n" + traceback.format_exc() + "\n```"
            st.markdown(ans)
        add_msg("assistant", ans)

    # tools
    with st.expander("도구"):
        c1, c2, c3 = st.columns(3)
        if c1.button("대화 초기화"): clear_msgs(); st.experimental_rerun()
        if c2.button("요약 보기"):
            msgs = sget("messages", [])[-8:]
            summ = " / ".join(f"{m['role']}: {m['content']}" for m in msgs)
            st.success(summ or "기록이 거의 없어요.")
        st.code(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH}", language="text")

# ---------------- entry ----------------
if __name__ == "__main__":
    render_app()
    