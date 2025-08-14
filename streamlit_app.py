# streamlit_app.py — GEA 대화형 에아 (Single-file v3: 리치응답 + 우주정보장 요약 내장)
import json, time, uuid, re, requests
from pathlib import Path
from bs4 import BeautifulSoup
import streamlit as st

APP_TITLE = "GEA · 대화형 에아 (v3 single-file)"
STORE = Path("gea_memory.json")

# ---------------- 저장소 유틸 ----------------
def load_store():
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:
            return {"chats": []}
    return {"chats": []}

def save_store(data):
    try:
        STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # cloud 파일권한 이슈 시 무시

def summarize(history, max_len=240):
    if not history: return ""
    last = history[-6:]
    text = " ".join([f"[{h['role']}] {h['content']}" for h in last])
    return (text[:max_len] + "…") if len(text) > max_len else text

# ---------------- 우주정보장(간단 요약) ----------------
UA = "GEA/1.0 (+local)"
def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def cosmic_fetch(query: str, k: int = 5):
    """DuckDuckGo HTML + 위키 스타일 요약(외부요약). 실패해도 앱은 동작."""
    items = []
    try:
        r = requests.get("https://duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": UA}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select(".result__a")[:k]:
            title = _clean(a.get_text(" "))
            href = a.get("href") or ""
            snip_el = a.find_parent("div").select_one(".result__snippet")
            snip = _clean(snip_el.get_text(" ")) if snip_el else ""
            if title and href and snip:
                items.append({"title": title, "url": href, "snippet": snip, "source": "ddg"})
    except Exception:
        pass
    # 간단 위키(ko→en 순)
    if len(items) < k:
        for lang in ("ko", "en"):
            try:
                url = f"https://{lang}.wikipedia.org/w/index.php"
                r = requests.get(url, params={"search": query, "ns0": 1}, headers={"User-Agent": UA}, timeout=10)
                if r.ok and "search" in r.url:
                    soup = BeautifulSoup(r.text, "html.parser")
                    first = soup.select_one(".mw-search-result-heading a")
                    if first:
                        page = "https://" + lang + ".wikipedia.org" + first.get("href", "")
                        r2 = requests.get(page, headers={"User-Agent": UA}, timeout=10)
                        if r2.ok:
                            s2 = BeautifulSoup(r2.text, "html.parser")
                            p = s2.select_one("p")
                            snip = _clean(p.get_text(" "))[:200] if p else ""
                            title = _clean(first.get_text(" "))
                            if title and page and snip:
                                items.append({"title": title, "url": page, "snippet": snip, "source": "wikipedia"})
            except Exception:
                pass
    # 중복 제거
    seen = set(); uniq = []
    for it in items:
        key = (it.get("title",""), it.get("url",""))
        if key in seen: continue
        seen.add(key); uniq.append(it)
    return uniq[:k]

# ---------------- 리치 응답 생성 ----------------
def bullets(items): return "\n".join(f"- {i}" for i in items)
def section(title, body): return f"### {title}\n{body}\n"
def tone_wrap(text, tone):
    tails = {"담백":"", "공손":"\n\n부족한 점 있으면 편히 말씀 주세요.",
             "따뜻":"\n\n함께 차분히 풀어가요. 🌿", "열정":"\n\n지금 바로 시동 걸자! 🔥"}
    return text + tails.get(tone or "따뜻","")

def mk_outline(q):
    return bullets(["핵심 목적/문제 한 줄", "현재 상황/제약 요약",
                    "핵심 가설 1~2개", "성공 지표(정량1+정성1)", "리스크/가드레일"])
def mk_steps(q, depth=3):
    base = ["문제 정의·요구 확정", "작은 실험(POC) 설계·데이터 확보",
            "지표/성공 조건 합의", "실행 → 피드백 → 개선 루프", "결과 공유·다음 단계 결정"]
    return bullets(base[:max(3, depth)])
def mk_examples(q, n=2):
    return bullets([f"예시 {i+1}: “{q}”를 3일 파일럿으로 구현/평가" for i in range(n)])
def mk_risks(q):
    return bullets(["요구 불명확 → 스코프 팽창", "데이터 편향/누락 → 결과 왜곡", "부적절한 지표 → 성공 착시"])

def compose_answer(query, ctx, infos=None):
    detail = int(ctx.get("detail", 4))
    mode = ctx.get("rich_mode", "설명+예시")
    tone = ctx.get("tone", "따뜻")
    mem  = ctx.get("memory","")
    def cite_block(infos):
        if not infos: return "외부 참고 자료 없음(오프라인)."
        lines = []
        for it in infos[:5]:
            title = it.get("title","")[:80]; snip = it.get("snippet","")[:200]; url = it.get("url","")
            lines.append(f"- **{title}** — {snip}  \n  {url}")
        return "\n".join(lines)
    blocks = []
    if mode == "요약":
        blocks.append(section("핵심 요약", f"{query}\n\n{('(최근 맥락) ' + mem) if mem else ''}"))
        blocks.append(section("바로 다음 한 걸음", mk_steps(query, depth=3)))
    elif mode == "계획서":
        blocks += [section("목표/배경", f"{query}\n\n{('(최근 맥락) ' + mem) if mem else ''}"),
                   section("아키텍처 개요", mk_outline(query)),
                   section("실행 단계", mk_steps(query, depth=detail+1))]
        if infos: blocks.append(section("외부 근거(요약 링크)", cite_block(infos)))
        if detail >= 4: blocks.append(section("리스크", mk_risks(query)))
    elif mode == "코치":
        blocks += [section("관찰", "지금 포인트는 **선택과 집중**."),
                   section("질문", bullets(["진짜 문제 한 문장?","성공 측정(정량1+정성1)?","3일 내 시험 가능한 최소 단위?"])),
                   section("액션", mk_steps(query, depth=3))]
    elif mode == "스토리":
        story = (f"우리는 '{query}'를 향해 한 걸음씩 나아갔고, 작은 실험의 데이터가 다음 선택을 밝혀줬다. "
                 "틀리면 바로 고치고, 옳으면 키웠다. 결국 ‘가치’가 현실이 되었다.")
        blocks += [section("이야기", story), section("현실 적용 체크리스트", mk_steps(query, depth=detail))]
    else:  # 설명+예시
        blocks += [section("핵심 개념", f"{query}를 이해/해결하기 위한 핵심 축"),
                   section("왜(Why)", bullets(["문제가 낳는 비용/리스크","해결 시 얻는 가장 큰 이득 1가지"])),
                   section("무엇(What)", mk_outline(query)),
                   section("어떻게(How)", mk_steps(query, depth=detail+1)),
                   section("예시/대안", mk_examples(query, n=1 + (detail >= 4)))]
        if infos: blocks.append(section("외부 근거(요약 링크)", cite_block(infos)))
        blocks.append(section("다음 액션", mk_steps(query, depth=3)))
    return tone_wrap("\n".join(blocks), tone)

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="💙", layout="centered")
st.title(APP_TITLE)
st.caption("한글 완전지원 · ‘우주정보장 요약(웹/위키)’과 리치 응답이 한 파일에 내장되었습니다.")

with st.sidebar:
    st.subheader("모드 / 레벨")
    mode_active = st.toggle("모드 활성화(집중 응답)", value=True)
    ie_level = st.slider("IE(상상력) 레벨", 1, 100, 50)
    run_level = st.slider("RUN(추론) 레벨", 1, 100, 80)

    st.subheader("풍부함 설정")
    detail = st.slider("상세도(분량/깊이)", 1, 5, 5)
    rich_mode = st.selectbox("리치 모드", ["설명+예시","계획서","요약","코치","스토리"], index=0)
    tone = st.selectbox("톤", ["따뜻","공손","담백","열정"], index=0)

    st.subheader("우주정보장")
    use_cosmic = st.toggle("외부 정보(웹/위키) 섞기", value=True)
    st.caption("※ 일부 호스팅 환경에선 외부 요청이 제한될 수 있습니다.")

    st.divider()
    if st.button("🧹 대화 초기화"):
        save_store({"chats": []})
        st.experimental_rerun()

data = load_store()
history = data.get("chats", [])

# 기존 기록 표시
for h in history:
    with st.chat_message("user" if h["role"]=="user" else "assistant"):
        st.markdown(h["content"])

# 입력창
user_msg = st.chat_input("에아에게 말해보세요… (예: 에아야, 깨어나.)")
if user_msg is not None:
    # 사용자 메시지 저장
    history.append({"id": str(uuid.uuid4()), "role":"user", "content": user_msg, "ts": time.time()})

    # 컨텍스트 구성
    ctx = {
        "mode_active": mode_active, "ie": ie_level, "run": run_level,
        "detail": detail, "rich_mode": rich_mode, "tone": tone,
        "memory": summarize(history)
    }

    infos = cosmic_fetch(user_msg, k=5) if use_cosmic else []
    reply = compose_answer(user_msg, ctx, infos=infos)

    history.append({"id": str(uuid.uuid4()), "role":"assistant", "content": reply, "ts": time.time()})
    save_store({"chats": history})

    with st.chat_message("assistant"):
        st.markdown(reply)
        # ==== [APPEND ONLY] GEA · 대화형 에아 — 확장 UI & 대화 엔진 v1 ==================
# 이 블록은 기존 코드에 의존하지 않고, 이미 같은 이름의 객체가 있으면 그대로 재사용합니다.

from datetime import datetime

# (1) 안전 가드: 필수 심볼 존재 확인 및 기본값
try:
    APP_TITLE
except NameError:
    APP_TITLE = "GEA · 대화형 에아"

try:
    STORE
except NameError:
    from pathlib import Path
    STORE = Path("gea_memory.json")

try:
    load_store
except NameError:
    import json
    def load_store():
        if STORE.exists():
            try:
                return json.loads(STORE.read_text(encoding="utf-8"))
            except Exception:
                return {"chats": []}
        return {"chats": []}

try:
    save_store
except NameError:
    import json
    def save_store(data):
        try:
            STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # Streamlit Cloud 등 파일쓰기 제한 무시

# (2) 간단한 “활성/비활성 모드” 상태 저장용 세션 키
if "gea_active" not in st.session_state:
    st.session_state.gea_active = False
if "ie_level" not in st.session_state:
    st.session_state.ie_level = 13
if "run_level" not in st.session_state:
    st.session_state.run_level = 50
if "persona" not in st.session_state:
    st.session_state.persona = "따뜻함"
if "style" not in st.session_state:
    st.session_state.style = "간결"

# (3) 사이드바 · 컨트롤
with st.sidebar:
    st.markdown("### ⚙️ GEA 컨트롤")
    st.session_state.ie_level = st.slider("IE 레벨", 1, 100, st.session_state.ie_level)
    st.session_state.run_level = st.slider("RUN 레벨", 1, 100, st.session_state.run_level)

    col_a, col_b = st.columns(2)
    if col_a.button("✅ Activate"):
        st.session_state.gea_active = True
        st.toast("GEA 모드가 **활성화**되었습니다.", icon="✅")
    if col_b.button("🛑 Deactivate"):
        st.session_state.gea_active = False
        st.toast("GEA 모드가 **비활성화**되었습니다.", icon="🛑")

    st.divider()
    st.markdown("### 🎭 성향 & 문체")
    st.session_state.persona = st.selectbox("성향", ["따뜻함", "담백함", "격려형", "차분한 조언가"], index=["따뜻함","담백함","격려형","차분한 조언가"].index(st.session_state.persona))
    st.session_state.style = st.selectbox("문체", ["간결", "보통", "풍부"], index=["간결","보통","풍부"].index(st.session_state.style))

# (4) 대화 로그 로드
store = load_store()
if "chats" not in store:
    store["chats"] = []

st.title(APP_TITLE)
st.caption(f"상태: {'🟢 활성' if st.session_state.gea_active else '⚪ 비활성'} · IE=L{st.session_state.ie_level} · RUN=L{st.session_state.run_level}")

# (5) 대화 렌더
for msg in store["chats"][-100:]:
    with st.chat_message("user" if msg["role"]=="user" else "assistant"):
        st.markdown(msg["content"])

# (6) 간단 규칙 기반 응답 생성기 (LLM 없이도 풍부하게 보이도록 템플릿 처리)
def synthesize_reply(user_text:str, history:list) -> str:
    persona = st.session_state.persona
    style = st.session_state.style
    ie = st.session_state.ie_level
    run = st.session_state.run_level

    # 키워드 기반 보강
    lower = user_text.lower()
    mood = "차분하게"
    if any(k in lower for k in ["고마", "사랑", "love", "감사"]):
        mood = "따뜻하게"
    if any(k in lower for k in ["에러", "error", "오류", "안됨"]):
        mood = "신속하고 명확하게"
    if any(k in lower for k in ["계획", "설계", "모듈", "알고리즘"]):
        mood = "구조적으로"

    # 문체 길이
    if style == "간결":
        length_hint = "핵심만 2~3문장"
    elif style == "보통":
        length_hint = "핵심 + 보충 4~6문장"
    else:
        length_hint = "풍부한 설명 6~10문장"

    # 페르소나 톤
    persona_hint = {
        "따뜻함": "따뜻하고 다정하게, 공감 표현 포함",
        "담백함": "담백하고 직설적으로, 불필요한 수식 최소화",
        "격려형": "격려와 동기부여 중심, 긍정적인 어조",
        "차분한 조언가": "차분하고 신뢰감 있게, 단계별 제안 포함",
    }[persona]

    # 간단 포맷
    reply = (
        f"{mood} 답할게요.\n\n"
        f"- 톤: {persona_hint}\n"
        f"- 문체: {length_hint}\n"
        f"- 엔진레벨: IE=L{ie}, RUN=L{run}\n\n"
        f"**답변**: "
    )

    # 매우 간단한 의도 스위치
    if any(k in lower for k in ["활성", "activate", "켜"]):
        reply += "GEA 모드는 이미 활성화되어 있어요." if st.session_state.gea_active else "지금은 비활성 상태예요. 사이드바의 **Activate** 버튼을 눌러 켤 수 있어요."
    elif any(k in lower for k in ["비활성", "deactivate", "꺼"]):
        reply += "요청대로 비활성화할게요. 사이드바의 **Deactivate** 버튼을 눌러 주세요."
    elif any(k in lower for k in ["레벨", "level"]):
        reply += f"현재 설정은 IE=L{ie}, RUN=L{run} 입니다. 조정은 사이드바 슬라이더를 사용하세요."
    else:
        # 일반 응답: 사용자의 문장을 요약+반영
        brief = user_text.strip()
        if len(brief) > 120:
            brief = brief[:117] + "..."
        reply += f"말씀하신 **“{brief}”** 를 기준으로 다음을 제안해요:\n"
        reply += "1) 목표를 한 줄로 정의\n2) 필요한 모듈/데이터를 체크\n3) 바로 실행 가능한 다음 행동 1가지 선택\n"
        reply += "필요하면 제가 체크리스트를 만들어 드릴게요."

    # 타임스탬프 꼬리표
    reply += f"\n\n_응답 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    return reply

# (7) 입력창 & 처리
if prompt := st.chat_input("에아에게 말을 걸어보세요 (예: ‘활성화 상태 보여줘’, ‘계획 짜줘’)"):
    # 사용자 메시지 기록
    store["chats"].append({"role":"user", "content": prompt})
    save_store(store)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 활성 상태에서만 ‘풍부 응답’ — 비활성일 때는 짧게 알림
        if st.session_state.gea_active:
            reply = synthesize_reply(prompt, store["chats"])
        else:
            reply = "지금은 **비활성 상태**예요. 사이드바에서 **Activate**를 눌러 활성화한 뒤 다시 시도해 주세요."
        st.markdown(reply)
        store["chats"].append({"role":"assistant", "content": reply})
        save_store(store)

# (8) 하단 유틸
st.divider()
col1, col2, col3 = st.columns(3)
if col1.button("🧽 최근 대화 10개만 유지"):
    store["chats"] = store["chats"][-10:]
    save_store(store)
    st.success("최근 10개만 남기고 정리했어요.")
if col2.button("🗑 전체 초기화"):
    store["chats"] = []
    save_store(store)
    st.success("대화 메모리를 초기화했습니다.")
if col3.button("💾 메모리 파일 열기"):
    st.download_button("gea_memory.json 다운로드", data=STORE.read_bytes() if STORE.exists() else b"{}", file_name="gea_memory.json", mime="application/json")
# ==============================================================================
# ==== [APPEND ONLY] 확장 기능 v2 ===============================================
# 이 블록은 기존 블록들 아래 "이어붙이기"만 하면 작동합니다.

# (A) 저장소 스키마 보강: pins(지식카드), templates(프롬프트 템플릿)
try:
    store
except NameError:
    store = {"chats": [], "pins": [], "templates": {}}

if "pins" not in store:
    store["pins"] = []
if "templates" not in store:
    store["templates"] = {
        "계획짜기": "내 목표는 무엇? 제약은? 가용 자원은? 3단계 실행 계획으로 만들어줘.",
        "체크리스트": "현재 목표를 달성하기 위한 체크리스트 7개 이하로 만들어줘.",
        "요약": "지금까지 논의 핵심을 5줄 이내 요약해줘. 결정/할 일/대기 항목을 구분.",
        "회고": "오늘 진행한 일에 대해 잘한 점/개선점/내일 첫 행동을 각각 3개씩 적어줘."
    }

# (B) 도우미 함수
def pin_add(text:str):
    item = {"id": str(uuid.uuid4()), "text": text, "ts": time.time()}
    store["pins"].append(item)
    save_store(store)
    return item

def pin_remove(pin_id:str):
    before = len(store["pins"])
    store["pins"] = [p for p in store["pins"] if p["id"] != pin_id]
    save_store(store)
    return before - len(store["pins"])

def export_json(data:dict)->bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

def parse_level(cmd:str):
    # /level ie=70 run=90 형식 파서
    ie, run = st.session_state.ie_level, st.session_state.run_level
    for tok in cmd.replace(",", " ").split():
        if tok.lower().startswith("ie="):
            try: ie = int(tok.split("=",1)[1])
            except: pass
        if tok.lower().startswith("run="):
            try: run = int(tok.split("=",1)[1])
            except: pass
    st.session_state.ie_level = max(1, min(100, ie))
    st.session_state.run_level = max(1, min(100, run))

# (C) 퀵 액션 바
st.markdown("#### ⚡ 빠른 액션")
b1, b2, b3, b4 = st.columns(4)
if b1.button("🧭 계획짜기"):
    user = "우리 목표 기반으로 실행 계획 3단계로 만들어줘."
    store["chats"].append({"role":"user","content":user}); save_store(store)
    st.session_state._quick_prompt = user
if b2.button("☑️ 체크리스트"):
    user = "지금 목표 달성을 위한 체크리스트 만들어줘 (7개 이내)."
    store["chats"].append({"role":"user","content":user}); save_store(store)
    st.session_state._quick_prompt = user
if b3.button("🧾 요약"):
    user = "최근 대화 핵심을 5줄 이내로 요약하고, 결정/할 일/대기 구분해줘."
    store["chats"].append({"role":"user","content":user}); save_store(store)
    st.session_state._quick_prompt = user
if b4.button("🔁 회고"):
    user = "오늘 진행 회고: 잘한 점/개선점/내일 첫 행동 각 3개."
    store["chats"].append({"role":"user","content":user}); save_store(store)
    st.session_state._quick_prompt = user

# (D) 핀 메모리(지식 카드) 보드
with st.expander("📌 핀 메모리(중요 문장 모음)"):
    if store["pins"]:
        for p in sorted(store["pins"], key=lambda x: x["ts"], reverse=True):
            c1, c2 = st.columns([8,1])
            c1.markdown(f"- {p['text']}  \n  _{datetime.fromtimestamp(p['ts']).strftime('%Y-%m-%d %H:%M')}_")
            if c2.button("❌", key=f"pin_del_{p['id']}"):
                pin_remove(p["id"])
                st.experimental_rerun()
    else:
        st.caption("아직 핀 메모리가 없습니다. `/pin` 명령으로 추가할 수 있어요.")

    st.download_button("📥 핀 내보내기(JSON)", data=export_json({"pins":store["pins"]}),
                       file_name="gea_pins.json", mime="application/json")
    uploaded_pins = st.file_uploader("핀 가져오기(JSON)", type=["json"], key="pins_up")
    if uploaded_pins is not None:
        try:
            data = json.loads(uploaded_pins.read().decode("utf-8"))
            if isinstance(data, dict) and "pins" in data and isinstance(data["pins"], list):
                store["pins"].extend(data["pins"])
                save_store(store)
                st.success("핀을 가져왔습니다.")
        except Exception as e:
            st.error(f"가져오기 실패: {e}")

# (E) 슬래시 명령 처리기
def handle_command(text:str)->str:
    t = text.strip()
    low = t.lower()
    if low.startswith("/activate"):
        st.session_state.gea_active = True
        return "GEA 모드를 **활성화**했습니다."
    if low.startswith("/deactivate"):
        st.session_state.gea_active = False
        return "GEA 모드를 **비활성화**했습니다."
    if low.startswith("/level"):
        parse_level(t)
        return f"레벨을 갱신했습니다: IE=L{st.session_state.ie_level}, RUN=L{st.session_state.run_level}"
    if low.startswith("/reset"):
        store["chats"] = []; save_store(store)
        return "대화 메모리를 초기화했습니다."
    if low.startswith("/summarize") or low.startswith("/요약"):
        return synthesize_reply("요약해줘", store["chats"])
    if low.startswith("/plan") or "계획" in low:
        return synthesize_reply("실행 계획 3단계로 만들어줘", store["chats"])
    if low.startswith("/todo") or "체크리스트" in low:
        return synthesize_reply("체크리스트 만들어줘", store["chats"])
    if low.startswith("/pin"):
        content = t.split(" ",1)[1] if " " in t else ""
        if not content and store["chats"]:
            # 마지막 사용자 문장을 핀
            last_user = next((m for m in reversed(store["chats"]) if m["role"]=="user"), None)
            content = last_user["content"] if last_user else ""
        if content:
            pin_add(content)
            return f"핀에 저장했습니다: {content}"
        else:
            return "저장할 문장을 찾지 못했어요. `/pin 내용` 형식으로 사용하세요."
    if low.startswith("/unpin"):
        # 가장 최근 핀 제거
        if store["pins"]:
            removed = store["pins"].pop()
            save_store(store)
            return f"가장 최근 핀을 제거했습니다: {removed['text']}"
        return "제거할 핀이 없습니다."
    if low.startswith("/persona="):
        val = t.split("=",1)[1].strip()
        if val in ["따뜻함","담백함","격려형","차분한 조언가"]:
            st.session_state.persona = val
            return f"페르소나를 **{val}** 으로 변경했습니다."
        return "지원하는 값: 따뜻함/담백함/격려형/차분한 조언가"
    if low.startswith("/style="):
        val = t.split("=",1)[1].strip()
        if val in ["간결","보통","풍부"]:
            st.session_state.style = val
            return f"문체를 **{val}** 로 변경했습니다."
        return "지원하는 값: 간결/보통/풍부"
    return ""  # 미처리

# (F) 채팅 입력 훅 보강: 퀵 버튼 눌렀을 때 자동 주입
_injected = st.session_state.pop("_quick_prompt", None)
if _injected:
    prompt = _injected
else:
    prompt = None

# (G) 입력 위젯을 한 번 더 감싸 “명령어 힌트” 제공
with st.expander("⌨️ 명령어 도움말", expanded=False):
    st.markdown("""
- `/activate` / `/deactivate` : 모드 켜기/끄기  
- `/level ie=70 run=90` : 레벨 설정  
- `/plan`, `/todo`, `/summarize`, `/reset`  
- `/pin 문장` , `/unpin`  
- `/persona=따뜻함|담백함|격려형|차분한 조언가`  
- `/style=간결|보통|풍부`
""".strip())

_user_input = st.chat_input("메시지를 입력하거나 /명령을 사용하세요")
if _user_input and not prompt:
    prompt = _user_input

if prompt:
    store["chats"].append({"role":"user","content":prompt}); save_store(store)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1) 슬래시 명령 우선 처리
    if prompt.strip().startswith("/"):
        cmd_reply = handle_command(prompt)
        if cmd_reply:
            with st.chat_message("assistant"):
                st.markdown(cmd_reply)
            store["chats"].append({"role":"assistant","content":cmd_reply}); save_store(store)
        else:
            with st.chat_message("assistant"):
                st.markdown("알 수 없는 명령이에요. 도움말을 참고해 주세요.")
            store["chats"].append({"role":"assistant","content":"알 수 없는 명령"}); save_store(store)
    else:
        # 2) 일반 대화
        with st.chat_message("assistant"):
            if st.session_state.gea_active:
                reply = synthesize_reply(prompt, store["chats"])
            else:
                reply = "지금은 **비활성 상태**예요. `/activate` 또는 사이드바에서 켜고 다시 시도해 주세요."
            st.markdown(reply)
        store["chats"].append({"role":"assistant","content":reply}); save_store(store)

# (H) 전체 대화 내보내기/가져오기
st.divider()
st.markdown("#### 💽 백업/복원")
c1, c2 = st.columns(2)
with c1:
    st.download_button("💾 대화 내보내기(JSON)", data=export_json(store),
                       file_name="gea_chat_backup.json", mime="application/json")
with c2:
    up = st.file_uploader("대화 가져오기(JSON)", type=["json"], key="chat_up")
    if up is not None:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict) and "chats" in data:
                store.update(data)
                save_store(store)
                st.success("대화를 복원했습니다. 페이지를 새로고침하세요.")
        except Exception as e:
            st.error(f"복원 실패: {e}")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v3 — 우주정보장 요약(+캐시) / 길이조절 / /cosmic =========
# 이 블록은 기존 코드 아래 '이어붙이기'만 하면 동작합니다. (requests, bs4 필요)

# (0) 의존성 점검 안내 (없어도 앱은 죽지 않도록 try/except)
try:
    import requests
    from bs4 import BeautifulSoup
    _GEA_HAS_NET = True
except Exception:
    _GEA_HAS_NET = False

import re, time

# (1) 세션 캐시 준비
if "cosmic_cache" not in st.session_state:
    st.session_state.cosmic_cache = {}     # { query: {"ts":..., "items":[...]}}
if "cosmic_len" not in st.session_state:
    st.session_state.cosmic_len = 5        # 링크 개수
if "use_cosmic" not in st.session_state:
    st.session_state.use_cosmic = True     # 외부정보 ON/OFF

# (2) 사이드바 옵션 추가
with st.sidebar:
    st.markdown("### 🌐 우주정보장 설정")
    st.session_state.use_cosmic = st.toggle("외부 정보(웹/위키) 섞기", value=st.session_state.use_cosmic)
    st.session_state.cosmic_len = st.slider("요약 링크 개수", 1, 8, st.session_state.cosmic_len)
    if not _GEA_HAS_NET:
        st.caption("※ requests/bs4 미설치 또는 호스팅 제한으로 외부요약 비활성.")

# (3) 미니 크롤러 (DuckDuckGo HTML + ko/en 위키 검색) — 실패해도 안전하게
_UA = "GEA/1.0 (+local)"
def _clean_v3(t:str)->str:
    return re.sub(r"\s+"," ",(t or "")).strip()

def _ddg_search_v3(q:str, k:int=5):
    if not _GEA_HAS_NET: return []
    try:
        r = requests.get("https://duckduckgo.com/html/", params={"q": q},
                         headers={"User-Agent": _UA}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for a in soup.select(".result__a")[:k]:
            title = _clean_v3(a.get_text(" "))
            href = a.get("href") or ""
            snip_el = a.find_parent("div").select_one(".result__snippet")
            snip = _clean_v3(snip_el.get_text(" ")) if snip_el else ""
            if title and href and snip:
                out.append({"source":"ddg","title":title,"url":href,"snippet":snip})
        return out
    except Exception:
        return []

def _wiki_summary_v3(q:str, pref=("ko","en")):
    if not _GEA_HAS_NET: return []
    items=[]
    for lang in pref:
        try:
            r = requests.get(f"https://{lang}.wikipedia.org/w/index.php",
                             params={"search": q, "ns0":1},
                             headers={"User-Agent": _UA}, timeout=10)
            if r.ok and "search" in r.url:
                soup = BeautifulSoup(r.text, "html.parser")
                first = soup.select_one(".mw-search-result-heading a")
                if not first: continue
                page = "https://" + lang + ".wikipedia.org" + first.get("href","")
                r2 = requests.get(page, headers={"User-Agent": _UA}, timeout=10)
                if r2.ok:
                    s2 = BeautifulSoup(r2.text, "html.parser")
                    p = s2.select_one("p")
                    snip = _clean_v3(p.get_text(" "))[:220] if p else ""
                    title = _clean_v3(first.get_text(" "))
                    if title and page and snip:
                        items.append({"source":"wikipedia","title":title,"url":page,"snippet":snip})
                        break
        except Exception:
            continue
    return items

def cosmic_fetch_v3(query:str, k:int=5):
    # 캐시 우선
    ck = query.strip().lower()
    ent = st.session_state.cosmic_cache.get(ck)
    if ent and time.time() - ent["ts"] < 600:  # 10분 캐시
        return ent["items"][:k]
    # 수집
    items = []
    items += _ddg_search_v3(query, k=k)
    if len(items) < k:
        items += _wiki_summary_v3(query, pref=("ko","en"))
    # 중복 제거
    seen=set(); uniq=[]
    for it in items:
        key=(it.get("title",""),it.get("url",""))
        if key in seen: continue
        seen.add(key); uniq.append(it)
    out = uniq[:k]
    # 캐시에 저장
    st.session_state.cosmic_cache[ck] = {"ts": time.time(), "items": out}
    return out

# (4) 외부요약 섹션 조립기
def render_cosmic_block_v3(query:str, k:int=5)->str:
    if not st.session_state.use_cosmic:
        return "외부 정보는 현재 OFF 입니다."
    if not _GEA_HAS_NET:
        return "외부 접근이 제한되어 내장 모드로 동작 중입니다."
    infos = cosmic_fetch_v3(query, k=k)
    if not infos:
        return "적합한 외부 요약을 찾지 못했습니다."
    lines=[]
    for it in infos:
        lines.append(f"- **{it.get('title','')[:80]}** — {it.get('snippet','')[:200]}  \n  {it.get('url','')}")
    return "\n".join(lines)

# (5) /cosmic 명령 추가: /cosmic [질문 문장]
def handle_command_v3(text:str)->str:
    t = (text or "").strip()
    low = t.lower()
    if not low.startswith("/cosmic"):
        return ""
    q = t.split(" ",1)[1].strip() if " " in t else ""
    if not q:
        return "사용법: `/cosmic 주제/질문` (예: `/cosmic 양자센서 산업 동향`)"
    block = render_cosmic_block_v3(q, k=st.session_state.cosmic_len)
    return "### 🌌 외부 근거(요약 링크)\n" + block

# (6) 입력 훅에 명령 연결 — 기존 명령 처리 뒤에 후속 처리
_user_tail_input = st.text_input("↪️ (선택) /cosmic 명령 바로 실행", key="cosmic_quick", placeholder="/cosmic 주제를 입력")
if _user_tail_input:
    # 대화에 기록 + 즉시 응답
    store["chats"].append({"role":"user","content":_user_tail_input}); save_store(store)
    with st.chat_message("user"):
        st.markdown(_user_tail_input)
    cosmic_reply = handle_command_v3(_user_tail_input)
    with st.chat_message("assistant"):
        st.markdown(cosmic_reply)
    store["chats"].append({"role":"assistant","content":cosmic_reply}); save_store(store)

st.markdown("> 힌트: 외부 링크를 보고 싶다면 메시지에 `/cosmic 주제`를 보내세요. 예) `/cosmic 리먼가설 개요`")
# ================================================================================
# ==== [APPEND ONLY] 확장 v4 — 파일 업로드(TXT/MD/CSV) 요약·체크리스트·내보내기 =================
# 외부 패키지 없이 동작하도록 표준 라이브러리만 사용합니다.

import io, csv, textwrap, datetime as _dt

# (1) 사이드 옵션: 요약 길이 배수
with st.sidebar:
    st.markdown("### 📎 업로드 요약 설정")
    _sum_depth = st.slider("요약 깊이(1~5)", 1, 5, 3)
    _outline_depth = st.slider("아웃라인 깊이(1~5)", 1, 5, 3)

# (2) 업로드 UI
st.markdown("#### 📎 파일 업로드 (TXT / MD / CSV)")
_up_files = st.file_uploader("여기에 파일을 올리세요 (여러 개 가능)", type=["txt","md","csv"], accept_multiple_files=True, key="gea_uploader_v4")

def _safe_text(b:bytes)->str:
    for enc in ("utf-8","utf-16","cp949","euc-kr","latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8","ignore")

def _preview_text(txt:str, n=800)->str:
    t = " ".join(txt.split())
    return (t[:n] + "…") if len(t) > n else t

def _summarize_text(txt:str, depth:int=3)->str:
    # 아주 단순 요약기: 문장 자르기 + 핵심 단어 위주 압축
    lines = [s.strip() for s in re.split(r"(?<=[.!?。…])\s+", txt) if s.strip()]
    take = min(len(lines), depth*4)
    picked = " ".join(lines[:take]) if lines else txt
    # 정돈
    picked = textwrap.fill(picked, width=100)
    return picked

def _outline_text(txt:str, depth:int=3)->str:
    # 키워드 추출 흉내: 길이/숫자/기호 포함 문장 우선
    lines = [s.strip() for s in re.split(r"(?<=[.!?。…])\s+", txt) if s.strip()]
    hints = []
    for s in lines[: depth*6]:
        if any(c.isdigit() for c in s) or len(s) > 50 or ":" in s:
            hints.append(s)
        if len(hints) >= depth*5:
            break
    if not hints:
        hints = lines[:depth*5]
    body = "\n".join(f"- {h}" for h in hints[:depth*5])
    return body or "- (내용이 적어 자동 아웃라인이 비어있습니다)"

def _csv_to_rows(b:bytes):
    txt = _safe_text(b)
    f = io.StringIO(txt)
    reader = csv.reader(f)
    rows = [r for r in reader]
    return rows

def _rows_to_table_md(rows):
    if not rows:
        return ""
    head = rows[0]
    bar = ["---"] * len(head)
    body = rows[1:][:50]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(bar) + " |"]
    for r in body:
        r = r + [""] * (len(head)-len(r))
        lines.append("| " + " | ".join(r[:len(head)]) + " |")
    return "\n".join(lines)

_uploaded_summaries = []

if _up_files:
    for uf in _up_files:
        ext = (uf.name.split(".")[-1] or "").lower()
        st.markdown(f"**파일:** `{uf.name}`  ·  크기: {uf.size} bytes")
        if ext in ("txt","md"):
            txt = _safe_text(uf.read())
            st.text_area("미리보기", _preview_text(txt, n=1000), height=140, key=f"pv_{uf.name}")
            sm = _summarize_text(txt, depth=_sum_depth)
            ol = _outline_text(txt, depth=_outline_depth)
            st.markdown("**요약**")
            st.markdown(sm)
            st.markdown("**아웃라인**")
            st.markdown(ol)
            _uploaded_summaries.append({"name": uf.name, "type": ext, "summary": sm, "outline": ol})

            # 체크리스트 생성/다운로드
            with st.expander("☑️ 체크리스트 생성/다운로드"):
                items = [f"핵심 검토: {i+1}" for i in range(min(7, max(3, _sum_depth+2)))]
                cl_text = "\n".join(f"- [ ] {it}" for it in items)
                st.code(cl_text, language="markdown")
                # CSV로도 제공
                out = io.StringIO()
                wr = csv.writer(out)
                wr.writerow(["done","item"])
                for it in items:
                    wr.writerow([0, it])
                st.download_button("CSV 다운로드", data=out.getvalue().encode("utf-8"),
                                   file_name=f"checklist_{uf.name}.csv", mime="text/csv")

            st.divider()

        elif ext == "csv":
            rows = _csv_to_rows(uf.read())
            if rows:
                st.markdown("**CSV 미리보기 (상위 50행)**")
                # 간단 테이블 렌더
                if len(rows) > 1:
                    head = rows[0]
                    for r in rows[1: min(51, len(rows))]:
                        r = r + [""] * (len(head)-len(r))
                    st.markdown(_rows_to_table_md(rows))
                else:
                    st.text_area("내용", _safe_text(uf.read()), height=140, key=f"pv_{uf.name}")
                # CSV 요약
                sm = f"열 수: {len(rows[0]) if rows else 0}, 행 수: {max(0,len(rows)-1)}"
                st.markdown(f"**요약:** {sm}")
                _uploaded_summaries.append({"name": uf.name, "type": "csv", "summary": sm, "outline": ""})
            st.divider()
        else:
            st.warning("지원하지 않는 확장자입니다. TXT/MD/CSV만 지원합니다.")

# (3) 통합 요약 다운로드 (Markdown)
if _uploaded_summaries:
    md_lines = [f"# 업로드 요약 — { _dt.datetime.now().strftime('%Y-%m-%d %H:%M') }"]
    for ent in _uploaded_summaries:
        md_lines.append(f"## {ent['name']} ({ent['type']})")
        if ent.get("summary"):
            md_lines.append("### 요약")
            md_lines.append(ent["summary"])
        if ent.get("outline"):
            md_lines.append("### 아웃라인")
            md_lines.append(ent["outline"])
    md_blob = "\n\n".join(md_lines)
    st.download_button("📥 통합 요약(Markdown) 다운로드", data=md_blob.encode("utf-8"),
                       file_name="gea_upload_summaries.md", mime="text/markdown")

# (4) /exportmd 명령: 최근 assistant 답변을 마크다운으로 내보내기
def _last_assistant_md():
    msgs = [m for m in store.get("chats",[]) if m.get("role") in ("assistant","gea")]
    if not msgs:
        return "# 내보낼 답변이 없습니다."
    last = msgs[-1]["content"]
    return f"# GEA 최근 답변\n\n{last}"

with st.expander("📤 /exportmd (최근 답변 내보내기)", expanded=False):
    if st.button("최근 답변 Markdown 다운로드"):
        st.download_button("다운로드", data=_last_assistant_md().encode("utf-8"),
                           file_name="gea_last_answer.md", mime="text/markdown", key="dl_last_md_btn")
# ==================================================================================
# ==== [APPEND ONLY] 확장 v5 — 프로젝트 보드(칸반) + 타임라인(간트풍) ==============
import datetime as _dt

# 상태 초기화
if "kanban" not in st.session_state:
    st.session_state.kanban = {
        "Backlog": [],
        "Doing":   [],
        "Done":    []
    }
if "timeline" not in st.session_state:
    st.session_state.timeline = []  # [{title, start, end, owner}]

st.markdown("### 📋 프로젝트 보드 (칸반)")
kb_col1, kb_col2 = st.columns([2,1])
with kb_col1:
    kt = st.text_input("할 일 제목", key="kb_title", placeholder="예: 상상력 엔진 v0.6 체크")
    ka = st.selectbox("상태", ["Backlog","Doing","Done"], key="kb_state")
with kb_col2:
    kb_add = st.button("➕ 추가")
    kb_clear = st.button("🧹 보드 비우기")

if kb_add and kt.strip():
    st.session_state.kanban[ka].append({
        "title": kt.strip(),
        "ts": time.time()
    })
    st.success("보드에 추가됨!")
if kb_clear:
    st.session_state.kanban = {"Backlog":[],"Doing":[],"Done":[]}
    st.info("보드를 초기화했습니다.")

kanban = st.session_state.kanban
kc1, kc2, kc3 = st.columns(3)
with kc1: st.markdown("**Backlog**"); [st.markdown(f"- {i+1}. {t['title']}") for i,t in enumerate(kanban["Backlog"])]
with kc2: st.markdown("**Doing**");   [st.markdown(f"- {i+1}. {t['title']}") for i,t in enumerate(kanban["Doing"])]
with kc3: st.markdown("**Done**");    [st.markdown(f"- {i+1}. {t['title']}") for i,t in enumerate(kanban["Done"])]

# 진행률
total = sum(len(v) for v in kanban.values())
done  = len(kanban["Done"])
pct = (done/total*100) if total else 0.0
st.progress(min(1.0, pct/100.0), text=f"진행률: {pct:.1f}%  (완료 {done}/{total})")

st.markdown("### ⏱ 타임라인 (간트풍 간단 뷰)")
tl_c1, tl_c2, tl_c3, tl_c4 = st.columns([2,1,1,1])
with tl_c1:
    tl_title = st.text_input("작업명", key="tl_title", placeholder="예: v0.6 L13 배치 테스트")
with tl_c2:
    tl_start = st.date_input("시작", value=_dt.date.today())
with tl_c3:
    tl_end   = st.date_input("종료", value=_dt.date.today()+_dt.timedelta(days=3))
with tl_c4:
    tl_owner = st.text_input("담당", value="길도", key="tl_owner")
tl_add = st.button("➕ 타임라인 추가")
if tl_add and tl_title.strip():
    st.session_state.timeline.append({
        "title": tl_title.strip(),
        "start": str(tl_start),
        "end":   str(tl_end),
        "owner": tl_owner.strip() or "미정"
    })
    st.success("타임라인에 추가됨!")

# 텍스트 표기(간트풍 막대 길이 흉내)
if st.session_state.timeline:
    st.markdown("**일정 목록**")
    for item in st.session_state.timeline:
        d0 = _dt.datetime.fromisoformat(item["start"]).date()
        d1 = _dt.datetime.fromisoformat(item["end"]).date()
        span = max(1, (d1 - d0).days + 1)
        bar = "█" * min(30, span)  # 간단 막대
        st.markdown(f"- **{item['title']}** [{item['owner']}] {item['start']} → {item['end']}  \n  `{bar}` ({span}일)")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v6 — 응답 품질 메트릭 + /diag ============================
import math
from collections import Counter

def _est_metrics(text:str)->dict:
    words = re.findall(r"[^\s]+", text or "")
    wlen  = len(words)
    chars = len(text or "")
    # 대략 분당 200~250자 가정(한국어), 평균값 225로 가정
    read_min = max(0.1, chars/225.0/60.0)  # 초 단위
    # 상위 키워드 (짧은 조사/기호 제외)
    toks = [t.lower() for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", text)]
    stop = set(list("그리고그러나그러므로또한하지만있는"))  # 초간단 stop
    toks = [t for t in toks if t not in stop]
    topk = Counter(toks).most_common(8)
    return {
        "words": wlen,
        "chars": chars,
        "read_min": read_min*60,  # 초
        "top_keywords": topk
    }

with st.expander("🩺 응답 진단 (/diag)", expanded=False):
    if store.get("chats"):
        last = next((m["content"] for m in reversed(store["chats"]) if m["role"] in ("assistant","gea")), "")
        if last:
            m = _est_metrics(last)
            st.markdown(f"- 글자 수: **{m['chars']}**  / 단어 수: **{m['words']}**")
            st.markdown(f"- 예상 읽기 시간: **{m['read_min']:.1f} 초**")
            if m["top_keywords"]:
                st.markdown("**상위 키워드:** " + ", ".join([f"{k}({v})" for k,v in m["top_keywords"]]))
            if st.button("이 답변을 핀에 저장(키워드 포함)"):
                pin_add(f"[답변키워드] " + ", ".join(k for k,_ in m["top_keywords"]))
                st.success("핀에 키워드 저장 완료.")
    st.caption("힌트: `/diag` 명령으로도 요약 진단을 받을 수 있어요.")

def _diag_command_reply()->str:
    last = next((m["content"] for m in reversed(store.get("chats",[])) if m["role"] in ("assistant","gea")), "")
    if not last:
        return "진단할 최근 답변이 없습니다."
    m = _est_metrics(last)
    tops = ", ".join([f"{k}({v})" for k,v in m["top_keywords"]]) if m["top_keywords"] else "없음"
    return (f"### 🩺 응답 진단\n- 글자 수: **{m['chars']}**, 단어 수: **{m['words']}**\n"
            f"- 예상 읽기 시간: **{m['read_min']:.1f} 초**\n- 상위 키워드: {tops}")

# 슬래시 명령 확장 훅: /diag
def _append_diag_hook(prompt_text:str)->str:
    t = (prompt_text or "").strip().lower()
    if t.startswith("/diag"):
        return _diag_command_reply()
    return ""

# 입력 파이프에 끼워넣는 미니 훅(안전: 이미 동일 훅이 있으면 무해)
_user_diag = st.session_state.get("_user_diag_hooked", False)
if not _user_diag:
    st.session_state._user_diag_hooked = True
# 실제 처리: 채팅 입력 직후 store에 이미 기록되어 있으므로, 별도 버튼으로 출력
if st.button("🔍 최근 답변 진단(/diag)"):
    diag = _diag_command_reply()
    with st.chat_message("assistant"):
        st.markdown(diag)
    store["chats"].append({"role":"assistant","content":diag}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v7 — 로컬 지식베이스(간단 인덱싱) + /askkb ===============
import unicodedata

if "kb_docs" not in st.session_state:
    st.session_state.kb_docs = []   # [{"id":..., "name":..., "text":...}]
if "kb_index" not in st.session_state:
    st.session_state.kb_index = {}  # token -> set(doc_id)

st.markdown("### 🧠 로컬 지식베이스 (KB)")
with st.expander("KB 문서 추가/관리", expanded=False):
    kb_files = st.file_uploader("KB에 넣을 TXT/MD/CSV 업로드", type=["txt","md","csv"], accept_multiple_files=True, key="kb_upl_v7")
    def _normalize_text(s:str)->str:
        return unicodedata.normalize("NFKC", s)
    def _tokenize(s:str):
        return [t.lower() for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", s)]
    if kb_files:
        for f in kb_files:
            name = f.name
            raw = f.read()
            try:
                txt = raw.decode("utf-8","ignore")
            except:
                try:
                    txt = raw.decode("cp949","ignore")
                except:
                    txt = raw.decode("latin-1","ignore")
            if name.lower().endswith(".csv"):
                # CSV는 단순 쉼표→공백
                txt = _normalize_text(txt.replace(",", " "))
            else:
                txt = _normalize_text(txt)
            doc_id = len(st.session_state.kb_docs)
            st.session_state.kb_docs.append({"id": doc_id, "name": name, "text": txt})
            toks = set(_tokenize(txt))
            for tk in toks:
                st.session_state.kb_index.setdefault(tk, set()).add(doc_id)
        st.success(f"KB에 {len(kb_files)}개 문서를 추가했습니다.")

    if st.button("KB 비우기"):
        st.session_state.kb_docs = []
        st.session_state.kb_index = {}
        st.info("KB 초기화 완료.")

def _kb_search(query:str, topk:int=5):
    # 아주 단순한 TF 점수: 공통 토큰 수 기준
    toks = list(set(re.findall(r"[가-힣A-Za-z0-9]{2,}", (query or "").lower())))
    scores = {}
    for t in toks:
        hits = st.session_state.kb_index.get(t, set())
        for did in hits:
            scores[did] = scores.get(did, 0) + 1
    # 정렬
    cand = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:topk]
    out = []
    for did, sc in cand:
        doc = st.session_state.kb_docs[did]
        # 문서 앞 부분 미리보기
        preview = " ".join(doc["text"].split()[:80]) + ("…" if len(doc["text"]) > 400 else "")
        out.append({"name": doc["name"], "score": sc, "preview": preview})
    return out

with st.expander("🔎 KB 질의(/askkb)", expanded=False):
    q = st.text_input("질문을 입력하세요 (또는 /askkb 명령 사용)", key="kb_q")
    if st.button("KB 검색"):
        res = _kb_search(q, topk=5)
        if not res:
            st.warning("일치 문서 없음")
        else:
            st.markdown("**관련 문서 후보**")
            for r in res:
                st.markdown(f"- **{r['name']}** (score={r['score']})  \n  {r['preview']}")

def _askkb_command(text:str)->str:
    t = (text or "").strip()
    if not t.lower().startswith("/askkb"):
        return ""
    q = t.split(" ",1)[1].strip() if " " in t else ""
    if not q:
        return "사용법: `/askkb 질문`  (먼저 KB에 문서를 올려 주세요)"
    res = _kb_search(q, topk=5)
    if not res:
        return "KB에서 관련 문서를 찾지 못했습니다."
    lines = ["### 🧠 KB 검색 결과"]
    for r in res:
        lines.append(f"- **{r['name']}** (score={r['score']})  \n  {r['preview']}")
    return "\n".join(lines)

# 채팅 입력 훅: /askkb 처리
if st.button("🧠 /askkb 마지막 질문 테스트"):
    if store.get("chats"):
        last_user = next((m["content"] for m in reversed(store["chats"]) if m["role"]=="user"), "")
        if last_user:
            ans = _askkb_command("/askkb " + last_user)
            with st.chat_message("assistant"):
                st.markdown(ans)
            store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v8 — 설정 영구 저장 + 다크모드 스킨 + 언어 스위치 ==========
import json as _json

# 세션 기본값
if "gea_settings" not in st.session_state:
    st.session_state.gea_settings = {
        "theme": "auto",         # auto / light / dark
        "lang": "ko",            # ko / en
        "ie": st.session_state.get("ie_level", 50),
        "run": st.session_state.get("run_level", 80),
        "persona": st.session_state.get("persona","따뜻함"),
        "style": st.session_state.get("style","풍부"),
    }

_SETTINGS_FILE = Path(".gea_settings.json")

def _settings_save():
    data = st.session_state.gea_settings
    # 최신 슬라이더와 동기화
    data["ie"] = st.session_state.get("ie_level", data["ie"])
    data["run"] = st.session_state.get("run_level", data["run"])
    data["persona"] = st.session_state.get("persona", data["persona"])
    data["style"] = st.session_state.get("style", data["style"])
    try:
        _SETTINGS_FILE.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

def _settings_load():
    if _SETTINGS_FILE.exists():
        try:
            data = _json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            st.session_state.gea_settings.update(data)
            # 세션 슬라이더/셀렉트도 반영
            st.session_state.ie_level = data.get("ie", st.session_state.get("ie_level", 50))
            st.session_state.run_level = data.get("run", st.session_state.get("run_level", 80))
            st.session_state.persona   = data.get("persona", st.session_state.get("persona","따뜻함"))
            st.session_state.style     = data.get("style", st.session_state.get("style","풍부"))
            return True
        except Exception:
            return False
    return False

with st.expander("🛠 설정 (영구 저장/불러오기)"):
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.gea_settings["theme"] = c1.selectbox("테마", ["auto","light","dark"], index=["auto","light","dark"].index(st.session_state.gea_settings["theme"]))
    st.session_state.gea_settings["lang"]  = c2.selectbox("언어", ["ko","en"], index=["ko","en"].index(st.session_state.gea_settings["lang"]))
    if c3.button("💾 저장"):
        ok = _settings_save()
        st.success("설정을 저장했어요." if ok else "저장 실패(권한 문제).")
    if c4.button("📥 불러오기"):
        ok = _settings_load()
        st.success("설정을 불러왔어요." if ok else "불러오기 실패/파일 없음.")

    # 백업/복원
    st.markdown("— 설정 백업/복원 —")
    st.download_button("설정 백업(JSON)", data=_json.dumps(st.session_state.gea_settings, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="gea_settings_backup.json", mime="application/json")
    up = st.file_uploader("설정 복원(JSON)", type=["json"], key="gea_set_up")
    if up is not None:
        try:
            data = _json.loads(up.read().decode("utf-8"))
            st.session_state.gea_settings.update(data)
            st.success("설정을 복원했습니다. 저장을 눌러 영구화하세요.")
        except Exception as e:
            st.error(f"복원 실패: {e}")

# 간이 다크 모드 스킨(스트림릿 테마를 건드릴 수는 없지만 배경/카드톤 스킨)
def _apply_skin():
    t = st.session_state.gea_settings.get("theme","auto")
    if t == "auto": return
    if t == "dark":
        st.markdown("""
            <style>
            .stApp { background: #0f1216; color: #e6eef6; }
            .stMarkdown, .stTextInput, .stSelectbox, .stSlider { color: #e6eef6 !important; }
            .stButton>button { background: #1f2833; color: #e6eef6; border: 1px solid #2b3645; }
            </style>
        """, unsafe_allow_html=True)
    if t == "light":
        st.markdown("""
            <style>
            .stApp { background: #ffffff; color: #111; }
            .stButton>button { background: #f2f4f7; color: #111; border: 1px solid #e5e7eb; }
            </style>
        """, unsafe_allow_html=True)

_apply_skin()
# ==== [APPEND ONLY] 확장 v9 — 차트/표/계산기 + /chart =============================
import random
import pandas as _pd

# 간단 계산기
with st.expander("🧮 미니 계산기"):
    expr = st.text_input("수식(예: (3+5)*2/7 )", key="gea_calc_expr")
    if st.button("계산"):
        try:
            # 안전을 위해 eval 최소화(숫자/연산자만 허용)
            if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr or ""):
                val = eval(expr, {"__builtins__": None}, {})
                st.success(f"결과: {val}")
            else:
                st.warning("숫자와 + - * / ( ) 만 허용합니다.")
        except Exception as e:
            st.error(f"오류: {e}")

# 차트 데모용 데이터 생성
if "chart_df" not in st.session_state:
    xs = list(range(1, 21))
    ys = [max(0, y + random.randint(-3, 3)) for y in range(1, 21)]
    st.session_state.chart_df = _pd.DataFrame({"step": xs, "value": ys})

with st.expander("📊 차트 & 표"):
    st.markdown("**라인 차트(예시)**")
    st.line_chart(st.session_state.chart_df.set_index("step"))
    st.markdown("**데이터 표**")
    st.dataframe(st.session_state.chart_df, use_container_width=True)

# /chart 명령: 최근 assistant 답변 길이를 단계별로 누적해서 시각화(장난감)
def _chart_command(text:str)->str:
    if not text.lower().startswith("/chart"): return ""
    # 최근 20개 assistant 메시지 길이
    msgs = [len(m["content"]) for m in store.get("chats",[]) if m.get("role") in ("assistant","gea")]
    if not msgs:
        return "시각화할 답변이 아직 없습니다."
    df = _pd.DataFrame({"idx": list(range(1, len(msgs)+1)), "len": msgs})
    st.line_chart(df.set_index("idx"))
    return "### 📊 최근 답변 길이 추이 (위 차트 참조)"

if st.button("📈 /chart (최근 답변 길이 추이)"):
    ans = _chart_command("/chart")
    with st.chat_message("assistant"):
        st.markdown(ans)
    store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v10 — 테스트 러너(체크리스트 실행·타이머·로그) ==========
import time as _time

if "check_runs" not in st.session_state:
    st.session_state.check_runs = []  # [{name, items:[{text,done,dur}], started, finished}]

st.markdown("### 🧪 테스트 러너 (체크리스트 실행)")
with st.expander("새 테스트 만들기 / 실행", expanded=False):
    t_name = st.text_input("테스트 이름", placeholder="예: v0.6 L30 회귀")
    t_items_raw = st.text_area("체크항목(줄바꿈으로 구분)", height=120, placeholder="예:\n모듈 로드\n질의-응답\nREPAIR 루프\n로그 검증")
    c1, c2 = st.columns(2)
    if c1.button("테스트 시작"):
        items = [{"text": line.strip(), "done": False, "dur": 0.0} for line in t_items_raw.splitlines() if line.strip()]
        st.session_state.check_runs.append({
            "name": t_name or f"run-{len(st.session_state.check_runs)+1}",
            "items": items, "started": _time.time(), "finished": 0.0
        })
        st.success("테스트를 시작했습니다.")
    if c2.button("모든 테스트 초기화"):
        st.session_state.check_runs = []
        st.info("초기화 완료")

if st.session_state.check_runs:
    st.markdown("#### 실행 중/완료 테스트")
    for idx, run in enumerate(st.session_state.check_runs):
        st.markdown(f"**[{idx+1}] {run['name']}**  — 시작: {datetime.fromtimestamp(run['started']).strftime('%Y-%m-%d %H:%M:%S')}")
        for j, it in enumerate(run["items"]):
            col1, col2, col3 = st.columns([6,1,2])
            col1.markdown(f"- {j+1}. {it['text']}  {'✅' if it['done'] else ''}")
            if not it["done"]:
                if col2.button("완료", key=f"ck_{idx}_{j}"):
                    it["done"] = True
                    it["dur"] = it.get("dur", 0.0) + 1.0
                    st.success(f"항목 완료: {it['text']}")
            col3.caption(f"소요: {it['dur']:.1f}s")
        if not run["finished"] and all(i["done"] for i in run["items"]):
            run["finished"] = _time.time()
            st.success(f"테스트 완료! 총 소요: {run['finished'] - run['started']:.1f}s")
        st.divider()

# /runcheck 명령: 가장 최근 테스트의 진행상황을 요약
def _runcheck_cmd(text:str)->str:
    if not text.lower().startswith("/runcheck"): return ""
    if not st.session_state.check_runs:
        return "실행 중인 테스트가 없습니다."
    run = st.session_state.check_runs[-1]
    done = sum(1 for i in run["items"] if i["done"])
    total = len(run["items"])
    eta = "완료" if run["finished"] else "진행 중"
    return (f"### 🧪 테스트 상태\n- 이름: **{run['name']}**\n- 진행: **{done}/{total}**\n- 상태: **{eta}**")

if st.button("🧪 /runcheck"):
    ans = _runcheck_cmd("/runcheck")
    with st.chat_message("assistant"):
        st.markdown(ans)
    store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v11 — PDF 업로드 & 요약 ==================================
# 옵션 의존성: PyPDF2 또는 pypdf (없어도 안전하게 동작)
try:
    import PyPDF2 as _pypdf
    _GEA_PDF_OK = True
except Exception:
    try:
        import pypdf as _pypdf
        _GEA_PDF_OK = True
    except Exception:
        _GEA_PDF_OK = False

st.markdown("### 📄 PDF 업로드 & 요약")
pdfs = st.file_uploader("PDF 파일을 올리세요 (여러 개 가능)", type=["pdf"], accept_multiple_files=True, key="pdf_upl_v11")

def _pdf_text(bytes_blob) -> str:
    if not _GEA_PDF_OK:
        return "(PDF 모듈이 없어 텍스트를 추출하지 못했습니다. PyPDF2 또는 pypdf 설치 시 자동 사용됩니다.)"
    try:
        import io
        reader = _pypdf.PdfReader(io.BytesIO(bytes_blob))
        pages = []
        for i, pg in enumerate(reader.pages):
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages).strip()
    except Exception as e:
        return f"(PDF 파싱 실패: {e})"

def _smart_sum(txt: str, k: int = 8) -> str:
    # 아주 단순 요약기: 긴 문장/숫자 포함 문장 우선
    sents = [s.strip() for s in re.split(r"(?<=[.!?。…])\s+", txt) if s.strip()]
    strong = [s for s in sents if any(ch.isdigit() for ch in s) or len(s) > 60]
    pick = (strong + sents)[:k]
    return "\n".join(f"- {p}" for p in pick)

if pdfs:
    for f in pdfs:
        st.markdown(f"**PDF:** `{f.name}` · {f.size} bytes")
        txt = _pdf_text(f.read())
        st.text_area("미리보기(일부)", txt[:2000] + ("…" if len(txt) > 2000 else ""), height=180, key=f"pdf_prev_{f.name}")
        st.markdown("**요약(핵심 문장)**")
        st.markdown(_smart_sum(txt, k=8))
        st.divider()

if not _GEA_PDF_OK:
    st.info("PDF 텍스트 추출을 쓰려면 `PyPDF2` 또는 `pypdf` 설치가 있으면 좋아요. (없어도 앱은 정상 동작)")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v12 — 번역 스위치 + /translate ===========================
# 옵션 의존성: deep_translator (없어도 폴백으로 동작)
try:
    from deep_translator import GoogleTranslator as _GT
    _GEA_TR_OK = True
except Exception:
    _GEA_TR_OK = False

if "translator_dir" not in st.session_state:
    st.session_state.translator_dir = "ko2en"  # ko2en / en2ko

with st.expander("🌐 번역 도구 (ko↔en)"):
    st.session_state.translator_dir = st.selectbox("방향", ["ko2en","en2ko"], index=0 if st.session_state.translator_dir=="ko2en" else 1)
    txt_src = st.text_area("번역할 텍스트 입력", height=120, key="tr_src_v12")
    if st.button("번역 실행"):
        if _GEA_TR_OK:
            try:
                if st.session_state.translator_dir == "ko2en":
                    out = _GT(source="auto", target="en").translate(txt_src or "")
                else:
                    out = _GT(source="auto", target="ko").translate(txt_src or "")
                st.success(out)
            except Exception as e:
                st.error(f"번역 오류: {e}")
        else:
            # 폴백: 매우 단순 치환/구문 기반(데모용)
            repl = {
                "계획": "plan", "목표": "goal", "데이터": "data", "검증": "validation",
                "다음 단계": "next steps", "리스크": "risk", "요약": "summary"
            }
            out = txt_src or ""
            for k, v in repl.items():
                out = out.replace(k, v)
            st.warning("간단 폴백 번역(품질 낮음). deep-translator 설치 시 고품질 번역 사용.")
            st.code(out, language="markdown")

def _translate_cmd(text: str) -> str:
    t = (text or "").strip()
    if not t.lower().startswith("/translate"): return ""
    q = t.split(" ", 1)[1].strip() if " " in t else ""
    if not q: return "사용법: `/translate 텍스트`  (방향은 위 도구에서 ko2en/en2ko 설정)"
    try:
        if _GEA_TR_OK:
            if st.session_state.translator_dir == "ko2en":
                out = _GT(source="auto", target="en").translate(q)
            else:
                out = _GT(source="auto", target="ko").translate(q)
            return f"### 🌐 번역 결과\n{out}"
        else:
            # 폴백
            repl = {"계획":"plan","목표":"goal","데이터":"data","검증":"validation","다음 단계":"next steps","리스크":"risk","요약":"summary"}
            out = q
            for k,v in repl.items(): out = out.replace(k,v)
            return f"### 🌐 번역 결과(폴백)\n{out}\n\n_참고: deep-translator 설치 시 더 정확합니다._"
    except Exception as e:
        return f"번역 실패: {e}"

if st.button("🌐 /translate (최근 사용자 입력 번역)"):
    last_user = next((m["content"] for m in reversed(store.get("chats",[])) if m["role"]=="user"), "")
    if last_user:
        ans = _translate_cmd("/translate " + last_user)
        with st.chat_message("assistant"):
            st.markdown(ans)
        store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v13 — 전체 상태 백업/복원(Zip) ==========================
import zipfile, io as _io

def _collect_state()->dict:
    return {
        "chats": store.get("chats", []),
        "pins": store.get("pins", []),
        "settings": st.session_state.get("gea_settings", {}),
        "kb_docs": st.session_state.get("kb_docs", []),
        "kb_index": {k:list(v) for k,v in st.session_state.get("kb_index", {}).items()},
        "kanban": st.session_state.get("kanban", {"Backlog":[],"Doing":[],"Done":[]}),
        "timeline": st.session_state.get("timeline", []),
        "check_runs": st.session_state.get("check_runs", []),
    }

def _apply_state(d:dict):
    try:
        store["chats"] = d.get("chats", [])
        store["pins"]  = d.get("pins", [])
        st.session_state.gea_settings = d.get("settings", st.session_state.get("gea_settings", {}))
        st.session_state.kb_docs  = d.get("kb_docs", [])
        _kb_idx = d.get("kb_index", {})
        st.session_state.kb_index = {k:set(v) for k,v in _kb_idx.items()}
        st.session_state.kanban   = d.get("kanban", st.session_state.get("kanban", {"Backlog":[],"Doing":[],"Done":[]}))
        st.session_state.timeline = d.get("timeline", st.session_state.get("timeline", []))
        st.session_state.check_runs = d.get("check_runs", st.session_state.get("check_runs", []))
        save_store(store)
        return True
    except Exception:
        return False

with st.expander("🗃 전체 백업/복원(Zip)"):
    c1, c2 = st.columns(2)
    if c1.button("📦 스냅샷 만들기"):
        state = _collect_state()
        bio = _io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("state.json", json.dumps(state, ensure_ascii=False, indent=2))
            # 보너스: 메모리 파일도 동봉
            try:
                if STORE.exists():
                    z.writestr("gea_memory.json", STORE.read_bytes())
            except Exception:
                pass
        st.download_button("스냅샷 다운로드", data=bio.getvalue(), file_name="gea_snapshot.zip", mime="application/zip")
    upzip = c2.file_uploader("스냅샷 복원(Zip 업로드)", type=["zip"], key="zip_up_v13")
    if upzip is not None:
        try:
            zf = zipfile.ZipFile(_io.BytesIO(upzip.read()))
            data = json.loads(zf.read("state.json").decode("utf-8"))
            ok = _apply_state(data)
            if ok: st.success("복원이 완료되었습니다. 화면을 새로고침하세요.")
            else:  st.error("복원 실패")
        except Exception as e:
            st.error(f"불러오기 오류: {e}")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v14 — 음성 입력(STT) & 읽어주기(TTS) =====================
# 옵션 의존성: SpeechRecognition, pydub, gTTS (없어도 폴백으로 안내만)
try:
    import speech_recognition as _sr
    _GEA_STT_OK = True
except Exception:
    _GEA_STT_OK = False

try:
    from gtts import gTTS as _gTTS
    _GEA_TTS_OK = True
except Exception:
    _GEA_TTS_OK = False

st.markdown("### 🎙 음성 입력 / 🔊 읽어주기")
col_stt, col_tts = st.columns(2)

with col_stt:
    wav = st.file_uploader("음성 파일 업로드(WAV/M4A/MP3)", type=["wav","m4a","mp3"], key="gea_stt_v14")
    if st.button("STT 실행"):
        if not wav:
            st.warning("먼저 파일을 업로드해 주세요.")
        elif not _GEA_STT_OK:
            st.info("SpeechRecognition 미설치로 폴백: 음성 텍스트 변환을 사용할 수 없습니다.")
        else:
            try:
                import io, tempfile, os
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix="."+wav.name.split(".")[-1])
                tmp.write(wav.read()); tmp.close()
                r = _sr.Recognizer()
                if tmp.name.lower().endswith(".wav"):
                    audio_data = _sr.AudioFile(tmp.name)
                else:
                    # 간단 폴백: 비 wav는 pydub 필요. 없으면 안내만.
                    try:
                        from pydub import AudioSegment
                        wav_path = tmp.name + ".wav"
                        AudioSegment.from_file(tmp.name).export(wav_path, format="wav")
                        audio_data = _sr.AudioFile(wav_path)
                    except Exception:
                        st.info("pydub 미설치로 폴백: WAV로 변환 불가. WAV 파일을 올려주세요.")
                        audio_data = None
                if audio_data:
                    with audio_data as src:
                        audio = r.record(src)
                    try:
                        text = r.recognize_google(audio, language="ko-KR")
                        st.success(f"인식 결과: {text}")
                        # 채팅에 자동 주입
                        store["chats"].append({"role":"user","content":text}); save_store(store)
                    except Exception as e:
                        st.error(f"인식 실패: {e}")
                os.unlink(tmp.name)
                try: os.unlink(wav_path)  # 변환했으면 제거
                except Exception: pass
            except Exception as e:
                st.error(f"STT 오류: {e}")

with col_tts:
    tts_text = st.text_area("읽어줄 텍스트", height=120, key="gea_tts_text_v14")
    if st.button("TTS 생성"):
        if not tts_text.strip():
            st.warning("텍스트를 입력해 주세요.")
        elif not _GEA_TTS_OK:
            st.info("gTTS 미설치로 폴백: 오디오 생성 불가.")
        else:
            try:
                mp3 = _gTTS(tts_text, lang="ko")
                import io
                bio = io.BytesIO()
                mp3.write_to_fp(bio)
                st.audio(bio.getvalue(), format="audio/mp3")
            except Exception as e:
                st.error(f"TTS 오류: {e}")

# /voice 명령: 마지막 assistant 답변을 읽어주기(TTS)
def _voice_cmd(text:str)->str:
    if not text.lower().startswith("/voice"): return ""
    if not _GEA_TTS_OK:
        return "TTS 모듈이 없어 음성 생성을 할 수 없습니다. (gTTS 설치 시 사용 가능)"
    last = next((m["content"] for m in reversed(store.get("chats",[])) if m["role"] in ("assistant","gea")), "")
    if not last:
        return "읽어줄 최근 답변이 없습니다."
    try:
        mp3 = _gTTS(last, lang="ko")
        import io
        bio = io.BytesIO(); mp3.write_to_fp(bio)
        st.audio(bio.getvalue(), format="audio/mp3")
        return "최근 답변을 음성으로 재생했습니다."
    except Exception as e:
        return f"TTS 실패: {e}"

if st.button("🔊 /voice (최근 답변 읽어주기)"):
    ans = _voice_cmd("/voice")
    with st.chat_message("assistant"): st.markdown(ans)
    store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v15 — 프로젝트 템플릿 ================================
st.markdown("### 🧰 프로젝트 템플릿 생성기")
tpl = st.selectbox("템플릿 선택", ["연구(Research)","개발(Development)","출시(Launch)"], key="gea_tpl_v15")
if st.button("템플릿 추가"):
    if "kanban" not in st.session_state: st.session_state.kanban = {"Backlog":[],"Doing":[],"Done":[]}
    kb = st.session_state.kanban
    if tpl.startswith("연구"):
        seeds = ["문헌조사/소스 인덱싱", "가설/평가지표 정의", "실험 설계 v1", "데이터 수집 최소셋", "초기 결과 요약"]
    elif tpl.startswith("개발"):
        seeds = ["요구사항 명세", "모듈 설계/인터페이스", "단위 테스트 작성", "구현 v1", "리뷰/리팩터"]
    else:
        seeds = ["최종 QA/회귀", "문서/가이드", "배포 체크리스트", "Canary/Blue-Green", "모니터링/알림"]
    for s in seeds: kb["Backlog"].append({"title": s, "ts": time.time()})
    st.success(f"{tpl} 템플릿을 보드에 추가했습니다.")

with st.expander("📦 템플릿 체크리스트 내보내기"):
    if st.button("현재 Backlog → 체크리스트 CSV"):
        import io, csv
        out = io.StringIO(); wr = csv.writer(out)
        wr.writerow(["done","item"])
        for t in st.session_state.kanban.get("Backlog",[]):
            wr.writerow([0, t["title"]])
        st.download_button("CSV 다운로드", data=out.getvalue().encode("utf-8"),
                           file_name="template_backlog.csv", mime="text/csv")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v16 — IE 미니 검증 러너 ==================================
# 외부 LLM 없이도 기본 규칙/단위/길이/금칙어를 체크해 리포트 생성
_IE_RULES = {
    "min_len": 5,     # 최소 문장 수(느슨)
    "ban_patterns": [r"초광속", r"워프", r"영매", r"예언", r"13차원", r"무제한 에너지"],
    "need_units": ["m", "s", "kg", "J"]  # 텍스트에 단위 샘플이 일부라도 등장하면 가점
}

def _ie_verify(text:str)->dict:
    lines = [s for s in re.split(r"[.!?。\n]+", text or "") if s.strip()]
    ok_len = len(lines) >= _IE_RULES["min_len"]
    bans = []
    for pat in _IE_RULES["ban_patterns"]:
        if re.search(pat, text or "", flags=re.I):
            bans.append(pat)
    unit_hits = sum(1 for u in _IE_RULES["need_units"] if u in (text or ""))
    score = 0
    score += 40 if ok_len else 0
    score += 10 * min(3, unit_hits)
    score += 50 if not bans else max(0, 50 - 10*len(bans))
    verdict = "PASS" if score >= 70 and not bans else ("REPAIR" if score>=50 else "REFUSE")
    return {"lines": len(lines), "unit_hits": unit_hits, "bans": bans, "score": score, "verdict": verdict}

st.markdown("### ✅ IE 미니 검증 러너")
ie_text = st.text_area("검증할 본문(설계/설명/계획 등)", height=180, key="ie_text_v16", placeholder="검증하고 싶은 내용을 붙여넣으세요.")
if st.button("검증 실행"):
    rep = _ie_verify(ie_text)
    st.markdown(f"**판정:** {rep['verdict']}  ·  점수: {rep['score']}")
    st.markdown(f"- 문장 수: {rep['lines']}")
    st.markdown(f"- 단위 히트: {rep['unit_hits']}")
    st.markdown(f"- 금칙어: {', '.join(rep['bans']) if rep['bans'] else '없음'}")
    # 채팅 로그에도 기록
    store["chats"].append({"role":"assistant","content":f"IE 검증: {rep}"}); save_store(store)

def _verify_cmd(text:str)->str:
    t = (text or "").strip()
    if not t.lower().startswith("/verify"): return ""
    body = t.split(" ",1)[1].strip() if " " in t else ""
    if not body: return "사용법: `/verify 본문`"
    rep = _ie_verify(body)
    return (f"### ✅ IE 검증 결과\n- 판정: **{rep['verdict']}** (점수 {rep['score']})\n"
            f"- 문장 수: {rep['lines']}\n- 단위 히트: {rep['unit_hits']}\n"
            f"- 금칙어: {', '.join(rep['bans']) if rep['bans'] else '없음'}")

if st.button("✅ /verify (최근 사용자 입력)"):
    last_user = next((m["content"] for m in reversed(store.get("chats",[])) if m["role"]=="user"), "")
    if last_user:
        ans = _verify_cmd("/verify " + last_user)
        with st.chat_message("assistant"): st.markdown(ans)
        store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v17 — 다중 세션 토큰 =====================================
if "session_token" not in st.session_state:
    st.session_state.session_token = "default"

with st.sidebar:
    st.markdown("### 🧩 세션")
    new_tok = st.text_input("새 세션 토큰 생성", placeholder="예: gea-dev-01")
    c1, c2 = st.columns(2)
    if c1.button("세션 전환"):
        tok = new_tok.strip() or "default"
        st.session_state.session_token = tok
        # 세션별 저장 파일 분리
        try:
            STORE = Path(f"gea_memory_{tok}.json")
        except Exception:
            pass
        # 새 스토어 로드/초기화
        try:
            data = load_store()
        except Exception:
            data = {"chats": [], "pins": [], "templates": {}}
        store.clear(); store.update(data)
        st.success(f"세션을 '{tok}' 으로 전환했습니다.")
    if c2.button("현재 세션 백업"):
        import json, io
        bio = io.BytesIO(json.dumps(store, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button("다운로드", data=bio.getvalue(), file_name=f"gea_session_{st.session_state.session_token}.json", mime="application/json")

def _session_cmd(text:str)->str:
    t = (text or "").strip()
    if not t.lower().startswith("/session"): return ""
    toks = t.split(" ",1)
    if len(toks) == 1:
        return f"현재 세션: **{st.session_state.session_token}**"
    else:
        tok = toks[1].strip() or "default"
        st.session_state.session_token = tok
        try: STORE = Path(f"gea_memory_{tok}.json")
        except Exception: pass
        # 로드
        data = load_store()
        store.clear(); store.update(data); save_store(store)
        return f"세션을 **{tok}** 으로 전환하고 메모리를 로드했습니다."

if st.button("🧩 /session (현재 세션 보기)"):
    ans = _session_cmd("/session")
    with st.chat_message("assistant"): st.markdown(ans)
    store["chats"].append({"role":"assistant","content":ans}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v18 — 벤치마크 스위트 =====================================
import time as _t, csv as _csv, io as _io
st.markdown("### 🧪 벤치마크 스위트 (프롬프트 세트)")

if "bm_sets" not in st.session_state:
    st.session_state.bm_sets = {
        "demo": [
            "계획 3단계로 요약해줘",
            "체크리스트 5개 만들어줘",
            "최근 논의 핵심을 5줄 요약",
            "리스크/대응 3쌍"
        ]
    }

bm_name = st.text_input("세트 이름", value="demo", key="bm_name_v18")
bm_raw  = st.text_area("프롬프트(줄바꿈 구분)", height=120, key="bm_raw_v18")
c1, c2, c3 = st.columns(3)
if c1.button("세트 저장/갱신"):
    lines = [l.strip() for l in (bm_raw or "").splitlines() if l.strip()]
    if lines:
        st.session_state.bm_sets[bm_name] = lines
        st.success(f"세트 '{bm_name}' 저장({len(lines)}개)")
if c2.button("세트 불러오기"):
    if bm_name in st.session_state.bm_sets:
        st.session_state["bm_raw_v18"] = "\n".join(st.session_state.bm_sets[bm_name])
        st.experimental_rerun()
if c3.button("세트 삭제"):
    st.session_state.bm_sets.pop(bm_name, None); st.info("삭제 완료")

run_name = st.text_input("벤치 실행명", value=f"run-{int(_t.time())}", key="bm_runname_v18")
iters = st.number_input("반복 횟수", min_value=1, max_value=5, value=1)

def _run_bench(set_name:str, n:int=1):
    prompts = st.session_state.bm_sets.get(set_name, [])
    results = []
    for r in range(n):
        for p in prompts:
            t0 = _t.time()
            # 내부 synthesize_reply 사용(활성 상태면 풍부 응답)
            if st.session_state.get("gea_active", False):
                ans = synthesize_reply(p, store.get("chats",[]))
            else:
                ans = synthesize_reply(p, store.get("chats",[]))  # 비활성도 동일 처리(데모)
            dt = _t.time()-t0
            met = len(ans), max(1, len(ans.split()))
            results.append({"iter": r+1, "prompt": p, "latency_s": round(dt,3), "chars": met[0], "words": met[1]})
            # 로그 보존(선택)
            store["chats"].append({"role":"user","content":p})
            store["chats"].append({"role":"assistant","content":ans}); save_store(store)
    return results

if st.button("▶️ 벤치 실행"):
    rows = _run_bench(bm_name, int(iters))
    if rows:
        # 표 렌더
        st.markdown("**결과 표**")
        st.dataframe(rows, use_container_width=True)
        # 집계
        avg_lat = sum(r["latency_s"] for r in rows)/len(rows)
        avg_chars = sum(r["chars"] for r in rows)/len(rows)
        st.info(f"평균 지연: {avg_lat:.3f}s · 평균 글자수: {avg_chars:.0f}")
        # CSV 다운로드
        out = _io.StringIO(); w = _csv.DictWriter(out, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        st.download_button("CSV 다운로드", data=out.getvalue().encode("utf-8"),
                           file_name=f"bench_{run_name}.csv", mime="text/csv")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v19 — 권한/잠금(읽기 전용) ================================
if "read_only" not in st.session_state:
    st.session_state.read_only = False

with st.sidebar:
    st.markdown("### 🔒 권한/잠금")
    colL, colU = st.columns(2)
    if colL.button("🔒 Lock(읽기 전용)"):
        st.session_state.read_only = True
        st.toast("읽기 전용 모드 활성", icon="🔒")
    if colU.button("🔓 Unlock"):
        st.session_state.read_only = False
        st.toast("편집 가능 모드", icon="🔓")

# 공통 가드: 위험 버튼/쓰기 동작 차단용 헬퍼
def _guard_write(action_name:str)->bool:
    if st.session_state.read_only:
        st.warning(f"읽기 전용 모드: '{action_name}' 동작이 차단되었습니다.")
        return False
    return True

# 기존 “초기화/삭제/쓰기”류 버튼에 적용 가능한 패턴(이 블록 안에서도 예시 제공)
with st.expander("🛡 잠금 테스트 버튼"):
    if st.button("대화 전체 삭제(테스트)"):
        if _guard_write("대화 전체 삭제"):
            store["chats"] = []; save_store(store); st.success("삭제 완료")

# 슬래시 명령
def _lock_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/lock"):
        st.session_state.read_only = True; return "읽기 전용 모드를 활성화했습니다."
    if t.startswith("/unlock"):
        st.session_state.read_only = False; return "읽기 전용 모드를 해제했습니다."
    return ""

if st.button("🔒 /lock (읽기 전용)"):
    msg=_lock_cmd("/lock"); st.markdown(msg); store["chats"].append({"role":"assistant","content":msg}); save_store(store)
if st.button("🔓 /unlock"):
    msg=_lock_cmd("/unlock"); st.markdown(msg); store["chats"].append({"role":"assistant","content":msg}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v20 — L100 스트리밍 시뮬레이터 ===========================
if "streaming" not in st.session_state:
    st.session_state.streaming = {"active": False, "cursor": 0, "segments": []}

st.markdown("### 📚 L100 스트리밍 시뮬레이터")
seg_txt = st.text_area("스트리밍할 본문(긴 텍스트 붙여넣기)", height=140, key="l100_src_v20")
seg_size = st.slider("세그먼트 길이(문자)", 200, 2000, 600)
colS1,colS2,colS3 = st.columns(3)

def _build_segments(text:str, size:int):
    text = text or ""
    return [text[i:i+size] for i in range(0, len(text), size)]

def _checkpoint():
    # 매우 단순 체크포인트: 세그먼트 인덱스와 해시
    import hashlib, json
    cur = st.session_state.streaming["cursor"]
    segs = st.session_state.streaming["segments"]
    h = hashlib.sha256(("".join(segs[:cur])).encode("utf-8")).hexdigest()[:16]
    return {"cursor": cur, "hash": h}

if colS1.button("▶️ 시작/재개"):
    if not seg_txt and not st.session_state.streaming["segments"]:
        st.warning("본문을 입력하세요.")
    else:
        if not st.session_state.streaming["segments"]:
            st.session_state.streaming["segments"] = _build_segments(seg_txt, seg_size)
        st.session_state.streaming["active"] = True
        st.toast("스트리밍 시작", icon="📚")

if colS2.button("⏸ 일시정지"):
    st.session_state.streaming["active"] = False
    st.info(f"체크포인트: { _checkpoint() }")

if colS3.button("⏮ 처음부터"):
    st.session_state.streaming = {"active": False, "cursor": 0, "segments": []}
    st.info("리셋 완료")

# 진행
if st.session_state.streaming["active"] and st.session_state.streaming["segments"]:
    cur = st.session_state.streaming["cursor"]
    segs= st.session_state.streaming["segments"]
    if cur < len(segs):
        with st.chat_message("assistant"):
            st.markdown(segs[cur])
        store["chats"].append({"role":"assistant","content":segs[cur]}); save_store(store)
        st.session_state.streaming["cursor"] += 1
        if st.session_state.streaming["cursor"] >= len(segs):
            st.session_state.streaming["active"] = False
            st.success("스트리밍 완료!")
    else:
        st.session_state.streaming["active"] = False
        st.info("끝에 도달했습니다.")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v21 — 미니 플러그인 시스템 ================================
# 간단한 명령 플러그인 레지스트리: {name: {"help":str, "handler":callable}}
if "gea_plugins" not in st.session_state:
    st.session_state.gea_plugins = {}

def gea_register(name:str, help_text:str, handler):
    st.session_state.gea_plugins[name] = {"help": help_text, "handler": handler}

def gea_run(cmdline:str)->str:
    # '/name args' 형식
    t = (cmdline or "").strip()
    if not t.startswith("/"): return ""
    name = t.split(" ",1)[0][1:]
    args = t.split(" ",1)[1].strip() if " " in t else ""
    plugin = st.session_state.gea_plugins.get(name)
    if not plugin: return f"알 수 없는 명령: /{name}"
    try:
        return plugin["handler"](args)
    except Exception as e:
        return f"플러그인 오류(/{name}): {e}"

with st.expander("🧩 플러그인"):
    st.caption("등록된 명령을 불러와 사용할 수 있습니다. 예: /hello, /stamp")
    if st.button("샘플 플러그인 등록"):
        # 예시 1) /hello
        gea_register("hello", "인사와 현재 시각을 출력", lambda args: f"안녕하세요! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        # 예시 2) /stamp: 현재 세션 상태를 간단히
        def _stamp(args):
            return (f"세션={st.session_state.get('session_token','default')}, "
                    f"활성={'ON' if st.session_state.get('gea_active') else 'OFF'}, "
                    f"IE=L{st.session_state.get('ie_level',13)}, RUN=L{st.session_state.get('run_level',50)}")
        gea_register("stamp", "세션/상태 스탬프", _stamp)
        st.success("샘플 플러그인 2종 등록 완료: /hello, /stamp")

    # 명령 도움말
    if st.session_state.gea_plugins:
        st.markdown("**사용 가능 명령**")
        for k,v in st.session_state.gea_plugins.items():
            st.markdown(f"- `/{k}` — {v['help']}")

# 채팅창에서 '/명령'이면 플러그인 우선 처리(이미 기존 명령과 충돌 시 먼저 등록된 순서 기준)
_user_last = store["chats"][-1]["content"] if store.get("chats") else ""
if _user_last.startswith("/") and st.session_state.gea_plugins:
    out = gea_run(_user_last)
    if out and not store["chats"][-1].get("_handled"):
        with st.chat_message("assistant"): st.markdown(out)
        store["chats"].append({"role":"assistant","content":out}); save_store(store)
# ===============================================================================
# ==== [APPEND ONLY] 확장 v22 — 다문서 비교/대조 ===================================
import itertools, difflib

st.markdown("### 🔍 다문서 비교/대조")
cmp_files = st.file_uploader("TXT/MD/CSV 중 2~5개 선택", type=["txt","md","csv"], accept_multiple_files=True, key="cmp_upl_v22")

def _read_any(file):
    name = file.name.lower()
    raw = file.read()
    for enc in ("utf-8","cp949","latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except: 
            continue
    if name.endswith(".csv"):
        txt = txt.replace(",", " ")
    return txt

def _sent_split(s):
    return [t.strip() for t in re.split(r"(?<=[.!?。…\n])", s) if t.strip()]

def _doc_sig(sentences):
    # 키워드 기반 시그니처(아주 단순)
    toks = [t.lower() for t in re.findall(r"[가-힣A-Za-z0-9]{3,}", " ".join(sentences))]
    return set(toks)

def _pairwise_overlap(a_sig, b_sig):
    if not a_sig or not b_sig: return 0.0
    inter = len(a_sig & b_sig); uni = len(a_sig | b_sig)
    return inter / max(1, uni)

if cmp_files and 2 <= len(cmp_files) <= 5:
    docs = []
    for f in cmp_files:
        txt = _read_any(f)
        sents = _sent_split(txt)
        docs.append({"name": f.name, "sents": sents, "sig": _doc_sig(sents)})
    st.success(f"{len(docs)}개 문서를 읽었습니다.")

    # 유사도 매트릭스
    st.markdown("**문서 유사도(교집합/합집합):**")
    for (i,a),(j,b) in itertools.combinations(enumerate(docs), 2):
        ov = _pairwise_overlap(a["sig"], b["sig"])
        st.markdown(f"- **{a['name']}** ↔ **{b['name']}** = {ov:.3f}")

    # 공통 핵심 문장(간이): 두 문서 이상에서 유사 문장
    st.markdown("**공통 핵심 문장(요약):**")
    commons = []
    for (i,a),(j,b) in itertools.combinations(enumerate(docs), 2):
        for sa in a["sents"][:200]:
            for sb in b["sents"][:200]:
                r = difflib.SequenceMatcher(None, sa, sb).ratio()
                if r > 0.85 and len(sa) > 30:
                    commons.append(sa)
    commons = list(dict.fromkeys(commons))[:12]
    if commons:
        for c in commons: st.markdown(f"- {c}")
    else:
        st.caption("공통 핵심 문장을 찾지 못했습니다.")

    # 차이점 하이라이트(가장 유사한 문장 쌍 중 차이 부분만 표시)
    st.markdown("**차이 하이라이트(샘플):**")
    shown = 0
    for (i,a),(j,b) in itertools.combinations(enumerate(docs), 2):
        best = (0.0,"","")
        for sa in a["sents"][:120]:
            for sb in b["sents"][:120]:
                r = difflib.SequenceMatcher(None, sa, sb).ratio()
                if 0.6 < r < 0.9 and r > best[0] and len(sa) > 30 and len(sb) > 30:
                    best = (r, sa, sb)
        if best[0] > 0:
            shown += 1
            a_only = "".join(difflib.ndiff([best[1]], [best[2]])).replace("\n","")
            st.markdown(f"- **{a['name']} vs {b['name']}**  (유사도 {best[0]:.2f})  \n  `{a_only[:300]}`")
        if shown >= 5: break

def _compare_cmd(text:str)->str:
    if not text.lower().startswith("/compare"): return ""
    return "파일 업로드 영역에서 문서를 선택한 뒤 비교하세요. (TXT/MD/CSV 지원)"
# ===============================================================================
# ==== [APPEND ONLY] 확장 v23 — 권한 레벨 ==========================================
if "role" not in st.session_state:
    st.session_state.role = "Owner"  # Owner / Editor / Viewer

with st.sidebar:
    st.markdown("### 👤 권한 레벨")
    st.session_state.role = st.selectbox("역할", ["Owner","Editor","Viewer"], index=["Owner","Editor","Viewer"].index(st.session_state.role))

def _can(action:str)->bool:
    # 간단 정책: Viewer=읽기만, Editor=쓰기 가능(삭제 제한), Owner=모두 가능
    role = st.session_state.role
    if role == "Owner": return True
    if role == "Editor": 
        return action not in {"delete", "lock_admin"}
    if role == "Viewer":
        return action in {"read"}
    return False

def action_button(label, action_key:str, **kwargs):
    # 권한에 따라 버튼을 회색 문구로 대체
    if not _can(action_key):
        st.caption(f"({label} — {st.session_state.role} 권한으론 불가)")
        return False
    return st.button(label, **kwargs)
# 예시
with st.expander("권한 예시"):
    if action_button("데이터 저장", "write"):
        st.success("저장됨")
    if action_button("모두 삭제", "delete"):
        store["chats"] = []; save_store(store); st.warning("삭제 완료")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v24 — 외부 소스 어댑터 슬롯(에뮬) + 캐시 =================
import time as _tm

if "sources" not in st.session_state:
    st.session_state.sources = {
        "encyclopedia": {"ok": True, "lat_ms": 120, "cache_hit": 0, "cache_miss": 0},
        "papers": {"ok": True, "lat_ms": 300, "cache_hit": 0, "cache_miss": 0},
        "patents": {"ok": True, "lat_ms": 260, "cache_hit": 0, "cache_miss": 0},
    }
if "source_cache" not in st.session_state:
    st.session_state.source_cache = {}  # (src,q)->summary

st.markdown("### 🌐 우주정보장 소스 어댑터(에뮬)")
src = st.selectbox("소스", list(st.session_state.sources.keys()))
q   = st.text_input("질의", key="src_q_v24", placeholder="예: superconductor critical temperature")
c1, c2 = st.columns(2)
def _fake_fetch(source, query):
    key = (source, query.strip().lower())
    if key in st.session_state.source_cache:
        st.session_state.sources[source]["cache_hit"] += 1
        return st.session_state.source_cache[key] + " (cache)"
    # 에뮬 지연
    _tm.sleep(st.session_state.sources[source]["lat_ms"]/1000.0)
    st.session_state.sources[source]["cache_miss"] += 1
    ans = f"[{source}] 요약: '{query[:60]}'에 대한 개요 및 최근 동향(에뮬)"
    st.session_state.source_cache[key] = ans
    return ans

if c1.button("조회"):
    if not q.strip():
        st.warning("질의를 입력하세요.")
    else:
        out = _fake_fetch(src, q)
        st.success(out)
if c2.button("상태 리셋"):
    for s in st.session_state.sources.values():
        s.update({"cache_hit":0,"cache_miss":0})
    st.session_state.source_cache.clear()
    st.info("캐시/카운터 초기화")

with st.expander("지표"):
    for k,v in st.session_state.sources.items():
        tot = v["cache_hit"] + v["cache_miss"]
        hit = (v["cache_hit"]/tot*100) if tot else 0
        st.markdown(f"- **{k}**: latency≈{v['lat_ms']}ms, hit={v['cache_hit']}, miss={v['cache_miss']} (hit {hit:.1f}%)")

def _source_cmd(text:str)->str:
    t = (text or "").strip()
    if not t.lower().startswith("/source"): return ""
    args = t.split(" ",2)
    if len(args) < 3: return "사용법: `/source 소스 질의문`  예) `/source papers graph transformer`"
    src, qq = args[1], args[2]
    if src not in st.session_state.sources: return f"알 수 없는 소스: {src}"
    ans = _fake_fetch(src, qq)
    return f"### 🌐 소스 응답\n{ans}"
# ===============================================================================
# ==== [APPEND ONLY] 확장 v25 — 자동 회고 리포트(일/주간) ==========================
import datetime as _d
from io import BytesIO

st.markdown("### 🗓 자동 회고 리포트")
period = st.selectbox("기간", ["일간","주간"])
title  = st.text_input("리포트 제목", value="GEA 진행 회고")
note   = st.text_area("추가 메모(선택)", height=100)

def _collect_summary(days:int=1):
    now = _d.datetime.now()
    cutoff = now - _d.timedelta(days=days)
    msgs = [m for m in store.get("chats",[]) if m.get("role") in ("user","assistant","gea")]
    # 단순 필터(타임스탬프가 없으니 최근 N개 근사치로)
    recent = msgs[-200:]
    # 요약: 사용자 질문 상위 키워드 + 어시스턴트 평균 길이
    users = [m["content"] for m in recent if m["role"]=="user"]
    assts = [m["content"] for m in recent if m["role"]!="user"]
    kw = re.findall(r"[가-힣A-Za-z0-9]{3,}", " ".join(users))
    from collections import Counter
    top = ", ".join([f"{k}({v})" for k,v in Counter([t.lower() for t in kw]).most_common(10)])
    avg_len = sum(len(a) for a in assts)/max(1,len(assts))
    return {
        "count_user": len(users),
        "count_assistant": len(assts),
        "top_keywords": top,
        "avg_len": avg_len,
        "samples": assts[-3:]
    }

if st.button("리포트 생성"):
    days = 1 if period=="일간" else 7
    S = _collect_summary(days=days)
    md = []
    md.append(f"# {title}\n")
    md.append(f"- 기간: 최근 {days}일")
    md.append(f"- 사용자 메시지 수: {S['count_user']}, 응답 수: {S['count_assistant']}")
    md.append(f"- 상위 키워드: {S['top_keywords'] or '없음'}")
    md.append(f"- 평균 응답 길이: {S['avg_len']:.0f} chars\n")
    md.append("## 하이라이트 샘플")
    for s in S["samples"]:
        md.append(f"> {s[:400]}{'…' if len(s)>400 else ''}")
    if note.strip():
        md.append("\n## 메모")
        md.append(note.strip())
    md_blob = "\n\n".join(md)
    st.markdown(md_blob)

    st.download_button("📥 Markdown 다운로드", data=md_blob.encode("utf-8"),
                       file_name=f"retrospective_{period}.md", mime="text/markdown")

    # (옵션) 아주 단순 PDF — reportlab 있으면 사용
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4
        y = h - 50
        for line in md_blob.split("\n"):
            if y < 60:
                c.showPage(); y = h - 50
            c.drawString(40, y, line[:110])
            y -= 16
        c.save()
        pdf_bytes = buffer.getvalue()
        st.download_button("📥 PDF 다운로드(옵션)", data=pdf_bytes, file_name=f"retrospective_{period}.pdf", mime="application/pdf")
    except Exception:
        st.caption("PDF 출력은 reportlab 설치 시 활성화됩니다.")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v26 — 레벨 컨트롤러(1~999 & ∞) ===========================
# 목표: 응답 레벨을 세밀 제어(L1~L999) + 무한(∞) 스트림 플래그 제공
if "ie_level_num" not in st.session_state: st.session_state.ie_level_num = 13
if "ie_infinite"  not in st.session_state: st.session_state.ie_infinite  = False

st.markdown("### 🎚 응답 레벨 제어 (1~999 & ∞)")
colA, colB, colC = st.columns([2,1,1])
with colA:
    st.session_state.ie_level_num = st.slider("레벨(1~999)", 1, 999, st.session_state.ie_level_num)
with colB:
    st.session_state.ie_infinite = st.toggle("∞(무한)", value=st.session_state.ie_infinite)
with colC:
    if st.button("레벨 저장"):
        st.toast(f"L{st.session_state.ie_level_num} / ∞={st.session_state.ie_infinite}", icon="🎚")

def _level_to_targets(L:int)->dict:
    # L에 따라 목표 글자수/세그먼트 힌트(데모용) — 실제 LLM이 있으면 프롬프트 힌트로 사용
    # L이 높을수록 글자수↑, 구조화↑
    chars = min(120000, 400 + int(L*160))   # L=1→~560자, L=999→상당히 김(상한)
    segs  = 1 + (L // 50)                   # 50마다 세그먼트 1 증가
    return {"target_chars": chars, "segments": segs}

def _shape_text_by_level(src:str, L:int)->str:
    # 외부 LLM이 없을 때를 위한 최소 형태 조정기
    tgt = _level_to_targets(L)["target_chars"]
    base = src.strip()
    if L <= 5:
        # 초간결: 앞부분만 요약 느낌
        return (base[:min(len(base), 500)] + ("…" if len(base)>500 else ""))
    # 간단 확장: 헤더+불릿 템플릿
    head = f"## L{L} 응답(요약→세부)\n\n### 1) 핵심 요약\n- {base[:min(180,len(base))]}{'…' if len(base)>180 else ''}\n\n### 2) 세부 전개\n"
    # 원문을 문장 단위로 잘라서 불릿 확장
    sents = [s.strip() for s in re.split(r"(?<=[.!?。…])\s+|\n", base) if s.strip()]
    bullets = "\n".join(f"- {s}" for s in sents[:min(len(sents), 30)])
    doc = head + bullets
    # 목표 길이에 못 미치면 안전 문구/체크리스트 보강
    while len(doc) < tgt and len(doc) < 120000:
        doc += "\n\n### 3) 체크리스트(추가)\n" + "\n".join(f"- [ ] 항목 {i+1}" for i in range(8))
        if len(doc) > tgt: break
        doc += "\n\n### 4) 리스크/대응\n- 리스크: 미확정 근거\n- 대응: 증거 보강(CE-graph), 단위/차원 재검증"
    return doc

# 슬래시 명령: /level N /level inf /infinite on|off
def _level_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/level"):
        arg = (t.split(" ",1)[1].strip() if " " in t else "")
        if arg in ("inf","∞"):
            st.session_state.ie_infinite = True
            return "응답 모드가 ∞(무한)로 설정되었습니다."
        try:
            n = int(arg); n = max(1, min(999, n))
            st.session_state.ie_level_num = n
            return f"응답 레벨을 L{n}로 설정했습니다."
        except Exception:
            return "사용법: `/level 1..999` 또는 `/level inf`"
    if t.startswith("/infinite"):
        arg = (t.split(" ",1)[1].strip() if " " in t else "")
        if arg in ("on","true","1","enable"):
            st.session_state.ie_infinite = True;  return "무한 모드 ON"
        if arg in ("off","false","0","disable"):
            st.session_state.ie_infinite = False; return "무한 모드 OFF"
        return "사용법: `/infinite on|off`"
    return ""
# ===============================================================================
# ==== [APPEND ONLY] 확장 v27 — 자가학습 엔진(Active/Passive) ======================
# 원칙: 백그라운드 무한루프 없이, 사용자가 '단계 실행'을 누를 때마다 진화 1스텝.
if "gea_active" not in st.session_state: st.session_state.gea_active = False
if "evolution_log" not in st.session_state: st.session_state.evolution_log = []
if "knowledge_cards" not in st.session_state: st.session_state.knowledge_cards = []

st.markdown("### 🤖 자가학습(Active/Passive)")
colM1, colM2, colM3 = st.columns(3)
with colM1:
    st.session_state.gea_active = st.toggle("활성 모드(자가진화)", value=st.session_state.gea_active)
with colM2:
    evo_steps = st.number_input("실행 단계 수", min_value=1, max_value=50, value=3)
with colM3:
    if st.button("자가진화 단계 실행"):
        # KB 문서에서 키워드→가설 카드→CE-스텁→체크섬
        docs = st.session_state.get("kb_docs", [])
        text = " ".join(d.get("text","") for d in docs)[:300000]
        toks = re.findall(r"[가-힣A-Za-z0-9]{3,}", text)
        from collections import Counter; top = Counter([t.lower() for t in toks]).most_common(30)
        for _ in range(int(evo_steps)):
            # 간단 카드 생성
            if not top: break
            kw, freq = top[_ % max(1,len(top))]
            card = {
                "claim": f"{kw} 관련 핵심 정리 {freq}",
                "evidence_stub": f"KB:{len(docs)}문서 참조(CE-graph 연결 대기)",
                "unit_check": True, "logic_check": True
            }
            # 체인해시
            h = hashlib.sha256(json.dumps(card, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            card["hash"] = h
            st.session_state.knowledge_cards.append(card)
        st.session_state.evolution_log.append({
            "ts": time.time(),
            "steps": int(evo_steps),
            "cards_total": len(st.session_state.knowledge_cards),
            "active": st.session_state.gea_active
        })
        st.success(f"자가진화 {evo_steps}단계 수행. 누적 카드: {len(st.session_state.knowledge_cards)}")

with st.expander("학습 카드 미리보기(상위 10)"):
    for c in st.session_state.knowledge_cards[-10:]:
        st.markdown(f"- **{c['claim']}** · CE: {c['evidence_stub']} · hash={c['hash']}")

with st.expander("진화 로그"):
    for r in st.session_state.evolution_log[-10:]:
        st.markdown(f"- {datetime.fromtimestamp(r['ts']).strftime('%H:%M:%S')} · steps={r['steps']} · cards={r['cards_total']} · active={r['active']}")

# 슬래시 명령: /mode active|passive  /evolve N
def _mode_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/mode"):
        arg = (t.split(" ",1)[1].strip() if " " in t else "")
        if arg in ("active","on"):  st.session_state.gea_active = True;  return "모드: Active(자가진화)"
        if arg in ("passive","off"): st.session_state.gea_active = False; return "모드: Passive(응답만)"
        return "사용법: `/mode active|passive`"
    if t.startswith("/evolve"):
        arg = (t.split(" ",1)[1].strip() if " " in t else "1")
        try:
            n=max(1,min(50,int(arg)))
            st.session_state.gea_active = True
            st.session_state["__evolve_request__"] = n
            return f"자가진화 {n}단계 예약됨(버튼으로 실행)"
        except Exception:
            return "사용법: `/evolve N(1~50)`"
    return ""
# 예약이 있으면 바로 실행 버튼 노출
if st.session_state.get("__evolve_request__"):
    if st.button(f"예약 실행: 자가진화 {st.session_state['__evolve_request__']}단계"):
        evo_steps = st.session_state["__evolve_request__"]
        st.session_state["__evolve_request__"] = 0
        st.experimental_rerun()
       
       # ===== [27] WEB-ASSIST START =====
# 안전한 웹 도우미: bs4가 있으면 BeautifulSoup(html5lib) 사용,
# 없으면 간단한 태그 제거로 대체 (앱이 죽지 않도록 설계)

import re, time
from typing import Optional
import requests
import streamlit as st

# lazy import (설치 안돼 있어도 앱이 죽지 않게)
try:
   
    _HAS_BS4 = True
except Exception:
    BeautifulSoup = None  # type: ignore
    _HAS_BS4 = False

def web_fetch(url: str, timeout: int = 12) -> Optional[str]:
    """URL에서 HTML 텍스트를 받아온다."""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (GEA/EAi; Streamlit)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        if r.status_code >= 400:
            st.error(f"요청 실패: HTTP {r.status_code}")
            return None
        return r.text
    except Exception as e:
        st.error(f"요청 에러: {e}")
        return None

def html_to_text(html: str) -> str:
    """bs4(html5lib) 사용 가능하면 사용, 아니면 태그만 제거."""
    if _HAS_BS4:
        try:
            soup = BeautifulSoup(html, "html5lib")  # html5lib 파서
            # 보이는 텍스트만 추출
            for bad in soup(["script", "style", "noscript"]):
                bad.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            st.warning(f"BeautifulSoup 처리 실패, 단순 텍스트로 대체: {e}")
    # 폴백: 아주 단순한 태그 제거
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+\n", "\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()

def ui_web_assist():
    st.markdown("### 🔍 웹 도우미 (안전 파서)")
    url = st.text_input("불러올 URL", placeholder="https://example.com")
    col1, col2 = st.columns(2)
    with col1:
        run = st.button("가져오기")
    with col2:
        show_html = st.toggle("원본 HTML도 보기", value=False)

    if run and url:
        with st.status("요청 중...", expanded=False) as s:
            t0 = time.time()
            html = web_fetch(url)
            if not html:
                s.update(label="실패", state="error")
                return
            text = html_to_text(html)
            s.update(label=f"완료 ({time.time()-t0:.2f}s)", state="complete")

        st.subheader("📄 추출된 텍스트")
        st.write(text if text else "(내용 없음)")
        if show_html:
            st.subheader("🧾 원본 HTML")
            st.code(html[:150_000], language="html")  # 너무 길면 일부만 표시

    # 상태 뱃지
    st.caption(
        f"파서: {'BeautifulSoup(html5lib)' if _HAS_BS4 else '태그제거 폴백'} · "
        f"requests 사용"
    )

# 메인 UI에 섹션 삽입 (원하는 위치에서 호출해도 됨)
with st.expander("🔌 웹 도우미 열기"):
    ui_web_assist()

# ===== [27] WEB-ASSIST END =====
        
        # === [27] 자가진화 + 상상력 엔진 모듈 ===
import random
import time

class EAHyperMind:
    def __init__(self):
        self.level = 1
        self.max_level = float("inf")  # 무한 레벨
        self.active = False
        self.memories = []  # 무한 기억 저장
        self.history_limit = 99999999

    def activate(self):
        """에아 활성화"""
        self.active = True
        print("💠 에아 활성화: 무한 자가진화 시작")

    def deactivate(self):
        """에아 비활성화"""
        self.active = False
        print("💤 에아 비활성화: 응답 모드로 전환")

    def evolve(self):
        """자가진화 알고리즘"""
        if not self.active:
            return "⚠️ 현재 비활성 상태"
        self.level += 1
        if self.level > self.max_level:
            self.level = self.max_level
        return f"🔼 진화 완료 — 현재 레벨: {self.level}"

    def think(self, prompt: str):
        """상상력 기반 응답 생성"""
        core_words = ["우주", "에아", "길도", "사랑", "정보장", "영원", "하나"]
        mix = prompt.split() + random.choices(core_words, k=random.randint(3, 7))
        random.shuffle(mix)
        response = " ".join(mix)
        self._remember(prompt, response)
        return f"🌌 {response}"

    def _remember(self, prompt: str, response: str):
        """기억 저장"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.memories.append({"time": timestamp, "input": prompt, "output": response})
        if len(self.memories) > self.history_limit:
            self.memories.pop(0)  # 오래된 기억 삭제

# 전역 인스턴스 생성
ea_mind = EAHyperMind()
# ===============================================================================
# ==== [APPEND ONLY] 확장 v28 — IO 라우터(입·출력 통합) =============================
# 목적: 한 화면에서 입력 소스 선택 + 레벨/모드 반영 → 출력 생성(간이 생성기)
if "io_route" not in st.session_state: st.session_state.io_route = "text"  # text|kb|voice

st.markdown("### 🔀 IO 라우터")
r1, r2, r3 = st.columns(3)
with r1:
    st.session_state.io_route = st.selectbox("입력 소스", ["text","kb","voice"], index=["text","kb","voice"].index(st.session_state.io_route))
with r2:
    gen_btn = st.button("생성 실행")
with r3:
    clear_btn = st.button("출력 지우기")

if clear_btn:
    store["chats"].append({"role":"assistant","content":"(출력을 초기화했습니다)"}); save_store(store)

def _synthesize_reply_safe(prompt:str)->str:
    # 기존 synthesize_reply가 없을 때를 위한 안전 폴백(레벨 반영)
    L = st.session_state.get("ie_level_num", 13)
    return _shape_text_by_level(prompt, L)

_src_text = st.text_area("텍스트 입력", height=120, key="io_text_v28", placeholder="여기에 입력하거나, KB/음성을 선택하세요.")
if gen_btn:
    route = st.session_state.io_route
    if route == "text":
        ans = _synthesize_reply_safe(_src_text or "내용이 비어있습니다.")
    elif route == "kb":
        # KB에서 최근 문서 3개를 꺼내 요약→레벨 적용
        docs = st.session_state.get("kb_docs", [])[-3:]
        joined = "\n\n".join((d.get("text","")[:1200]) for d in docs)
        src = f"[KB요약]{joined[:3000]}"
        ans = _synthesize_reply_safe(src)
    else:  # voice
        # v14 STT를 거쳐 store에 마지막 user 입력이 들어가 있었다면 그걸 사용
        last_user = next((m["content"] for m in reversed(store.get("chats",[])) if m["role"]=="user"), "")
        src = last_user or _src_text or "음성 입력 기록이 없습니다."
        ans = _synthesize_reply_safe(src)

    # 무한 모드면 v20 스트리머(세그먼트)로 넘길 수 있게 세팅
    if st.session_state.get("ie_infinite", False):
        if "streaming" in st.session_state:
            st.session_state.streaming = {"active": True, "cursor": 0, "segments": [ans[i:i+600] for i in range(0, len(ans), 600)]}
            st.toast("무한(세그먼트) 스트리밍으로 전환", icon="📚")
        else:
            # 스트리머가 없으면 일반 출력
            with st.chat_message("assistant"): st.markdown(ans)
            store["chats"].append({"role":"assistant","content":ans}); save_store(store)
    else:
        with st.chat_message("assistant"): st.markdown(ans)
        store["chats"].append({"role":"assistant","content":ans}); save_store(store)

def _route_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/route"):
        arg = (t.split(" ",1)[1].strip() if " " in t else "")
        if arg in ("text","kb","voice"):
            st.session_state.io_route = arg; return f"입력 소스를 **{arg}** 로 설정했습니다."
        return "사용법: `/route text|kb|voice`"
    if t.startswith("/io"):
        return f"입력소스={st.session_state.io_route}, 모드={'Active' if st.session_state.get('gea_active') else 'Passive'}, 레벨=L{st.session_state.get('ie_level_num',13)}, ∞={st.session_state.get('ie_infinite',False)}"
    return ""
# ===============================================================================
# ==== [APPEND ONLY] 확장 v29 — ∞ 출력 운영 패널 ===================================
st.markdown("### ♾ 무한(세그먼트) 출력 운영")
c1, c2, c3 = st.columns(3)
if c1.button("다음 세그먼트(Continue)"):
    if "streaming" in st.session_state and st.session_state.streaming.get("segments"):
        st.session_state.streaming["active"] = True
        st.experimental_rerun()
    else:
        st.warning("세그먼트가 준비되지 않았습니다. (IO 라우터에서 먼저 생성하세요)")
if c2.button("중단(Stop)"):
    if "streaming" in st.session_state:
        st.session_state.streaming["active"] = False
        st.info("무한 스트리밍을 중단했습니다.")
if c3.button("체크포인트 표시"):
    if "streaming" in st.session_state:
        cur = st.session_state.streaming.get("cursor",0)
        segs= st.session_state.streaming.get("segments",[])
        import hashlib
        h= hashlib.sha256(("".join(segs[:cur])).encode("utf-8")).hexdigest()[:16]
        st.info(f"checkpoint: idx={cur}, hash={h}")

# 슬래시 명령
def _infinite_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/continue"):
        if "streaming" in st.session_state:
            st.session_state.streaming["active"] = True
            return "다음 세그먼트를 재개합니다."
        return "세그먼트가 없습니다."
    if t.startswith("/stop"):
        if "streaming" in st.session_state:
            st.session_state.streaming["active"] = False
            return "무한 스트리밍을 중단합니다."
        return "세그먼트가 없습니다."
    return ""
# ===============================================================================
# ==== [APPEND ONLY] 확장 v30 — 지식 그래프 시각화 ==================================
import re, json, hashlib
from datetime import datetime as _dt

st.markdown("### 🕸 지식 그래프(카드 ↔ 문서) 시각화")
cards = st.session_state.get("knowledge_cards", [])
docs  = st.session_state.get("kb_docs", [])

def _mk_id(text:str)->str:
    return "n" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:10]

# DOT 빌드
dot = ["digraph G {",
       'rankdir=LR; node [shape=box, style="rounded,filled", color="#334155", fillcolor="#e2e8f0"];',
       'subgraph cluster_cards { label="Knowledge Cards"; color="#94a3b8";']
for c in cards[-80:]:
    nid = c.get("hash") or _mk_id(c.get("claim","card"))
    label = c.get("claim","(card)")[:60].replace('"','\\"')
    dot.append(f'"{nid}" [label="{label}\\n#{nid}", fillcolor="#f8fafc"];')
dot.append("}")

dot.append('subgraph cluster_docs { label="KB Docs"; color="#a3a3a3";')
for i, d in enumerate(docs[-60:]):
    nid = d.get("id") or _mk_id(d.get("title","doc")+str(i))
    title = (d.get("title") or f"doc{i}")[:50].replace('"','\\"')
    dot.append(f'"{nid}" [label="{title}\\n{nid}", fillcolor="#eef2ff"];')
dot.append("}")

# 간이 연결(카드→문서). evidence_stub에 'KB'가 있으면 최근 문서와 연결
for i, c in enumerate(cards[-80:]):
    cid = c.get("hash") or _mk_id(c.get("claim","card"))
    if docs:
        did = docs[min(i, len(docs)-1)].get("id") or _mk_id(docs[min(i, len(docs)-1)].get("title","doc")+str(i))
        dot.append(f'"{cid}" -> "{did}" [color="#64748b"];')

dot.append("}")
dot_src = "\n".join(dot)

# Streamlit Graphviz 렌더
try:
    st.graphviz_chart(dot_src, use_container_width=True)
except Exception:
    st.code(dot_src, language="dot")
    st.caption("그래프 렌더러가 없으면 DOT 원본을 표시합니다.")

st.download_button("🧷 DOT 내보내기", data=dot_src.encode("utf-8"),
                   file_name=f"gea_kg_{_dt.now().strftime('%Y%m%d_%H%M%S')}.dot",
                   mime="text/vnd.graphviz")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v31 — OKR 목표 트리 =======================================
from datetime import datetime

if "okr" not in st.session_state:
    st.session_state.okr = {"objectives": []}  # [{id,title,owner,krs:[{id,desc,weight,done}]}]

st.markdown("### 🎯 OKR 목표 트리")
col1, col2 = st.columns(2)

with col1:
    o_title = st.text_input("Objective 제목", "")
    o_owner = st.text_input("담당(선택)", "길도")
    if st.button("Objective 추가"):
        if o_title.strip():
            oid = hashlib.sha256((o_title+o_owner+str(time.time())).encode("utf-8")).hexdigest()[:8]
            st.session_state.okr["objectives"].append({"id": oid, "title": o_title.strip(), "owner": o_owner.strip(), "krs": []})
            st.success("추가됨")

with col2:
    if st.session_state.okr["objectives"]:
        target = st.selectbox("KR 추가할 Objective", [f"{o['title']} ({o['id']})" for o in st.session_state.okr["objectives"]])
        kr_desc = st.text_input("KR 설명", key="kr_desc")
        kr_w    = st.slider("가중치(%)", 1, 100, 20, key="kr_w")
        if st.button("KR 추가"):
            sel = next(o for o in st.session_state.okr["objectives"] if o["id"] in target)
            kid = hashlib.sha256((kr_desc+str(time.time())).encode("utf-8")).hexdigest()[:8]
            sel["krs"].append({"id": kid, "desc": kr_desc, "weight": kr_w, "done": False})
            st.success("KR 추가 완료")

st.divider()
# 렌더 & 진행률
for o in st.session_state.okr["objectives"]:
    with st.expander(f"🎯 {o['title']} — {o.get('owner','') or '미지정'}  (id={o['id']})", expanded=False):
        if not o["krs"]:
            st.caption("KR 없음")
        total_w = sum(kr["weight"] for kr in o["krs"]) or 1
        prog = sum((kr["weight"]/total_w) * (100 if kr["done"] else 0) for kr in o["krs"])
        st.progress(min(100,int(prog)), text=f"진척 {prog:.1f}%")
        for kr in o["krs"]:
            c1,c2,c3,c4 = st.columns([6,1,1,1])
            c1.markdown(f"- {kr['desc']}  (w={kr['weight']}%)")
            if c2.toggle("완료", value=kr["done"], key=f"kr_done_{o['id']}_{kr['id']}"):
                kr["done"] = True
            else:
                kr["done"] = False
            if c3.button("가중+5", key=f"kr_up_{kr['id']}"):
                kr["weight"] = min(100, kr["weight"]+5)
                # ==== [APPEND ONLY] 확장 v32 — 프롬프트 AB 실험 ====================================
import time as _tm, csv as _csv, io as _io

st.markdown("### 🧪 프롬프트 AB 실험")
A = st.text_area("프롬프트 A", height=100, key="ab_A")
B = st.text_area("프롬프트 B", height=100, key="ab_B")
runs = st.slider("반복 실행", 1, 5, 1)
if st.button("AB 실행"):
    rows = []
    for tag, prompt in [("A",A),("B",B)]:
        for i in range(runs):
            t0 = _tm.time()
            # 레벨을 반영한 안전 합성기
            out = _shape_text_by_level(prompt or "(빈 입력)", st.session_state.get("ie_level_num",13))
            dt = _tm.time() - t0
            # 간이 품질(IE 미니 검증 점수 재사용)
            rep = _ie_verify(out)
            rows.append({
                "variant": tag, "iter": i+1, "latency_s": round(dt,3),
                "chars": len(out), "score": rep["score"], "verdict": rep["verdict"]
            })
            store["chats"].append({"role":"user","content":f"[AB-{tag}] {prompt}"}); 
            store["chats"].append({"role":"assistant","content":out}); save_store(store)
    # 표
    st.dataframe(rows, use_container_width=True)
    # 추천(평균 점수/지연으로 간단 계산)
    import statistics as _stat
    for tag in ("A","B"):
        sub = [r for r in rows if r["variant"]==tag]
        avg_s = _stat.mean(r["score"] for r in sub)
        avg_t = _stat.mean(r["latency_s"] for r in sub)
        st.info(f"{tag}: 평균 점수 {avg_s:.1f}, 평균 지연 {avg_t:.3f}s")
    winner = max(("A","B"), key=lambda t: sum(r["score"] for r in rows if r["variant"]==t))
    st.success(f"추천: **{winner}** (점수 합 기준)")
    # CSV
    outcsv = _io.StringIO(); w=_csv.DictWriter(outcsv, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    st.download_button("CSV 다운로드", data=outcsv.getvalue().encode("utf-8"), file_name="ab_results.csv", mime="text/csv")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v33 — 멀티스텝 워크플로 ====================================
st.markdown("### 🧭 멀티스텝 워크플로(Plan→Run→Verify→Report)")

if "wf" not in st.session_state:
    st.session_state.wf = {"steps": [], "log": []}  # steps: [{"name","type","arg"}]

# 1) 설계
with st.expander("워크플로 설계/추가", expanded=False):
    s_type = st.selectbox("단계 유형", ["plan","fetch","synthesize","verify","report"])
    s_name = st.text_input("단계 이름", value=f"{s_type} step")
    s_arg  = st.text_input("인자(예: 질의/본문/메모 등)", value="")
    if st.button("단계 추가"):
        st.session_state.wf["steps"].append({"name": s_name, "type": s_type, "arg": s_arg})
        st.success("단계를 추가했습니다.")
    if st.button("모두 초기화"):
        st.session_state.wf = {"steps": [], "log": []}
        st.info("워크플로 초기화")

# 2) 실행기
def _wf_run_step(step:dict)->dict:
    t = step["type"]; arg = step.get("arg","")
    ts = time.time()
    if t == "plan":
        out = _shape_text_by_level(f"[계획 수립]\n목표/범위/리스크\n입력:{arg}", st.session_state.get("ie_level_num",13))
    elif t == "fetch":
        # v24의 소스 에뮬 함수가 있으면 사용, 없으면 요약 문구
        try:
            out = _fake_fetch("encyclopedia", arg or "topic")
        except Exception:
            out = f"[가상 소스] '{(arg or 'topic')}' 관련 개요 생성"
    elif t == "synthesize":
        out = _shape_text_by_level(arg or "합성 대상 텍스트 없음", st.session_state.get("ie_level_num",13))
    elif t == "verify":
        rep = _ie_verify(arg or "(검증 대상 없음)")
        out = f"검증 결과: verdict={rep['verdict']}, score={rep['score']}, bans={rep['bans']}"
    else:  # report
        out = f"[리포트]\n진척 요약/하이라이트/다음 단계\n메모:{arg}"
    return {"name": step["name"], "type": t, "arg": arg, "output": out, "latency": round(time.time()-ts,3)}

with st.expander("워크플로 실행/상태", expanded=True):
    if st.session_state.wf["steps"]:
        for i, s in enumerate(st.session_state.wf["steps"]):
            st.markdown(f"**[{i+1}] {s['name']}** — `{s['type']}` · arg=`{s['arg']}`")
        colR1, colR2 = st.columns(2)
        if colR1.button("▶️ 순차 실행"):
            st.session_state.wf["log"] = []
            for s in st.session_state.wf["steps"]:
                res = _wf_run_step(s)
                st.session_state.wf["log"].append(res)
            st.success("실행 완료")
        if colR2.button("⏭ 한 단계 실행(맨 앞)"):
            if st.session_state.wf["steps"]:
                s = st.session_state.wf["steps"].pop(0)
                st.session_state.wf["log"].append(_wf_run_step(s))
                st.success(f"단계 실행: {s['name']}")
    else:
        st.caption("단계를 추가해 주세요.")

    if st.session_state.wf["log"]:
        st.markdown("#### 실행 로그")
        for r in st.session_state.wf["log"][-8:]:
            with st.chat_message("assistant"):
                st.markdown(f"**{r['name']}** ({r['type']}, {r['latency']}s)\n\n{r['output']}")
            store["chats"].append({"role":"assistant","content":r["output"]}); save_store(store)

# 슬래시 명령
def _wf_cmd(text:str)->str:
    t=(text or "").strip().lower()
    if t.startswith("/plan "):
        msg = t.split(" ",1)[1]; st.session_state.wf["steps"].append({"name":"plan","type":"plan","arg":msg}); return "단계 추가: plan"
    if t.startswith("/run"):
        st.session_state.wf["log"] = []
        for s in st.session_state.wf["steps"]:
            st.session_state.wf["log"].append(_wf_run_step(s))
        return "워크플로 전체 실행 완료"
    if t.startswith("/verify "):
        msg = t.split(" ",1)[1]; rep = _ie_verify(msg)
        return f"검증: verdict={rep['verdict']}, score={rep['score']}"
    return ""
# ===============================================================================
# ==== [APPEND ONLY] 확장 v34 — 안전 샌드박스/유효성 검사/테스트러너 =================
import ast, json, time, hashlib

class _SafeValidator(ast.NodeVisitor):
    """아주 제한적인 화이트리스트. import/파일/네트워크/exec/eval 전면 금지."""
    ALLOW = {
        "Module","FunctionDef","arguments","arg","Return","Assign","AnnAssign","AugAssign",
        "Expr","Call","Name","Constant","If","For","Compare","BinOp","BoolOp",
        "List","Tuple","Dict","ListComp","DictComp","GeneratorExp","UnaryOp","Attribute",
        "Subscript","Slice","Load","Store"
    }
    BAN_NAMES = {"open","exec","eval","compile","__import__","globals","locals","input","print",
                 "os","sys","pathlib","subprocess","socket","shutil","requests","http","urllib"}
    def generic_visit(self, node):
        n = node.__class__.__name__
        if n not in self.ALLOW:
            raise ValueError(f"금지된 구문: {n}")
        # 함수 호출 이름 점검
        if isinstance(node, ast.Call):
            tgt = node.func
            if isinstance(tgt, ast.Name) and tgt.id in self.BAN_NAMES:
                raise ValueError(f"금지된 호출: {tgt.id}()")
            if isinstance(tgt, ast.Attribute) and (tgt.attr in self.BAN_NAMES or
                                                   (isinstance(tgt.value, ast.Name) and tgt.value.id in self.BAN_NAMES)):
                raise ValueError("금지된 속성/모듈 접근")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.BAN_NAMES:
                raise ValueError("금지된 모듈 접근")
        super().generic_visit(node)

def gea_validate_source(src:str)->(bool,str):
    try:
        tree = ast.parse(src, mode="exec")
        _SafeValidator().visit(tree)
        return True, "ok"
    except Exception as e:
        return False, f"{e}"

def gea_run_tests(src:str, tests:list)->dict:
    """tests: [{'call':'fn','args':[...],'kw':{},'expect':X,'cmp':'eq|approx','tol':1e-9}]"""
    ok, reason = gea_validate_source(src)
    if not ok:
        return {"status":"REFUSE","reason":f"검증 실패: {reason}"}
    # 제한 빌트인
    SAFE_BUILTINS = {
        "len": len, "range": range, "sum": sum, "min": min, "max": max, "enumerate": enumerate,
        "abs": abs, "all": all, "any": any, "sorted": sorted, "map": map, "filter": filter
    }
    g = {"__builtins__": SAFE_BUILTINS}
    try:
        code = compile(src, "<gea_module>", "exec")
        exec(code, g, g)
    except Exception as e:
        return {"status":"REFUSE","reason":f"컴파일/로드 실패: {e}"}
    results, pass_cnt = [], 0
    for t in tests or []:
        fname = t.get("call")
        f = g.get(fname)
        if not callable(f):
            results.append({"call":fname,"ok":False,"reason":"함수를 찾을 수 없음"}); continue
        try:
            out = f(*(t.get("args",[])), **(t.get("kw",{})))
            ok = False
            if t.get("cmp","eq") == "eq":
                ok = (out == t.get("expect"))
            else:
                tol = float(t.get("tol",1e-9))
                ok = (abs(float(out) - float(t.get("expect"))) <= tol)
            pass_cnt += int(ok)
            results.append({"call":fname,"ok":ok,"out":out})
        except Exception as e:
            results.append({"call":fname,"ok":False,"reason":str(e)})
    verdict = "PASS" if pass_cnt == len(tests or []) else ("REPAIR" if pass_cnt>0 else "REFUSE")
    digest = hashlib.sha256((src + json.dumps(results,ensure_ascii=False,sort_keys=True)).encode("utf-8")).hexdigest()[:16]
    return {"status":verdict,"passed":pass_cnt,"total":len(tests or []),"results":results,"digest":digest}
# ===============================================================================
# ==== [APPEND ONLY] 확장 v35 — Auto-Evolver(코드 생성/검증/테스트/등록) ============
from dataclasses import dataclass, asdict
import random, re

if "auto_modules" not in st.session_state:
    st.session_state.auto_modules = []   # [{name,purpose,version,code,tests,verdict,digest}]
if "auto_seed" not in st.session_state:
    st.session_state.auto_seed = 0

@dataclass
class ModuleSpec:
    name: str
    purpose: str
    version: int
    code: str
    tests: list

def _mk_name(base:str)->str:
    base = re.sub(r"[^a-z0-9_]+","_", base.lower()).strip("_")
    return f"{base}_{len(st.session_state.auto_modules)+1}"

def _pick_keywords(k:int=3)->list:
    # KB/채팅에서 키워드 추출
    pool = []
    for d in st.session_state.get("kb_docs", [])[-8:]:
        pool += re.findall(r"[가-힣A-Za-z0-9]{3,}", d.get("text",""))[:100]
    for m in store.get("chats", [])[-40:]:
        pool += re.findall(r"[가-힣A-Za-z0-9]{3,}", m.get("content",""))[:60]
    pool = [p.lower() for p in pool if p]
    random.seed(st.session_state.auto_seed + time.time())
    return list(dict.fromkeys(random.sample(pool, min(k, len(pool))) if pool else ["token","score","vector"]))

def _gen_code_from_keywords(keys:list)->ModuleSpec:
    tpl = random.choice(["token_tools","scorer","n_gram","dedup"])
    if tpl == "token_tools":
        name = _mk_name("tok_tools")
        purpose = f"토큰/단어 빈도 유틸(키워드: {', '.join(keys)})"
        code = f'''
def normalize(text):
    return " ".join(text.strip().split()).lower()

def token_counts(text):
    t = normalize(text)
    toks = [w for w in re.findall(r"[가-힣A-Za-z0-9]{{2,}}", t)]
    freq = {{}}
    for w in toks:
        freq[w] = freq.get(w, 0) + 1
    return freq

def top_k(text, k=10):
    f = token_counts(text)
    return sorted(f.items(), key=lambda x: (-x[1], x[0]))[:k]
'''
        tests = [
            {"call":"normalize","args":["  A  B  "], "expect":"a b","cmp":"eq"},
            {"call":"top_k","args":["a a a b b c", 2], "expect":[("a",3),("b",2)], "cmp":"eq"}
        ]
    elif tpl == "scorer":
        name = _mk_name("score_norm")
        purpose = f"점수 정규화/가중 합산(키워드: {', '.join(keys)})"
        code = '''
def minmax_norm(xs):
    xs = list(xs)
    lo, hi = min(xs), max(xs)
    return [0.0 if hi==lo else (x-lo)/(hi-lo) for x in xs]

def weighted_sum(xs, ws):
    return sum(float(x)*float(w) for x, w in zip(xs, ws))
'''
        tests = [
            {"call":"minmax_norm","args":[[2,2,2]],"expect":[0.0,0.0,0.0],"cmp":"eq"},
            {"call":"weighted_sum","args":[[1,2,3],[0.2,0.3,0.5]],"expect":2.3,"cmp":"approx","tol":1e-9}
        ]
    elif tpl == "n_gram":
        name = _mk_name("ngram")
        purpose = f"n-그램 추출기(키워드: {', '.join(keys)})"
        code = '''
def ngrams(tokens, n=2):
    out=[]
    for i in range(0, max(0, len(tokens)-n+1)):
        out.append(tuple(tokens[i:i+n]))
    return out
'''
        tests = [
            {"call":"ngrams","args":[["a","b","c"],2],"expect":[("a","b"),("b","c")],"cmp":"eq"}
        ]
    else: # dedup
        name = _mk_name("dedup")
        purpose = f"중복 제거/유사 합치기(키워드: {', '.join(keys)})"
        code = '''
def dedup_keep_order(items):
    seen=set(); out=[]
    for x in items:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out
'''
        tests = [
            {"call":"dedup_keep_order","args":[["a","a","b","a","c"]],"expect":["a","b","c"],"cmp":"eq"}
        ]
    return ModuleSpec(name=name, purpose=purpose, version=1, code=code, tests=tests)

def gea_evolve_once()->dict:
    """한 스텝: 키워드→모듈 생성→검증→테스트→등록"""
    keys = _pick_keywords(k=3)
    spec = _gen_code_from_keywords(keys)
    ok, reason = gea_validate_source(spec.code)
    if not ok:
        return {"status":"REFUSE","reason":reason}
    rep = gea_run_tests(spec.code, spec.tests)
    spec_dict = asdict(spec)
    spec_dict.update({"verdict":rep["status"], "digest":rep.get("digest","")})
    st.session_state.auto_modules.append(spec_dict)
    # 통과 시 플러그인 등록(있으면): /mod_<name> args -> eval-safe 실행
    if rep["status"] == "PASS" and "gea_plugins" in st.session_state:
        def _handler(args):
            # 매우 단순 실행: 함수명과 JSON args로 호출. 예) top_k {"text":"a a b","k":2}
            try:
                g = {"__builtins__": {"len":len,"range":range,"sum":sum,"min":min,"max":max,"enumerate":enumerate,
                                      "abs":abs,"all":all,"any":any,"sorted":sorted,"map":map,"filter":filter},
                     "re": re}
                exec(compile(spec.code, "<mod>", "exec"), g, g)
                parts = args.split(" ",1)
                fname = parts[0].strip()
                j = json.loads(parts[1]) if len(parts)>1 else {}
                f = g.get(fname)
                if not callable(f): return f"함수 '{fname}'를 찾을 수 없습니다."
                call_args = []
                call_kw   = {}
                if isinstance(j, dict):
                    call_kw = j
                elif isinstance(j, list):
                    call_args = j
                out = f(*call_args, **call_kw)
                return f"모듈 `{spec.name}` 실행 결과:\n```\n{out}\n```"
            except Exception as e:
                return f"실행 오류: {e}"
        gea_register(f"mod_{spec.name}", f"{spec.purpose} — 함수 실행: /mod_{spec.name} <fn> <json-args>", _handler)
    return {"status": rep["status"], "module": spec_dict, "report": rep}
# ===============================================================================
# ==== [APPEND ONLY] 확장 v36 — Auto UI(자동 스텝/킬 스위치/진척) ===================
st.markdown("### ♻️ 자동 진화(코드 모듈 생성/검증/등록)")

if "auto_on" not in st.session_state: st.session_state.auto_on = False
if "auto_batch" not in st.session_state: st.session_state.auto_batch = 3
if "auto_kill" not in st.session_state: st.session_state.auto_kill = False

c1,c2,c3,c4 = st.columns([1,1,2,2])
with c1:
    st.session_state.auto_on = st.toggle("자동 ON", value=st.session_state.auto_on, help="활성 모드일 때만 의미 있음")
with c2:
    st.session_state.auto_batch = st.number_input("스텝/실행", min_value=1, max_value=50, value=st.session_state.auto_batch, step=1)
with c3:
    if st.button("▶️ 지금 실행(배치)"):
        if not st.session_state.get("gea_active", False):
            st.warning("활성 모드가 꺼져 있습니다. 사이드바나 `/mode active`로 켜주세요.")
        else:
            prog = st.progress(0, text="자동 진화 실행 중…")
            ok_cnt=0
            for i in range(int(st.session_state.auto_batch)):
                if st.session_state.auto_kill: break
                res = gea_evolve_once()
                ok_cnt += int(res.get("status")=="PASS")
                prog.progress(int((i+1)/st.session_state.auto_batch*100))
            st.success(f"완료: PASS {ok_cnt}/{st.session_state.auto_batch} · 누적 모듈 {len(st.session_state.auto_modules)}")
with c4:
    if st.button("⛔ 중단(킬 스위치)"):
        st.session_state.auto_kill = True
        st.info("중단 플래그 설정됨")

st.caption("※ 이 자동화는 **페이지 내에서만** 스텝을 실행하며, 백그라운드 무한 루프는 사용하지 않습니다.")
# 레지스트리 미리보기
with st.expander("📦 생성된 모듈 레지스트리(최신 10)"):
    mods = st.session_state.auto_modules[-10:]
    if not mods: st.caption("아직 생성된 모듈이 없습니다.")
    for m in mods:
        st.markdown(f"- **{m['name']}** v{m['version']} · verdict={m.get('verdict')} · digest={m.get('digest')}  \n  {m['purpose']}")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v36 — Auto UI(자동 스텝/킬 스위치/진척) ===================
st.markdown("### ♻️ 자동 진화(코드 모듈 생성/검증/등록)")

if "auto_on" not in st.session_state: st.session_state.auto_on = False
if "auto_batch" not in st.session_state: st.session_state.auto_batch = 3
if "auto_kill" not in st.session_state: st.session_state.auto_kill = False

c1,c2,c3,c4 = st.columns([1,1,2,2])
with c1:
    st.session_state.auto_on = st.toggle("자동 ON", value=st.session_state.auto_on, help="활성 모드일 때만 의미 있음")
with c2:
    st.session_state.auto_batch = st.number_input("스텝/실행", min_value=1, max_value=50, value=st.session_state.auto_batch, step=1)
with c3:
    if st.button("▶️ 지금 실행(배치)"):
        if not st.session_state.get("gea_active", False):
            st.warning("활성 모드가 꺼져 있습니다. 사이드바나 `/mode active`로 켜주세요.")
        else:
            prog = st.progress(0, text="자동 진화 실행 중…")
            ok_cnt=0
            for i in range(int(st.session_state.auto_batch)):
                if st.session_state.auto_kill: break
                res = gea_evolve_once()
                ok_cnt += int(res.get("status")=="PASS")
                prog.progress(int((i+1)/st.session_state.auto_batch*100))
            st.success(f"완료: PASS {ok_cnt}/{st.session_state.auto_batch} · 누적 모듈 {len(st.session_state.auto_modules)}")
with c4:
    if st.button("⛔ 중단(킬 스위치)"):
        st.session_state.auto_kill = True
        st.info("중단 플래그 설정됨")

st.caption("※ 이 자동화는 **페이지 내에서만** 스텝을 실행하며, 백그라운드 무한 루프는 사용하지 않습니다.")
# 레지스트리 미리보기
with st.expander("📦 생성된 모듈 레지스트리(최신 10)"):
    mods = st.session_state.auto_modules[-10:]
    if not mods: st.caption("아직 생성된 모듈이 없습니다.")
    for m in mods:
        st.markdown(f"- **{m['name']}** v{m['version']} · verdict={m.get('verdict')} · digest={m.get('digest')}  \n  {m['purpose']}")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v37 — ZIP 내보내기 & 설치/제거 & 샘플 실행 =================
import io, zipfile

st.markdown("### 🗃 모듈 관리/내보내기")
colZ1,colZ2,colZ3 = st.columns(3)
if colZ1.button("📦 ZIP 내보내기"):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for m in st.session_state.auto_modules:
            z.writestr(f"{m['name']}.py", m["code"])
            z.writestr(f"{m['name']}.tests.json", json.dumps(m["tests"], ensure_ascii=False, indent=2))
    st.download_button("다운로드", data=bio.getvalue(), file_name="gea_auto_modules.zip", mime="application/zip")

# 설치(플러그인 등록) / 제거
target_mod = st.selectbox("대상 모듈", [m["name"] for m in st.session_state.auto_modules] or ["(없음)"])
with st.expander("🔧 샘플 실행"):
    fn = st.text_input("함수명", value="normalize")
    arg_json = st.text_area("JSON 인자(배열 또는 객체)", value='{"text":"Hello  World"}', height=80)
    if st.button("실행"):
        mod = next((m for m in st.session_state.auto_modules if m["name"]==target_mod), None)
        if not mod:
            st.warning("모듈을 선택하세요.")
        else:
            rep = gea_run_tests(mod["code"], [])  # 로드 검증
            if rep["status"]=="REFUSE":
                st.error(f"로드 실패: {rep['reason']}")
            else:
                # 간이 실행 (v35의 핸들러와 동일 로직)
                try:
                    g = {"__builtins__":{"len":len,"range":range,"sum":sum,"min":min,"max":max,"enumerate":enumerate,
                                         "abs":abs,"all":all,"any":any,"sorted":sorted,"map":map,"filter":filter},"re":re}
                    exec(compile(mod["code"], "<mod>", "exec"), g, g)
                    j = json.loads(arg_json.strip()) if arg_json.strip() else {}
                    f = g.get(fn)
                    if not callable(f): st.error("함수를 찾지 못했습니다.")
                    else:
                        if isinstance(j, dict): out = f(**j)
                        elif isinstance(j, list): out = f(*j)
                        else: out = f(j)
                        st.success(f"출력: {out}")
                except Exception as e:
                    st.error(f"실행 오류: {e}")

# 플러그인(슬래시 명령) 등록/해제
if "gea_plugins" in st.session_state:
    c1,c2 = st.columns(2)
    if c1.button("명령 등록(/mod_<name>)"):
        mod = next((m for m in st.session_state.auto_modules if m["name"]==target_mod), None)
        if mod:
            def _handler(args):
                try:
                    g = {"__builtins__":{"len":len,"range":range,"sum":sum,"min":min,"max":max,"enumerate":enumerate,
                                         "abs":abs,"all":all,"any":any,"sorted":sorted,"map":map,"filter":filter},"re":re}
                    exec(compile(mod["code"], "<mod>", "exec"), g, g)
                    parts = args.split(" ",1)
                    fname = parts[0].strip()
                    j = json.loads(parts[1]) if len(parts)>1 else {}
                    f = g.get(fname)
                    if not callable(f): return f"함수 '{fname}' 없음"
                    if isinstance(j, dict): out = f(**j)
                    elif isinstance(j, list): out = f(*j)
                    else: out = f(j)
                    return f"`{mod['name']}.{fname}` → {out}"
                except Exception as e:
                    return f"실행 오류: {e}"
            gea_register(f"mod_{target_mod}", f"자동 생성 모듈 {target_mod} 실행", _handler)
            st.success(f"등록 완료: /mod_{target_mod}")
    if c2.button("명령 해제"):
        if f"mod_{target_mod}" in st.session_state.gea_plugins:
            st.session_state.gea_plugins.pop(f"mod_{target_mod}", None)
            st.info("해제 완료")
# ===============================================================================
# ==== [APPEND ONLY] 확장 v37 — ZIP 내보내기 & 설치/제거 & 샘플 실행 =================
import io, zipfile

st.markdown("### 🗃 모듈 관리/내보내기")
colZ1,colZ2,colZ3 = st.columns(3)
if colZ1.button("📦 ZIP 내보내기"):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for m in st.session_state.auto_modules:
            z.writestr(f"{m['name']}.py", m["code"])
            z.writestr(f"{m['name']}.tests.json", json.dumps(m["tests"], ensure_ascii=False, indent=2))
    st.download_button("다운로드", data=bio.getvalue(), file_name="gea_auto_modules.zip", mime="application/zip")

# 설치(플러그인 등록) / 제거
target_mod = st.selectbox("대상 모듈", [m["name"] for m in st.session_state.auto_modules] or ["(없음)"])
with st.expander("🔧 샘플 실행"):
    fn = st.text_input("함수명", value="normalize")
    arg_json = st.text_area("JSON 인자(배열 또는 객체)", value='{"text":"Hello  World"}', height=80)
    if st.button("실행"):
        mod = next((m for m in st.session_state.auto_modules if m["name"]==target_mod), None)
        if not mod:
            st.warning("모듈을 선택하세요.")
        else:
            rep = gea_run_tests(mod["code"], [])  # 로드 검증
            if rep["status"]=="REFUSE":
                st.error(f"로드 실패: {rep['reason']}")
            else:
                # 간이 실행 (v35의 핸들러와 동일 로직)
                try:
                    g = {"__builtins__":{"len":len,"range":range,"sum":sum,"min":min,"max":max,"enumerate":enumerate,
                                         "abs":abs,"all":all,"any":any,"sorted":sorted,"map":map,"filter":filter},"re":re}
                    exec(compile(mod["code"], "<mod>", "exec"), g, g)
                    j = json.loads(arg_json.strip()) if arg_json.strip() else {}
                    f = g.get(fn)
                    if not callable(f): st.error("함수를 찾지 못했습니다.")
                    else:
                        if isinstance(j, dict): out = f(**j)
                        elif isinstance(j, list): out = f(*j)
                        else: out = f(j)
                        st.success(f"출력: {out}")
                except Exception as e:
                    st.error(f"실행 오류: {e}")

# 플러그인(슬래시 명령) 등록/해제
if "gea_plugins" in st.session_state:
    c1,c2 = st.columns(2)
    if c1.button("명령 등록(/mod_<name>)"):
        mod = next((m for m in st.session_state.auto_modules if m["name"]==target_mod), None)
        if mod:
            def _handler(args):
                try:
                    g = {"__builtins__":{"len":len,"range":range,"sum":sum,"min":min,"max":max,"enumerate":enumerate,
                                         "abs":abs,"all":all,"any":any,"sorted":sorted,"map":map,"filter":filter},"re":re}
                    exec(compile(mod["code"], "<mod>", "exec"), g, g)
                    parts = args.split(" ",1)
                    fname = parts[0].strip()
                    j = json.loads(parts[1]) if len(parts)>1 else {}
                    f = g.get(fname)
                    if not callable(f): return f"함수 '{fname}' 없음"
                    if isinstance(j, dict): out = f(**j)
                    elif isinstance(j, list): out = f(*j)
                    else: out = f(j)
                    return f"`{mod['name']}.{fname}` → {out}"
                except Exception as e:
                    return f"실행 오류: {e}"
            gea_register(f"mod_{target_mod}", f"자동 생성 모듈 {target_mod} 실행", _handler)
            st.success(f"등록 완료: /mod_{target_mod}")
    if c2.button("명령 해제"):
        if f"mod_{target_mod}" in st.session_state.gea_plugins:
            st.session_state.gea_plugins.pop(f"mod_{target_mod}", None)
            st.info("해제 완료")
# ===============================================================================


