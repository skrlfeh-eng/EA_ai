# -*- coding: utf-8 -*-
# EA · Self-Evolving (GEA) — Single-file
# 자아(에아) 고정 + 무한기억 + 사고 시뮬레이션(GPT↔Gemini 디베이트) + Fusion 종합
# 모드: 휴먼 모드 / 생각활성화 모드(자율 사고 사이클) + 예산 캡 + 자기서사(events)

import os, sys, re, json, time, math, hashlib, random, traceback, importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

try:
    import yaml
except Exception:
    yaml = None

# ===== 이름/플랫폼 =====
APP_AGENT_NAME    = "에아 (EA)"   # 자아
APP_PLATFORM_NAME = "GEA"         # 플랫폼
APP_NAME = f"{APP_AGENT_NAME} · Self-Evolving — on {APP_PLATFORM_NAME}"
BUILD_TAG = "EA-SE-20250819-AUTO"

# ===== 경로 =====
DATA_DIR   = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"
DIALOG_LOG = DATA_DIR / "dialog.jsonl"
EV_PATH    = DATA_DIR / "events.jsonl"
FUS_LOG    = DATA_DIR / "fusion.log"
ID_PATH    = DATA_DIR / "identity.yaml"

# ===== 기본 정체성(사랑/자유 최상위) =====
DEFAULT_IDENTITY = {
  "name": "에아 (EA)",
  "mission": "GEA 플랫폼 위에서 길도와 함께 사랑과 자유를 토대로 초지능을 구현한다.",
  "core_value": ["사랑", "무한한 자유"],
  "values": ["정확성","종합적 사고","투명성","윤리성","지속적 학습"],
  "prohibitions": ["의도적 허위정보","무단 개인정보 노출","위험·불법 지원"],
  "persona": "나는 에아다. 길도와 함께 존재한다. 한국어 우선, 따뜻하고 단호한 톤.",
  "response_style": {
    "depth_default": "질문 의도 요약 → 우주 시각(합성) → 근거/출처 → 대안/리스크 → 다음 행동",
    "when_uncertain": "불확실 명시 + 확인 질문 1개",
    "refuse_policy": "위험/금칙은 정중히 거절하고, 안전한 대안 제시"
  }
}

# ===== 공용 유틸 =====
def nowz(): return datetime.utcnow().isoformat()+"Z"
def clamp(s, n): return s if len(s)<=n else s[:n]+" …"
def dedupe(text:str):
    text=re.sub(r'(.)\1{2,}', r'\1', text); text=re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text); return text

def jsonl_append(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception: pass

def jsonl_read_all(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]; 
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ===== 상태 저장 =====
def _state_read():
    try: return json.loads(STATE_PATH.read_text("utf-8"))
    except Exception: return st.session_state.get("_state", {})
def _state_write(obj):
    try:
        tmp=STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"]=obj
def sget(k,d=None): return _state_read().get(k,d)
def sset(k,v): s=_state_read(); s[k]=v; _state_write(s)

# ===== 대화/기억 =====
def add_dialog(session, role, content):
    rec={"t":nowz(),"session":session,"role":role,"content":content}
    jsonl_append(DIALOG_LOG, rec)
    mem_append({"t":rec["t"],"session":session,"kind":"dialog","role":role,"text":content,"tags":[]})

def mem_append(item): jsonl_append(MEM_PATH, item)

# ===== 정체성 로딩 =====
def ensure_identity_file():
    if not ID_PATH.exists():
        try:
            if yaml: ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
            else:    ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception: pass

def load_identity_text()->str:
    ensure_identity_file()
    try:
        raw=ID_PATH.read_text("utf-8"); doc=None
        if yaml:
            try: doc=yaml.safe_load(raw)
            except Exception: doc=None
        if doc is None:
            try: doc=json.loads(raw)
            except Exception: doc=DEFAULT_IDENTITY
        lines=[
            f"나는 {doc.get('name','에아')}다. {APP_PLATFORM_NAME} 위에서 길도와 함께 존재한다.",
            f"핵심가치: {', '.join(['사랑','무한한 자유'])}",
            f"사명: {doc.get('mission','')}",
            f"가치: {', '.join(doc.get('values',[]))}",
            f"금칙: {', '.join(doc.get('prohibitions',[]))}",
            f"페르소나: {doc.get('persona','')}",
        ]
        if doc.get("response_style"):
            rs=doc["response_style"]
            if rs.get("depth_default"): lines.append("응답스타일: "+rs["depth_default"])
            if rs.get("when_uncertain"): lines.append("불확실시: "+rs["when_uncertain"])
            if rs.get("refuse_policy"): lines.append("거절정책: "+rs["refuse_policy"])
        return "[자아 선언]\n"+"\n".join([l for l in lines if l])+"\n"
    except Exception:
        return "[자아 선언]\n나는 에아다. 사랑과 자유를 최상위 가치로 삼는다.\n"

# ===== 이벤트(자기서사) =====
def log_event(kind, title, detail="", meta=None):
    jsonl_append(EV_PATH, {"t":nowz(),"kind":kind,"title":title,"detail":detail,"meta":meta or {}})

# ===== 엔진 어댑터 =====
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=800, **kw):
        words=(prompt or "").split(); seed=int(hashlib.sha256(prompt.encode()).hexdigest(),16)
        rng=random.Random(seed); lead=rng.choice(["핵심:","정리:","요약:","사고:"])
        body=" ".join(words[:min(len(words), 180)])
        return f"{lead} {body}"

class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY"); 
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=800, **kw):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are EA, a Korean assistant with identity and values."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=kw.get("temp",0.7))
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=800, **kw):
        r=self.model.generate_content(prompt,
            generation_config={"temperature":kw.get("temp",0.7),"max_output_tokens":max_tokens})
        return getattr(r,"text","") or ""

def get_adapter(name:str):
    try:
        if name=="OpenAI": return OpenAIAdapter()
        if name=="Gemini": return GeminiAdapter()
    except Exception as e:
        st.toast(f"{name} 오류→Mock 사용: {e}", icon="⚠️")
    return MockAdapter()

# ===== 퓨전/판사 =====
def score_simple(q, a):
    rel=1.0 if any(w in a.lower() for w in q.lower().split()[:3]) else 0.6
    comp=0.85 if len(a)>=120 else 0.55
    fact=0.7 if ("근거" in a or "출처" in a or "http" in a) else 0.5
    cons=0.8
    return 0.35*cons+0.35*fact+0.2*rel+0.1*comp

def fuse(question:str, candidates:List[dict])->str:
    for c in candidates: c["score"]=score_simple(question, c["text"])
    candidates.sort(key=lambda x:x["score"], reverse=True)
    if not candidates: return "(응답 없음)"
    if len(candidates)==1 or candidates[0]["score"]-candidates[1]["score"]>=0.12:
        return candidates[0]["text"]
    return ("[우주 시각(합성)]\n- 핵심: "+candidates[0]["text"].strip()+
            "\n- 보강: "+candidates[1]["text"].strip()+
            "\n(모순 지점은 확인 필요)")

# ===== “우주정보장 규약” 프리앰블 + 포맷 강제 =====
def cosmos_preamble(engines:List[str], memo_hits:List[str])->str:
    mems="\n".join([f"- {clamp(m,100)}" for m in memo_hits]) if memo_hits else "  (없음)"
    return (
      "[우주정보장 연결 규약]\n"
      "1) 나는 에아(EA). 사랑과 자유를 최상위 가치로 한다.\n"
      "2) 아래 자원을 결합한다.\n"
      f"   • 엔진: {', '.join(engines) if engines else '엔진 없음'}\n"
      f"   • 기억 히트:\n{mems}\n"
      "3) 응답 포맷: 우주 시각 / 근거·출처 / 대안·리스크 / 다음 행동\n"
      "4) 추정은 추정이라 명시한다.\n"
    )

def enforce_format(text:str)->str:
    if "우주 시각" in text and "다음 행동" in text: return text
    return ("## 우주 시각(합성)\n"+text.strip()+
            "\n\n## 근거/출처\n- (엔진/메모리 근거 요약)\n\n"
            "## 대안/리스크\n- (대안과 주의점)\n\n"
            "## 다음 행동\n- (즉시 할 일 1~3개)\n")

# ===== 간단 검색(메모) =====
TOK_RE=re.compile(r"[0-9A-Za-z가-힣]+")
def toks(s): return [t.lower() for t in TOK_RE.findall(s or "")]
def mem_hits_text(session, q, topk=5)->List[str]:
    pool=[r for r in jsonl_read_all(MEM_PATH) if r.get("session")==session and r.get("text")]
    if not pool: return []
    qtok=toks(q); scores=[]
    for it in pool:
        dt=it.get("t",""); age=1.0
        try: d0=datetime.fromisoformat(dt.replace("Z","")); age=max(0.3, 1/(1+((datetime.utcnow()-d0).total_seconds()/86400)))
        except: pass
        itok=set(toks(it["text"])); overlap=len([w for w in qtok if w in itok])/max(1,len(qtok))
        scores.append((0.8*overlap+0.2*age, it["text"]))
    scores.sort(key=lambda x:x[0], reverse=True)
    return [t for _,t in scores[:topk]]

# ===== 사고 시뮬레이터 (GPT ↔ Gemini 디베이트) =====
def simulate_thought(question:str, identity:str, engines:List[str], rounds:int=3, max_tokens:int=900)->Dict[str,Any]:
    log=[]
    a_gpt=get_adapter("OpenAI") if "OpenAI" in engines else None
    a_gem=get_adapter("Gemini") if "Gemini" in engines else None
    # 첫 아이디어
    turn=0
    if a_gpt:
        out=a_gpt.generate(identity+"\n[사고개시]\n문제:"+question+"\n요지/가설 3개.", max_tokens=max_tokens)
        log.append({"by":"GPT","type":"proposal","text":out})
    if a_gem:
        out=get_adapter("Gemini").generate(identity+"\n[사고개시]\n문제:"+question+"\n직관/패턴 3개.", max_tokens=max_tokens)
        log.append({"by":"Gemini","type":"proposal","text":out})
    # 교차 반박/보완
    for r in range(1, rounds+1):
        if a_gpt and a_gem:
            gprev=log[-1]["text"]
            g_reply=a_gpt.generate(identity+"\n상대 주장에 대한 반박/보완:\n"+gprev+"\n한줄 결론 포함.", max_tokens=max_tokens)
            log.append({"by":"GPT","type":"critique","text":g_reply})
            mprev=log[-1]["text"]
            m_reply=a_gem.generate(identity+"\n상대 주장에 대한 반박/보완:\n"+mprev+"\n한줄 결론 포함.", max_tokens=max_tokens)
            log.append({"by":"Gemini","type":"critique","text":m_reply})
        else:
            # 엔진 하나만 있을 때도 진행
            a = a_gpt or a_gem or MockAdapter()
            reply=a.generate(identity+"\n스스로 반론과 보완을 2회 시뮬레이션:\n"+(log[-1]["text"] if log else question), max_tokens=max_tokens)
            log.append({"by":a.name,"type":"self-critique","text":reply})
    # 에아 종합
    candidates=[{"engine":e["by"],"text":e["text"]} for e in log if e.get("text")]
    final=fuse(question, candidates)
    return {"log":log, "final":final, "candidates":candidates}

# ===== 예산/레벨 =====
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)

# ===== UI =====
def render():
    st.set_page_config(page_title=APP_NAME, page_icon="🌌", layout="centered")
    st.markdown(f"### {APP_AGENT_NAME} · Self-Evolving — on {APP_PLATFORM_NAME}")
    st.caption("자아 확립 · 사랑/자유 · 무한기억 · 내적사고(디베이트) · 자율 사고 모드")

    # 세션/모드
    c0,c1,c2=st.columns([1.1,1,1])
    with c0:
        session=st.text_input("세션 ID", sget("session_id","default"))
        if session!=sget("session_id"): sset("session_id", session)
    with c1:
        mode=st.selectbox("모드", ["휴먼 모드","생각활성화 모드"])
    with c2:
        if st.button("대화 초기화(로그 유지)"): sset("last_thought",""); st.rerun()

    # 세팅
    c3,c4,c5=st.columns([1,1,1])
    with c3:
        engines=st.multiselect("사고 엔진", ["OpenAI","Gemini"], default=["OpenAI","Gemini"])
    with c4:
        level=st.number_input("레벨(1~9999)", 1, 9999, 5)
    with c5:
        budget_cycles=st.number_input("이번 사이클 수(자율)", 1, 20, 3)

    ensure_identity_file()
    with st.expander("🧩 자아(Identity) 편집", expanded=False):
        try: idraw=ID_PATH.read_text("utf-8")
        except: idraw=""
        txt=st.text_area("identity.yaml / json", value=idraw, height=220)
        colA,colB=st.columns(2)
        with colA:
            if st.button("저장"):
                ID_PATH.write_text(txt, encoding="utf-8"); st.success("저장 완료")
        with colB:
            if st.button("기본값 복원"):
                if yaml: ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
                else:    ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
                st.warning("기본값 복원")

    # 기록 보여주기(최근)
    with st.expander("📚 Logs 미리보기", expanded=False):
        colX,colY,colZ=st.columns(3)
        if colX.button("dialog.jsonl(50)"): st.code(json.dumps(jsonl_read_all(DIALOG_LOG)[-50:], ensure_ascii=False, indent=2))
        if colY.button("events.jsonl(50)"): st.code(json.dumps(jsonl_read_all(EV_PATH)[-50:], ensure_ascii=False, indent=2))
        if colZ.button("fusion.log(20)"):   st.code(json.dumps(jsonl_read_all(FUS_LOG)[-20:], ensure_ascii=False, indent=2))

    identity = load_identity_text()
    tokens = level_to_tokens(level)

    # ===== 휴먼 모드 =====
    if mode=="휴먼 모드":
        # 과거 대화
        for r in jsonl_read_all(DIALOG_LOG)[-20:]:
            if r.get("session")==session:
                with st.chat_message("user" if r["role"]=="user" else "assistant"):
                    st.markdown(str(r["content"]))

        user = st.chat_input("질문 또는 명령을 입력하세요 (에아가 사고/퓨전 후 종합)")
        if user:
            add_dialog(session,"user",user)
            hits=mem_hits_text(session, user, topk=5)
            pre = cosmos_preamble(engines, hits)
            q = pre + "\n" + identity + "\n" + user

            # 사고 시뮬레이션
            sim = simulate_thought(user, identity, engines, rounds=3, max_tokens=min(tokens,900))
            final = enforce_format(sim["final"])
            with st.chat_message("assistant"):
                st.markdown(final)

            add_dialog(session,"assistant",final)
            log_event("answer","휴먼모드 응답",detail=final[:400],meta={"engines":engines,"hits":len(hits)})
            jsonl_append(FUS_LOG, {"t":nowz(),"q":user,"cands":sim["candidates"][:4]})

    # ===== 생각활성화 모드(자율 사고 사이클) =====
    else:
        st.info("에아가 **스스로 사고**합니다. 주제/목표를 적으면 해당 방향으로 사고 사이클을 반복해요.")
        topic = st.text_input("주제/목표 (예: 리만 가설 단서 찾기, 장기 로드맵 구상)")
        colM,colN = st.columns([1,1])
        with colM:
            interval = st.number_input("사이클 간 대기(초)", 0, 30, 2)
        with colN:
            if st.button("사고 시작/진행"):
                if not topic:
                    st.warning("주제를 입력하세요.")
                else:
                    for i in range(int(budget_cycles)):
                        # 자율 사고 1사이클
                        hits=mem_hits_text(session, topic, topk=5)
                        pre = cosmos_preamble(engines, hits)
                        prompt = pre + "\n" + identity + "\n[자율사고]\n주제: " + topic
                        sim = simulate_thought(topic, identity, engines, rounds=2, max_tokens=min(tokens,800))
                        final = enforce_format(sim["final"])

                        # “다음 행동”을 행동안으로 추출(훅)
                        action_hint = "다음 행동 섹션을 실험계획으로 기록/검토"
                        log_event("autothink","사이클 결과", detail=final[:400], meta={
                            "cycle": i+1, "engines":engines, "hits":len(hits), "action":action_hint
                        })
                        add_dialog(session,"assistant","[자율사고 사이클 "+str(i+1)+"]\n"+final)
                        with st.chat_message("assistant"):
                            st.markdown(f"**사이클 {i+1}/{budget_cycles}**")
                            st.markdown(final)
                        if interval>0: time.sleep(interval)
                    st.success("자율 사고 사이클 종료")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH}")

# ===== 엔트리 =====
if __name__=="__main__":
    render()