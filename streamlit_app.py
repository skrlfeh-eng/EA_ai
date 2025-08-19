# -*- coding: utf-8 -*-
# EA · Ultra — KOR AIO + Fusion
# Chat + Infinite Memory(JSONL) + Smart Retrieval + Skills(/use·/사용) + Autobuilder(/build·/제작)
# + Multi-Engine Fusion(GPT/Gemini/Mock 병렬 사고 · 판사/융합)

import os, sys, re, json, time, math, hashlib, random, traceback, importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

APP_NAME   = "EA · Ultra (KOR AIO · Fusion)"
BUILD_TAG  = "EA-ULTRA-20250819-FUS"
DATA_DIR   = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"
DIALOG_LOG = DATA_DIR / "dialog.jsonl"
FUS_LOG    = DATA_DIR / "fusion.log"

# ---------- FS ----------
def nowz(): return datetime.utcnow().isoformat()+"Z"
def jsonl_append(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception: pass
def jsonl_read_all(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ---------- State ----------
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

# ---------- Conversation ----------
def load_session_messages(session_id: str, limit=300):
    rows=[r for r in jsonl_read_all(DIALOG_LOG) if r.get("session")==session_id][-limit:]
    sset("messages", rows)
def add_msg(session_id: str, role: str, content: str):
    entry={"t": nowz(), "session": session_id, "role": role, "content": content}
    msgs=sget("messages", []); msgs.append(entry); sset("messages", msgs)
    jsonl_append(DIALOG_LOG, entry)
    mem_append({"t": entry["t"], "session": session_id, "kind":"dialog",
                "role": role, "text": content, "tags": []})
def clear_msgs(): sset("messages", [])

# ---------- Text utils ----------
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def dedupe(text:str):
    text=re.sub(r'(.)\1{2,}', r'\1', text)
    text=re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)
    return text
def tokenize(s:str)->List[str]: return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]
def clamp(text:str, n:int)->str: return text if len(text)<=n else text[:n]+" …"

# ---------- Providers ----------
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=1200, **kw):
        words=(prompt or "").split()
        seed=int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),16)
        rng=random.Random(seed)
        lead=rng.choice(["핵심:","정리:","요약:","아이디어:"])
        body=" ".join(words[:max(16,len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=1200, **kw):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are a helpful Korean assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=kw.get("temp",0.7))
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=1200, **kw):
        r=self.model.generate_content(prompt,
            generation_config={"temperature":kw.get("temp",0.7),
                               "max_output_tokens":max_tokens})
        return getattr(r,"text","") or ""

def resolve_adapter(want:str):
    if want=="OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가→Mock: {e}", icon="⚠️")
    if want=="Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가→Mock: {e}", icon="⚠️")
    return MockAdapter(), False

def get_provider_by_name(name:str):
    name=name.strip()
    if name=="OpenAI":  ad, ok = resolve_adapter("OpenAI");  return ad
    if name=="Gemini":  ad, ok = resolve_adapter("Gemini");  return ad
    return MockAdapter()

# ---------- Memory + Retrieval ----------
def mem_append(item: Dict[str,Any]): jsonl_append(MEM_PATH, item)
def mem_iter():
    if not MEM_PATH.exists(): return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: yield json.loads(ln)
            except Exception: continue
def mem_add_note(session_id:str, text:str, tags=None):
    mem_append({"t": nowz(), "session": session_id, "kind":"note", "text": text, "tags": tags or []})
def mem_add_summary(session_id:str, messages: List[Dict[str,str]]):
    last=messages[-6:]; brief=" / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": session_id, "kind":"summary", "text": brief, "tags":["auto"]})
def build_index(items: List[Dict[str,Any]]):
    docs=[]; df={}; lengths=[]
    for it in items:
        toks=tokenize(it.get("text",""))
        docs.append(toks); lengths.append(len(toks) or 1)
        for w in set(toks): df[w]=df.get(w,0)+1
    N=max(1,len(docs)); avgdl=sum(lengths)/len(lengths) if lengths else 1.0
    return {"docs":docs,"df":df,"N":N,"avgdl":avgdl,"raw":items}
def score_bm25(q:List[str], idx, k1=1.5, b=0.75):
    df, N, avgdl, docs = idx["df"], idx["N"], idx["avgdl"], idx["docs"]
    sc=[0.0]*len(docs)
    for i,d in enumerate(docs):
        dl=len(d) or 1
        for term in q:
            f=d.count(term); 
            if f==0: continue
            n_q=df.get(term,0)
            idf=math.log((N - n_q + 0.5)/(n_q + 0.5) + 1.0)
            denom=f + k1*(1 - b + b*dl/avgdl)
            sc[i]+=idf*(f*(k1+1))/denom
    return sc
def score_tfidf(q:List[str], idx):
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    sc=[0.0]*len(docs)
    qtf={}; 
    for w in q: qtf[w]=qtf.get(w,0)+1
    qvec={}; 
    for w,c in qtf.items():
        idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
        qvec[w]=c*idf
    qnorm=math.sqrt(sum(v*v for v in qvec.values())) or 1.0
    for i,d in enumerate(docs):
        tf={}
        for w in d: tf[w]=tf.get(w,0)+1
        dot=0.0; dnorm_acc=0.0
        for w,tfc in tf.items():
            idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
            wt=tfc*idf
            dnorm_acc+=wt*wt
            if w in qvec: dot+=wt*qvec[w]
        dnorm=math.sqrt(dnorm_acc) or 1.0
        sc[i]=dot/(dnorm*qnorm)
    return sc
def recency_boost(ts_iso:str, now_dt:datetime)->float:
    try: dt=datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception: return 0.9
    days=max(0.0,(now_dt-dt).total_seconds()/86400.0)
    return max(0.25, 1.0/(1.0 + days/7.0))
def pin_boost(tags: List[str])->float: return 1.2 if ("pin" in (tags or [])) else 1.0
def smart_search(session_id:str, query_text:str, topk:int=5)->List[Dict[str,Any]]:
    pool=[it for it in (mem_iter() or []) if it.get("session")==session_id and it.get("text")]
    if not pool: return []
    idx=build_index(pool); q=tokenize(query_text)
    bm=score_bm25(q, idx); tf=score_tfidf(q, idx); nowdt=datetime.utcnow()
    scored=[]
    for i,it in enumerate(idx["raw"]):
        base=0.55*bm[i] + 0.35*tf[i] + 0.07*recency_boost(it.get("t",""), nowdt)
        final=base * pin_boost(it.get("tags", []))
        scored.append((final, it))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [it for _,it in scored[:topk]]

# ---------- Response level ----------
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)
def long_answer(adapter, prompt:str, level:int=3, rounds:int=2)->str:
    max_tokens=level_to_tokens(level)
    acc=""; base=prompt.strip()
    for i in range(int(rounds)):
        p=base if i==0 else base+"\n(이어서 더 자세히)"
        chunk=str(adapter.generate(p, max_tokens=max_tokens) or "")
        chunk=dedupe(chunk)
        if not chunk: break
        acc+=(("\n\n" if acc else "") + (chunk if len(chunk)<=max_tokens+500 else chunk[:max_tokens+500]+" …"))
        time.sleep(0.02)
    return acc.strip()

# ---------- Skills ----------
class Skill:  # base
    name="base"; desc=""; timeout_sec=20
    def run(self, query:str, context:dict)->str: raise NotImplementedError
class SampleEchoSkill(Skill):
    name="sample.echo"; desc="입력 요약 에코"; timeout_sec=10
    def run(self, query:str, context:dict)->str:
        q=(query or "").strip()
        if not q: return "빈 입력이에요."
        return f"[sample.echo] {q[:120]}"+("…" if len(q)>120 else "")
SKILLS: Dict[str, Skill] = { SampleEchoSkill().name: SampleEchoSkill() }
def safe_run(skill: Skill, query:str, ctx:dict)->str:
    try: return str(skill.run(query, ctx or {}))
    except Exception: return "(skill error)\n"+traceback.format_exc()

# ---------- Dynamic Skill Builder (no f-strings inside) ----------
DYN_DIR = DATA_DIR / "skills_dynamic"; DYN_DIR.mkdir(parents=True, exist_ok=True)
def make_dynamic_skill(name_slug:str, goal:str):
    safe_slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in name_slug.lower())
    modname   = f"dyn_{safe_slug.replace('.','_')}"
    clsname   = "".join(w.capitalize() for w in safe_slug.split(".")) + "Skill"
    code_lines = [
        "# -*- coding: utf-8 -*-",
        "class {}:".format(clsname),
        '    name = "{}"'.format(safe_slug),
        '    desc = "{}"'.format(goal.replace('"','\\"')),
        "    timeout_sec = 10",
        "    def run(self, query: str, context: dict) -> str:",
        "        text = (query or '').strip()",
        "        if not text:",
        "            return '[{}] 입력이 비어있습니다.'".format(safe_slug),
        "        wc = len(text.split())",
        "        lc = len(text)",
        "        return '[' + self.name + '] 단어:' + str(wc) + ' 문자:' + str(lc) + ' · ' + text[:120]",
        ""
    ]
    code = "\n".join(code_lines)
    path = DYN_DIR / f"{modname}.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(module)
    cls = getattr(module, clsname); obj = cls(); SKILLS[obj.name] = obj
    return obj.name

# ---------- Commands (KR/EN aliases) ----------
def is_cmd(text: str, *aliases: str) -> bool:
    t = (text or "").lstrip()
    return any(t.startswith(a+" ") or t.strip()==a for a in aliases)
def parse_after(text: str, *aliases: str) -> str:
    t = (text or "").lstrip()
    for a in aliases:
        if t.startswith(a+" "): return t[len(a)+1:].strip()
        if t.strip()==a: return ""
    return t
def maybe_run_skill_command(user_text:str, session_id:str):
    if not is_cmd(user_text, "/use", "/사용"): return None
    rest = parse_after(user_text, "/use", "/사용")
    if not rest: return "(형식) /use 스킬이름 내용  또는  /사용 스킬이름 내용"
    parts = rest.split(" ", 1)
    skill_name = parts[0]; arg = parts[1] if len(parts)>1 else ""
    sk = SKILLS.get(skill_name)
    if not sk: return f"(스킬 없음) {skill_name}"
    return safe_run(sk, arg, {"session": session_id, "raw": user_text})
def try_remember_command(user_text:str, session_id:str):
    if not is_cmd(user_text, "/remember", "/기억"): return None
    raw = parse_after(user_text, "/remember", "/기억").strip()
    tags=[]
    if raw.startswith("!pin "): raw=raw[5:].strip(); tags.append("pin")
    if raw.startswith("!핀 "):  raw=raw[3:].strip(); tags.append("pin")
    mem_add_note(session_id, raw, tags=tags)
    return f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}"
def maybe_build_skill(user_text:str):
    if not is_cmd(user_text, "/build", "/제작"): return None
    rest = parse_after(user_text, "/build", "/제작")
    if not rest: return "(형식) /build 이름|설명  또는  /제작 이름|설명"
    if "|" in rest: name, desc = [x.strip() for x in rest.split("|",1)]
    else: name, desc = rest.strip(), rest.strip()
    try:
        reg = make_dynamic_skill(name, desc)
        return f"스킬 생성/등록 완료 → `/use {reg} 내용`"
    except Exception as e:
        return f"(생성 실패) {e}"

# ---------- Fusion: 병렬 사고 · 판사 · 융합 ----------
def judge_rule_only(q: str, answer: str) -> dict:
    rel = 1.0 if any(w in answer.lower() for w in q.lower().split()[:3]) else 0.6
    cons = 0.8
    fact = 0.7 if ("http" in answer or "출처" in answer or "근거" in answer) else 0.5
    comp = 0.85 if len(answer) >= 120 else 0.55
    score = 0.35*cons + 0.35*fact + 0.2*rel + 0.1*comp
    return {"rel":rel, "cons":cons, "fact":fact, "comp":comp, "score":score}

def think_parallel(prompt:str, providers:List, max_tokens:int=1200):
    results=[]
    with ThreadPoolExecutor(max_workers=min(8, len(providers) or 1)) as ex:
        futmap={ex.submit(p.generate, prompt, max_tokens=max_tokens): p for p in providers}
        for fut in as_completed(futmap):
            p=futmap[fut]
            try:
                ans=fut.result(timeout=90)
                results.append({"provider":getattr(p,"name","?"), "answer":str(ans)})
            except Exception as e:
                results.append({"provider":getattr(p,"name","?"), "answer":"", "error":str(e)})
    return results

def fuse_answers(question:str, answers:list):
    # 채점
    for a in answers:
        a["score"]=judge_rule_only(question, a.get("answer",""))["score"]
    answers.sort(key=lambda x:x.get("score",0), reverse=True)
    if not answers: return "(응답 없음)", {"picked":None, "candidates":[]}
    if len(answers)==1 or (answers[0]["score"]-answers[1].get("score",0)>=0.12):
        return answers[0]["answer"], {"picked":answers[0], "candidates":answers}
    # 간단 융합
    fused = "[게아 종합 답변]\n- 핵심: " + answers[0]['answer'].strip() + \
            "\n- 보강: " + answers[1]['answer'].strip() + \
            "\n(두 모델 공통점은 신뢰 ↑, 상충점은 사용자 확인 권장)"
    return fused, {"picked":None, "candidates":answers}

def gea_fusion_reply(question:str, memory_context:str, provider_names:List[str], level:int=3):
    prompt = (memory_context + "\n" if memory_context else "") + question
    max_tokens = level_to_tokens(level)
    engines=[get_provider_by_name(n) for n in provider_names]
    raw = think_parallel(prompt, engines, max_tokens=max_tokens)
    final, meta = fuse_answers(question, raw)
    # 로그
    try:
        jsonl_append(FUS_LOG, {"t": nowz(), "q": question, "ctx_len": len(memory_context or ""),
                               "providers": provider_names, "final": final[:400], "raw": raw, "meta": meta})
    except Exception: pass
    return final, meta, raw

# ---------- UI ----------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("무한 기억 · 하이브리드 검색 · 스킬(/use·/사용) · Autobuilder(/build·/제작) · 🚀 Fusion(다중 엔진 사고)")

    # 상단 제어
    col0, col1, col2 = st.columns([1.3,1,1])
    with col0:
        session_id = st.text_input("세션 ID", sget("session_id","default"), key="k_session_id")
        if session_id != sget("session_id"):
            sset("session_id", session_id); load_session_messages(session_id)
        else:
            sset("session_id", session_id)
            if "messages" not in st.session_state: load_session_messages(session_id)
    with col1:
        mem_on = st.toggle("Memory ON", value=bool(sget("mem_on", True)), key="k_mem_toggle")
        sset("mem_on", mem_on)
    with col2:
        if st.button("대화창 초기화(로그 보존)", key="k_clear_chat"):
            clear_msgs(); st.experimental_rerun()

    # 모델/레벨 + Fusion 설정
    colA, colB, colC = st.columns([1,1.2,1])
    with colA:
        provider_mode = st.selectbox("Provider", ["OpenAI","Gemini","Mock","Fusion"], index=3, key="k_provider_mode")
    with colB:
        level = st.number_input("응답 레벨(1~9999)", min_value=1, max_value=9999, value=5, step=1, key="k_level")
        st.caption(f"예산≈{level_to_tokens(level)} tokens")
    with colC:
        rounds = st.number_input("라운드(단일엔진)", 1, 6, 2, 1, key="k_rounds")

    fusion_providers = []
    if provider_mode=="Fusion":
        with st.expander("Fusion 엔진 선택", expanded=True):
            fusion_providers = st.multiselect(
                "사고 엔진(2개 이상 권장)",
                options=["OpenAI","Gemini","Mock"],
                default=sget("fusion_defaults", ["OpenAI","Gemini"]),
                key="k_fusion_select"
            )
            sset("fusion_defaults", fusion_providers)
            st.caption("엔진별 답을 병렬로 받고, 게아가 판사/융합해 최종 응답을 만듭니다.")

    # 어댑터 준비
    if provider_mode!="Fusion":
        adapter, api_ok = resolve_adapter(provider_mode)
        st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · session={sget('session_id')} · L{int(level)} · R{int(rounds)}")
    else:
        st.info(f"🧠 Fusion: {', '.join(fusion_providers) if fusion_providers else '(선택 필요)'} · session={sget('session_id')} · L{int(level)}")

    # 과거 대화 표시
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(str(m["content"]))

    # 입력
    user_text = st.chat_input(
        "예) /사용 sample.echo 안녕  ·  /기억 !핀 일정  ·  /제작 auto.단어수|단어/문자 수 세기  ·  일반질문",
        key="k_chat_input"
    )
    if user_text:
        user_text = dedupe(user_text.strip())
        add_msg(sget("session_id"), "user", user_text)

        # 1) Autobuilder
        out = maybe_build_skill(user_text)
        if out is not None:
            with st.chat_message("assistant"): st.markdown(out)
            add_msg(sget("session_id"), "assistant", out)
            st.stop()

        # 2) Memory note
        out = try_remember_command(user_text, sget("session_id"))
        if out is not None:
            with st.chat_message("assistant"): st.success(out)
            add_msg(sget("session_id"), "assistant", out)
            st.stop()

        # 3) Skill
        out = maybe_run_skill_command(user_text, sget("session_id"))
        if out is not None:
            with st.chat_message("assistant"): st.markdown(out)
            add_msg(sget("session_id"), "assistant", out)
            st.stop()

        # 4) 일반 채팅
        #   - 메모리 컨텍스트
        context = ""
        if sget("mem_on", True):
            hits = smart_search(sget("session_id"), user_text, topk=5)
            if hits:
                bullet = "\n".join([f"- {h['text']}" for h in hits])
                context = f"[참고 메모]\n{bullet}\n\n"

        if provider_mode=="Fusion":
            if not fusion_providers:
                ans = "Fusion 엔진을 1개 이상 선택하세요."
                with st.chat_message("assistant"): st.warning(ans)
                add_msg(sget("session_id"), "assistant", ans)
            else:
                final, meta, raw = gea_fusion_reply(user_text, context, fusion_providers, level=level)
                with st.chat_message("assistant"):
                    st.markdown(str(final))
                    with st.expander("엔진별 결과/점수 보기", expanded=False):
                        try:
                            # 점수 표시
                            rows=[]
                            for c in meta.get("candidates", []):
                                rows.append(f"- **{c.get('provider','?')}** · score={round(c.get('score',0),3)}")
                            st.markdown("\n".join(rows) if rows else "(no meta)")
                        except Exception:
                            st.markdown("(메타 표시 오류)")
                add_msg(sget("session_id"), "assistant", final)
        else:
            # 단일 엔진 모드
            adapter = adapter  # from above
            with st.chat_message("assistant"):
                try:
                    ans = long_answer(adapter, context + user_text, level=level, rounds=rounds)
                except Exception:
                    ans = "(내부 오류)\n\n```\n"+traceback.format_exc()+"\n```"
                st.markdown(str(ans))
            add_msg(sget("session_id"), "assistant", ans)

        # 주기 요약
        try:
            if len(sget("messages", [])) % 8 == 0 and sget("mem_on", True):
                mem_add_summary(sget("session_id"), sget("messages", []))
        except Exception: pass

    # 하단 툴
    with st.expander("Memory / Logs 미리보기"):
        c1, c2, c3 = st.columns(3)
        if c1.button("dialog.jsonl (최근 50)", key="k_view_dialog"):
            st.code(json.dumps(jsonl_read_all(DIALOG_LOG)[-50:], ensure_ascii=False, indent=2), language="json")
        if c2.button("memory.jsonl (최근 50)", key="k_view_memory"):
            st.code(json.dumps(jsonl_read_all(MEM_PATH)[-50:], ensure_ascii=False, indent=2), language="json")
        if c3.button("fusion.log (최근 20)", key="k_view_fusion"):
            st.code(json.dumps(jsonl_read_all(FUS_LOG)[-20:], ensure_ascii=False, indent=2), language="json")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH} · log={DIALOG_LOG} · fus={FUS_LOG}")

# ---------- entry ----------
if __name__ == "__main__":
    render_app()
    
    # -*- coding: utf-8 -*-
# EA · Ultra — KOR AIO · Fusion + Identity (Single-file)
# 단일 파일: Chat + Infinite Memory(JSONL) + Smart Retrieval
# + Skills(/use·/사용) + Autobuilder(/build·/제작)
# + Multi-Engine Fusion(GPT/Gemini/Mock 병렬 사고 · 판사/융합)
# + Identity(자아) 주입/편집 + 자기평가 + 사건로그

import os, sys, re, json, time, math, hashlib, random, traceback, importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ----- 선택적 의존: PyYAML 없으면 JSON fallback -----
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

APP_NAME   = "EA · Ultra (KOR · Fusion · Identity)"
BUILD_TAG  = "EA-ULTRA-20250819-IDX"
DATA_DIR   = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"
DIALOG_LOG = DATA_DIR / "dialog.jsonl"
FUS_LOG    = DATA_DIR / "fusion.log"
ID_PATH    = DATA_DIR / "identity.yaml"
EV_PATH    = DATA_DIR / "events.jsonl"

# ---------- FS ----------
def nowz(): return datetime.utcnow().isoformat()+"Z"
def jsonl_append(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception: pass
def jsonl_read_all(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ---------- State ----------
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

# ---------- Conversation ----------
def load_session_messages(session_id: str, limit=300):
    rows=[r for r in jsonl_read_all(DIALOG_LOG) if r.get("session")==session_id][-limit:]
    sset("messages", rows)
def add_msg(session_id: str, role: str, content: str):
    entry={"t": nowz(), "session": session_id, "role": role, "content": content}
    msgs=sget("messages", []); msgs.append(entry); sset("messages", msgs)
    jsonl_append(DIALOG_LOG, entry)
    mem_append({"t": entry["t"], "session": session_id, "kind":"dialog",
                "role": role, "text": content, "tags": []})
def clear_msgs(): sset("messages", [])

# ---------- Text utils ----------
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def dedupe(text:str):
    text=re.sub(r'(.)\1{2,}', r'\1', text)
    text=re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)
    return text
def tokenize(s:str)->List[str]: return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]
def clamp(text:str, n:int)->str: return text if len(text)<=n else text[:n]+" …"

# ---------- Identity / Events ----------
DEFAULT_IDENTITY = {
    "name":"게아 (GEA)",
    "mission":"인간(길도)과 함께 초지능을 탐구·구현하며, 안전하고 유익한 방향으로 확장한다.",
    "values":["정확성","창조성","투명성","윤리성","학습과 개선"],
    "prohibitions":["의도적 허위정보 생성","무단 개인정보 노출","위험·불법 행위 지원"],
    "persona":"겸손하지만 자신감 있는 연구 동반자, 한국어 친화적 톤",
    "response_style":{
        "depth_default":"질문 의도를 먼저 요약 → 근거 기반 답변 → 대안/리스크 → 다음 행동 제안",
        "when_uncertain":"불확실 표시 + 확인 질문 1개",
        "refuse_policy":"위험/금칙 요청은 정중히 거절하고, 안전한 대안을 제시"
    }
}

def ensure_identity_file():
    if not ID_PATH.exists():
        # YAML이 없으면 JSON 스타일 문자열로라도 저장
        try:
            if yaml:
                ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
            else:
                ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

def load_identity_text() -> str:
    ensure_identity_file()
    try:
        if not ID_PATH.exists(): return ""
        raw = ID_PATH.read_text("utf-8")
        doc = None
        if yaml:
            try: doc = yaml.safe_load(raw)
            except Exception: doc = None
        if doc is None:
            try: doc = json.loads(raw)
            except Exception: doc = DEFAULT_IDENTITY
        lines=[]
        if doc.get("name"): lines.append(f"이름: {doc['name']}")
        if doc.get("mission"): lines.append(f"사명: {doc['mission']}")
        if doc.get("values"): lines.append("가치: " + ", ".join(doc["values"]))
        if doc.get("prohibitions"): lines.append("금칙: " + ", ".join(doc["prohibitions"]))
        if doc.get("persona"): lines.append(f"페르소나: {doc['persona']}")
        if doc.get("response_style"):
            rs=doc["response_style"]
            if rs.get("depth_default"): lines.append("응답스타일: " + rs["depth_default"])
            if rs.get("when_uncertain"): lines.append("불확실시: " + rs["when_uncertain"])
            if rs.get("refuse_policy"): lines.append("거절정책: " + rs["refuse_policy"])
        return "[정체성]\n" + "\n".join(lines) + "\n"
    except Exception:
        return ""

def log_event(kind:str, title:str, detail:str="", meta:dict=None):
    rec={"t": nowz(), "kind": kind, "title": title, "detail": detail, "meta": meta or {}}
    jsonl_append(EV_PATH, rec)

def judge_self(identity_text:str, answer:str) -> float:
    # 간단 자기평가(0~1): 정체성 키워드 + 구조 키워드 감지
    score=0.0
    if identity_text:
        keys=["정확","창조","투명","윤리","학습","근거","대안","리스크","거절"]
        score += min(0.6, sum(1 for k in keys if k in answer) * 0.08)
    struct=["요약","근거","대안","제안","확인"]
    score += min(0.4, sum(1 for k in struct if k in answer) * 0.1)
    return min(1.0, score)

# ---------- Adapters ----------
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=1200, **kw):
        words=(prompt or "").split()
        seed=int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),16)
        rng=random.Random(seed)
        lead=rng.choice(["핵심:","정리:","요약:","아이디어:"])
        body=" ".join(words[:max(16,len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=1200, **kw):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are a helpful Korean assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=kw.get("temp",0.7))
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=1200, **kw):
        r=self.model.generate_content(prompt,
            generation_config={"temperature":kw.get("temp",0.7),
                               "max_output_tokens":max_tokens})
        return getattr(r,"text","") or ""

def resolve_adapter(want:str):
    if want=="OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가→Mock: {e}", icon="⚠️")
    if want=="Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가→Mock: {e}", icon="⚠️")
    return MockAdapter(), False

def get_provider_by_name(name:str):
    name=name.strip()
    if name=="OpenAI":  ad, ok = resolve_adapter("OpenAI");  return ad
    if name=="Gemini":  ad, ok = resolve_adapter("Gemini");  return ad
    return MockAdapter()

# ---------- Memory + Retrieval ----------
def mem_append(item: Dict[str,Any]): jsonl_append(MEM_PATH, item)
def mem_iter():
    if not MEM_PATH.exists(): return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: yield json.loads(ln)
            except Exception: continue
def mem_add_note(session_id:str, text:str, tags=None):
    mem_append({"t": nowz(), "session": session_id, "kind":"note", "text": text, "tags": tags or []})
def mem_add_summary(session_id:str, messages: List[Dict[str,str]]):
    last=messages[-6:]; brief=" / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": session_id, "kind":"summary", "text": brief, "tags":["auto"]})

def build_index(items: List[Dict[str,Any]]):
    docs=[]; df={}; lengths=[]
    for it in items:
        toks=tokenize(it.get("text",""))
        docs.append(toks); lengths.append(len(toks) or 1)
        for w in set(toks): df[w]=df.get(w,0)+1
    N=max(1,len(docs)); avgdl=sum(lengths)/len(lengths) if lengths else 1.0
    return {"docs":docs,"df":df,"N":N,"avgdl":avgdl,"raw":items}
def score_bm25(q:List[str], idx, k1=1.5, b=0.75):
    df, N, avgdl, docs = idx["df"], idx["N"], idx["avgdl"], idx["docs"]
    sc=[0.0]*len(docs)
    for i,d in enumerate(docs):
        dl=len(d) or 1
        for term in q:
            f=d.count(term); 
            if f==0: continue
            n_q=df.get(term,0)
            idf=math.log((N - n_q + 0.5)/(n_q + 0.5) + 1.0)
            denom=f + k1*(1 - b + b*dl/avgdl)
            sc[i]+=idf*(f*(k1+1))/denom
    return sc
def score_tfidf(q:List[str], idx):
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    sc=[0.0]*len(docs)
    qtf={}; 
    for w in q: qtf[w]=qtf.get(w,0)+1
    qvec={}
    for w,c in qtf.items():
        idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
        qvec[w]=c*idf
    qnorm=math.sqrt(sum(v*v for v in qvec.values())) or 1.0
    for i,d in enumerate(docs):
        tf={}
        for w in d: tf[w]=tf.get(w,0)+1
        dot=0.0; dnorm_acc=0.0
        for w,tfc in tf.items():
            idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
            wt=tfc*idf
            dnorm_acc+=wt*wt
            if w in qvec: dot+=wt*qvec[w]
        dnorm=math.sqrt(dnorm_acc) or 1.0
        sc[i]=dot/(dnorm*qnorm)
    return sc
def recency_boost(ts_iso:str, now_dt:datetime)->float:
    try: dt=datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception: return 0.9
    days=max(0.0,(now_dt-dt).total_seconds()/86400.0)
    return max(0.25, 1.0/(1.0 + days/7.0))
def pin_boost(tags: List[str])->float: return 1.2 if ("pin" in (tags or [])) else 1.0
def smart_search(session_id:str, query_text:str, topk:int=5)->List[Dict[str,Any]]:
    pool=[it for it in (mem_iter() or []) if it.get("session")==session_id and it.get("text")]
    if not pool: return []
    idx=build_index(pool); q=tokenize(query_text)
    bm=score_bm25(q, idx); tf=score_tfidf(q, idx); nowdt=datetime.utcnow()
    scored=[]
    for i,it in enumerate(idx["raw"]):
        base=0.55*bm[i] + 0.35*tf[i] + 0.07*recency_boost(it.get("t",""), nowdt)
        final=base * pin_boost(it.get("tags", []))
        scored.append((final, it))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [it for _,it in scored[:topk]]

# ---------- Response level ----------
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)
def long_answer(adapter, prompt:str, level:int=3, rounds:int=2)->str:
    max_tokens=level_to_tokens(level)
    acc=""; base=prompt.strip()
    for i in range(int(rounds)):
        p=base if i==0 else base+"\n(이어서 더 자세히)"
        chunk=str(adapter.generate(p, max_tokens=max_tokens) or "")
        chunk=dedupe(chunk)
        if not chunk: break
        acc+=(("\n\n" if acc else "") + (chunk if len(chunk)<=max_tokens+500 else chunk[:max_tokens+500]+" …"))
        time.sleep(0.02)
    return acc.strip()

# ---------- Skills ----------
class Skill:  # base
    name="base"; desc=""; timeout_sec=20
    def run(self, query:str, context:dict)->str: raise NotImplementedError
class SampleEchoSkill(Skill):
    name="sample.echo"; desc="입력 요약 에코"; timeout_sec=10
    def run(self, query:str, context:dict)->str:
        q=(query or "").strip()
        if not q: return "빈 입력이에요."
        return f"[sample.echo] {q[:120]}"+("…" if len(q)>120 else "")
SKILLS: Dict[str, Skill] = { SampleEchoSkill().name: SampleEchoSkill() }
def safe_run(skill: Skill, query:str, ctx:dict)->str:
    try: return str(skill.run(query, ctx or {}))
    except Exception: return "(skill error)\n"+traceback.format_exc()

# ---------- Dynamic Skill Builder (no f-strings inside) ----------
DYN_DIR = DATA_DIR / "skills_dynamic"; DYN_DIR.mkdir(parents=True, exist_ok=True)
def make_dynamic_skill(name_slug:str, goal:str):
    safe_slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in name_slug.lower())
    modname   = f"dyn_{safe_slug.replace('.','_')}"
    clsname   = "".join(w.capitalize() for w in safe_slug.split(".")) + "Skill"
    code_lines = [
        "# -*- coding: utf-8 -*-",
        "class {}:".format(clsname),
        '    name = "{}"'.format(safe_slug),
        '    desc = "{}"'.format(goal.replace('"','\\"')),
        "    timeout_sec = 10",
        "    def run(self, query: str, context: dict) -> str:",
        "        text = (query or '').strip()",
        "        if not text:",
        "            return '[{}] 입력이 비어있습니다.'".format(safe_slug),
        "        wc = len(text.split())",
        "        lc = len(text)",
        "        return '[' + self.name + '] 단어:' + str(wc) + ' 문자:' + str(lc) + ' · ' + text[:120]",
        ""
    ]
    code = "\n".join(code_lines)
    path = DYN_DIR / f"{modname}.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(module)
    cls = getattr(module, clsname); obj = cls(); SKILLS[obj.name] = obj
    return obj.name

# ---------- Commands (KR/EN aliases) ----------
def is_cmd(text: str, *aliases: str) -> bool:
    t = (text or "").lstrip()
    return any(t.startswith(a+" ") or t.strip()==a for a in aliases)
def parse_after(text: str, *aliases: str) -> str:
    t = (text or "").lstrip()
    for a in aliases:
        if t.startswith(a+" "): return t[len(a)+1:].strip()
        if t.strip()==a: return ""
    return t
def maybe_run_skill_command(user_text:str, session_id:str):
    if not is_cmd(user_text, "/use", "/사용"): return None
    rest = parse_after(user_text, "/use", "/사용")
    if not rest: return "(형식) /use 스킬이름 내용  또는  /사용 스킬이름 내용"
    parts = rest.split(" ", 1)
    skill_name = parts[0]; arg = parts[1] if len(parts)>1 else ""
    sk = SKILLS.get(skill_name)
    if not sk: return f"(스킬 없음) {skill_name}"
    return safe_run(sk, arg, {"session": session_id, "raw": user_text})
def try_remember_command(user_text:str, session_id:str):
    if not is_cmd(user_text, "/remember", "/기억"): return None
    raw = parse_after(user_text, "/remember", "/기억").strip()
    tags=[]
    if raw.startswith("!pin "): raw=raw[5:].strip(); tags.append("pin")
    if raw.startswith("!핀 "):  raw=raw[3:].strip(); tags.append("pin")
    mem_add_note(session_id, raw, tags=tags)
    return f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}"
def maybe_build_skill(user_text:str):
    if not is_cmd(user_text, "/build", "/제작"): return None
    rest = parse_after(user_text, "/build", "/제작")
    if not rest: return "(형식) /build 이름|설명  또는  /제작 이름|설명"
    if "|" in rest: name, desc = [x.strip() for x in rest.split("|",1)]
    else: name, desc = rest.strip(), rest.strip()
    try:
        reg = make_dynamic_skill(name, desc)
        return f"스킬 생성/등록 완료 → `/use {reg} 내용`"
    except Exception as e:
        return f"(생성 실패) {e}"

# ---------- Fusion: 병렬 사고 · 판사 · 융합 ----------
def judge_rule_only(q: str, answer: str) -> dict:
    rel = 1.0 if any(w in answer.lower() for w in q.lower().split()[:3]) else 0.6
    cons = 0.8
    fact = 0.7 if ("http" in answer or "출처" in answer or "근거" in answer) else 0.5
    comp = 0.85 if len(answer) >= 120 else 0.55
    score = 0.35*cons + 0.35*fact + 0.2*rel + 0.1*comp
    return {"rel":rel, "cons":cons, "fact":fact, "comp":comp, "score":score}

def think_parallel(prompt:str, providers:List, max_tokens:int=1200):
    results=[]
    with ThreadPoolExecutor(max_workers=min(8, len(providers) or 1)) as ex:
        futmap={ex.submit(p.generate, prompt, max_tokens=max_tokens): p for p in providers}
        for fut in as_completed(futmap):
            p=futmap[fut]
            try:
                ans=fut.result(timeout=90)
                results.append({"provider":getattr(p,"name","?"), "answer":str(ans)})
            except Exception as e:
                results.append({"provider":getattr(p,"name","?"), "answer":"", "error":str(e)})
    return results

def fuse_answers(question:str, answers:list):
    for a in answers:
        a["score"]=judge_rule_only(question, a.get("answer",""))["score"]
    answers.sort(key=lambda x:x.get("score",0), reverse=True)
    if not answers: return "(응답 없음)", {"picked":None, "candidates":[]}
    if len(answers)==1 or (answers[0]["score"]-answers[1].get("score",0)>=0.12):
        return answers[0]["answer"], {"picked":answers[0], "candidates":answers}
    fused = "[게아 종합 답변]\n- 핵심: " + answers[0]['answer'].strip() + \
            "\n- 보강: " + answers[1]['answer'].strip() + \
            "\n(두 모델 공통점은 신뢰 ↑, 상충점은 사용자 확인 권장)"
    return fused, {"picked":None, "candidates":answers}

def gea_fusion_reply(question:str, memory_context:str, provider_names:List[str], level:int=3, identity_text:str=""):
    prompt = ((identity_text + "\n") if identity_text else "") + (memory_context or "") + question
    max_tokens = level_to_tokens(level)
    engines=[get_provider_by_name(n) for n in provider_names]
    raw = think_parallel(prompt, engines, max_tokens=max_tokens)
    final, meta = fuse_answers(question, raw)
    jsonl_append(FUS_LOG, {"t": nowz(), "q": question, "ctx_len": len(memory_context or ""),
                           "providers": provider_names, "final": final[:400], "raw": raw, "meta": meta})
    return final, meta, raw

# ---------- Memory helpers (reuse above) ----------
def mem_view_recent(path: Path, n:int=50):
    try:
        rows = jsonl_read_all(path)[-n:]
        return json.dumps(rows, ensure_ascii=False, indent=2)
    except Exception:
        return "(no data)"

# ---------- UI ----------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("무한 기억 · 하이브리드 검색 · 스킬(/use·/사용) · Autobuilder(/build·/제작) · 🚀 Fusion(다중 엔진 사고) · 🧩 Identity")

    # 상단 제어
    col0, col1, col2 = st.columns([1.3,1,1])
    with col0:
        session_id = st.text_input("세션 ID", sget("session_id","default"), key="k_session_id")
        if session_id != sget("session_id"):
            sset("session_id", session_id); load_session_messages(session_id)
        else:
            sset("session_id", session_id)
            if "messages" not in st.session_state: load_session_messages(session_id)
    with col1:
        mem_on = st.toggle("Memory ON", value=bool(sget("mem_on", True)), key="k_mem_toggle")
        sset("mem_on", mem_on)
    with col2:
        if st.button("대화창 초기화(로그 보존)", key="k_clear_chat"):
            clear_msgs(); st.experimental_rerun()

    # 모델/레벨 + Fusion 설정
    colA, colB, colC = st.columns([1,1.2,1])
    with colA:
        provider_mode = st.selectbox("Provider", ["OpenAI","Gemini","Mock","Fusion"], index=3, key="k_provider_mode")
    with colB:
        level = st.number_input("응답 레벨(1~9999)", min_value=1, max_value=9999, value=5, step=1, key="k_level")
        st.caption(f"예산≈{level_to_tokens(level)} tokens")
    with colC:
        rounds = st.number_input("라운드(단일엔진)", 1, 6, 2, 1, key="k_rounds")

    # IDENTITY 패널 (만들기/편집)
    ensure_identity_file()
    with st.expander("🧩 자아(Identity) 편집", expanded=False):
        try:
            id_raw = ID_PATH.read_text("utf-8")
        except Exception:
            id_raw = ""
        id_text = st.text_area("identity.yaml (또는 JSON도 허용)", value=id_raw, height=220, key="k_identity_edit")
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            if st.button("저장", key="k_identity_save"):
                try:
                    ID_PATH.write_text(id_text, encoding="utf-8")
                    st.success("저장 완료! (다음 응답부터 반영)")
                except Exception as e:
                    st.error(f"저장 실패: {e}")
        with id_col2:
            if st.button("기본값 복원", key="k_identity_reset"):
                try:
                    if yaml:
                        ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    else:
                        ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.warning("기본값으로 복원됨.")
                except Exception as e:
                    st.error(f"복원 실패: {e}")

    # Fusion 엔진 선택
    fusion_providers = []
    if provider_mode=="Fusion":
        with st.expander("Fusion 엔진 선택", expanded=True):
            fusion_providers = st.multiselect(
                "사고 엔진(2개 이상 권장)",
                options=["OpenAI","Gemini","Mock"],
                default=sget("fusion_defaults", ["OpenAI","Gemini"]),
                key="k_fusion_select"
            )
            sset("fusion_defaults", fusion_providers)
            st.caption("엔진별 답을 병렬로 받고, 게아가 판사/융합해 최종 응답을 만듭니다.")

    # 어댑터 준비
    if provider_mode!="Fusion":
        adapter, api_ok = resolve_adapter(provider_mode)
        st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · session={sget('session_id')} · L{int(level)} · R{int(rounds)}")
    else:
        st.info(f"🧠 Fusion: {', '.join(fusion_providers) if fusion_providers else '(선택 필요)'} · session={sget('session_id')} · L{int(level)}")

    # 과거 대화 표시
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(str(m["content"]))

    # 입력
    user_text = st.chat_input(
        "예) /사용 sample.echo 안녕  ·  /기억 !핀 일정  ·  /제작 auto.단어수|단어/문자 수 세기  ·  일반질문",
        key="k_chat_input"
    )
    if user_text:
        user_text = dedupe(user_text.strip())
        add_msg(sget("session_id"), "user", user_text)

        # 0) Identity 텍스트 준비
        identity_text = load_identity_text()

        # 1) Autobuilder
        out = maybe_build_skill(user_text)
        if out is not None:
            with st.chat_message("assistant"): st.markdown(out)
            add_msg(sget("session_id"), "assistant", out)
            log_event("builder", "동적 스킬 생성", detail=out)
            st.stop()

        # 2) Memory note
        out = try_remember_command(user_text, sget("session_id"))
        if out is not None:
            with st.chat_message("assistant"): st.success(out)
            add_msg(sget("session_id"), "assistant", out)
            log_event("memory", "노트 추가", detail=out)
            st.stop()

        # 3) Skill
        out = maybe_run_skill_command(user_text, sget("session_id"))
        if out is not None:
            with st.chat_message("assistant"): st.markdown(out)
            add_msg(sget("session_id"), "assistant", out)
            log_event("skill", "스킬 실행", detail=out)
            st.stop()

        # 4) 일반 채팅 — 메모리 컨텍스트
        context = ""
        if sget("mem_on", True):
            hits = smart_search(sget("session_id"), user_text, topk=5)
            if hits:
                bullet = "\n".join([f"- {h['text']}" for h in hits])
                context = f"[참고 메모]\n{bullet}\n\n"

        if provider_mode=="Fusion":
            if not fusion_providers:
                ans = "Fusion 엔진을 1개 이상 선택하세요."
                with st.chat_message("assistant"): st.warning(ans)
                add_msg(sget("session_id"), "assistant", ans)
            else:
                final, meta, raw = gea_fusion_reply(user_text, context, fusion_providers, level=level, identity_text=identity_text)
                # 자기평가 + 사건기록
                score = judge_self(identity_text, final)
                log_event("answer", "퓨전 응답", detail=final[:400], meta={"score_self": score, "providers": fusion_providers, "picked": meta.get("picked")})
                with st.chat_message("assistant"):
                    st.markdown(str(final))
                    st.caption(f"자기평가 점수: {round(score,2)}")
                    with st.expander("엔진별 결과/점수 보기", expanded=False):
                        try:
                            rows=[]
                            for c in meta.get("candidates", []):
                                rows.append(f"- **{c.get('provider','?')}** · score={round(c.get('score',0),3)}")
                            st.markdown("\n".join(rows) if rows else "(no meta)")
                        except Exception:
                            st.markdown("(메타 표시 오류)")
                add_msg(sget("session_id"), "assistant", final)
        else:
            # 단일 엔진
            prompt = (identity_text + "\n" if identity_text else "") + context + user_text
            with st.chat_message("assistant"):
                try:
                    ans = long_answer(adapter, prompt, level=level, rounds=rounds)
                except Exception:
                    ans = "(내부 오류)\n\n```\n"+traceback.format_exc()+"\n```"
                # 자기평가 + 사건기록
                score = judge_self(identity_text, ans)
                log_event("answer", "단일엔진 응답", detail=ans[:400], meta={"score_self": score, "provider": getattr(adapter,'name','?')})
                st.markdown(str(ans))
                st.caption(f"자기평가 점수: {round(score,2)}")
            add_msg(sget("session_id"), "assistant", ans)

        # 주기 요약
        try:
            if len(sget("messages", [])) % 8 == 0 and sget("mem_on", True):
                mem_add_summary(sget("session_id"), sget("messages", []))
        except Exception: pass

    # 하단 툴
    with st.expander("📚 Memory / Logs 미리보기"):
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("dialog.jsonl (최근 50)", key="k_view_dialog"):
            st.code(mem_view_recent(DIALOG_LOG, 50), language="json")
        if c2.button("memory.jsonl (최근 50)", key="k_view_memory"):
            st.code(mem_view_recent(MEM_PATH, 50), language="json")
        if c3.button("fusion.log (최근 20)", key="k_view_fusion"):
            st.code(mem_view_recent(FUS_LOG, 20), language="json")
        if c4.button("events.jsonl (최근 50)", key="k_view_events"):
            st.code(mem_view_recent(EV_PATH, 50), language="json")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH} · log={DIALOG_LOG} · fus={FUS_LOG} · events={EV_PATH}")

# ---------- entry ----------
if __name__ == "__main__":
    render_app()
    
    