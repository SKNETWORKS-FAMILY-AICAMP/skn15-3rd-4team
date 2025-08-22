# gpt기반 챗봇 대화주제 변경에 대한 정의가 확실하지 않은 상태
# 질문을 하고 다음 질문을 했을때 명사가 없는 경우 오류 발생 가능
# 지시대명사가 있으면 명사를 앞에 안붙혀도 이전 질문기반으로 대답
# 경로 수정해야함 

# rag_chatbot_mk.py — Multi-mode + Auto Topic Shift (Deictic-aware) + Active Context + Topic Label Injection + Meta-memory + RAG + MCQ line-break fix
import os
import re
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from langchain_core.runnables import Runnable, RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.retrievers import TavilySearchAPIRetriever

# =========================
# 0) 환경 변수 로딩 (KEYS.env 고정)
# =========================
ENV_PATH = Path("/home/mk/workspace/KEYS.env")
if ENV_PATH.exists():
    vals = dotenv_values(ENV_PATH)
    oa = vals.get("OPENAI_API_KEY") or vals.get("OPENAI_KEY")
    if oa:
        os.environ["OPENAI_API_KEY"] = oa.strip().strip('"').strip("'")
    tv = vals.get("TAVILY_API_KEY") or vals.get("TAVILY_KEY")
    if tv:
        os.environ["TAVILY_API_KEY"] = tv.strip().strip('"').strip("'")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MAX_HISTORY_CHARS = 4000
STICKY_MARGIN = 0.05  # 토픽 유지 여유

# =========================
# 1) Streamlit 설정 + 화면 로그
# =========================
st.set_page_config(page_title="RAG 기반 멀티모드 챗봇", layout="wide")
st.title("RAG 기반 멀티모드 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    '<div id="chatlog" style="max-height:72vh; overflow-y:auto; padding-right:8px;">',
    unsafe_allow_html=True,
)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 2) 유틸: MCQ 포맷 보정 + 하드 브레이크
# =========================
def format_mcq(text: str) -> str:
    patterns = [
        (r'\s*①', r'\n①'), (r'\s*②', r'\n②'), (r'\s*③', r'\n③'), (r'\s*④', r'\n④'),
        (r'\s*⑤', r'\n⑤'), (r'\s*⑥', r'\n⑥'), (r'\s*⑦', r'\n⑦'), (r'\s*⑧', r'\n⑧'),
        (r'\s*⑨', r'\n⑨'), (r'\s*⑩', r'\n⑩'),
        (r'\s*1\)', r'\n①'), (r'\s*2\)', r'\n②'), (r'\s*3\)', r'\n③'), (r'\s*4\)', r'\n④'),
        (r'\s*5\)', r'\n⑤'), (r'\s*6\)', r'\n⑥'),
    ]
    for pat, rep in patterns:
        text = re.sub(pat, rep, text)
    text = re.sub(r'\s*(Answer\s*:\s*[^\n]+)', r'\n\1', text)
    text = re.sub(r'(Answer\s*:\s*[^\n]+)\s*(?=Question\s*\d+\.)', r'\1\n\n', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()

def to_hard_breaks(md: str) -> str:
    md = md.replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'(?<!\n)\n(?!\n)', '  \n', md)

def get_choice_labels(n: int):
    digits = "①②③④⑤⑥⑦⑧⑨⑩"
    n = max(2, min(n, 10))
    return list(digits[:n])

# =========================
# 3) 임베딩 (캐시)
# =========================
@st.cache_resource
def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# =========================
# 4) 지시/메타/모호 → 활성맥락 + 도메인 라벨 주입
# =========================
DEFAULT_DEICTIC_TERMS = [
    "이것","그것","저것","이거","그거","저거","이거요","그거요",
    "이건","그건","저건","이런","그런","저런","이런거","그런거","저런거",
    "여기","거기","저기","이쪽","그쪽","저쪽",
    "지금","방금","아까","방금 전","이때","그때","저때",
    "그 문서","그 자료","그 내용","그 문제","이 문제","위 내용","앞에 말한","말한거","말씀하신",
    "그거 다시","그거 뭐였지","그러면","그럼","그렇다면"
]

def build_deictic_regex(extra_terms=None):
    terms = set(DEFAULT_DEICTIC_TERMS)
    if extra_terms:
        terms.update(t.strip() for t in extra_terms if t and t.strip())
    pat = r"(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + r")"
    return re.compile(pat)

_EXTRA_ENV = os.getenv("DEICTIC_EXTRA", "")
_extra_from_env = [s for s in _EXTRA_ENV.split(",")] if _EXTRA_ENV else []
_extra_from_file = []
cfg_path = Path("./config/deictic_terms.txt")
if cfg_path.exists():
    try:
        _extra_from_file = [
            line.strip() for line in cfg_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except Exception:
        _extra_from_file = []

_DEICTIC_RE = build_deictic_regex(_extra_from_env + _extra_from_file)

def adjust_topic_threshold(user_text: str, base_threshold: float) -> float:
    eff = float(base_threshold)
    t = (user_text or "").strip()
    if _DEICTIC_RE.search(t): eff -= 0.12
    if len(t) <= 10:          eff -= 0.05
    return max(0.45, min(eff, 0.95))

def _normalize_meta_text(text: str) -> str:
    if not text: return ""
    t = text.strip()
    t = re.sub(r"[ㅣl|]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

_MEMORY_QUERY_RE = re.compile(
    r"(?:"
    r"(?:전에|방금|아까)\s*(?:내가\s*)?(?:뭐|무엇)(?:라|라고)?\s*(?:말|질문|물어(?:봤|보았)|물었|했)\w*(?:더라|드라|지|나요|니|였니|었니)?\??"
    r"|(?:이전|직전|마지막)\s*(?:질문|말|발화)\s*(?:이|은)?\s*(?:뭐였|무엇이었)\w*"
    r"|(?:what\s+did\s+i\s+(?:ask|say)\s+(?:before|earlier)|previous\s+question|last\s+question)"
    r")",
    re.IGNORECASE,
)

def _regex_is_meta_query(text: str) -> bool:
    return bool(_MEMORY_QUERY_RE.search(_normalize_meta_text(text)))

@st.cache_resource
def _meta_intent_refs():
    examples = [
        "내가 전에 뭐라 질문했더라?",
        "전에 내가 뭐라고 말했지?",
        "이전 질문이 뭐였지?",
        "마지막으로 한 질문 알려줘",
        "방금 내가 뭐라 했지",
        "What did I ask before?",
        "What did I say earlier?",
        "What was my previous question?",
        "Show my last question",
    ]
    emb = get_embedder()
    vecs = [np.array(emb.embed_query(x), dtype=float) for x in examples]
    norms = [np.linalg.norm(v) + 1e-12 for v in vecs]
    return vecs, norms

def _semantic_is_meta_query(text: str, th: float = 0.82) -> bool:
    t = _normalize_meta_text(text)
    if not t: return False
    v = np.array(get_embedder().embed_query(t), dtype=float)
    vnorm = np.linalg.norm(v) + 1e-12
    refs, norms = _meta_intent_refs()
    sims = [(v @ refs[i]) / (vnorm * norms[i]) for i in range(len(refs))]
    return max(sims) >= th

def is_meta_memory_query(text: str) -> bool:
    t = _normalize_meta_text(text)
    return _regex_is_meta_query(t) or _semantic_is_meta_query(t)

# --- Domain label extraction (lightweight) ---
STOPWORDS_KO = {
    "지금","질문","하고","있잖아","추천","해줘","해주세요","주세요","좀","관련","도움",
    "책","교재","문제","어떻게","알려줘","뭐","뭐야","요","것","그것","저것",
    "이거","그거","저거","이건","그건","저건","정보","요약","자료","설명","방법","그러면","그럼"
}
SQLD_PAT_KO = re.compile(r"(SQLD|SQL\-?D|SQL\s*개발자|데이터베이스|정규화|조인|쿼리|DDL|DML)", re.IGNORECASE)

def _tokenize_ko(text: str):
    toks = re.findall(r"[A-Za-z0-9가-힣]+", text or "")
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS_KO]

def derive_topic_label_from_texts(texts: list[str]) -> str:
    joined = " ".join(texts[-5:])  # 최근 최대 5개
    if SQLD_PAT_KO.search(joined):
        return "SQLD (SQL 개발자)"
    return ""

# 모호질문 리스트 (풍부)
GENERIC_AMBIGUOUS_TERMS = {
    "문제","예제","연습","선수","감독","순위","리스트","추천","방법","정의","설명",
    "자료","요약","강의","교재","책","문제집","기출","자료집","수업","강좌","코스",
    "포지션","유명한","정보","비교","특징","기본","개념"
}

def rewrite_with_active_context(user_text: str, scope_hint: str, topic_label: str) -> str:
    """
    질문이 짧거나 지시표현/일반명사 위주로 모호하면
    1) [활성주제: ...] 태그
    2) topic_label(예: 'SQLD (SQL 개발자)')을 질문에 직접 주입하여 해석을 강제.
    """
    t = (user_text or "").strip()
    if not t:
        return user_text

    short = len(t) <= 14
    has_deictic = bool(_DEICTIC_RE.search(t))
    has_generic = any(term in t for term in GENERIC_AMBIGUOUS_TERMS)
    ambiguous = short or has_deictic or has_generic

    if not ambiguous:
        return t  # 모호하지 않으면 원문 유지

    label_prefix = ""
    if topic_label and topic_label.lower() not in t.lower():
        label_prefix = f"{topic_label} 관련 "

    tag = f"[활성주제: {scope_hint}]" if scope_hint else ""
    return f"{label_prefix}{t} {tag} — 범위를 변경하지 않았다면 위 활성주제/라벨에 한정해 답하라."

# =========================
# 5) 대화 메모리 + 자동 주제 전환 (+ 활성 맥락/라벨)
# =========================
class MemoryManager:
    def __init__(self, base_dir: str = "./mycache/conversations"):
        os.makedirs(base_dir, exist_ok=True)
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = uuid.uuid4().hex
        self.base_dir = base_dir
        self.path = Path(base_dir) / f"{st.session_state['session_id']}.json"
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.memory = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.memory = []
        else:
            self.memory = []

    def save(self):
        try:
            self.path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            st.warning(f"메모리 저장 실패: {e}")

    def reset(self):
        st.session_state["session_id"] = uuid.uuid4().hex
        self.path = Path(self.base_dir) / f"{st.session_state['session_id']}.json"
        self.memory = []
        self.save()

    def add_turn(self, user_text: str, assistant_text: str):
        self.memory.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user_text,
            "assistant": assistant_text
        })
        self.save()

    def last_user_text(self):
        for t in reversed(self.memory):
            if t.get("user"):
                return t["user"]
        return None

    def calc_similarity(self, user_text: str):
        last = self.last_user_text()
        if not last:
            return None
        emb = get_embedder()
        v_new = np.array(emb.embed_query(user_text), dtype=float)
        v_old = np.array(emb.embed_query(last), dtype=float)
        return float(v_new @ v_old) / (np.linalg.norm(v_new) * np.linalg.norm(v_old) + 1e-12)

    def scope_hint(self, k: int = 3) -> str:
        texts = [t.get("user", "") for t in self.memory if t.get("user")]
        if not texts: return ""
        hint = " / ".join(texts[-k:])
        return hint[:500]

    def topic_label(self) -> str:
        texts = [t.get("user", "") for t in self.memory if t.get("user")]
        if not texts: return ""
        return derive_topic_label_from_texts(texts)

    def is_new_topic(self, user_text: str, threshold: float = 0.75) -> bool:
        if is_meta_memory_query(user_text):
            return False
        last = self.last_user_text()
        if not last:
            return False
        emb = get_embedder()
        v_new = np.array(emb.embed_query(user_text), dtype=float)
        v_old = np.array(emb.embed_query(last), dtype=float)
        sim = float(v_new @ v_old) / (np.linalg.norm(v_new) * np.linalg.norm(v_old) + 1e-12)
        eff_threshold = adjust_topic_threshold(user_text, threshold)
        st.session_state["__topic_debug"] = (sim, eff_threshold)
        return sim < (eff_threshold - STICKY_MARGIN)

    def render_capped(self, max_chars: int = MAX_HISTORY_CHARS) -> str:
        lines = []
        for turn in self.memory:
            lines.append(f"[User @ {turn['time']}] {turn['user']}")
            lines.append(f"[Assistant] {turn['assistant']}")
        s = "\n".join(lines)
        return s[-max_chars:] if len(s) > max_chars else s

memory = MemoryManager()

# =========================
# 6) 사이드바 UI
# =========================
with st.sidebar:
    def mask(s): return (s[:4] + "..." + s[-4:]) if s else "None"
    st.markdown("### 🔑 Keys")
    st.write("OpenAI:", bool(OPENAI_API_KEY), mask(OPENAI_API_KEY or ""))
    st.write("Tavily:", bool(TAVILY_API_KEY), mask(TAVILY_API_KEY or ""))

    st.markdown("### 🧠 Memory")
    use_memory = st.checkbox("이전 대화 기억 사용", value=True)
    auto_topic = st.checkbox("자동 주제 전환 감지", value=True)
    topic_threshold = st.slider("주제 전환 임계값(유사도)", 0.00, 1.00, 0.72, 0.01)
    topic_debug = st.checkbox("토픽 디버그 표시", value=False)
    context_debug = st.checkbox("맥락 디버그 표시", value=False)
    if st.button("메모리 수동 초기화"):
        memory.reset()
        st.success("메모리를 초기화했습니다.")
    if st.button("🧹 대화 모두 지우기"):
        st.session_state.messages = []
        st.success("화면 채팅 로그를 비웠습니다. (메모리는 그대로)")

    st.markdown("### 🧩 응답 모드")
    response_mode = st.selectbox(
        "원하는 출력 형태",
        ["일반 대화", "요약", "MCQ(객관식)", "OX(참/거짓)", "단답형 퀴즈", "혼합(객관+OX+단답)"],
        index=0
    )
    if response_mode in ["MCQ(객관식)", "OX(참/거짓)", "단답형 퀴즈", "혼합(객관+OX+단답)"]:
        num_q = st.slider("문항 수", 1, 15, 5, 1)
    else:
        num_q = 0
    if response_mode in ["MCQ(객관식)", "혼합(객관+OX+단답)"]:
        num_choices = st.slider("보기 개수(객관식)", 2, 6, 4, 1)
    else:
        num_choices = 0

    st.markdown("### 📄 파일 입력")
    uploaded_file = st.file_uploader("파일 업로드", type=['pdf', 'txt'])
    use_semantic_chunk_txt = st.checkbox("TXT에 시맨틱 청킹 사용(비용↑)", value=False)
    chunk_size = st.number_input("청크 크기", min_value=200, max_value=4000, value=1000, step=50)
    chunk_overlap = st.number_input("청크 오버랩", min_value=0, max_value=1000, value=100, step=10)

# =========================
# 7) 로컬 캐시 경로
# =========================
os.makedirs("./mycache/files", exist_ok=True)
os.makedirs("./mycache/embedding", exist_ok=True)
store = LocalFileStore("./mycache/embedding")

# =========================
# 8) RAG 구성 요소
# =========================
class FallbackRetriever(Runnable):
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
    def invoke(self, input, config=None):
        try:
            results = self.primary.invoke(input, config=config)
            if results:
                return results
        except Exception:
            pass
        return self.fallback.invoke(input, config=config)

join_docs = RunnableLambda(lambda docs: "\n".join(getattr(d, "page_content", str(d)) for d in (docs or [])))

def _openai_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

def make_vectorstore_from_documents(split_documents):
    cached = CacheBackedEmbeddings.from_bytes_store(_openai_embeddings(), store)
    return FAISS.from_documents(documents=split_documents, embedding=cached)

@st.cache_resource(show_spinner=True)
def process_pdf_to_retriever(file_path: str, chunk_size: int, chunk_overlap: int):
    docs = PDFPlumberLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = make_vectorstore_from_documents(splitter.split_documents(docs))
    return vectorstore.as_retriever()

@st.cache_resource(show_spinner=True)
def process_txt_to_retriever(text: str, use_semantic: bool, chunk_size: int, chunk_overlap: int):
    if use_semantic:
        splitter = SemanticChunker(_openai_embeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=70)
        docs = splitter.create_documents([text])
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([text])
    vectorstore = make_vectorstore_from_documents(docs)
    return vectorstore.as_retriever()

def _safe_tavily_retriever(k: int = 3):
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        return TavilySearchAPIRetriever(k=k)
    else:
        class EmptyRetriever(Runnable):
            def invoke(self, input, config=None): return []
        st.sidebar.warning("Tavily 키가 없어 웹 폴백 검색은 비활성화됩니다.")
        return EmptyRetriever()

# =========================
# 9) 프롬프트 (history/active_context 반영 + 태그/라벨 우선)
# =========================
def build_prompt_text(mode: str) -> str:
    header_common = """[지시]
- 현재 질문과 무관한 과거 히스토리는 **무시**하라(자동 주제 전환 시 이전 주제는 제외됨).
- 아래의 <chat_history>와 <active_context>를 참고하되, 현재 질문과 직접 관련 있을 때만 사용하라.
- 질문이 모호하거나 일반명사 위주일 경우, 사용자가 범위를 바꾸었다고 **명시하지 않으면**
  <active_context>의 범위를 **기본값**으로 가정하라.
- 질문 문자열에 **[활성주제: …]** 태그나 **'… 관련' 도메인 라벨**이 보이면, 그 범위를 **최우선으로 적용**하라.
"""

    history_block = """
<chat_history>
{history}
</chat_history>

<active_context>
{active_context}
</active_context>
"""

    if mode == "일반 대화":
        return f"""당신은 유능한 도우미입니다.

{header_common}
- 아래의 <information> 컨텍스트를 우선 활용하되, 충분치 않으면 일반 지식으로 보완하라.
- 간결하고 정확하게 답하라.
{history_block}
<information>
{{context}}
</information>

#Question:
{{question}}
"""

    if mode == "요약":
        return f"""다음 정보를 한국어로 핵심만 간결하게 요약하라. 필요시 항목별 불릿을 사용하라.
{header_common}
{history_block}
<information>
{{context}}
</information>

요약 대상/요청:
{{question}}
"""

    if mode == "MCQ(객관식)":
        return f"""당신은 객관식 문항 생성기입니다.

{header_common}
- 총 {{num_q}}개의 객관식 문항을 만들어라.
- 각 문항은 보기 {{num_choices}}개로 구성하고, 보기 표시는 다음 라벨을 **이 순서**로 사용하라: {{choice_labels_text}}
- 문제문 다음 줄부터 각 보기를 **한 줄에 하나씩** 출력하라.
- 마지막 줄에 정답을 'Answer: ①' 형식으로 명시하라.
- 문항 사이에는 빈 줄 1줄을 둔다.
{history_block}
<information>
{{context}}
</information>

#Question(주제/요청):
{{question}}

#예시(형식만 참고)
Question 1. …문제문…
① 보기1
② 보기2
③ 보기3
④ 보기4
Answer: ③
"""

    if mode == "OX(참/거짓)":
        return f"""당신은 OX(참/거짓) 문제 생성기입니다.

{header_common}
- 총 {{num_q}}개의 진술문을 만들고, 각 문항의 정답을 'Answer: O' 또는 'Answer: X' 한 줄로 명시하라.
- 문항 사이에는 빈 줄 1줄을 둔다.
{history_block}
<information>
{{context}}
</information>

#Question(주제/요청):
{{question}}

#예시
Q1. …진술문…
Answer: O

Q2. …진술문…
Answer: X
"""

    if mode == "단답형 퀴즈":
        return f"""당신은 단답형 퀴즈 생성기입니다.

{header_common}
- 총 {{num_q}}개의 단답형 문항을 만들고, 각 문항의 정답을 'Answer: …' 한 줄로 명시하라.
- 문항 사이에는 빈 줄 1줄을 둔다.
{history_block}
<information>
{{context}}
</information>

#Question(주제/요청):
{{question}}

#예시
Q1. …문제문…
Answer: …

Q2. …문제문…
Answer: …
"""

    # 혼합(객관+OX+단답)
    return f"""당신은 혼합형 퀴즈 생성기입니다.

{header_common}
- 총 {{num_q}}개 문항을 생성하되, 대략 50%는 객관식, 25%는 OX, 25%는 단답형으로 섞어라.
- 객관식의 보기 수는 {{num_choices}}개이며, 보기 라벨은 다음을 사용: {{choice_labels_text}}
- 각 유형의 출력 형식을 엄격히 지켜라:
  - 객관식: 문제문 -> 각 보기(한 줄씩) -> 'Answer: ①' 형태
  - OX: 'Qn. …' -> 'Answer: O/X'
  - 단답형: 'Qn. …' -> 'Answer: …'
- 문항 사이에는 빈 줄 1줄을 둔다.
{history_block}
<information>
{{context}}
</information>

#Question(주제/요청):
{{question}}
"""

# =========================
# 10) 체인
# =========================
def create_chain(retriever, mode: str, num_q: int, num_choices: int, choice_labels_text: str):
    tavily = _safe_tavily_retriever(k=3)
    fallback = FallbackRetriever(retriever, tavily)
    prompt = ChatPromptTemplate.from_template(build_prompt_text(mode))
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    return (
        RunnableMap({
            "context": fallback | join_docs,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: memory.render_capped(MAX_HISTORY_CHARS) if use_memory else ""),
            "active_context": RunnableLambda(lambda _: memory.scope_hint()),
            "num_q": RunnableLambda(lambda _: num_q),
            "num_choices": RunnableLambda(lambda _: num_choices),
            "choice_labels_text": RunnableLambda(lambda _: choice_labels_text),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

# =========================
# 11) 업로드 처리 & 리트리버
# =========================
retriever = None
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    save_path = os.path.join("./mycache/files", f"{uuid.uuid4().hex}.{ext}")
    try:
        file_bytes = uploaded_file.read()
        if ext == "pdf":
            with open(save_path, "wb") as fw: fw.write(file_bytes)
            retriever = process_pdf_to_retriever(save_path, int(chunk_size), int(chunk_overlap))
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore")
            with open(save_path, "w", encoding="utf-8") as fw: fw.write(text)
            retriever = process_txt_to_retriever(text, use_semantic_chunk_txt, int(chunk_size), int(chunk_overlap))
        else:
            st.error("pdf/txt만 지원합니다.")
    except Exception as e:
        st.error(f"파일 처리 오류: {e}")

if retriever is None:
    retriever = _safe_tavily_retriever(k=3)

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. /home/mk/workspace/KEYS.env 확인.")
    st.stop()

# =========================
# 12) 채팅 처리
# =========================
user_input = st.chat_input("원하는 모드와 함께 아무 주제나 질문하세요. (예: 'SQLD 핵심 요약', '객관식 5문제', '야구 규칙 OX' 등)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if is_meta_memory_query(user_input):
        prev_q = memory.last_user_text()
        reply = f"당신이 전에 물어본 것은 다음입니다:\n\n> {prev_q}" if prev_q else "직전에 저장된 사용자 질문이 없어요."
        with st.chat_message('assistant'):
            st.markdown(f"**🧑 질문:** {user_input}\n\n{reply}")
        st.session_state.messages.append({"role": "assistant", "content": f"**🧑 질문:** {user_input}\n\n{reply}"})
        memory.add_turn(user_input, reply)
        st.stop()

    if use_memory and auto_topic:
        try:
            if memory.is_new_topic(user_input, threshold=topic_threshold):
                memory.reset()
                st.toast("🔄 새 주제로 전환되었습니다.", icon="🔁")
        except Exception as e:
            st.sidebar.warning(f"주제 전환 감지 경고: {e}")

    if topic_debug and "__topic_debug" in st.session_state:
        sim, thr = st.session_state["__topic_debug"]
        st.sidebar.info(f"유사도: {sim:.3f} / 적용 임계값: {thr:.2f}")

    choice_labels = get_choice_labels(num_choices) if num_choices >= 2 else []
    choice_labels_text = " ".join(choice_labels) if choice_labels else ""
    chain = create_chain(retriever, response_mode, num_q, num_choices, choice_labels_text)

    # ★ 활성 맥락 + 도메인 라벨을 반영한 '실제 질의'
    scope_hint = memory.scope_hint()
    topic_lbl = memory.topic_label()
    effective_query = rewrite_with_active_context(user_input, scope_hint, topic_lbl)
    if context_debug:
        st.sidebar.caption(f"scope_hint: {scope_hint}")
        st.sidebar.caption(f"topic_label: {topic_lbl}")
        st.sidebar.caption(f"effective_query: {effective_query}")

    try:
        response_stream = chain.stream(effective_query)
        with st.chat_message('assistant'):
            container = st.empty()
            ai_answer = ""
            for token in response_stream:
                ai_answer += token
                preview = ai_answer if response_mode not in ["MCQ(객관식)", "혼합(객관+OX+단답)"] else to_hard_breaks(format_mcq(ai_answer))
                container.markdown(f"**🧑 질문:** {user_input}\n\n{preview}")

            if response_mode in ["MCQ(객관식)", "혼합(객관+OX+단답)"]:
                final_text = to_hard_breaks(format_mcq(ai_answer))
            else:
                final_text = ai_answer

            final_render = f"**🧑 질문:** {user_input}\n\n{final_text}"
            container.markdown(final_render)

        st.session_state.messages.append({"role": "assistant", "content": final_render})
        memory.add_turn(user_input, final_text)

    except Exception as e:
        err = f"응답 생성 중 오류: {e}"
        st.chat_message("assistant").error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
