# chatbot_engine.py
# Core engine: env loading, memory, topic-label injection, deictic hard-guard,
# active-context rewriting, RAG chain, formatting helpers.

from __future__ import annotations
import os, re, json, uuid
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple, List

import numpy as np
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


# ---------- Formatting helpers (exported) ----------
def format_mcq(text: str) -> str:
    patterns = [
        (r"\s*①", r"\n①"), (r"\s*②", r"\n②"), (r"\s*③", r"\n③"), (r"\s*④", r"\n④"),
        (r"\s*⑤", r"\n⑤"), (r"\s*⑥", r"\n⑥"), (r"\s*⑦", r"\n⑦"), (r"\s*⑧", r"\n⑧"),
        (r"\s*⑨", r"\n⑨"), (r"\s*⑩", r"\n⑩"),
        (r"\s*1\)", r"\n①"), (r"\s*2\)", r"\n②"), (r"\s*3\)", r"\n③"), (r"\s*4\)", r"\n④"),
        (r"\s*5\)", r"\n⑤"), (r"\s*6\)", r"\n⑥"),
    ]
    for pat, rep in patterns:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\s*(Answer\s*:\s*[^\n]+)", r"\n\1", text)
    text = re.sub(r"(Answer\s*:\s*[^\n]+)\s*(?=Question\s*\d+\.)", r"\1\n\n", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

def to_hard_breaks(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # single LF -> markdown hard break "  \n"
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", md)

def get_choice_labels(n: int) -> list[str]:
    digits = "①②③④⑤⑥⑦⑧⑨⑩"
    n = max(2, min(n, 10))
    return list(digits[:n])


# ---------- Memory ----------
class MemoryManager:
    def __init__(self, session_id: str, base_dir: str = "./mycache/conversations"):
        self.session_id = session_id or uuid.uuid4().hex
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = Path(self.base_dir) / f"{self.session_id}.json"
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.memory: list[dict] = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.memory = []
        else:
            self.memory = []

    def save(self):
        try:
            self.path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def reset(self):
        self.session_id = uuid.uuid4().hex
        self.path = Path(self.base_dir) / f"{self.session_id}.json"
        self.memory = []
        self.save()

    def add_turn(self, user_text: str, assistant_text: str):
        self.memory.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user_text,
            "assistant": assistant_text
        })
        self.save()

    def last_user_text(self) -> Optional[str]:
        for t in reversed(self.memory):
            if t.get("user"):
                return t["user"]
        return None

    def render_capped(self, max_chars: int = 4000) -> str:
        lines = []
        for turn in self.memory:
            lines.append(f"[User @ {turn['time']}] {turn['user']}")
            lines.append(f"[Assistant] {turn['assistant']}")
        s = "\n".join(lines)
        return s[-max_chars:] if len(s) > max_chars else s

    def recent_user_texts(self, k: int = 5) -> list[str]:
        return [t.get("user", "") for t in self.memory if t.get("user")][-k:]


# ---------- Engine ----------
class ChatBotEngine:
    def __init__(
        self,
        session_id: str,
        keys_env_path: str = "/home/mk/workspace/KEYS.env",
        conversations_dir: str = "./mycache/conversations",
        embed_cache_dir: str = "./mycache/embedding",
        max_history_chars: int = 4000,
    ):
        self.keys_env_path = keys_env_path
        self.ensure_env()
        self.max_history_chars = max_history_chars

        self.memory = MemoryManager(session_id=session_id, base_dir=conversations_dir)
        os.makedirs(embed_cache_dir, exist_ok=True)
        self.embed_store = LocalFileStore(embed_cache_dir)

        # 🔒 도메인 라벨 캐시(예: "SQLD (SQL 개발자)")
        self.cached_label: str = ""

    # ----- ENV -----
    def ensure_env(self):
        p = Path(self.keys_env_path)
        if p.exists():
            vals = dotenv_values(p)
            oa = vals.get("OPENAI_API_KEY") or vals.get("OPENAI_KEY")
            if oa: os.environ["OPENAI_API_KEY"] = oa.strip().strip('"').strip("'")
            tv = vals.get("TAVILY_API_KEY") or vals.get("TAVILY_KEY")
            if tv: os.environ["TAVILY_API_KEY"] = tv.strip().strip('"').strip("'")

    # ----- Embedder (cached) -----
    @staticmethod
    @lru_cache(maxsize=1)
    def get_embedder_cached(api_key: str):
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    def get_embedder(self):
        return self.get_embedder_cached(os.getenv("OPENAI_API_KEY") or "")

    # ----- Deictic / meta / label -----
    DEFAULT_DEICTIC_TERMS = [
        "이것","그것","저것","이거","그거","저거","이거요","그거요",
        "이건","그건","저건","이런","그런","저런","이런거","그런거","저런거",
        "여기","거기","저기","이쪽","그쪽","저쪽",
        "지금","방금","아까","방금 전","이때","그때","저때",
        "그 문서","그 자료","그 내용","그 문제","이 문제","위 내용","앞에 말한","말한거","말씀하신",
        "그거 다시","그거 뭐였지","그러면","그럼","그렇다면"
    ]
    _DEICTIC_RE = re.compile("|".join(map(re.escape, sorted(DEFAULT_DEICTIC_TERMS, key=len, reverse=True))))

    @staticmethod
    def adjust_topic_threshold(user_text: str, base_threshold: float) -> float:
        eff = float(base_threshold)
        t = (user_text or "").strip()
        if ChatBotEngine._DEICTIC_RE.search(t): eff -= 0.12
        if len(t) <= 10: eff -= 0.05
        return max(0.45, min(eff, 0.95))

    @staticmethod
    def _normalize_meta_text(text: str) -> str:
        if not text: return ""
        t = text.strip()
        t = re.sub(r"[ㅣl|]+$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    _MEMORY_QUERY_RE = re.compile(
        r"(?:(?:전에|방금|아까)\s*(?:내가\s*)?(?:뭐|무엇)(?:라|라고)?\s*(?:말|질문|물어(?:봤|보았)|물었|했)\w*(?:더라|드라|지|나요|니|였니|었니)?\??"
        r"|(?:이전|직전|마지막)\s*(?:질문|말|발화)\s*(?:이|은)?\s*(?:뭐였|무엇이었)\w*"
        r"|(?:what\s+did\s+i\s+(?:ask|say)\s+(?:before|earlier)|previous\s+question|last\s+question))",
        re.IGNORECASE,
    )

    def is_meta_memory_query(self, text: str) -> bool:
        t = self._normalize_meta_text(text)
        if self._MEMORY_QUERY_RE.search(t):
            return True
        # semantic fallback
        examples = [
            "내가 전에 뭐라 질문했더라?",
            "이전 질문이 뭐였지?",
            "What did I ask before?",
            "What was my previous question?",
        ]
        emb = self.get_embedder()
        v = np.array(emb.embed_query(t), dtype=float)
        vnorm = np.linalg.norm(v) + 1e-12
        sims = []
        for ex in examples:
            r = np.array(emb.embed_query(ex), dtype=float)
            sims.append((v @ r) / ((np.linalg.norm(r) + 1e-12) * vnorm))
        return max(sims) >= 0.82

    STOPWORDS_KO = {
        "지금","질문","하고","있잖아","추천","해줘","해주세요","주세요","좀","관련","도움",
        "책","교재","문제","어떻게","알려줘","뭐","뭐야","요","것","그것","저것",
        "이거","그거","저거","이건","그건","저건","정보","요약","자료","설명","방법","그러면","그럼"
    }
    SQLD_PAT_KO = re.compile(r"(SQLD|SQL\-?D|SQL\s*개발자|데이터베이스|정규화|조인|쿼리|DDL|DML)", re.IGNORECASE)

    @staticmethod
    def _tokenize_ko(text: str):
        toks = re.findall(r"[A-Za-z0-9가-힣]+", text or "")
        return [t for t in toks if len(t) > 1 and t not in ChatBotEngine.STOPWORDS_KO]

    def derive_topic_label_from_texts(self, texts: List[str]) -> str:
        joined = " ".join(texts[-5:])
        if self.SQLD_PAT_KO.search(joined):
            return "SQLD (SQL 개발자)"
        return ""

    GENERIC_AMBIGUOUS_TERMS = {
        "문제","예제","연습","선수","감독","순위","리스트","추천","방법","정의","설명",
        "자료","요약","강의","교재","책","문제집","기출","자료집","수업","강좌","코스",
        "포지션","유명한","정보","비교","특징","기본","개념"
    }

    def rewrite_with_active_context(self, user_text: str, scope_hint: str, topic_label: str) -> str:
        t = (user_text or "").strip()
        if not t:
            return user_text
        short = len(t) <= 14
        has_deictic = bool(self._DEICTIC_RE.search(t))
        has_generic = any(term in t for term in self.GENERIC_AMBIGUOUS_TERMS)
        ambiguous = short or has_deictic or has_generic
        if not ambiguous:
            return t
        label_prefix = ""
        if topic_label and topic_label.lower() not in t.lower():
            label_prefix = f"{topic_label} 관련 "
        tag = f"[활성주제: {scope_hint}]" if scope_hint else ""
        return f"{label_prefix}{t} {tag} — 범위를 변경하지 않았다면 위 활성주제/라벨에 한정해 답하라."

    # ----- Topic detection -----
    def calc_similarity_with_last(self, user_text: str) -> Optional[float]:
        last = self.memory.last_user_text()
        if not last:
            return None
        emb = self.get_embedder()
        v_new = np.array(emb.embed_query(user_text), dtype=float)
        v_old = np.array(emb.embed_query(last), dtype=float)
        return float(v_new @ v_old) / (np.linalg.norm(v_new) * np.linalg.norm(v_old) + 1e-12)

    def should_reset_topic(
        self,
        user_text: str,
        threshold: float = 0.72,
        sticky_margin: float = 0.05
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        # 🔒 하드 가드: 지시표현이 있으면 절대 리셋하지 않음
        if self._DEICTIC_RE.search(user_text or ""):
            return False, (1.0, threshold)  # debug용 값
        if self.is_meta_memory_query(user_text):
            return False, None
        last = self.memory.last_user_text()
        if not last:
            return False, None
        emb = self.get_embedder()
        v_new = np.array(emb.embed_query(user_text), dtype=float)
        v_old = np.array(emb.embed_query(last), dtype=float)
        sim = float(v_new @ v_old) / (np.linalg.norm(v_new) * np.linalg.norm(v_old) + 1e-12)
        eff = self.adjust_topic_threshold(user_text, threshold)
        return sim < (eff - sticky_margin), (sim, eff)

    # ----- Scope & label helpers -----
    def scope_hint(self, k: int = 3) -> str:
        texts = self.memory.recent_user_texts(k)
        if not texts: return ""
        hint = " / ".join(texts)
        return hint[:500]

    def topic_label(self) -> str:
        # 최근 발화에서 라벨을 뽑아 갱신, 없으면 캐시 유지
        derived = self.derive_topic_label_from_texts(self.memory.recent_user_texts(5))
        if derived:
            self.cached_label = derived
            return derived
        return self.cached_label

    # 전체 주제 리셋(라벨 캐시 포함)
    def reset_topic(self):
        self.memory.reset()
        self.cached_label = ""

    # ----- RAG components -----
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

    def _openai_embeddings(self):
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    def make_vectorstore_from_documents(self, split_documents):
        cached = CacheBackedEmbeddings.from_bytes_store(self._openai_embeddings(), self.embed_store)
        return FAISS.from_documents(documents=split_documents, embedding=cached)

    def build_retriever_from_pdf(self, file_path: str, chunk_size: int, chunk_overlap: int):
        docs = PDFPlumberLoader(file_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = self.make_vectorstore_from_documents(splitter.split_documents(docs))
        return vectorstore.as_retriever()

    def build_retriever_from_text(self, text: str, use_semantic: bool, chunk_size: int, chunk_overlap: int):
        if use_semantic:
            splitter = SemanticChunker(self._openai_embeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=70)
            docs = splitter.create_documents([text])
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.create_documents([text])
        vectorstore = self.make_vectorstore_from_documents(docs)
        return vectorstore.as_retriever()

    def safe_tavily_retriever(self, k: int = 3):
        if os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
            return TavilySearchAPIRetriever(k=k)
        class EmptyRetriever(Runnable):
            def invoke(self, input, config=None): return []
        return EmptyRetriever()

    # ----- Prompt / Chain -----
    @staticmethod
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
        # 혼합
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

    def create_chain(self, retriever, mode: str, num_q: int, num_choices: int, choice_labels_text: str):
        tavily = self.safe_tavily_retriever(k=3)
        fallback = ChatBotEngine.FallbackRetriever(retriever, tavily)
        prompt = ChatPromptTemplate.from_template(self.build_prompt_text(mode))
        llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        return (
            RunnableMap({
                "context": fallback | RunnableLambda(lambda x: "\n".join(
                    getattr(d, "page_content", str(d)) for d in (x or [])
                )),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: self.memory.render_capped(self.max_history_chars)),
                "active_context": RunnableLambda(lambda _: self.scope_hint()),
                "num_q": RunnableLambda(lambda _: num_q),
                "num_choices": RunnableLambda(lambda _: num_choices),
                "choice_labels_text": RunnableLambda(lambda _: choice_labels_text),
            })
            | prompt
            | llm
            | StrOutputParser()
        )
