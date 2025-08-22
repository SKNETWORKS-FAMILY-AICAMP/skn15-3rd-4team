# app.py
# Streamlit UI that uses ChatBotEngine

import os, uuid
import streamlit as st

from chatbot_engine_mk_not_yet import (
    ChatBotEngine,
    format_mcq, to_hard_breaks, get_choice_labels,
)

# ---------- Page / Session ----------
st.set_page_config(page_title="RAG 기반 멀티모드 챗봇", layout="wide")
st.title("RAG 기반 멀티모드 챗봇")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = ChatBotEngine(session_id=st.session_state.session_id)
ENGINE: ChatBotEngine = st.session_state.engine

# ---------- Chat log (scrollable) ----------
st.markdown('<div id="chatlog" style="max-height:72vh; overflow-y:auto; padding-right:8px;">', unsafe_allow_html=True)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    def mask(s): return (s[:4] + "..." + s[-4:]) if s else "None"
    st.markdown("### 🔑 Keys")
    st.write("OpenAI:", bool(os.getenv("OPENAI_API_KEY")), mask(os.getenv("OPENAI_API_KEY") or ""))
    st.write("Tavily:", bool(os.getenv("TAVILY_API_KEY")), mask(os.getenv("TAVILY_API_KEY") or ""))

    st.markdown("### 🧠 Memory")
    use_memory = st.checkbox("이전 대화 기억 사용", value=True)
    auto_topic = st.checkbox("자동 주제 전환 감지", value=True)
    topic_threshold = st.slider("주제 전환 임계값(유사도)", 0.00, 1.00, 0.72, 0.01)
    topic_debug = st.checkbox("토픽 디버그 표시", value=False)
    context_debug = st.checkbox("맥락 디버그 표시", value=False)

    if st.button("메모리 수동 초기화"):
        ENGINE.reset_topic()
        st.success("메모리와 라벨 캐시를 초기화했습니다.")
    if st.button("🧹 화면 채팅 로그 지우기"):
        st.session_state.messages = []
        st.success("화면 채팅 로그를 비웠습니다. (엔진 메모리는 그대로)")

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
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf", "txt"])
    use_semantic_chunk_txt = st.checkbox("TXT에 시맨틱 청킹 사용(비용↑)", value=False)
    chunk_size = st.number_input("청크 크기", min_value=200, max_value=4000, value=1000, step=50)
    chunk_overlap = st.number_input("청크 오버랩", min_value=0, max_value=1000, value=100, step=10)

# ---------- Retriever ----------
retriever = None
if uploaded_file:
    import os
    ext = uploaded_file.name.split(".")[-1].lower()
    os.makedirs("./mycache/files", exist_ok=True)
    save_path = os.path.join("./mycache/files", f"{uuid.uuid4().hex}.{ext}")
    try:
        file_bytes = uploaded_file.read()
        if ext == "pdf":
            with open(save_path, "wb") as fw: fw.write(file_bytes)
            retriever = ENGINE.build_retriever_from_pdf(save_path, int(chunk_size), int(chunk_overlap))
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore")
            with open(save_path, "w", encoding="utf-8") as fw: fw.write(text)
            retriever = ENGINE.build_retriever_from_text(text, use_semantic_chunk_txt, int(chunk_size), int(chunk_overlap))
        else:
            st.error("pdf/txt만 지원합니다.")
    except Exception as e:
        st.error(f"파일 처리 오류: {e}")

if retriever is None:
    retriever = ENGINE.safe_tavily_retriever(k=3)

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. /home/mk/workspace/KEYS.env 를 확인하세요.")
    st.stop()

# ---------- Chat Input ----------
user_input = st.chat_input("무엇이든 물어보세요. (예: 'SQLD 공부법', '객관식 5문제', '야구 규칙 OX' 등)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 1) 메타 메모리 질문은 즉시 처리
    if ENGINE.is_meta_memory_query(user_input):
        prev_q = ENGINE.memory.last_user_text()
        reply = f"당신이 전에 물어본 것은 다음입니다:\n\n> {prev_q}" if prev_q else "직전에 저장된 사용자 질문이 없어요."
        render = f"**🧑 질문:** {user_input}\n\n{reply}"
        with st.chat_message("assistant"):
            st.markdown(render)
        st.session_state.messages.append({"role": "assistant", "content": render})
        ENGINE.memory.add_turn(user_input, reply)
        st.stop()

    # 2) 자동 주제 전환 (지시표현 하드가드 포함)
    if use_memory and auto_topic:
        try:
            reset, dbg = ENGINE.should_reset_topic(user_input, threshold=topic_threshold, sticky_margin=0.05)
            if topic_debug and dbg:
                sim, thr = dbg
                st.sidebar.info(f"유사도: {sim:.3f} / 적용 임계값: {thr:.2f}")
            if reset:
                ENGINE.reset_topic()
                st.toast("🔄 새 주제로 전환되었습니다.", icon="🔁")
        except Exception as e:
            st.sidebar.warning(f"주제 전환 감지 경고: {e}")

    # 3) 체인 생성 & 실제 질의(활성맥락 + 도메인 라벨 주입)
    choice_labels = get_choice_labels(num_choices) if num_choices >= 2 else []
    choice_labels_text = " ".join(choice_labels) if choice_labels else ""
    chain = ENGINE.create_chain(retriever, response_mode, num_q, num_choices, choice_labels_text)

    scope_hint = ENGINE.scope_hint()
    topic_lbl = ENGINE.topic_label()  # ← 캐시 포함
    effective_query = ENGINE.rewrite_with_active_context(user_input, scope_hint, topic_lbl)
    if context_debug:
        st.sidebar.caption(f"scope_hint: {scope_hint}")
        st.sidebar.caption(f"topic_label: {topic_lbl or '(없음)'}")
        st.sidebar.caption(f"effective_query: {effective_query}")

    # 4) 실행 + 렌더
    try:
        response_stream = chain.stream(effective_query)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response_stream:
                ai_answer += token
                preview = ai_answer
                if response_mode in ["MCQ(객관식)", "혼합(객관+OX+단답)"]:
                    preview = to_hard_breaks(format_mcq(preview))
                container.markdown(f"**🧑 질문:** {user_input}\n\n{preview}")

            final_text = ai_answer
            if response_mode in ["MCQ(객관식)", "혼합(객관+OX+단답)"]:
                final_text = to_hard_breaks(format_mcq(final_text))
            final_render = f"**🧑 질문:** {user_input}\n\n{final_text}"
            container.markdown(final_render)

        st.session_state.messages.append({"role": "assistant", "content": final_render})
        ENGINE.memory.add_turn(user_input, final_text)

    except Exception as e:
        err = f"응답 생성 중 오류: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
