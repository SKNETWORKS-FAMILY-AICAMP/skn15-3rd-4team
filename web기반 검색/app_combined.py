# app_combined.py
import os, uuid
import streamlit as st
from dotenv import load_dotenv

# ------------------- Load tools -------------------
from tools.tavily_tool import TavilySearchTool
from tools.google_tool import GoogleImageSearchTool
from chatbot_engine_mk_not_yet import ChatBotEngine, format_mcq, to_hard_breaks, get_choice_labels

# Load environment variables
load_dotenv()

# ------------------- Streamlit Page -------------------
st.set_page_config(page_title="RAG + MCP 챗봇", layout="wide")
st.title("RAG 기반 + MCP Tool 검색 챗봇")

# ------------------- Session Setup -------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = ChatBotEngine(session_id=st.session_state.session_id)
ENGINE: ChatBotEngine = st.session_state.engine

# ------------------- Tool Instances -------------------
tavily_tool = TavilySearchTool()
google_tool = GoogleImageSearchTool()

# ------------------- Sidebar -------------------
with st.sidebar:
    st.markdown("### 🔑 Keys")
    def mask(s): return (s[:4]+"..."+s[-4:]) if s else "None"
    st.write("OpenAI:", bool(os.getenv("OPENAI_API_KEY")), mask(os.getenv("OPENAI_API_KEY")))
    st.write("Tavily:", bool(os.getenv("TAVILY_API_KEY")), mask(os.getenv("TAVILY_API_KEY")))

    # Memory options
    use_memory = st.checkbox("이전 대화 기억 사용", value=True)
    auto_topic = st.checkbox("자동 주제 전환 감지", value=True)

    if st.button("메모리 초기화"):
        ENGINE.reset_topic()
        st.success("메모리 초기화 완료")
    if st.button("🧹 채팅 로그 지우기"):
        st.session_state.messages = []
        st.success("화면 채팅 로그 비움 (엔진 메모리는 유지)")

# ------------------- Chat UI -------------------
st.markdown("### 💬 RAG 기반 챗봇")
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 1) 자동 주제 전환
    if use_memory and auto_topic:
        try:
            reset, _ = ENGINE.should_reset_topic(user_input)
            if reset:
                ENGINE.reset_topic()
                st.toast("🔄 새 주제로 전환되었습니다.", icon="🔁")
        except Exception as e:
            st.sidebar.warning(f"주제 전환 감지 오류: {e}")

    # 2) RAG 챗봇 응답 생성
    retriever = ENGINE.safe_tavily_retriever(k=3)
    chain = ENGINE.create_chain(retriever)
    try:
        response_stream = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response_stream:
                ai_answer += token
                container.markdown(ai_answer)
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
    except Exception as e:
        err = f"RAG 챗봇 응답 중 오류: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})

# ------------------- MCP Tool 검색 UI -------------------
st.markdown("### 🔍 MCP Tool 검색")
mcp_query = st.text_input("검색어를 입력하고 Enter", key="mcp_query")
if mcp_query:
    st.subheader("Tavily 검색 결과")
    try:
        docs = tavily_tool.run(mcp_query)
        if not docs:
            st.write("검색 결과가 없습니다.")
        for d in docs:
            st.markdown(f"**[{d['title']}]({d['url']})**")
            snippet = (d['content'] or "")[:200] + "..."
            st.write(snippet)
    except Exception as e:
        st.write("Tavily API 호출 실패:", e)

    st.subheader("Google 이미지 검색")
    try:
        images = google_tool.run(mcp_query)
        if images:
            cols = st.columns(3)
            for idx, img in enumerate(images):
                with cols[idx % 3]:
                    st.image(img["image_url"], use_column_width=True)
                    st.caption(img["title"])
        else:
            st.write("이미지 검색 결과가 없습니다.")
    except Exception as e:
        st.write("Google 이미지 API 호출 실패:", e)
