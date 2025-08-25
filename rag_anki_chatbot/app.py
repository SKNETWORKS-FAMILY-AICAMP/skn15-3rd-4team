# app.py
import streamlit as st
import tempfile
import os
from langchain_core.messages import AIMessage, HumanMessage

# agent.py에서 Agent 실행기 함수를 가져옵니다.
from agent import get_agent_executor
from tools import upload_document, list_anki_decks, check_document_status

st.set_page_config(page_title="RAG Anki 챗봇", layout="wide")
st.title("단위프로젝트 챗봇 만들기")

# --- 세션 ID 생성 ---
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]  # 짧은 세션 ID

# --- Agent 및 세션 상태 초기화 ---

# Agent 실행기는 리소스가 크므로 캐싱하여 한 번만 생성합니다.
@st.cache_resource
def initialize_agent():
    return get_agent_executor()

agent_executor = initialize_agent()

# 세션 상태에 메시지 기록이 없으면 초기화합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- 사이드바 ---
with st.sidebar:
    st.header("📚 메뉴")
    
    # 🔥 세션 정보 표시
    st.info(f"🔑 세션 ID: {st.session_state.session_id}")
    st.info(f"💬 대화 기록: {len(st.session_state.messages)}개")
    
    # 파일 업로드 섹션
    st.subheader("📄 문서 업로드")
    uploaded_file = st.file_uploader(
        "학습 자료를 업로드하세요",
        type=['txt', 'pdf', 'docx', 'md'],
        help="업로드된 문서는 '문서에서 찾아줘' 같은 질문으로 검색할 수 있습니다."
    )
    
    if uploaded_file is not None:
        if st.button("문서 업로드 📤"):
            try:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                
                # 🔥 개선: 세션 ID를 함께 전달하여 세션별 문서 관리
                with st.spinner("문서를 처리 중입니다..."):
                    result = upload_document(tmp_file_path, uploaded_file.name, st.session_state.session_id)
                    st.success(result)
                    
                    # 업로드된 파일 목록에 추가 (중복 방지)
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"문서 업로드 중 오류 발생: {e}")
    
    # 업로드된 파일 목록
    if st.session_state.uploaded_files:
        st.subheader("📋 업로드된 문서")
        for file_name in st.session_state.uploaded_files:
            st.text(f"• {file_name}")
        
        # 🔥 추가: 문서 현황 상세 확인 버튼
        if st.button("문서 현황 상세보기"):
            with st.spinner("문서 현황을 확인 중..."):
                status = check_document_status(st.session_state.session_id)
                st.text(status)
    
    st.divider()
    
    # Anki 관련 정보
    st.subheader("🃏 Anki 정보")
    if st.button("덱 목록 확인"):
        with st.spinner("Anki 덱을 확인 중..."):
            deck_list = list_anki_decks()
            st.text(deck_list)
    
    st.divider()
    
    # 채팅 기록 관리
    st.subheader("🗑️ 기록 관리")
    if st.button("채팅 기록 초기화"):
        st.session_state.messages = []
        st.success("채팅 기록이 초기화되었습니다.")
        st.rerun()
    
    # 🔥 새로운 세션 시작 버튼 추가
    if st.button("새 세션 시작"):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.success(f"새 세션이 시작되었습니다! (ID: {st.session_state.session_id})")
        st.rerun()
    
    # 사용법 안내
    st.divider()
    st.subheader("💡 사용법")
    st.markdown("""
    **기본 대화:**
    - 일반적인 질문을 자유롭게 하세요
    
    **Anki 카드 생성:**
    - 학습하고 싶은 내용에 대해 대화한 후
    - "저장해줘" 또는 "카드 만들어줘"
    
    **이전 질문 내용 검색:**
    - "옛날에 내가 트렌스포머와 관련된 질문했던거 보여줘"
    - "저장한 카드 내용 보여줘"
    
    **문서 검색:**
    - 문서 업로드 후
    - "문서에서 찾아줘"
    - "업로드한 파일에 대해 알려줘"
    
    **세션 관리:**
    - 각 세션별로 문서가 독립 관리됩니다
    - 새 세션 시작으로 깨끗한 환경에서 시작 가능
    """)

# --- 메인 채팅 화면 ---

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (예: '저장해줘', '문서에서 찾아줘')"):
    # 사용자의 메시지를 HumanMessage로 변환하여 기록에 추가하고 화면에 표시
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent를 실행하고 스트리밍 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # 🔥 핵심 수정: 전체 대화 기록과 세션 ID를 Agent에게 전달
                stream = agent_executor.stream({
                    "messages": st.session_state.messages.copy(),  # 전체 대화 기록 전달
                    "query": prompt,  # 현재 질문
                    "session_id": st.session_state.session_id  # 세션 ID 전달
                })
                
                # 스트림에서 응답 처리
                response_container = st.empty()
                full_response = ""
                
                for chunk in stream:
                    # 각 노드의 결과를 확인하고 최종 응답만 표시
                    if "synthesizer" in chunk:
                        ai_message_chunk = chunk["synthesizer"]["messages"][-1]
                        full_response = ai_message_chunk.content
                        response_container.markdown(full_response + "▌")
                    
                    elif "saver" in chunk:
                        ai_message_chunk = chunk["saver"]["messages"][-1]
                        full_response = ai_message_chunk.content
                        response_container.markdown(full_response + "▌")
                
                # 최종 응답 표시
                if full_response:
                    response_container.markdown(full_response)
                    
                    # 완성된 AI 응답을 AIMessage로 변환하여 기록에 추가
                    ai_message = AIMessage(content=full_response)
                    st.session_state.messages.append(ai_message)
                else:
                    error_msg = "응답을 생성하지 못했습니다. 다시 시도해주세요."
                    response_container.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
            
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))