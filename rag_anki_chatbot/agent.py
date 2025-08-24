# agent.py
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, END
from tools import web_search, anki_card_saver, document_search 
from utils import parse_anki_cards

load_dotenv()

# --- 1. DB 설정 및 Retriever 생성 ---
VECTORSTORE_DSN = "postgresql+psycopg2://play:123@localhost:5432/play"
COLLECTION_NAME = "play_anki_cards"
DOCUMENT_COLLECTION_NAME = "play_documents"  # 문서용 컬렉션 추가

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVector(
    connection=VECTORSTORE_DSN,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
)
db_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 문서용 벡터 스토어 추가
document_vector_store = PGVector(
    connection=VECTORSTORE_DSN,
    embeddings=embeddings,
    collection_name=DOCUMENT_COLLECTION_NAME,
    use_jsonb=True,
)
document_retriever = document_vector_store.as_retriever(search_kwargs={"k": 5})

# --- 2. LangGraph 상태(State) 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    route: str
    context: str
    conversation_history: str  # 전체 대화 기록을 명시적으로 저장

# --- 3. 노드(Node) 함수 정의 ---
def router_node(state: AgentState) -> dict:
    print("--- 1. ROUTER ---")
    query = state["query"]
    
    # 🔥 대폭 개선된 라우터: 더 정확한 키워드 기반 판단
    prompt_router = ChatPromptTemplate.from_template(
        """당신은 사용자의 질문을 분석하여 가장 적합한 정보 소스로 안내하는 라우팅 전문가입니다.
        사용자의 질문을 보고 아래 4가지 카테고리 중 가장 적절한 것 하나만 골라 답하세요.
        
        중요: 반드시 정확히 다음 4개 단어 중 하나만 출력하세요:
        "Anki 저장" 또는 "데이터베이스" 또는 "문서 검색" 또는 "웹 검색"
        
        1순위: "Anki 저장"
        - 키워드: "저장", "카드 만들어", "안키", "요약", "문제 출제", "복습 카드"
        - 예시: "이 내용 저장해줘", "카드 만들어줘", "안키에 저장"
        
        2순위: "문서 검색"** (업로드된 문서 관련) 
        - 키워드: "문서에서", "파일에서", "업로드", "올린", "첨부", "문서 내용", "문서에 있는"
        - 예시: "문서에서 찾아줘", "업로드한 파일", "올린 문서에서", "첨부 파일"
        
        3순위: "데이터베이스"** (이전 대화/기록 관련)
        - 키워드: "이전에", "지난번", "저장한 카드", "예전에", "과거", "기록", "전에"
        - 예시: "지난번에 얘기한 거", "이전 대화", "저장된 내용"
        
        4순위: "웹 검색"** (그 외 모든 일반 질문)
        - 위 3가지에 해당하지 않는 모든 질문
        - 예시: "날씨", "뉴스", "일반 지식", "최신 정보"
        
        **분석 과정:**
        1. 질문에 "문서", "파일", "업로드", "올린" 등이 있는가? → 문서 검색
        2. 질문에 "저장", "카드", "안키" 등이 있는가? → Anki 저장  
        3. 질문에 "이전", "지난번", "저장한 카드" 등이 있는가? → 데이터베이스
        4. 그 외 → 웹 검색
        
        [사용자 질문]: "{query}"
        
        분석:
        - 문서 관련 키워드 포함 여부: 
        - 저장 관련 키워드 포함 여부:
        - 이전 대화 관련 키워드 포함 여부:
        
        [최종 라우팅 결과]: """
    )
    
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")  # 온도 0으로 일관성 확보
    routing_chain = prompt_router | llm
    
    try:
        response = routing_chain.invoke({"query": query})
        route_text = response.content.strip()
        
        # 🔥 라우팅 결과 정리 (마지막 줄에서 추출)
        lines = route_text.split('\n')
        route = lines[-1].strip()
        
        # 결과에서 실제 라우팅 키워드만 추출
        if "Anki 저장" in route:
            route = "Anki 저장"
        elif "문서 검색" in route:
            route = "문서 검색"
        elif "데이터베이스" in route:
            route = "데이터베이스"
        elif "웹 검색" in route:
            route = "웹 검색"
        else:
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ["문서에서", "파일에서", "업로드", "올린", "첨부", "문서 내용"]):
                route = "문서 검색"
                print("   🎯 키워드 기반 강제 라우팅: 문서 검색")
            elif any(keyword in query_lower for keyword in ["저장", "카드", "안키", "anki"]):
                route = "Anki 저장"
                print("   🎯 키워드 기반 강제 라우팅: Anki 저장")
            elif any(keyword in query_lower for keyword in ["이전에", "지난번", "저장한", "예전에"]):
                route = "데이터베이스"
                print("   🎯 키워드 기반 강제 라우팅: 데이터베이스")
            else:
                route = "웹 검색"
                print("   🎯 키워드 기반 강제 라우팅: 웹 검색")
        
        print(f"   -> 질문: '{query[:50]}...'")
        print(f"   -> 라우팅 결과: {route}")
        print(f"   -> LLM 전체 응답:\n{route_text}")
        
        return {"route": route}
    
    except Exception as e:
        print(f"   ❌ 라우터 오류: {e}")
        # 오류 시 키워드 기반 라우팅으로 폴백
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["문서에서", "파일에서", "업로드", "올린"]):
            fallback_route = "문서 검색"
        elif any(keyword in query_lower for keyword in ["저장", "카드"]):
            fallback_route = "Anki 저장"
        else:
            fallback_route = "웹 검색"
        
        print(f"   🔄 폴백 라우팅: {fallback_route}")
        return {"route": fallback_route}

# 라우터 관련 함수들
def db_retriever_node(state: AgentState) -> dict:
    print("--- 2-1. DB RETRIEVER ---")
    query = state["query"]
    documents = db_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in documents])
    print("   -> Context from DB retrieved.")
    return {"context": context}

def document_retriever_node(state: AgentState) -> dict:
    print("--- 2-2. DOCUMENT RETRIEVER ---")
    query = state["query"]
    context = document_search(query=query)  # tools.py에서 구현할 함수
    print("   -> Context from Documents retrieved.")
    return {"context": context}

def web_retriever_node(state: AgentState) -> dict:
    print("--- 2-3. WEB RETRIEVER ---")
    query = state["query"]
    context = web_search(query=query)
    print("   -> Context from Web retrieved.")
    return {"context": context}

def save_anki_card_node(state: AgentState) -> dict:
    print("--- 2-4. ANKI SAVER ---")
    messages = state["messages"]
    
    # 현재 "저장해줘" 메시지를 제외한 모든 이전 대화를 포함
    relevant_messages = []
    
    for msg in messages[:-1]:  # 마지막 저장 요청 메시지 제외
        if isinstance(msg, HumanMessage):
            relevant_messages.append(f"사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            relevant_messages.append(f"어시스턴트: {msg.content}")
    
    # 대화 기록이 없으면 에러 메시지
    if not relevant_messages:
        return {"messages": [AIMessage(content="❌ 저장할 대화 내용이 없습니다. 먼저 학습할 내용에 대해 대화를 나눠주세요.")]}
    
    conversation_text = "\n\n".join(relevant_messages)
    
    print(f"📝 처리할 대화 기록 수: {len(relevant_messages)}개")
    print(f"📝 대화 기록 샘플:\n{conversation_text[:200]}...")

    # 🔥 더욱 개선된 카드 생성 프롬프트
    prompt_card_generator = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 전문 교육자이자 문제 출제자입니다. 
         주어진 대화 기록을 바탕으로 학습에 효과적인 Anki 카드를 생성하세요.
         
         **중요 원칙:**
         1. 대화에서 다룬 모든 핵심 개념을 포괄해야 합니다
         2. 단순 암기가 아닌 이해를 돕는 카드를 만드세요
         3. 대화에 없는 내용은 추가하지 마세요
         4. 구체적인 예시나 설명이 있다면 반드시 포함하세요"""),
        
        ("user",
         """아래 대화 기록을 분석하여 학습용 Anki 카드를 생성하세요.

         **필수 출력 형식:**

         앞면
         [개념명 또는 질문]

         뒷면
         [상세한 설명, 정의, 예시 포함]

         앞면
         Question: [4지선다 문제]
         A. [선택지 1]
         B. [선택지 2] 
         C. [선택지 3]
         D. [선택지 4]

         뒷면
         Answer: [정답 (A/B/C/D)]
         해설: [정답 이유와 오답 설명]

         **요구사항:**
         - 대화에서 언급된 핵심 개념마다 개념 카드 1개씩 생성
         - 각 주요 주제마다 4지선다 문제 카드 1개씩 생성
         - 최소 2개, 최대 6개 카드 생성
         - 모든 내용은 아래 대화 기록에 근거해야 함

         **대화 기록:**
         {conversation}
         """),
    ])
    
    card_generation_llm = ChatOpenAI(model="gpt-4.1")
    card_chain = prompt_card_generator | card_generation_llm
    
    try:
        generated_text = card_chain.invoke({"conversation": conversation_text}).content
        
        # 디버깅용 print
        print("\n" + "="*50)
        print("🎯 LLM이 생성한 카드 텍스트:")
        print(generated_text)
        print("="*50 + "\n")
        
        cards = parse_anki_cards(generated_text)
        
        if not cards:
            return {"messages": [AIMessage(content="카드 생성에 실패했습니다. 대화 내용을 다시 확인해주세요.")]}

        # 카드 저장 및 성공 메시지 생성
        saved_count = 0
        card_summaries = []
        failed_saves = []
        
        for i, card in enumerate(cards, 1):
            try:
                # Anki에 저장
                result = anki_card_saver(
                    front=card['front'], 
                    back=card['back'], 
                    deck="기본", 
                    tags=["chatbot-generated", "conversation-based"]
                )
                
                # 벡터DB에도 저장 (검색 가능하도록)
                vector_store.add_texts(
                    texts=[f"{card['front']}\n{card['back']}"], 
                    collection_name=COLLECTION_NAME
                )
                
                saved_count += 1
                
                # 카드 요약 (앞면의 첫 30자)
                front_preview = card['front'].replace('\n', ' ')[:40]
                card_summaries.append(f"{i}. {front_preview}...")
                
            except Exception as e:
                failed_saves.append(f"카드 {i}: {str(e)}")
        
        # 성공 메시지 생성
        success_msg = f"""✅ **대화 내용이 Anki 카드로 변환되었습니다!**

**저장 결과:**
- 생성된 카드: **{len(cards)}개**
- 성공적으로 저장: **{saved_count}개**
- 저장 위치: Anki **'기본'** 덱

**생성된 카드 목록:**
{chr(10).join(card_summaries)}

**포함된 대화 범위:**
- 총 {len(relevant_messages)}개 메시지 분석
- 핵심 개념과 설명을 카드로 변환

**다음 단계:**
- Anki에서 즉시 복습 가능
- 향후 대화에서 이 내용 참조 가능"""

        if failed_saves:
            success_msg += f"\n\n⚠️ **저장 실패:** {'; '.join(failed_saves)}"

        return {"messages": [AIMessage(content=success_msg)]}
        
    except Exception as e:
        error_msg = f"❌ 카드 생성 중 오류가 발생했습니다: {str(e)}"
        print(f"ERROR in save_anki_card_node: {e}")
        return {"messages": [AIMessage(content=error_msg)]}

def synthesizer_node(state: AgentState) -> dict:
    print("--- 3. SYNTHESIZER ---")
    context = state["context"]
    messages = state["messages"]
    
    # 시스템 프롬프트
    system_prompt = """당신은 사용자와 지속적인 대화를 나누는 지능형 AI 어시스턴트입니다.

핵심 역할:
1. 이전 대화 내용을 기억하고 맥락을 유지합니다.
2. 사용자의 후속 질문("그게 뭐야?", "더 자세히", "예를 들어")에 맥락적으로 답변합니다.
3. 검색된 정보와 대화 기록을 종합하여 유용한 답변을 제공합니다.

답변 스타일:
- 친근하고 자연스러운 대화체
- 복잡한 개념은 쉽게 설명  
- 필요시 구체적 예시나 비유 활용
- 이전 대화를 자연스럽게 연결"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", """
[검색된 정보/컨텍스트]
{context}

[이전 대화 기록]
{history}

[현재 질문]
{query}

위 정보를 종합하여 자연스럽고 도움이 되는 답변을 해주세요.
특히 이전 대화의 맥락을 고려하여 답변해주세요.
""")
    ])
    
    llm = ChatOpenAI(model="gpt-4.1")
    chain = prompt | llm
    
    # 대화 기록을 문자열로 변환 (현재 질문 제외)
    history_messages = []
    for msg in messages[:-1]:  # 마지막 메시지(현재 질문) 제외
        role = "사용자" if isinstance(msg, HumanMessage) else "어시스턴트"
        history_messages.append(f"{role}: {msg.content}")
    
    history_str = "\n".join(history_messages[-10:])  # 최근 10개 메시지만 사용
    current_query = messages[-1].content if messages else state["query"]
    
    # 디버깅 정보
    print(f"   -> Context: {len(context) if context else 0} chars")
    print(f"   -> History: {len(history_messages)} messages")
    print(f"   -> Query: {current_query[:50]}...")
    
    try:
        response = chain.invoke({
            "history": history_str,
            "context": context if context else "관련 정보를 찾지 못했습니다.",
            "query": current_query
        })
        
        return {"messages": [AIMessage(content=response.content)]}
    
    except Exception as e:
        error_response = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        return {"messages": [AIMessage(content=error_response)]}

# --- 4. 그래프(Graph) 구성 ---
def get_agent_executor():
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("router", router_node)
    graph.add_node("db_retriever", db_retriever_node)
    graph.add_node("document_retriever", document_retriever_node)  # 문서 검색 노드 추가
    graph.add_node("web_retriever", web_retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("saver", save_anki_card_node)
    
    # 시작점 설정
    graph.set_entry_point("router")
    
    # 라우팅 조건부 엣지
    graph.add_conditional_edges(
        "router", 
        lambda state: state["route"], 
        {
            "Anki 저장": "saver", 
            "데이터베이스": "db_retriever", 
            "문서 검색": "document_retriever",  # 문서 검색 라우트 추가
            "웹 검색": "web_retriever"
        }
    )
    
    # 각 retriever에서 synthesizer로 연결
    graph.add_edge("db_retriever", "synthesizer")
    graph.add_edge("document_retriever", "synthesizer")  # 문서 검색 후 synthesizer로
    graph.add_edge("web_retriever", "synthesizer")
    
    # 종료 엣지
    graph.add_edge("synthesizer", END)
    graph.add_edge("saver", END)
    
    return graph.compile()