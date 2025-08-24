# tools.py
import json
import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import chardet
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

ANKI_CONNECT_URL = "http://192.168.160.1:8765"
tavily_search_tool = TavilySearchResults(max_results=3)

# 문서 검색을 위한 벡터스토어 설정
VECTORSTORE_DSN = "postgresql+psycopg2://play:123@localhost:5432/play"
DOCUMENT_COLLECTION_NAME = "play_documents"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
document_vector_store = PGVector(
    connection=VECTORSTORE_DSN,
    embeddings=embeddings,
    collection_name=DOCUMENT_COLLECTION_NAME,
    use_jsonb=True,
)

def web_search(query: str) -> str:
    """웹 검색 도구"""
    try:
        print(f"🖥️  웹 검색 수행: {query}")
        results = tavily_search_tool.invoke({"query": query})
        return f"'{query}'에 대한 웹 검색 결과입니다.\n\n{results}"
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"

def document_search(query: str) -> str:
    """
    🔧 디버깅이 강화된 문서 검색 함수
    - 점수 계산 과정 상세 로깅
    - 임계값 문제 해결
    - 더 관대한 필터링
    """
    try:
        print(f"📄 문서 검색 수행: '{query}'")
        
        # 1. 초기 검색
        retriever = document_vector_store.as_retriever(search_kwargs={"k": 15})
        documents = retriever.invoke(query)
        
        if not documents:
            print("   ❌ 검색된 문서가 없음")
            return "업로드된 문서에서 관련 정보를 찾을 수 없습니다. 먼저 문서를 업로드해주세요."
        
        print(f"   🔍 초기 검색 결과: {len(documents)}개 문서")
        
        # 2. 🔥 디버깅: 문서 내용 미리보기
        for i, doc in enumerate(documents[:3], 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            source = doc.metadata.get('source', '알 수 없음')
            print(f"   📋 문서 {i}: {source} - {preview}...")
        
        # 3. 관련성 점수 계산 (디버깅 강화)
        scored_docs = []
        query_embedding = embeddings.embed_query(query)
        print(f"   🧮 쿼리 임베딩 생성 완료 (차원: {len(query_embedding)})")
        
        for i, doc in enumerate(documents):
            try:
                # 문서 임베딩 계산
                doc_embedding = embeddings.embed_query(doc.page_content)
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    [query_embedding], [doc_embedding]
                )[0][0]
                
                # 🔥 디버깅: 점수 출력
                source = doc.metadata.get('source', '알 수 없음')
                chunk_id = doc.metadata.get('chunk_id', 0)
                print(f"   📊 문서 {i+1} 점수: {similarity:.4f} ({source}, 청크 {chunk_id})")
                
                scored_docs.append({
                    'document': doc,
                    'score': similarity,
                    'source': source,
                    'chunk_id': chunk_id
                })
                
            except Exception as e:
                print(f"   ❌ 문서 {i+1} 점수 계산 실패: {e}")
                continue
        
        if not scored_docs:
            return "문서 유사도 계산에 실패했습니다."
        
        # 4. 점수 정렬
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 🔥 점수 분포 분석
        scores = [doc['score'] for doc in scored_docs]
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"   📈 점수 분포: 최고 {max_score:.4f}, 최저 {min_score:.4f}, 평균 {avg_score:.4f}")
        
        # 5. 🔥 더 관대한 임계값 설정
        # 기존의 0.7, 0.5는 너무 높았음
        PRIMARY_THRESHOLD = 0.3    # 0.7 → 0.3
        FALLBACK_THRESHOLD = 0.1   # 0.5 → 0.1
        
        relevant_docs = [doc for doc in scored_docs if doc['score'] >= PRIMARY_THRESHOLD]
        print(f"   ✅ 1차 필터링 (임계값 {PRIMARY_THRESHOLD}): {len(relevant_docs)}개 문서")
        
        # 관련성이 높은 문서가 없으면 임계값을 더 낮춤
        if not relevant_docs:
            relevant_docs = [doc for doc in scored_docs if doc['score'] >= FALLBACK_THRESHOLD]
            print(f"   🔄 2차 필터링 (임계값 {FALLBACK_THRESHOLD}): {len(relevant_docs)}개 문서")
        
        # 그래도 없으면 상위 5개라도 보여주기
        if not relevant_docs:
            relevant_docs = scored_docs[:5]
            print(f"   🆘 강제 선택: 상위 {len(relevant_docs)}개 문서 (점수 무관)")
        
        # 6. 중복 제거 (간소화)
        unique_docs = remove_duplicate_content(relevant_docs)
        print(f"   🔄 중복 제거 후: {len(unique_docs)}개 문서")
        
        # 7. 상위 결과만 선택
        top_docs = unique_docs[:5]
        
        # 8. 컨텍스트 구성
        context_parts = []
        for i, doc_info in enumerate(top_docs, 1):
            doc = doc_info['document']
            score = doc_info['score']
            source = doc_info['source']
            content = doc.page_content.strip()
            
            # 🔥 점수 표시를 더 친화적으로
            score_desc = "높음" if score >= 0.5 else "중간" if score >= 0.3 else "낮음"
            
            context_part = f"""📋 **문서 {i}** (관련성: {score_desc}) - {source}
{content}

"""
            context_parts.append(context_part)
        
        # 최종 결과
        if context_parts:
            context = f"""🔍 **'{query}' 검색 결과** ({len(context_parts)}개 관련 문서 발견)

{"".join(context_parts)}
📌 **검색 정보**: {len(documents)}개 문서 중 {len(context_parts)}개 문서를 선별했습니다.
📊 **점수 범위**: {max_score:.3f} ~ {min_score:.3f} (평균 {avg_score:.3f})"""
            
            print(f"   📊 최종 컨텍스트: {len(context)}자, {len(context_parts)}개 문서 포함")
            return context
        else:
            return f"'{query}'와 관련된 정보를 찾을 수 없습니다."
        
    except Exception as e:
        print(f"❌ 문서 검색 오류: {e}")
        import traceback
        traceback.print_exc()  # 상세한 에러 정보
        return f"문서 검색 중 오류가 발생했습니다: {e}"

# 🔥 간소화된 중복 제거 함수
def remove_duplicate_content(scored_docs: list) -> list:
    """중복 제거를 더 간단하고 안정적으로 처리"""
    if len(scored_docs) <= 1:
        return scored_docs
    
    unique_docs = []
    seen_contents = set()
    
    for doc_info in scored_docs:
        content = doc_info['document'].page_content
        # 내용의 첫 100자를 기준으로 중복 체크 (더 간단한 방식)
        content_key = content[:100].strip().lower()
        
        if content_key not in seen_contents:
            unique_docs.append(doc_info)
            seen_contents.add(content_key)
    
    return unique_docs

# 🔥 추가: 문서 현황 확인 함수
def check_document_status() -> str:
    """업로드된 문서 현황을 확인하는 함수"""
    try:
        # 전체 문서 검색
        retriever = document_vector_store.as_retriever(search_kwargs={"k": 100})
        all_docs = retriever.invoke("*")
        
        if not all_docs:
            return "❌ 업로드된 문서가 없습니다."
        
        # 문서별 통계
        sources = {}
        for doc in all_docs:
            source = doc.metadata.get('source', '알 수 없음')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        status_msg = f"📊 **업로드된 문서 현황** (총 {len(all_docs)}개 청크)\n\n"
        
        for source, count in sources.items():
            status_msg += f"📄 {source}: {count}개 청크\n"
        
        # 샘플 내용 미리보기
        status_msg += f"\n**샘플 내용:**\n"
        for i, doc in enumerate(all_docs[:3], 1):
            preview = doc.page_content[:50].replace('\n', ' ')
            source = doc.metadata.get('source', '알 수 없음')
            status_msg += f"{i}. [{source}] {preview}...\n"
        
        return status_msg
        
    except Exception as e:
        return f"❌ 문서 현황 확인 중 오류: {e}"

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트의 Jaccard 유사도를 계산합니다.
    """
    # 단어 단위로 분할
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Jaccard similarity = |A ∩ B| / |A ∪ B|
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def upload_document(file_path: str, file_name: str) -> str:
    """
    🔥 개선된 문서 업로드 - 더 작은 청크 크기로 정확도 향상
    """
    try:
        print(f"📁 문서 업로드 중: {file_name}")
        
        # 인코딩 감지 및 파일 읽기
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        content = raw_data.decode(encoding, errors='replace')
        
        # NULL 문자 제거
        content = content.replace('\x00', '')

        # 🔥 더 작은 청크 크기로 변경 (정확도 향상)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,        # 1000 → 600으로 축소
            chunk_overlap=150,     # 200 → 150으로 조정
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        chunks = text_splitter.split_text(content)
        metadatas = [{"source": file_name, "chunk_id": i} for i in range(len(chunks))]
        
        print(f"   📄 총 {len(chunks)}개 청크로 분할됨 (청크 크기: 600자)")
        
        # 배치 처리

        document_vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas,
            collection_name=DOCUMENT_COLLECTION_NAME
        )
            
         

        success_msg = f"""✅ **'{file_name}' 문서 업로드 완료!**"""
        
        return success_msg

    except Exception as e:
        return f"문서 업로드 중 오류가 발생했습니다: {e}"

def anki_card_saver(front: str, back: str, deck: str = "기본", tags: list = None) -> str:
    """
    AnkiConnect API를 사용하여 Anki 데스크톱 프로그램에 새 카드를 추가합니다.
    """
    print(f"🃏 Anki 카드 저장 시도: {front[:30]}...")
    
    def anki_request(action, **params):
        return {'action': action, 'version': 6, 'params': params}

    try:
        # 1. 중복 카드 확인 - 더 정확한 검색을 위해 앞면 내용의 핵심 키워드로 검색
        front_keywords = front.replace('\n', ' ')[:50]  # 앞면의 첫 50자로 중복 검사
        
        query = f'deck:"{deck}" front:*{front_keywords}*'
        find_payload = anki_request('findNotes', query=query)
        response = requests.post(ANKI_CONNECT_URL, json=find_payload)
        response.raise_for_status()
        
        print(f"   -> Anki 'findNotes' 응답: {response.json()}")
        
        existing_notes = response.json().get('result', [])
        if existing_notes:
            message = f"유사한 카드가 '{deck}' 덱에 이미 존재하여 새로 추가하지 않았습니다."
            print(f"   -> {message}")
            return message

        # 2. 새 노트(카드) 추가
        note_params = {
            'note': {
                'deckName': deck,
                'modelName': 'Basic',
                'fields': {
                    'Front': front,
                    'Back': back.replace("\n", "<br>")  # HTML 줄바꿈으로 변환
                },
                'tags': tags if tags else []
            }
        }
        
        add_payload = anki_request('addNote', **note_params)
        response = requests.post(ANKI_CONNECT_URL, json=add_payload)
        response.raise_for_status()

        print(f"   -> Anki 'addNote' 응답: {response.json()}")

        response_data = response.json()
        if error := response_data.get('error'):
            raise Exception(f"AnkiConnect 오류: {error}")
        
        note_id = response_data.get('result')
        message = f"✅ Anki 카드를 '{deck}' 덱에 성공적으로 추가했습니다. (ID: {note_id})"
        print(f"   -> {message}")
        return message

    except requests.exceptions.ConnectionError:
        error_msg = "❌ AnkiConnect에 연결할 수 없습니다. Anki 프로그램이 실행 중이고 AnkiConnect 애드온이 설치되었는지 확인하세요."
        print(f"   -> {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ Anki 카드 저장 중 오류 발생: {e}"
        print(f"   -> {error_msg}")
        return error_msg

def list_anki_decks() -> str:
    """Anki의 모든 덱 목록을 가져오는 도구"""
    def anki_request(action, **params):
        return {'action': action, 'version': 6, 'params': params}
    
    try:
        payload = anki_request('deckNames')
        response = requests.post(ANKI_CONNECT_URL, json=payload)
        response.raise_for_status()
        
        decks = response.json().get('result', [])
        if decks:
            return f"사용 가능한 Anki 덱 목록:\n" + "\n".join([f"- {deck}" for deck in decks])
        else:
            return "사용 가능한 덱이 없습니다."
            
    except Exception as e:
        return f"덱 목록 조회 중 오류 발생: {e}"