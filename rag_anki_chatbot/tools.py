# tools.py
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
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
EPHEMERAL_STORES = {}  # key: file_name, value: FAISS VectorStore

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
    업로드된 인메모리 FAISS 인덱스(EPHEMERAL_STORES)에서만 검색.
    - DB는 사용하지 않음
    - 기존의 상세 디버깅/점수 계산 로직은 유지·간소화
    """
    try:
        print(f"📄 인메모리 문서 검색: '{query}'")

        if not EPHEMERAL_STORES:
            return "업로드한 문서를 찾을 수 없습니다. 먼저 문서를 업로드해주세요."

        # 각 파일 인덱스에서 k개씩 수집
        k_per_file = 5
        gathered = []
        for fname, store in EPHEMERAL_STORES.items():
            try:
                docs = store.as_retriever(search_kwargs={"k": k_per_file}).get_relevant_documents(query)
                gathered.extend(docs)
            except Exception as e:
                print(f"   ❌ '{fname}' 검색 실패: {e}")

        if not gathered:
            return f"'{query}'에 대한 관련 내용을 찾지 못했습니다."

        # 점수 계산(코사인 유사도)
        q_emb = embeddings.embed_query(query)
        scored = []
        for i, d in enumerate(gathered):
            try:
                d_emb = embeddings.embed_query(d.page_content)
                sim = cosine_similarity([q_emb], [d_emb])[0][0]
                scored.append({
                    "document": d,
                    "score": float(sim),
                    "source": d.metadata.get("source", "알 수 없음"),
                    "chunk_id": d.metadata.get("chunk_id", i),
                })
            except Exception as e:
                print(f"   ❌ 점수계산 실패: {e}")

        if not scored:
            return "문서 유사도 계산에 실패했습니다."

        scored.sort(key=lambda x: x['score'], reverse=True)

        # 임계값(관대)
        PRIMARY = 0.3
        FALLBACK = 0.1
        picked = [x for x in scored if x['score'] >= PRIMARY] or \
                 [x for x in scored if x['score'] >= FALLBACK] or \
                 scored[:5]

        # 중복 제거(간단)
        seen = set()
        unique = []
        for x in picked:
            key = (x['source'], str(x['document'].page_content)[:100].lower())
            if key not in seen:
                seen.add(key)
                unique.append(x)

        top = unique[:5]
        if not top:
            return f"'{query}'와 관련된 정보를 찾지 못했습니다."

        # 컨텍스트 구성
        parts = []
        scores = [x['score'] for x in scored]
        for i, x in enumerate(top, 1):
            lvl = "높음" if x['score'] >= 0.5 else "중간" if x['score'] >= 0.3 else "낮음"
            content = x['document'].page_content.strip()
            parts.append(
                f"📋 **문서 {i}** (관련성: {lvl}) - {x['source']} (청크 {x['chunk_id']})\n{content}\n"
            )

        return (
            f"🔍 **'{query}' 검색 결과** ({len(parts)}개 관련 문서)\n\n" +
            "\n".join(parts) +
            f"\n📌 **검색 정보**: 후보 {len(scored)}개 중 상위 {len(top)}개 선별\n" +
            f"📊 **점수 범위**: {max(scores):.3f} ~ {min(scores):.3f} (평균 {sum(scores)/len(scores):.3f})"
        )

    except Exception as e:
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
    🔥 영구 DB에 저장하지 않고, 현재 프로세스 메모리(모듈 전역)에만 인덱스 생성
    - PDF/Docx는 로더로 텍스트 추출
    - txt/md는 그대로 디코딩
    - FAISS 인메모리 인덱스 생성 → EPHEMERAL_STORES[file_name]에 저장
    """
    try:
        print(f"📁 문서 업로드(인메모리): {file_name}")

        ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
        texts = []

        if ext == "pdf":
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
        elif ext in ("docx", "doc"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
        else:
            # txt, md 등 일반 텍스트
            with open(file_path, 'rb') as f:
                raw = f.read()
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
            content = raw.decode(enc, errors='replace').replace('\x00', '')
            texts = [content]

        if not texts:
            return f"❌ '{file_name}'에서 텍스트를 추출하지 못했습니다."

        # 기존과 동일한 분할 정책(정확도 ↑)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))

        if not chunks:
            return f"❌ '{file_name}'에서 생성된 청크가 없습니다."

        metadatas = [{"source": file_name, "chunk_id": i} for i in range(len(chunks))]

        # 🔥 인메모리 FAISS 인덱스 생성 (DB에 쓰지 않음)
        vs = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)

        # 전역 저장(세션/프로세스 한정)
        EPHEMERAL_STORES[file_name] = vs

        return f"✅ **'{file_name}' 업로드 완료!** (인메모리 인덱스, 청크 {len(chunks)}개)"

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