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

# 🔥 개선: 세션별 문서 저장소 관리
class DocumentManager:
    """문서 업로드와 검색을 관리하는 클래스"""
    
    def __init__(self):
        self.stores = {}  # session_id -> FAISS store 매핑
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    def upload_document(self, file_path: str, file_name: str, session_id: str = "default") -> str:
        """문서를 업로드하고 FAISS 인덱스를 생성"""
        try:
            print(f"📁 문서 업로드(세션: {session_id}): {file_name}")

            ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
            texts = []

            # 파일 형식별 텍스트 추출
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
                enc = chardet.detect(raw)['encoding'] or 'utf-8'
                content = raw.decode(enc, errors='replace').replace('\x00', '')
                texts = [content] if content.strip() else []

            if not texts:
                return f"❌ '{file_name}'에서 텍스트를 추출하지 못했습니다."

            # 텍스트 분할
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

            # FAISS 인덱스 생성
            vs = FAISS.from_texts(texts=chunks, embedding=self.embeddings, metadatas=metadatas)

            # 세션별 저장소에 저장
            if session_id not in self.stores:
                self.stores[session_id] = {}
            
            self.stores[session_id][file_name] = vs
            
            return f"✅ **'{file_name}' 업로드 완료!** (세션: {session_id}, 청크: {len(chunks)}개)"

        except Exception as e:
            return f"문서 업로드 중 오류가 발생했습니다: {e}"

    def search_documents(self, query: str, session_id: str = "default") -> str:
        """세션별로 업로드된 문서에서 검색"""
        try:
            print(f"📄 문서 검색 (세션: {session_id}): '{query}'")

            if session_id not in self.stores or not self.stores[session_id]:
                return "업로드된 문서가 없습니다. 먼저 문서를 업로드해주세요."

            # 각 파일 인덱스에서 검색
            k_per_file = 5
            gathered = []
            
            for fname, store in self.stores[session_id].items():
                try:
                    docs = store.as_retriever(search_kwargs={"k": k_per_file}).get_relevant_documents(query)
                    gathered.extend(docs)
                except Exception as e:
                    print(f"   ❌ '{fname}' 검색 실패: {e}")

            if not gathered:
                return f"'{query}'에 대한 관련 내용을 찾지 못했습니다."

            # 점수 계산 (코사인 유사도)
            q_emb = self.embeddings.embed_query(query)
            scored = []
            
            for i, d in enumerate(gathered):
                try:
                    d_emb = self.embeddings.embed_query(d.page_content)
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

            # 점수 기준으로 정렬
            scored.sort(key=lambda x: x['score'], reverse=True)

            # 임계값 적용
            PRIMARY = 0.3
            FALLBACK = 0.1
            picked = [x for x in scored if x['score'] >= PRIMARY] or \
                     [x for x in scored if x['score'] >= FALLBACK] or \
                     scored[:5]

            # 중복 제거
            unique = self._remove_duplicates(picked)
            top = unique[:5]
            
            if not top:
                return f"'{query}'와 관련된 정보를 찾지 못했습니다."

            # 결과 포맷팅
            parts = []
            scores = [x['score'] for x in scored]
            
            for i, x in enumerate(top, 1):
                level = "높음" if x['score'] >= 0.5 else "중간" if x['score'] >= 0.3 else "낮음"
                content = x['document'].page_content.strip()
                parts.append(
                    f"📋 **문서 {i}** (관련성: {level}) - {x['source']} (청크 {x['chunk_id']})\n{content}\n"
                )

            return (
                f"🔍 **'{query}' 검색 결과** ({len(parts)}개 관련 문서)\n\n" +
                "\n".join(parts) +
                f"\n📌 **검색 정보**: 후보 {len(scored)}개 중 상위 {len(top)}개 선별\n" +
                f"📊 **점수 범위**: {max(scores):.3f} ~ {min(scores):.3f} (평균 {sum(scores)/len(scores):.3f})"
            )

        except Exception as e:
            return f"문서 검색 중 오류가 발생했습니다: {e}"

    def _remove_duplicates(self, scored_docs: list) -> list:
        """중복 문서 제거"""
        if len(scored_docs) <= 1:
            return scored_docs
        
        unique_docs = []
        seen_contents = set()
        
        for doc_info in scored_docs:
            content = doc_info['document'].page_content
            content_key = content[:100].strip().lower()
            
            if content_key not in seen_contents:
                unique_docs.append(doc_info)
                seen_contents.add(content_key)
        
        return unique_docs

    def get_document_status(self, session_id: str = "default") -> str:
        """업로드된 문서 현황 확인"""
        try:
            if session_id not in self.stores or not self.stores[session_id]:
                return "❌ 업로드된 문서가 없습니다."
            
            total_chunks = 0
            status_msg = f"📊 **업로드된 문서 현황** (세션: {session_id})\n\n"
            
            for fname, store in self.stores[session_id].items():
                # 각 파일의 청크 수 계산 (임시로 1개 문서 검색해서 전체 인덱스 크기 추정)
                try:
                    sample_docs = store.as_retriever(search_kwargs={"k": 100}).get_relevant_documents("*")
                    chunk_count = len(sample_docs)
                    total_chunks += chunk_count
                    status_msg += f"📄 {fname}: {chunk_count}개 청크\n"
                    
                    # 샘플 미리보기
                    if sample_docs:
                        preview = sample_docs[0].page_content[:50].replace('\n', ' ')
                        status_msg += f"   → 샘플: {preview}...\n"
                        
                except Exception as e:
                    status_msg += f"📄 {fname}: 오류 ({e})\n"
            
            status_msg += f"\n**총 청크 수**: {total_chunks}개"
            return status_msg
            
        except Exception as e:
            return f"❌ 문서 현황 확인 중 오류: {e}"

# 🔥 전역 문서 매니저 인스턴스
doc_manager = DocumentManager()

# 기존 Anki 관련 DB는 유지 (이전 대화 검색용)
VECTORSTORE_DSN = "postgresql+psycopg2://play:123@localhost:5432/play"
ANKI_COLLECTION_NAME = "play_anki_cards"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
anki_vector_store = PGVector(
    connection=VECTORSTORE_DSN,
    embeddings=embeddings,
    collection_name=ANKI_COLLECTION_NAME,
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

def document_search(query: str, session_id: str = "default") -> str:
    """문서 검색 - 개선된 DocumentManager 사용"""
    return doc_manager.search_documents(query, session_id)

def upload_document(file_path: str, file_name: str, session_id: str = "default") -> str:
    """문서 업로드 - 개선된 DocumentManager 사용"""
    return doc_manager.upload_document(file_path, file_name, session_id)

def check_document_status(session_id: str = "default") -> str:
    """문서 현황 확인 - 개선된 DocumentManager 사용"""
    return doc_manager.get_document_status(session_id)

def anki_card_saver(front: str, back: str, deck: str = "기본", tags: list = None) -> str:
    """
    AnkiConnect API를 사용하여 Anki 데스크톱 프로그램에 새 카드를 추가합니다.
    """
    print(f"🃏 Anki 카드 저장 시도: {front[:30]}...")
    
    def anki_request(action, **params):
        return {'action': action, 'version': 6, 'params': params}

    try:
        # 1. 중복 카드 확인
        front_keywords = front.replace('\n', ' ')[:50]
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
                    'Back': back.replace("\n", "<br>")
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
        
        # 🔥 개선: Anki 카드를 벡터DB에도 저장 (이전 대화 검색 가능하도록)
        try:
            anki_vector_store.add_texts(
                texts=[f"Anki 카드: {front}\n답변: {back}"], 
                metadatas=[{"type": "anki_card", "deck": deck, "anki_id": note_id}]
            )
        except Exception as db_error:
            print(f"   ⚠️ 벡터DB 저장 실패: {db_error}")
        
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