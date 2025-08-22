import asyncio, hashlib, requests
from typing import Optional, List
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP  # ← FastMCP 사용

ANKI_URL = "http://172.22.16.1:8765" # 로컬에서 사용시 http://127.0.0.1:8765

def anki(action, **params): # Anki와 대화하는 함수
    payload = {"action": action, "version": 6, "params": params}
    r = requests.post(ANKI_URL, json=payload, timeout=5)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data["result"]

def short_sha(text: str) -> str: #--- 질문 텍스트를 16자 해시로 요약. 질문 텍스트를 16자 해시로 요약.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

class UpsertInput(BaseModel): #---입력 데이터 틀
    type: str = "basic"     # 확장 대비(지금은 Basic 모델만 사용).
    deck: str = "기본"   # 영어 UI면 "Default"로 바꾸세요
    question: str   # 질문 텍스트.
    answer: str     # 정답 텍스트.
    tags: Optional[List[str]] = []      # 태그(검색/정리용).
    dedupe_key: Optional[str] = None        # 직접 중복키를 넣어줄 수도 있고, 비어 있으면 해시로 자동 생성.

app = FastMCP("anki-mcp-min")       # “anki-mcp-min”이라는 MCP 서버 앱을 하나 띄울 준비.

@app.tool("anki.upsert_note", description="Q/A를 Anki에 카드로 저장")
async def upsert_note(
    type: str = "basic",
    deck: str = "기본",
    question: str = "",
    answer: str = "",
    tags: Optional[List[str]] = None,
    dedupe_key: Optional[str] = None,
) -> dict:
    args = UpsertInput(
        type=type,
        deck=deck,
        question=question,
        answer=answer,
        tags=tags or [],
        dedupe_key=dedupe_key,
    )
    dedupe = args.dedupe_key or short_sha(args.question)

    existing = anki("findNotes", query=f'"{dedupe}"')
    fields = {"Front": args.question, "Back": f"{args.answer}\n\n[{dedupe}]"}
    if not existing:
        res = anki("addNotes", notes=[{
            "deckName": args.deck,
            "modelName": "Basic",
            "fields": fields,
            "tags": (args.tags or []) + [f"dedupe:{dedupe}"]
        }])
        return {"status": "created", "note_id": (res[0] if res else None), "dedupe_key": dedupe}
    else:
        nid = existing[0]
        anki("updateNoteFields", note={"id": nid, "fields": fields})
        try:
            anki("addTags", notes=[nid], tags=" ".join((args.tags or []) + [f"dedupe:{dedupe}"]))
        except Exception:
            pass
        return {"status": "updated", "note_id": nid, "dedupe_key": dedupe}

async def main():       # MCP 1.13.0 환경에서 표준입출력(stdio) 로 서버를 실행하는 한 줄. 클라이언트(call_tool.py)가 이 서버를 스폰해서 통신함.
    await app.run_stdio_async()

if __name__ == "__main__": # 다른 파일에서 임포트할 땐 실행되지 않고, 터미널에서 python mcp_anki_min.py 했을 때만 서버 시작.
    asyncio.run(main())
