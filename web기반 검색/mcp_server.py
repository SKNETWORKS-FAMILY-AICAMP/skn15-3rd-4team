# mcp_server.py
import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict
from mcp.server.fastmcp import FastMCP

# -------------------------------
# 환경 변수 불러오기
# -------------------------------
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# -------------------------------
# FastMCP 서버 초기화
# -------------------------------
mcp = FastMCP("web_search")

# -------------------------------
# Tavily 문서 검색 툴
# -------------------------------
@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> List[Dict]:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False
    }
    res = requests.post(url, json=payload)
    res.raise_for_status()
    data = res.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content")
        })
    return results

# -------------------------------
# Google 이미지 검색 툴
# -------------------------------
@mcp.tool()
def google_image_search(query: str, num: int = 5) -> List[Dict]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": num
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()

    items = []
    for it in data.get("items", []):
        items.append({
            "title": it.get("title"),
            "image_url": it.get("link"),
            "page_url": it.get("image", {}).get("contextLink")
        })
    return items

# -------------------------------
# FastMCP 서버 실행
# -------------------------------
if __name__ == "__main__":
    print("🚀 MCP 서버 실행 중...")
    mcp.run(transport="stdio")
