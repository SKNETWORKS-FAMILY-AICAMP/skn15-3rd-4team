import os
import requests
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class TavilySearchTool:
    def run(self, query: str):
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 5
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
