import os
import requests
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class GoogleImageSearchTool:
    def run(self, query: str):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "searchType": "image",
            "num": 6
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        items = []
        for it in res.json().get("items", []):
            items.append({
                "title": it.get("title"),
                "image_url": it.get("link"),
                "page_url": it.get("image", {}).get("contextLink")
            })
        return items
