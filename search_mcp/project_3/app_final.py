import streamlit as st
from tools.tavily_tool import TavilySearchTool
from tools.google_tool import GoogleImageSearchTool
from tools.mcp_tool import MCPTool

# Tool 인스턴스 생성
tavily_tool = MCPTool(TavilySearchTool())
google_tool = MCPTool(GoogleImageSearchTool())

st.title("🌐 웹 기반 MCP 흉내 챗봇")

query = st.text_input("검색어를 입력하세요:")

if query:
    st.subheader("🔍 문서 검색 결과 (Tavily)")
    docs = tavily_tool.run(query)
    for d in docs:
        st.markdown(f"**[{d['title']}]({d['url']})**")
        snippet = (d['content'] or "")[:200] + "..."
        st.write(snippet)

    st.subheader("🖼 이미지 검색 결과 (Google)")
    images = google_tool.run(query)
    cols = st.columns(3)
    for idx, img in enumerate(images):
        with cols[idx % 3]:
            st.image(img["image_url"], use_column_width=True)
            st.caption(img["title"])
