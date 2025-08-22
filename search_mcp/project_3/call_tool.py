import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def main():
    params = StdioServerParameters(command="python", args=["mcp_anki_min.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # ✅ 초기화가 필수!
            await session.initialize()

            tools = await session.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])

            payload = {
                "type":"basic",
                "deck":"기본",  # 영어 UI면 "Default"
                "question":"Transformer란?",
                "answer":"Self-Attention 기반 구조.",
                "tags":["nlp","demo"]
            }
            res = await session.call_tool("anki.upsert_note", payload)
            print("RESULT:", res.content[0].text)

asyncio.run(main())