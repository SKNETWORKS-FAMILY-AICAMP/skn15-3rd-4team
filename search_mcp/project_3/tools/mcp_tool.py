# MCP Tool 흉내
class MCPTool:
    def __init__(self, tool):
        self.tool = tool

    def run(self, query: str):
        # 실제 Tool 실행
        return self.tool.run(query)
