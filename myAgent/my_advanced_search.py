import os
from typing import Optional, List, Dict, Any
from hello_agents import ToolRegistry

class MyAdvancedSearchTool:
    """ 自定义搜索工具类, 多源数据搜索和智能结果整合 """
    def __init__(self):
        self.name = "my_advanced_search"
        self.description = "一个高级搜索工具，支持多源数据搜索和智能结果整合。"
        self.search_sources = []
        self._setup_search_sources()

    def _setup_search_sources(self):
        """ 初始化搜索数据源 """
        if os.getenv("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                self.search_sources.append("tavily")
                print("✅ Tavily 数据源已启用。")
            except ImportError:
                print("⚠️ Tavily 库未安装，跳过 Tavily 数据源。")
        if os.getenv("SERPAPI_API_KEY"):
            try:
                import serpapi
                self.search_sources.append("serpapi")
                print("✅ SerpAPI 数据源已启用。")
            except ImportError:
                print("⚠️ SerpAPI 库未安装，跳过 SerpAPI 数据源。")
        if self.search_sources:
            print(f"🔍 可用搜索数据源: {', '.join(self.search_sources)}")
        else:
            print("❌ 未检测到任何可用的搜索数据源。请配置环境变量以启用搜索功能。")

    def search(self, query: str) -> str:
        """ 执行多源搜索并整合结果 """
        if not query.strip():
            return "❌ 请输入有效的搜索查询。"
        if not self.search_sources:
            return "❌ 未配置任何搜索数据源，无法执行搜索。"
        print(f"🔎 执行搜索查询: {query}")
        for source in self.search_sources:
            try:
                if source == "tavily":
                    result = self._search_with_tavily(query)
                    if result and "未找到" not in result:
                        return f"📊 Tavily 搜索结果:\n{result}"
                elif source == "serpapi":
                    result = self._search_with_serpapi(query)
                    if result and "未找到" not in result:
                        return f"🌐 SerpAPI 搜索结果:\n{result}"
            except Exception as e:
                print(f"⚠️ 搜索数据源 {source} 出现错误: {e}")
                continue
        return "❌ 所有搜索数据源均未返回有效结果。"

    def _search_with_tavily(self, query: str) -> str:
        """ 使用 Tavily 进行搜索 """
        response = self.tavily_client.search(query=query, max_results=3)
        if response.get("answer"):
            result = f"💡 Tavily 回答: {response['answer']}\n\n"
        else:
            result = ""
        result += "🔗 相关链接:\n"
        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:150]}...\n\n"
        return result

    def _search_with_serpapi(self, query: str) -> str:
        """ 使用 SerpAPI 进行搜索 """
        import serpapi

        search = serpapi.GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 3
        })

        results = search.get_dict()

        result = "🔗 Google搜索结果：\n"
        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"    {res.get('snippet', '')}\n\n"
        return result


def create_advanced_search_registry():
    """ 创建并返回高级搜索工具注册表 """
    registry = ToolRegistry()
    advanced_search_tool = MyAdvancedSearchTool()
    registry.register_function(
        name="advanced_search",
        description="一个高级搜索工具，支持多源数据搜索和智能结果整合。",
        func=advanced_search_tool.search
    )
    return registry


def test_advanced_search():
    """测试高级搜索工具"""

    registry = create_advanced_search_registry()

    print("🔍 测试高级搜索工具\n")

    test_queries = [
        "Python编程语言的历史",
        "人工智能的最新发展",
        "2025年科技趋势"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = registry.execute_tool("advanced_search", query)
        print(f"结果: {result}\n")
        print("-" * 60 + "\n")

def test_api_configuration():
    """测试API配置检查"""
    print("🔧 测试API配置检查:")

    search_tool = MyAdvancedSearchTool()

    result = search_tool.search("机器学习算法")
    print(f"搜索结果: {result}")

def test_with_agent():
    """测试与Agent的集成"""
    print("\n🤖 与Agent集成测试:")
    print("高级搜索工具已准备就绪，可以与Agent集成使用")

    # 显示工具描述
    registry = create_advanced_search_registry()
    tools_desc = registry.get_tools_description()
    print(f"工具描述:\n{tools_desc}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_advanced_search()
    test_api_configuration()
    test_with_agent()