import os
import inspect
import json
from symbol import parameters
from typing import Dict, Any, Callable, Optional, List
from serpapi import SerpApiClient
from dotenv import load_dotenv
load_dotenv()


def search(query: str, gl: str = "cn", hl: str = "zh-cn") -> str:
    """
    使用 Google 搜索查询实时信息。

    Args:
        query: 搜索关键词
        gl: 地理位置 (默认 'cn' 中国)
        hl: 语言 (默认 'zh-cn' 简体中文)
    """
    try:
        api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError("SERPAPI_API_KEY environment variable not set.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": gl,
            "hl": hl,
            "num": 5
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # error 处理
        if "error" in results:
            return f"Search Error: {results['error']}"

        # 1. 优先返回 Answer Box (直接答案)
        if "answer_box" in results:
            box = results["answer_box"]
            if "answer" in box: return f"直接答案: {box['answer']}"
            if "snippet" in box: return f"精选摘要: {box['snippet']}"
            if "snippet_highlighted_words" in box: return f"重点: {box['snippet_highlighted_words']}"

        # 2. 其次返回 Knowledge Graph (知识图谱)
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            title = kg.get("title", "")
            desc = kg.get("description", "")
            if desc: return f"知识卡片 ({title}): {desc}"

        # 3. 最后返回 Organic Results (自然搜索结果)
        if "organic_results" in results:
            snippets = []
            for i, res in enumerate(results["organic_results"][:3]):
                title = res.get("title", "No Title")
                snippet = res.get("snippet", "No Content")
                link = res.get("link", "")
                snippets.append(f"[{i + 1}] {title}\n    摘要: {snippet}\n    来源: {link}")

            if snippets:
                return "搜索结果:\n" + "\n\n".join(snippets)

        return f"未找到关于 '{query}' 的相关信息。"

    except Exception as e:
        return f"An error occurred during the search: {str(e)}"

from typing import Dict, Any

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, func: Callable, name: str=None, description: str=None):

        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "No description provided."

        sig = inspect.signature(func)
        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = 'string'
            if param.annotation == int: param_type = 'integer'
            if param.annotation == float: param_type = 'number'
            if param.annotation == bool: param_type = 'boolean'

            parameters[param_name] = {"type": param_type, "description": f"f Parameter {param_name}"}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            'name': tool_name,
            'description': tool_desc.strip(),
            'parameters': {
                'type': 'object',
                'properties': parameters,
                'required': required
            }
        }

        self.tools[tool_name] = {
            'func': func,
            'schema': schema
        }

        print(f"✅ 工具 '{tool_name}' 已注册")


    def execute(self, tool_name: str, **kwargs) -> str:
        """
        统一执行入口，负责参数分发和异常捕获
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered.")
        func = self.tools[tool_name]['func']
        try:
            sig = inspect.signature(func)
            valid_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }
            return func(**valid_kwargs)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


    def get_tool_prompt(self) -> str:
        """
        生成给 LLM 看的工具描述 Prompt
        """
        prompt_lines = ['# 可用工具库:']
        for name, info in self.tools.items():
            schema = info['schema']
            params_desc = ", ".join([f"{k}" for k in schema['parameters']['properties'].keys()])
            prompt_lines.append(f"- {name}({params_desc}): {schema['description']}")
        prompt_lines.append("- finish(answer): 当你收集到足够信息可以回答用户问题时，调用此工具提交最终答案。")
        return "\n".join(prompt_lines)


if __name__ == "__main__":
    executor = ToolExecutor()

    executor.registerTool(search)

    print("\n--- System Prompt ---")
    print(executor.get_tool_prompt())

    # Example usage
    print("\n--- 模拟执行 ---")

    # 场景 1: 正常搜索
    llm_action_name = "search"
    llm_action_args = {"query": "NVIDIA H100 GPU specs"}

    result = executor.execute(llm_action_name, **llm_action_args)
    print(f"Observation:\n{result}")

    # 场景 2: 尝试调用不存在的参数 (测试鲁棒性)
    print("\n--- 鲁棒性测试 (多余参数) ---")
    llm_bad_args = {"query": "Python", "fake_param": "ignore_me"}
    # executor 会自动过滤 fake_param，防止报错
    result_safe = executor.execute("search", **llm_bad_args)
    print(f"Observation (Safe):\n{result_safe[:100]}...")