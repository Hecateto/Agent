"""
智能搜索助手
LangGraph + Tavily API
"""

import asyncio
import json
import os
import datetime
from typing import TypedDict, Annotated, List, Optional, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

def get_current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d %A")

class QueryAnalysis(BaseModel):
    """用户查询分析结果"""
    summary: str = Field(description="用户查询的简要总结")
    search_query: str = Field(description="优化后的搜索查询词(Query)")
    needs_search: bool = Field(description="是否需要联网搜索以获取答案", default=True)
    is_exit: bool = Field(
        description="用户是否表达了结束对话、再见或离开的意图",
        default=False
    )

class SearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    analysis: Optional[QueryAnalysis]
    search_context: str
    step: str

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.5,
)

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY 环境变量未设置")
tavily_client = TavilyClient(api_key=tavily_api_key)

# Prompts
# ===================================================

PROMPT_ANALYZE = """
你是一个搜索意图分析专家。
当前时间是：{current_date}
请分析用户的最新输入。

要求：
1. 阅读完整的对话历史。
2. 如果用户是在闲聊，needs_search 设为 false。
3. 如果用户在询问事实、新闻、知识，needs_search 设为 true，并生成最好的中文搜索关键词。
4. 如果用户表达了结束、告别、停止对话的意图，请将 is_exit 设为 true。
5. 请务必返回合法的 JSON 格式，不要包含 Markdown 代码块（如 ```json）。

JSON 格式示例：
{{
    "summary": "分析摘要",
    "search_query": "...",
    "needs_search": true/false,
    "is_exit": true/false
}}
"""

PROMPT_ANSWER = """
你是一个智能知识助手。
当前时间是：{current_date}

请基于以下提供的【搜索结果上下文】来回答用户的【问题】。
用户问题: {user_query}

【搜索结果上下文】:
{search_context}

要求：
1. **准确性**：严格基于搜索结果回答，不要编造信息。
2. **引用**：在回答中适当引用来源（例如 [1], [2]）。
3. **结构**：如果内容较多，使用要点符号列表。
4. **补充**：如果搜索结果无法完全回答问题，请诚实说明。
5. **兜底**：如果搜索结果为空，请基于你的通用知识尝试回答，并告知用户这是基于通用知识。
"""

# Nodes
# ===================================================
async def analyze_node(state: SearchState) -> dict:
    """分析用户意图 """
    messages = state["messages"]
    system_prompt = PROMPT_ANALYZE.format(current_date=get_current_date())

    try:
        conversation = [SystemMessage(content=system_prompt)] + messages
        response = await llm.ainvoke(conversation)
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        analysis = QueryAnalysis(**data)
    except Exception as e:
        print(f"⚠️ JSON 解析失败，回退到默认搜索模式: {e}")
        analysis = QueryAnalysis(
            summary="解析失败，默认搜索",
            search_query=messages[-1].content,
            needs_search=True
        )
    return {
        "analysis": analysis,
    }

def _format_search_results(tavily_response:dict) -> str:
    """格式化 Tavily 返回的 JSON"""
    results = []
    if tavily_response.get("answer"):
        results.append(f"--- 智能摘要 ---\n{tavily_response['answer']}\n")

    raw_results = tavily_response.get("results", [])
    if raw_results:
        results.append("--- 详细来源 ---")
        for i, res in enumerate(raw_results[:5], 1):
            title = res.get("title", "无标题")
            content = res.get("content", "无内容")
            url = res.get("url", "#")
            results.append(f"[{i}] {title}\n摘要: {content}\n链接: {url}\n")
    return "\n".join(results) if results else "无搜索结果。"

async def search_node(state: SearchState) -> dict:
    """执行搜索"""
    analysis = state["analysis"]
    if not analysis or not analysis.needs_search:
        return {
            "search_context": "无需搜索。",
            "step": "skip_search"
        }

    query = analysis.search_query
    print(f"🔍 智能搜索: {query}")

    try:
        response = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            include_answer=True,
            max_results=3
        )
        context = _format_search_results(response)
    except Exception as e:
        print(f"执行搜索时出错: {e}")
        context = "搜索失败，无法获取结果。"

    return {
        "search_context": context,
    }

async def answer_node(state: SearchState) -> dict:
    """生成最终回答"""
    analysis = state.get("analysis")
    context = state.get("search_context", "")
    messages = state["messages"]

    user_query = messages[-1].content

    if not analysis or not analysis.needs_search:
        response = await llm.ainvoke(messages)
    else:
        system_msg = PROMPT_ANSWER.format(
            current_date=get_current_date(),
            user_query=user_query,
            search_context=context
        )
        final_msg = [SystemMessage(content=system_msg)] + messages
        response = await llm.ainvoke(final_msg)

    return {
        "messages": [response]
    }

# Graph
# ==================================================
def route(state: SearchState) -> Literal["search", "generate"]:
    """根据分析结果决定下一步"""
    analysis = state["analysis"]
    if analysis and analysis.needs_search:
        return "search"
    return "generate"

def create_graph():
    workflow = StateGraph(SearchState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("search", search_node)
    workflow.add_node("generate", answer_node)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges(
        "analyze",
        route
    )
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=InMemorySaver())

async def main():
    app = create_graph()
    print("\n🌐 LangGraph 深度搜索助手已就绪")
    print("-----------------------------------")
    config = {"configurable": {"thread_id": "user-session-1144"}}

    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            if user_input.lower() in ["exit", "quit", "q", "再见"]:
                print("👋 再见！")
                break
            if not user_input:
                continue
            print("🤖 思考中...", end="", flush=True)

            should_exit = False
            async for event in app.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            ):
                for node_name, state_update in event.items():
                    if node_name == "analyze":
                        analysis = state_update.get("analysis")
                        if analysis.is_exit:
                            should_exit = True
                        elif analysis.needs_search:
                            print(f"\n   ↳ 🎯 意图识别: {analysis.summary}")
                            print(f"   ↳ 🔑 搜索关键词: {analysis.search_query}")
                    elif node_name == "search":
                        context = state_update.get("search_context", "")
                        if context:
                            print(f"   ↳ 📚 检索到资料 (长度: {len(context)} 字符)")
                    elif node_name == "generate":
                        last_msg = state_update["messages"][-1]
                        print(f"\n🤖 回答: {last_msg.content}")
                        print("-"*40)
            if should_exit:
                print("👋 智能助手下线。")
                break
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❗ 出错了: {e}")

if __name__ == "__main__":
    asyncio.run(main())