from dotenv import load_dotenv
load_dotenv()

"""
ContextBuilder 基础使用示例
"""

from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.core.message import Message
from datetime import datetime

def main():

    # 创建 ContextConfig 对象
    print("Creating ContextConfig...")
    config = ContextConfig(
        max_tokens=3000,
        reserve_ratio=0.2,
        min_relevance=0,    # 最小相关性阈值, 0表示不过滤
        enable_compression=True
    )
    builder = ContextBuilder(config=config)

    # 模拟对话历史
    print("Building conversation history...")
    conversation_history = [
        Message(content="我正在开发一个数据分析工具", role="user", timestamp=datetime.now()),
        Message(content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?", role="assistant",
                timestamp=datetime.now()),
        Message(content="我打算使用Python和Pandas,已经完成了CSV读取模块", role="user", timestamp=datetime.now()),
        Message(content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。", role="assistant",
                timestamp=datetime.now()),
    ]


    # 构建上下文
    print("Building context...")
    context = builder.build(
        user_query="如何优化Pandas的内存占用?",
        conversation_history=conversation_history,
        system_instructions="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例"
    )
    print(f"Context built: {context}")

    # 准备消息列表
    messages = [
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': '请回答'}
    ]

    from myAgent.my_llm import MyLLM
    print("Invoking LLM...")
    llm = MyLLM()
    response = llm.invoke(messages)
    print(f"LLM Response: {response}")


if __name__ == "__main__":
    main()

