from dotenv import load_dotenv
load_dotenv()

"""
ContextBuilder 与 Agent 集成
1. 上下文感知的 Agent
2. 自动构建优化的上下文
3. 记忆管理
"""

from hello_agents import SimpleAgent
from myAgent.my_llm import MyLLM
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.core.message import Message
from datetime import datetime

class ContextAwareAgent(SimpleAgent):
    def __init__(self, name: str, llm: MyLLM, **kwargs):
        super().__init__(name=name, llm=llm, **kwargs)
        self.context_builder = ContextBuilder(
            config=ContextConfig(
                max_tokens=3000,
                reserve_ratio=0.2,
                min_relevance=0,
                enable_compression=True
            )
        )
        self.conversation_history = []

    def run(self, user_input: str, **kwargs) -> str:
        optimizer_context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self.system_prompt
        )
        messages = [
            {'role': 'system', 'content': optimizer_context},
            {'role': 'user', 'content': user_input}
        ]
        response = self.llm.invoke(messages)

        self.conversation_history.append(
            Message(content=user_input, role="user", timestamp=datetime.now()),
        )
        self.conversation_history.append(
            Message(content=response, role="assistant", timestamp=datetime.now()),
        )
        return response


def main():
    llm = MyLLM()
    agent = ContextAwareAgent(
        name="ContextAwareAgent",
        llm=llm,
        system_prompt="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理"
    )
    response = agent.run("如何优化Pandas的内存占用?")
    print(f"Agent Response: {response}")
    response = agent.run("请给出一个示例代码")
    print(f"Agent Response: {response}")


if __name__ == "__main__":
    main()


