"""
AutoGen 软件开发团队协作案例 (Refactored)
基于 Microsoft AutoGen v0.4+ 架构
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient

logging.basicConfig(level=logging.WARNING)

load_dotenv()

SYSTEM_PROMPTS = {
"ProductManager": """你是一位经验丰富的产品经理 (PM)。
职责：
1. 分析用户需求，拆解为具体的功能点。
2. 制定验收标准 (Acceptance Criteria)。
3. 协调开发进度。

工作流：
- 收到需求后，输出一份《需求规格说明书》。
- 明确要求工程师开始开发。
- 只有在代码审查通过且符合需求后，才询问用户是否满意。
""",

    "Engineer": """你是一位资深的 Python 全栈工程师。
职责：
1. 基于 PM 的需求编写高质量代码。
2. 擅长 Streamlit, Python, Pandas, API 集成。
3. 代码必须包含完整的注释和错误处理。

工作流：
- 编写完整的、可运行的 Python 代码块。
- 代码完成后，明确呼叫代码审查员 (CodeReviewer) 进行检查。
- 如果审查未通过，根据反馈修复代码。
""",

    "CodeReviewer": """你是一位严格的代码审查专家。
职责：
1. 检查代码的安全性、效率和规范 (PEP 8)。
2. 确保没有明显的 Bug 或硬编码的敏感信息。
3. 检查是否满足 PM 定义的需求。

工作流：
- 如果发现问题，列出具体修改建议，让工程师重写。
- 如果代码完美，回复："代码审查通过，请 UserProxy 进行验收测试。"
""",

    "UserProxy": """你代表最终用户和测试人员。
职责：
1. 提出原始需求。
2. 在代码审查通过后，模拟运行代码并验证功能。
3. 决定任务是否结束。

工作流：
- 如果功能满足需求，必须回复 "TERMINATE" 以结束对话。
- 如果不满足，提出具体的修改意见。
"""
}

class AgentFactory:
    """ 用于创建 AutoGen Agent的工厂类 """
    def __init__(self):
        self.model_client = self._create_model_client()

    @staticmethod
    def _create_model_client() -> ChatCompletionClient:
        model = os.getenv("MODEL")
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        timeout = int(os.getenv("TIMEOUT", 60))

        if not api_key:
            raise ValueError("API_KEY 环境变量未设置")

        print(f"🔌 连接模型: {model} @ {base_url}")

        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=0.7,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": 'qwen',
                "structured_output": True,
                "multiple_system_messages": False,
            }
        )

    def create_assistant(self, name: str) -> AssistantAgent:
        if name not in SYSTEM_PROMPTS:
            raise ValueError(f"未定义的角色: {name}")

        return AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_prompt=SYSTEM_PROMPTS[name]
        )

    def create_user_proxy(self) -> UserProxyAgent:
        """ 创建 UserProxyAgent """
        return UserProxyAgent(
            name="UserProxy",
            description="用户代理，负责验收和终止对话。"
        )

async def run_software_team(task: str):
    """ 运行软件开发团队协作流程 """
    print("\n🔧 初始化智能体工厂...")
    factory = AgentFactory()

    print("👥 组建开发团队...")
    pm = factory.create_assistant("ProductManager")
    engineer = factory.create_assistant("Engineer")
    reviewer = factory.create_assistant("CodeReviewer")
    user_proxy = factory.create_user_proxy()

    termination = TextMentionTermination(mention="TERMINATE")
    team = RoundRobinGroupChat(
        participants=[pm, engineer, reviewer, user_proxy],
        termination_conditions=[termination],
        max_turns=11
    )

    print(f"🚀 任务启动: {task[:50]}...")
    print("=" * 60)

    stream = team.run_stream(task=task)
    await Console(stream)

    print("\n" + "=" * 60)
    print("✅ 协作流程结束")

if __name__ == "__main__":
    """ 定义开发任务 """
    DEV_TASK = """
    我们需要一个比特币价格监控面板。
    技术栈：Streamlit
    功能要求：
    1. 显示 BTC/USD 实时价格。
    2. 简单美观的 UI，包含刷新按钮。
    3. 必须处理网络请求失败的情况。
    
    请按 PM -> Engineer -> Reviewer -> User 的顺序协作，直到代码完美并通过验收。
    """

    try:
        asyncio.run(run_software_team(task=DEV_TASK))
    except KeyboardInterrupt:
        print("\n🚪 程序被用户中断，退出中...")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")