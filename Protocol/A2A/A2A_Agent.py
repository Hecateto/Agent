"""
A2A Protocol + Agent
"""
from dotenv import load_dotenv
load_dotenv()
from hello_agents.protocols import A2AClient, A2AServer
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import Tool, ToolParameter
import time, threading

# Technical Expert Agent
tech_expert = A2AServer("tech_expert", "技术专家", version='1.0')
@tech_expert.skill("answer")
def answer_tech_question(text: str) -> str:
    import re
    match = re.search(r'answer\s+(.+)', text, re.IGNORECASE)
    question = match.group(1).strip() if match else text
    answer = f"技术回答：关于'{question}'，你可以参考以下技术文档..."
    return answer

# Sales Advisor Agent
sales_advisor = A2AServer("sales_advisor", "销售顾问", version='1.0')
@sales_advisor.skill("answer")
def answer_sales_question(text: str) -> str:
    import re
    match = re.search(r'answer\s+(.+)', text, re.IGNORECASE)
    question = match.group(1).strip() if match else text
    answer = f"销售建议：关于'{question}'，我们目前有以下优惠活动..."
    return answer

# A2A Agent Service
print("="*60)
print("🚀 启动专业 Agent 服务")
print("="*60)
threading.Thread(target=lambda: tech_expert.run(port=6000), daemon=True).start()
threading.Thread(target=lambda: sales_advisor.run(port=6001), daemon=True).start()
time.sleep(2)

print("✓ 技术专家 Agent 启动在 http://localhost:6000")
print("✓ 销售顾问 Agent 启动在 http://localhost:6001")

print("\n⏳ 等待服务启动...")
time.sleep(2)

# A2A Tool
# class A2ATool(Tool):
#     def __init__(self, name: str, description: str, agent_url: str, skill_name: str = "answer"):
#         self.agent_url = agent_url
#         self.skill_name = skill_name
#         self.client = A2AClient(agent_url)
#         self._name = name
#         self._description = description
#         self._parameters = [
#             ToolParameter(
#                 name="question",
#                 description="The question to ask the agent.",
#                 type="string",
#                 required=True
#             )
#         ]
#
#     @property
#     def name(self) -> str:
#         return self._name
#
#     @property
#     def description(self) -> str:
#         return self._description
#
#     @property
#     def get_parameters(self) -> list[ToolParameter]:
#         return self._parameters
#
#     def run(self, **kwargs) -> str:
#         question = kwargs.get("question", "")
#         response = self.client.execute_skill(self.skill_name, f"answer {question}")
#         if response.get('status') == 'success':
#             return response.get('result', 'No response')
#         else:
#             return f"Error: {response.get('error', 'Unknown error')}"

from hello_agents.tools import A2ATool

tech_tool = A2ATool(
    name="tech_expert",
    description="技术专家，回答技术相关问题",
    agent_url="http://localhost:6000"
)
sales_tool = A2ATool(
    name="sales_advisor",
    description="销售顾问，提供销售相关建议",
    agent_url="http://localhost:6001"
)

print("\n" + "="*60)
print("🤖 创建接待员 SimpleAgent")
print("="*60)

llm = HelloAgentsLLM()
receptionist = SimpleAgent(
    name="接待员",
    llm=llm,
    system_prompt="""
    你是客服接待员，负责：
    1. 分析客户问题类型（技术问题 or 销售问题）
    2. 使用合适的工具（例如 tech_expert 或 sales_advisor）获取答案
    3. 整理答案并返回给客户
    
    请保持礼貌和专业。
    """
)

receptionist.add_tool(tech_tool)
receptionist.add_tool(sales_tool)

print("✓ 接待员 Agent 创建完成")
print(f"✓ 已集成 A2A 工具: {receptionist.tool_registry.list_tools()}")

def handle_customer_query(question: str) -> str:
    print(f"\n客户问题: {question}")
    print("=" * 50)
    response = receptionist.run(question)
    print(f"\n客服回复: {response}")
    print("=" * 50)

# 测试不同类型的问题
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 处理客户咨询")
    print("=" * 60)
    handle_customer_query("你们的API如何调用？")
    handle_customer_query("企业版的价格是多少？")
    handle_customer_query("如何集成到我的Python项目中？")