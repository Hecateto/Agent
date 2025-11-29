import json

from llm import LLM
from dotenv import load_dotenv
from typing import List, Dict
import re

load_dotenv()

PLANNER_SYSTEM_PROMPT = """
你是一个顶级的AI规划专家。
你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

# 要求:
1. 每个步骤必须是独立的、可执行的子任务。
2. 步骤之间必须有严格的逻辑顺序。
3. 不需要输出具体的执行过程，只需列出步骤标题。

# 输出格式:
必须严格输出标准的 JSON 对象，格式如下：
```json
{
    "plan": [
        "步骤1的具体描述",
        "步骤2的具体描述",
        "步骤3的具体描述"
    ]
}
"""

class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def plan(self, question: str) -> List[str]:
        print(f"📋 [Planner] 正在分析问题并生成计划...")

        messages = [
            {'role': 'system', 'content': PLANNER_SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]

        response_text = self.llm.think(messages=messages) or ""

        try:
            clean_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            if clean_text.startswith("["):
                plan_list = json.loads(clean_text)
            else:
                data = json.loads(clean_text)
                plan_list = data.get("plan", [])
            if not isinstance(plan_list, list):
                raise ValueError("计划不是一个列表")
            print(f"✅ [Planner] 计划生成成功，共 {len(plan_list)} 步。")
            return plan_list
        except Exception as e:
            print(f"❌ [Planner] 计划解析失败: {e}")
            print(f"原始响应: {response_text}")
            return []

EXECUTOR_SYSTEM_PROMPT = """ 
你是一位执行专家。你的任务是根据给定的计划步骤，结合已有的历史信息，计算或推理出当前步骤的结果。

# 规则:
1. 专注于解决“当前步骤”。
2. 必须参考“历史步骤与结果”中的数据，不要重复计算已经得出的结论。
3. 输出必须简洁明了，直接给出当前步骤的结论或数值。 
"""

USER_SYSTEM_PROMPT = """
# 原始问题:
{question}

# 历史步骤与结果:
{history}

# 当前需要执行的步骤:
{step}

请给出该步骤的执行结果:
"""

class Executor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def execute_step(self, step: str, question: str, history: str, step_idx: int, total_steps: int) -> str:
        print(f"\n👉 [Executor] 执行步骤 {step_idx}/{total_steps}: {step}")

        history = history if history else "（无历史记录，这是第一步）"
        USER_PROMPT = USER_SYSTEM_PROMPT.format(
            question=question,
            history=history,
            step=step
        )

        messages = [
            {'role': 'system', 'content': EXECUTOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT}
        ]

        result = self.llm.think(messages=messages) or ""

        print(f"💡 [Result]: {result}")
        return result

class PlanAndSolveAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(self, question: str):
        print(f"\n{'=' * 40}\n🤖 开始处理任务: {question}\n{'=' * 40}")
        plan = self.planner.plan(question)
        if not plan:
            print("❌ 无法生成有效的计划，任务终止。")
            return

        history = ""
        final_answer = ""

        for idx, step in enumerate(plan, start=1):
            step_result = self.executor.execute_step(
                question=question,
                step=step,
                history=history,
                step_idx=idx,
                total_steps=len(plan)
            )
            history += f"步骤 {idx}: {step}\n结果: {step_result}\n\n"
            final_answer = step_result
        print(f"\n🎉 任务完成！最终答案: {final_answer}\n{'=' * 40}")


if __name__ == "__main__":
    llm = LLM()
    agent = PlanAndSolveAgent(llm)
    # math_question = "一个水池有两个进水管和一个出水管。第一个进水管单独打开，4小时可以注满水池；第二个进水管单独打开，6小时可以注满水池；出水管单独打开，3小时可以排空水池。如果三个管道同时打开，水池需要多少小时才能注满？"
    math_question = "给出0/1背包问题的定义和一个动态规划求解该问题的Python代码示例。在生成代码后尝试从时空复杂度角度优化该代码。"
    agent.run(math_question)