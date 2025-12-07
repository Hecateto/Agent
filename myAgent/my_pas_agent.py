import ast
from typing import Optional, Dict, List
from hello_agents import HelloAgentsLLM, Config, SimpleAgent
from my_llm import MyLLM

MY_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

MY_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

MY_SUMMARIZER_PROMPT = """ 
你是一个顶级的AI总结专家。你的任务是根据已经执行的步骤和结果，为用户的问题提供最终的、完整的答案。

原始问题:
{question}

执行历史:
{history}

要求:

如果问题是数学或逻辑题，请整合步骤给出清晰的解题过程和最终答案。

如果问题是编程题，请将历史步骤中的代码片段整合为一个完整的、可运行的代码块，并包含必要的注释。

不要输出多余的废话，直接给出最终结果。 
"""

class MyPlanAndSolveAgent(SimpleAgent):
    """
    结合规划和执行能力的 Agent，能够将复杂任务分解为多个步骤并逐步解决。
    """
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            config: Optional[Config] = None,
            planner_prompt: Optional[str] = None,
            executor_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, config)
        self.planner_prompt = planner_prompt if planner_prompt else MY_PLANNER_PROMPT
        self.executor_prompt = executor_prompt if executor_prompt else MY_EXECUTOR_PROMPT
        self.summarizer_prompt = MY_SUMMARIZER_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # Plan
        print("\n📋 正在制定计划...")
        plan = self._make_plan(input_text, **kwargs)
        if not plan:
            return "❌ 制定计划失败，无法继续。"
        print(f"✅ 计划已生成，共 {len(plan)} 个步骤:")
        for i, step in enumerate(plan):
            print(f"  {i + 1}. {step}")

        # Solve
        print("\n🚀 开始执行计划...")
        step_history: List[Dict[str, str]] = []

        for i, step in enumerate(plan):
            print(f"\n👉 正在执行步骤 {i + 1}/{len(plan)}: {step}")
            history_text = self._format_history(step_history)

            prompt = self.executor_prompt.format(
                question=input_text,
                plan=str(plan),
                history=history_text,
                current_step=step
            )

            messages = [{'role': 'user', 'content': prompt}]
            step_result = self.llm.invoke(messages=messages, **kwargs).strip()

            step_result = step_result.replace("```python", "").replace("```", "").strip()
            print(f"💡 步骤结果: {step_result}")

            step_history.append({
                "step": step,
                "result": step_result
            })
        print(f"\n🏁 所有步骤执行完毕, 正在整合最终答案...")

        history_text = self._format_history(step_history)
        summary_prompt = self.summarizer_prompt.format(
            question=input_text,
            history=history_text
        )
        messages = [{'role': 'user', 'content': summary_prompt}]
        final_answer = self.llm.invoke(messages=messages, **kwargs).strip()

        return final_answer

    def _make_plan(self, question: str, **kwargs) -> Optional[List[str]]:
        """ 使用 LLM 制定行动计划，将复杂问题分解为多个步骤。 """
        prompt = self.planner_prompt.format(question=question)
        messages = [{'role': 'user', 'content': prompt}]
        response = self.llm.invoke(messages=messages, **kwargs).strip()
        return self._parse_plan_output(response)

    @staticmethod
    def _parse_plan_output(response: str) -> List[str] | str:
        """ 解析 LLM 输出的计划，提取步骤列表。 """
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start == -1 or end == 0:
                print(f"⚠️ 无法在响应中找到列表格式: {response}")
                return [response]
            list_str = response[start:end]
            plan = ast.literal_eval(list_str)
            if isinstance(plan, list):
                return [str(step) for step in plan]
            else:
                return response
        except Exception as e:
            print(f"⚠️ 解析计划时出错: {e}")
            return [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('```')]

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """ 格式化历史步骤和结果，供 Executor 使用。 """
        if not history:
            return "无(这是第一个步骤)"
        formatted = ""
        for i, item in enumerate(history):
            formatted += f"步骤 {i + 1}: {item['step']}\n结果: {item['result']}\n---\n"
        return formatted

def test_plan_and_solve():
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 50)
    print("测试 Plan-and-Solve Agent")
    print("=" * 50)

    llm = MyLLM()

    agent = MyPlanAndSolveAgent(
        name="P&S助手",
        llm=llm
    )

    # 测试案例 1: 逻辑推理/数学问题
    question1 = "A有5个苹果，B的苹果是A的3倍，C的苹果比A和B的总和少2个。请问他们三个人一共有多少个苹果？"

    result1 = agent.run(question1)
    print(f"\n🏆 最终答案:\n{result1}\n")

    print("-" * 50)

    # 测试案例 2: 代码任务
    question2 = "我想在Python中把一个名为'data.csv'的文件读取出来，删除包含空值的行，然后保存为'clean_data.json'。请给出具体的代码实现步骤，最后生成完整代码。"

    result2 = agent.run(question2)
    print(f"\n🏆 最终答案:\n{result2}")


if __name__ == "__main__":
    test_plan_and_solve()