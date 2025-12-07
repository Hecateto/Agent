from typing import Optional, Dict
from hello_agents import HelloAgentsLLM, Config, SimpleAgent

MY_REFLECTION_PROMPTS = {
    "initial": """
    你是一个智能助手。请根据以下要求完成任务。

    任务: {task}

    请提供一个完整、准确的回答。
    """,

    "reflect": """
    你是一个务实的审核员。请仔细审查以下回答的质量，找出错误、逻辑漏洞或可以改进的地方。

    # 原始任务:
    {task}

    # 待审查的回答:
    {content}
    
    **审查标准：**
    1. **准确性**：内容是否有事实错误？
    2. **完整性**：是否回答了用户的核心问题？
    3. **不要吹毛求疵**：适度优化个人写作风格、修辞优美度或“可以写得更好”这类主观建议。
    4. **收敛原则**：如果回答已经清晰、准确且通过了基本验收，请直接放行。
    
    请先列出必须修改的**硬伤**（如果有）。
    
    **【最终判定】**
    - 如果存在事实错误或严重遗漏，请在最后一行输出：[需要改进]
    - 如果内容准确且逻辑通顺，请在最后一行输出：[无需改进]
    """,

    "refine": """
    你是一个专业的编辑。请根据反馈意见，重新编写并优化回答。

    # 原始任务:
    {task}

    # 上一轮的回答:
    {last_attempt}

    # 批评意见:
    {feedback}

    请输出改进后的最终回答（只输出回答内容，不要包含其他解释）。
    """
}

class MyReflectionAgent(SimpleAgent):
    """
    基于反思机制的 Agent，能够通过多轮反思和改进来提升回答质量。
    """
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            config: Optional[Config] = None,
            system_prompt: Optional[str] = None,
            prompts: Optional[Dict[str, str]] = None,
            max_reflections: int = 5
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_reflections = max_reflections
        self.prompts = prompts if prompts else MY_REFLECTION_PROMPTS

    def run(self, input_text: str, **kwargs) -> str:

        print(f"🤖 {self.name} 正在处理: {input_text}")

        # Initial answer
        print("\n📝 生成初始回答...")
        messages = [{'role': 'user', 'content': self.prompts["initial"].format(task=input_text)}]
        current_answer = self.llm.invoke(messages, **kwargs).strip()
        print(f"✅ 初始版本:\n{current_answer[:100]}... (略)")

        # Reflection loop
        for i in range(self.max_reflections):
            print(f"\n--- 🔄 反思轮次 {i + 1}/{self.max_reflections} ---")

            # Reflect
            reflect_msg = [{
                'role': 'user',
                'content': self.prompts['reflect'].format(
                    task=input_text,
                    content=current_answer
                )
            }]

            feedback = self.llm.invoke(reflect_msg, **kwargs).strip()
            print(f"\n反馈意见:\n{feedback}")

            # Check for completion
            if "无需改进" in feedback:
                print("✨ 回答已达到要求，停止反思。")
                break

            # Refine
            refine_msg = [{
                'role': 'user',
                'content': self.prompts['refine'].format(
                    task=input_text,
                    last_attempt=current_answer,
                    feedback=feedback
                )
            }]
            current_answer = self.llm.invoke(refine_msg, **kwargs).strip()
            print(f"✅ 改进版本: {current_answer[:100]}... (略)")

        return current_answer


def test_reflection():
    from my_llm import MyLLM
    from dotenv import load_dotenv
    load_dotenv()

    llm = MyLLM()

    # 测试1: 默认提示词
    print("=== 测试1: 通用反思助手 ===")
    general_agent = MyReflectionAgent(
        name="通用反思助手",
        llm=llm
    )
    result = general_agent.run("写一段话简单介绍人工智能")
    print(f"\n🏆 最终结果:\n{result}\n")

    # 测试2: 自定义提示词
    print("=== 测试2: 代码生成反思助手 ===")
    code_prompts = {
        "initial": "你是Python专家。请编写函数: {task}。只输出代码。",
        "reflect": "请审查代码效率:\n任务:{task}\n代码:\n{content}\n如果完美回复'无需改进'。",
        "refine": "根据反馈重写:\n任务:{task}\n反馈:{feedback}\n旧代码:{last_attempt}"
    }

    code_agent = MyReflectionAgent(
        name="代码专家",
        llm=llm,
        prompts=code_prompts,
        max_reflections=2
    )
    code_agent.run("写一个斐波那契数列函数")


if __name__ == "__main__":
    test_reflection()