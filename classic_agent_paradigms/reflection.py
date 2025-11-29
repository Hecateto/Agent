import re
from typing import List, Dict, Optional, Any
from llm import LLM

GENERATOR_SYSTEM_PROMPT = """
你是一位资深的 Python 程序员。
你的任务是根据用户的要求编写高质量的 Python 代码。

# 编码规范:
1. 包含完整的函数签名和文档字符串 (docstring)。
2. 遵循 PEP 8 编码规范。
3. 代码必须是可运行的，不要使用伪代码。
"""

REFLECTOR_SYSTEM_PROMPT = """
你是一位极其严格的代码评审专家和资深算法工程师。
你的任务是审查代码，找出逻辑错误、性能瓶颈或安全隐患。

# 评审标准:
1. **正确性**: 代码是否能完成任务？
2. **效率**: 时间复杂度和空间复杂度是否最优？
3. **风格**: 是否符合 PEP 8？

# 输出要求:
- 如果代码完美或已达到最优，请仅输出: "无需改进"。
- 否则，请列出具体的改进建议，并简要说明理由。
"""

REFINER_USER_TEMPLATE = """
# 原始任务:
{task}

# 上一轮的代码:
```python
{last_code}
评审员反馈:
{feedback}

请根据以上反馈，生成优化后的新版本代码。 请直接输出代码，不要包含多余的解释。
"""

def clean_code_block(text: str) -> str:
    """ 清洗 LLM 输出，去除 python 和 标记，只保留代码本身。 """
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    return text.strip()

class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(self, role: str, content: str):
        """
        role: 'generator', 'reflector'
        """
        self.records.append({"role": role, "content": content})

    def get_last_code(self) -> Optional[str]:
        for r in reversed(self.records):
            if r['role'] == 'generator':
                return r['content']
        return None

class ReflectionAgent:
    def __init__(self, llm: LLM, max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations
        self.memory = Memory()

    def run(self, task: str):
        print(f"\n{'=' * 40}\n🤖 开始 Reflection 任务: {task}\n{'=' * 40}")
        initial_code = self._generate_initial_code(task)
        self.memory.add('generator', initial_code)
        print(f"✅ 初始代码生成完毕。")

        for i in range(self.max_iterations):
            print(f"\n--- 🔄 第 {i + 1}/{self.max_iterations} 轮优化 ---")

            last_code = self.memory.get_last_code()
            feedback = self._reflect(task, last_code)
            self.memory.add('reflector', feedback)

            preview_feedback = feedback.replace('\n', ' ')[:100]
            print(f"🧐 [Reflector] 反馈: {preview_feedback}...")

            if self._is_perfect(feedback):
                print(f"\n🎉 代码已达最优，流程结束。")
                break

            print(f"🛠️ [Generator] 正在根据反馈优化代码...")
            refined_code = self._refine(task, last_code, feedback)
            self.memory.add('generator', refined_code)

        final_code = self.memory.get_last_code()
        print(f"\n{'=' * 40}\n📦 最终交付代码:\n{'=' * 40}\n{final_code}\n{'=' * 40}")
        return final_code

    def _generate_initial_code(self, task: str) -> str:
        messages = [
            {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': f"任务: {task}\n请直接输出代码。"}
        ]
        response = self.llm.think(messages=messages)
        return clean_code_block(response)

    def _reflect(self, task: str, code: str) -> str:
        user_msg = f"任务: {task}\n\n待审查代码:\n```python\n{code}\n```"
        messages = [
            {'role': 'system', 'content': REFLECTOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_msg}
        ]
        return self.llm.think(messages=messages)

    def _refine(self, task: str, last_code: str, feedback: str) -> str:
        user_msg = REFINER_USER_TEMPLATE.format(task=task, last_code=last_code, feedback=feedback)
        messages = [
            {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_msg}
        ]
        response = self.llm.think(messages=messages)
        return clean_code_block(response)

    @staticmethod
    def _is_perfect(feedback: str) -> bool:
        keywords = ["无需改进", "无需修改", "没有改进建议", "代码完美",
                    "no need for improvement", "perfect", "optimal"]
        clean_feedback_start = re.sub(r'[^\w\s]', '', feedback)[:100].strip().lower() # 移除标点符号并转换为小写
        for k in keywords:
            if k in clean_feedback_start:
                return True
        return False

if __name__ == "__main__":
    try:
        my_llm = LLM()
        agent = ReflectionAgent(llm=my_llm, max_iterations=3)
        task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
        agent.run(task)
    except Exception as e:
        print(f"❌ 运行时出错: {e}")
