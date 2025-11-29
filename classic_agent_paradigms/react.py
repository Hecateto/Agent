import json
import re
from typing import List, Dict, Tuple, Optional

from llm import LLM
from tools import ToolExecutor, search

REACT_SYSTEM_PROMPT = """
你是一个智能助手，可以调用外部工具来解决问题。

# 可用工具:
{tools_desc}

# 思考与行动格式:
请严格按照以下 ReAct 格式进行回复（Thought 和 Action 必须交替出现）：

Thought: <你的思考过程，分析当前状态和下一步计划>
Action: ```json
{{
    "name": "工具名称",
    "args": {{ "参数名": "参数值" }}
}}
"""

class ReActAgent:
    def __init__(self, llm: LLM, tool_executor: ToolExecutor, max_steps: int=5):
        self.llm = llm
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.messages: List[Dict[str, str]] = []

    def run(self, question: str):

        tools_desc = self.tool_executor.get_tool_prompt()
        system_content = REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc)

        self.messages.append({'role': 'system', 'content': system_content})
        self.messages.append({'role': 'user', 'content': question})

        cur_step = 0
        print(f"🚀 开始任务: {question}")

        while cur_step < self.max_steps:
            cur_step += 1
            print(f"\n--- 第 {cur_step} 步 ---")

            response_text = self.llm.think(messages=self.messages)
            if not response_text:
                print("❌ 错误：LLM 返回为空，终止流程。")
                break
            self.messages.append({'role': 'assistant', 'content': response_text})

            thought, action_json = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action_json:
                print("⚠️ 警告: 未检测到有效 Action，尝试让 LLM 继续...")
                self.messages.append({"role": "user", "content": "System Error: 请严格遵循 JSON Action 格式输出。"})
                continue

            tool_name, tool_args = action_json.get("name"), action_json.get("args")
            if tool_name == "finish":
                final_answer = tool_args.get("answer", "任务完成 (无具体答案)")
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            print(f"🎬 行动: {tool_name} {tool_args}")

            observation = self.tool_executor.execute(tool_name=tool_name, **tool_args)
            print(f"👀 观察: {observation[:200]}..." if len(observation) > 200 else f"👀 观察: {observation}")
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})

        print("❌ 已达到最大步数，任务失败。")
        return None

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        解析 LLM 输出，提取 Thought 和 JSON Action
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction|\Z)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None

        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not json_match:
            # 尝试直接匹配 Action: 后的花括号内容
            json_match = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)

        action_json = None
        if json_match:
            try:
                action_str = json_match.group(1)
                # 清理可能存在的注释或非标准 JSON 字符
                action_json = json.loads(action_str)
            except json.JSONDecodeError:
                print("❌ JSON 解析失败")
                pass

        return thought, action_json

if __name__ == "__main__":
    llm = LLM()
    executor = ToolExecutor()
    executor.registerTool(search)
    agent = ReActAgent(llm=llm, tool_executor=executor)
    question = "小米最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question=question)



