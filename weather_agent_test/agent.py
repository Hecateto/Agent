import json
import re

from openai import OpenAI

from utils import *


class ReActAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv('BASE_URL'),
            api_key=os.getenv('API_KEY')
        )
        self.model = os.getenv('MODEL')
        self.history = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    def step(self, user_input=None):
        if user_input:
            self.history.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.1,
                stream=False
            )
        except Exception as e:
            print_colored("System", f"LLM 调用失败: {e}")
            return "Error"

        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})

        # 解析 Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction|$)', content, re.DOTALL)
        if thought_match:
            print_colored("Thought", thought_match.group(1).strip())

        # 解析 JSON Action
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

        if not json_match:
            json_match = re.search(r'Action:\s*(\{.*\})', content, re.DOTALL)

        if json_match:
            try:
                action_data = json.loads(json_match.group(1))
                tool_name = action_data.get("name")
                args = action_data.get("args", {})

                print_colored("Action", f"调用 {tool_name} 参数: {args}")

                if tool_name == "finish":
                    return args.get("answer")

                return self._execute_tool(tool_name, args)

            except json.JSONDecodeError:
                print_colored("System", "JSON 解析失败，要求模型重试...")
                return "Observation: Error: Action 格式错误，必须是合法的 JSON。"
        else:
            print_colored("System", "未检测到 Action，结束或重试...")
            return "Observation: Error: 请严格遵循 Thought-Action (JSON) 格式。"

    def _execute_tool(self, tool_name, args):
        if tool_name in AVAILABLE_TOOLS:
            try:
                result = AVAILABLE_TOOLS[tool_name](**args)
            except Exception as e:
                result = f"Error: 工具执行失败: {str(e)}"
        else:
            result = f"Error: 未知工具 {tool_name}"

        print_colored("Observation", str(result))
        self.history.append({"role": "user", "content": f"Observation: {result}"})
        return None

if __name__ == "__main__":

    agent = ReActAgent()

    user_query = "你好，查一下今天南京的天气，并推荐一些适合今天去的景点。"
    print_colored("User", user_query)
    print('='*50)

    response = agent.step(user_query)
    max_steps = 10
    step_count = 0

    while response is None and step_count < max_steps:
        response = agent.step()
        step_count += 1

    print('='*50)
    if response:
        print_colored("Answer", response)
    else:
        print_colored("System", "任务超时或失败。")