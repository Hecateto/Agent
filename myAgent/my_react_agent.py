from typing import Optional, List

from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry

MY_REACT_PROMPT = """
你是一个具备推理和行动能力的AI助手。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: [你的思考过程]
Action: [工具名][Parameter] 
   - 例如: calculate[20+5] 或 search[Python release date]
   - 如果你有足够信息回答问题，请使用: Finish[这里填入具体的回答内容]

## 规则
1. Action 必须严格匹配格式：ToolName[Input]
2. **不要**在 Action 外部添加多余的括号或引号。
3. 遇到问题先思考(Thought)，再行动(Action)。

## 当前任务
Question: {question}

## 执行历史
{history}

开始：
"""

class MyReActAgent(ReActAgent):
    """
    基于 ReActAgent 的自定义智能体，支持工具调用和多轮推理。
    """
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            tool_registry: ToolRegistry,
            config: Optional[Config] = None,
            system_prompt: Optional[str] = None,
            custom_prompt: Optional[str] = None,
            max_steps: int = 5
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        self.current_history: List[str] = []
        self.max_steps = max_steps

    def run(self, input_text: str, **kwargs) -> str:

        print(f"🤖 {self.name} 正在处理: {input_text}")
        self.current_history = []

        for step in range(self.max_steps):
            print(f"\n--- 步骤 {step + 1} ---")
            tools_desc = self.tool_registry.get_tools_description()
            history_text = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_text
            )
            messages = [{'role': 'user', 'content': prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            print(f"📝 LLM: \n{response_text.strip()}")

            thought, action = self._parse_output(response_text)

            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 任务完成: {final_answer}")
                self.add_message(Message(input_text, 'user'))
                self.add_message(Message(final_answer, 'assistant'))
                return final_answer

            if action:
                tool_name, tool_input = self._parse_action(action)
                print(f"🛠️ 调用工具: {tool_name} 参数: {tool_input}")
                try:
                    observation = self.tool_registry.execute_tool(tool_name, tool_input)
                except Exception as e:
                    observation = f"工具调用失败: {e}"
                print(f"👀 观测结果: {observation}")
                self.current_history.append(f"Thought: {thought}")
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")
            else:
                print("⚠️ 未检测到有效的 Action，结束推理。")
                return response_text

        return "❌ 超过最大推理步骤，任务失败。"

