from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message, ToolRegistry
import re

TOOL_CALL_PROMPT = """
## 可用工具
你可以使用以下工具来帮助回答问题：
{tools_desc}

## 工具调用格式
当需要使用工具时，请使用以下格式：
`[TOOL_CALL:{{tool_name}}:{{parameters}}]`

例如：
`[TOOL_CALL:search:Python编程]` 
或 
`[TOOL_CALL:memory:recall=用户信息]`

工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。
"""

class MySimpleAgent(SimpleAgent):
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_use: bool = True
    ):
        """
        基于 SimpleAgent 的自定义智能体，支持工具调用。
        :param name: 智能体名称
        :param llm: 模型实例
        :param system_prompt: 系统提示
        :param config: 配置项
        :param tool_registry: 工具注册表
        :param enable_tool_use: 是否启用工具调用
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_use = enable_tool_use and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_use else '禁用'}")

    def run(self, input_text: str, max_tool_iters: int=3, **kwargs) -> str:
        """
        处理输入文本，支持工具调用。
        :param input_text: 用户输入文本
        :param max_tool_iters: 最大工具调用迭代次数
        :param kwargs: 传递给 LLM 的其他参数
        :return: LLM 的响应文本
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")
        messages = []
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({'role': 'system', 'content': enhanced_system_prompt})

        for msg in self._history:
            messages.append({'role': msg.role, 'content': msg.content})

        messages.append({'role': 'user', 'content': input_text})

        if not self.enable_tool_use:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, 'user'))
            self.add_message(Message(response, 'assistant'))
            print(f"💬 {self.name} 回复完成")
            return response

        return self._run_with_tools(messages, input_text, max_tool_iters, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """
        构建增强的系统提示，包含工具信息。
        :return: 增强的系统提示文本
        """
        base_prompt = self.system_prompt or "你是一个智能助手。"
        if not self.enable_tool_use or not self.tool_registry:
            return base_prompt

        tools_desc = self.tool_registry.get_tools_description()
        if not tools_desc or tools_desc == "暂无可用工具":
            return base_prompt

        tools_section = TOOL_CALL_PROMPT.format(tools_desc=tools_desc)

        return base_prompt + tools_section

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iters: int, **kwargs) -> str:
        """
        处理输入文本，支持工具调用的迭代逻辑。
        :param messages: 历史消息列表
        :param input_text: 用户输入文本
        :param max_tool_iters: 最大工具调用迭代次数
        :param kwargs: 传递给 LLM 的其他参数
        :return: LLM 的最终响应文本
        """
        cur_iter = 0
        final_response = ""

        while cur_iter < max_tool_iters:
            response = self.llm.invoke(messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    clean_response = clean_response.replace(call['original'], "")

                messages.append({'role': 'assistant', 'content': clean_response.strip()})

                tool_results_text = "\n".join(tool_results)
                messages.append({'role': 'user', 'content': f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                cur_iter += 1
                continue

            final_response = response
            break

        if cur_iter >= max_tool_iters and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(final_response, 'assistant'))
        print(f"💬 {self.name} 回复完成")

        return final_response

    @staticmethod
    def _parse_tool_calls(text: str) -> list:
        """解析文本中的工具调用指令。"""
        pattern = r'\[TOOL_CALL:(\w+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append(
                {
                    'tool_name': tool_name.strip(),
                    'parameters': parameters.strip(),
                    'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
                }
            )
        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行指定的工具调用。"""
        if not self.tool_registry:
            return f"工具注册表未配置，无法执行工具 {tool_name}。"

        try:
            if tool_name == 'calculator':
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"未找到名为 {tool_name} 的工具。"
                result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 执行结果：{result}"
        except Exception as e:
            return f"执行工具 {tool_name} 时出错：{e}"

    @staticmethod
    def _parse_tool_parameters(tool_name: str, parameters: str) -> dict:
        """解析工具调用的参数字符串为字典。"""
        param_dict = {}
        if '=' in parameters:
            if ',' in parameters:
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}
        return param_dict

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式处理输入文本"""
        print(f"🌊 {self.name} 开始流式处理: {input_text}")
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        for msg in self._history:
            messages.append({'role': msg.role, 'content': msg.content})
        messages.append({'role': 'user', 'content': input_text})

        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk
        print()

        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(full_response, 'assistant'))
        print(f"💬 {self.name} 流式回复完成")

    # Tool management methods
    # ========================
    def has_tools(self) -> bool:
        """检查是否有注册的工具。"""
        return self.enable_tool_use and self.tool_registry is not None

    def add_tool(self, tool) -> None:
        """向工具注册表添加工具。"""
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
            self.enable_tool_use = True
        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 {tool.name} 已添加到注册表。")

    def remove_tool(self, tool_name: str) -> bool:
        """从工具注册表中移除工具。"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            print(f"🗑️ 工具 {tool_name} 已从注册表中移除。")
            return True
        return False

    def list_tools(self) -> list:
        """列出所有注册的工具。"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []