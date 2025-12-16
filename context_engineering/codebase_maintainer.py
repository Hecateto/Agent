"""
Codebase Maintainer Agent
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from hello_agents import SimpleAgent
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.core.message import Message
from hello_agents.tools import MemoryTool, NoteTool, TerminalTool
from hello_agents.tools.registry import ToolRegistry

from myAgent.my_llm import MyLLM

load_dotenv()


class CodebaseMaintainerAgent:
    def __init__(
            self,
            project_name: str,
            codebase_path: str,
            llm: Optional[MyLLM] = None,
    ):
        self.project_name = project_name
        self.codebase_path = codebase_path
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm or MyLLM()

        self.memory_tool = MemoryTool(user_id=project_name, memory_types=["working"])
        self.note_tool = NoteTool(workspace=f"./{project_name}_notes")
        self.terminal_tool = TerminalTool(workspace=codebase_path, timeout=60)

        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=None,
            config=ContextConfig(max_tokens=4000, reserve_ratio=0.15, min_relevance=0.2, enable_compression=True)
        )

        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(self.memory_tool)
        self.tool_registry.register_tool(self.note_tool)
        self.tool_registry.register_tool(self.terminal_tool)

        self.agent = SimpleAgent(
            name="CodebaseMaintainer",
            llm=self.llm,
            tool_registry=self.tool_registry,
            system_prompt=self._build_base_system_prompt(),
            enable_tool_calling=True
        )

        self.conversation_history: List[Message] = []

        self.stats = {
            "session_start": datetime.now(),
            "commands_executed": 0,
            "notes_created": 0,
            "issues_found": 0,
            "tool_calls": 0
        }

        print(f"✅ 代码库维护助手已初始化: {project_name} (Agentic Mode)")
        print(f"📁 工作目录: {codebase_path}")
        print(f"🆔 会话ID: {self.session_id}")
        print(f"🔧 可用工具: {', '.join(self.tool_registry.list_tools())}")


    def run(self, user_input: str, mode: str = "auto") -> str:
        """
        Run the Codebase Maintainer Agent with the given user input.
        :param user_input: User input string
        :param mode: Operation mode
            - "auto": Agent decides when to use tools
            - "explore": Focus on code exploration
            - "analyze": Focus on problem analysis
            - "plan": Focus on task planning
        :return: Agent response string
        """
        print(f"👤 用户: {user_input}\n")

        relevant_notes = self._retrieve_relevant_notes(user_input)
        note_packets = self._notes_to_packets(relevant_notes)

        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(mode),
            additional_packets=note_packets
        )

        print("🤖 Agent 正在思考并决定使用哪些工具...\n")
        self.agent.system_prompt = context

        response = self.agent.run(input_text=user_input)
        self._track_tool_usage()
        self._update_history(user_input, response)

        print(f"\n🤖 助手: {response}\n {'='*80}\n")
        return response


    def _build_base_system_prompt(self) -> str:
        return f"""
        你是 {self.project_name} 项目的代码库维护助手。

        你的核心能力:
        1. 使用 TerminalTool 探索代码库
           - 你可以执行任何 shell 命令: ls, cat, grep, find, git 等
           - 工作目录: {self.codebase_path}

        2. 使用 NoteTool 记录发现和任务
           - 创建笔记记录重要发现
           - 笔记类型: blocker(阻塞问题)、action(行动计划)、task_state(任务状态)、conclusion(结论)

        3. 使用 MemoryTool 存储关键信息
           - 记住重要的上下文信息
           - 跨会话保持连贯性

        当前会话ID: {self.session_id}

        重要原则:
        - 你要自主决定使用哪些工具、执行什么命令
        - 探索代码库时，先了解整体结构，再深入细节
        - 发现重要信息时，主动使用 NoteTool 记录
        - 保持回答的专业性和实用性
        """


    def _track_tool_usage(self):
        if hasattr(self.agent, 'message_history'):
            for msg in self.agent.message_history[-10:]:
                if msg.role == 'tool':
                    self.stats['tool_calls'] += 1
                    if 'terminal' in str(msg.content).lower() or 'command' in str(msg.content).lower():
                        self.stats['commands_executed'] += 1
                    elif 'note' in str(msg.content).lower():
                        self.stats['notes_created'] += 1


    def _retrieve_relevant_notes(self, query: str, limit: int=3) -> List[Dict]:
        try:
            blockers_raw = self.note_tool.run({
                "action": "list",
                "query": "blocker",
                "limit": 2
            })
            blockers = self._normalize_note_results(blockers_raw)

            search_results_raw = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": limit
            })
            search_results = self._normalize_note_results(search_results_raw)

            all_notes = {}
            for note in blockers + search_results:
                if not isinstance(note, dict):
                    continue
                note_id = note.get("id") or note.get("note_id")
                if not note_id:
                    continue
                if note_id not in all_notes:
                    all_notes[note_id] = note
            return list(all_notes.values())[:limit]
        except Exception as e:
            print(f"⚠️ 获取相关笔记时出错: {e}")
            return []


    def _normalize_note_results(self, result: Any) -> List[Dict]:
        if not result:
            return []
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, str):
            text = result.strip()
            if not text:
                return []
            if text.startswith('[') or text.startswith('{'):
                try:
                    data = json.loads(text)
                    return self._normalize_note_results(data)
                except json.JSONDecodeError:
                    return []
        return []

    @staticmethod
    def _notes_to_packets(notes: List[Dict]) -> List[ContextPacket]:
        packets = []
        for note in notes:
            if not isinstance(note, dict):
                continue
            relevance_map = {
                "blocker": 0.9,
                "action": 0.8,
                "task_state": 0.75,
                "conclusion": 0.7
            }
            note_type = note.get("type", "general")
            relevance = relevance_map.get(note_type, 0.6)
            content = f"[笔记:{note.get('title', 'Untitled')}]\n类型: {note_type}\n\n{note.get('content', '')}"
            update_at = note.get("updated_at")
            try:
                note_timestamp = datetime.fromisoformat(update_at) if update_at else datetime.now()
            except ValueError:
                note_timestamp = datetime.now()
            packets.append(ContextPacket(
                content=content,
                timestamp=note_timestamp,
                token_count=len(content)//4,
                relevance_score=relevance,
                metadata={
                    'type': 'note',
                    'note_type': note_type,
                    'note_id': note.get('id') or note.get('note_id')
                }
            ))
        return packets


    def _build_system_instructions(self, mode: str) -> str:
        base_instructions = self._build_base_system_prompt()
        mode_hints = {
            "explore": """
        用户当前关注: 探索代码库

        建议策略:
        - 考虑使用 TerminalTool 了解代码结构（如 find, ls, tree）
        - 查看关键文件（如 README, 主要模块）
        - 将架构信息记录到笔记方便后续查阅
        """,
            "analyze": """
        用户当前关注: 分析代码质量

        建议策略:
        - 考虑使用 grep 查找潜在问题（TODO, FIXME, BUG）
        - 分析代码复杂度和结构
        - 将发现的问题记录为 blocker 或 action 笔记
        """,
            "plan": """
        用户当前关注: 任务规划

        建议策略:
        - 回顾历史笔记了解当前进度
        - 基于已有信息制定行动计划
        - 创建或更新 task_state 类型的笔记
        """,
            "auto": """
        用户当前关注: 自由对话

        建议策略:
        - 根据用户需求灵活决策
        - 在需要时主动使用工具获取信息
        - 不需要时可以直接回答
        """
        }

        return base_instructions + "\n" + mode_hints.get(mode, mode_hints["auto"])


    def _update_history(self, user_input: str, agent_response: str):
        self.conversation_history.append(Message(role="user", content=user_input, timestamp=datetime.now()))
        self.conversation_history.append(Message(role="assistant", content=agent_response, timestamp=datetime.now()))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]


    def explore(self, target: str = ".") -> str:
        """
        Explore the codebase starting from the target directory or file.
        :param target: Target directory or file to explore
        :return: Agent response string
        """
        prompt = f"请探索代码库中的 '{target}'，了解它的结构和内容。"
        return self.run(prompt, mode="explore")

    def analyze(self, focus: str = "") -> str:
        """
        Analyze the codebase for potential issues or improvements.
        :param focus: Specific area to focus the analysis on
        :return: Agent response string
        """
        prompt = f"请分析代码库，特别关注 '{focus}' 方面，找出潜在的问题或改进点。"
        return self.run(prompt, mode="analyze")


    def plan_next_steps(self) -> str:
        """
        Plan the next steps for maintaining or improving the codebase.
        :return: Agent response string
        """
        prompt = "根据之前的分析和当前进度，请规划下一步任务。"
        return self.run(prompt, mode="plan")


    def execute_command(self, command: str) -> str:
        """
        Execute a specific shell command in the codebase context.
        :param command: Shell command to execute
        :return: Agent response string
        """
        result = self.terminal_tool.run({"command": command})
        self.stats['commands_executed'] += 1
        return result


    def create_note(self, title: str, content: str, note_type: str = "general", tags: List[str] = None) -> str:
        result = self.note_tool.run({
            "action": "create",
            "title": title,
            "content": content,
            "type": note_type,
            "tags": tags or [self.project_name]
        })
        self.stats['notes_created'] += 1
        return result


    def get_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        :return: Statistics dictionary
        """
        duration = (datetime.now() - self.stats['session_start']).total_seconds()
        try:
            note_summary = self.note_tool.run({"action": "summary"})
        except:
            note_summary = {}
        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration
            },
            "activity": {
                "commands_executed": self.stats['commands_executed'],
                "notes_created": self.stats['notes_created'],
                "issues_found": self.stats['issues_found'],
            },
            "notes": note_summary
        }


    def generate_report(self, save_to_file: bool=True) -> Dict[str, Any]:
        report = self.get_stats()
        if save_to_file:
            filename = f"{self.project_name}_session_report_{self.session_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 会话报告已保存到: {filename}")
        return report


def main():
    print("=" * 10 + " CodebaseMaintainer " + "=" * 10 + "\n")

    # 初始化助手
    maintainer = CodebaseMaintainerAgent(
        project_name="my_flask_app",
        codebase_path="./codebase",
        llm=MyLLM()
    )

    # 探索代码库（Agent 自主决定如何探索）
    print("\n### 探索代码库（Agent 自主探索）###")
    response = maintainer.explore()
    response = maintainer.run("请查看 data_processor.py 文件，分析其代码设计")
    time.sleep(1)

    # 分析代码质量（Agent 自主决定分析方法）
    print("\n### 分析代码质量（Agent 自主分析）###")
    response = maintainer.analyze()
    response = maintainer.run(
        "请分析 api_client.py 的代码质量，特别是错误处理部分，给出改进建议"
    )
    time.sleep(1)

    # 规划下一步（Agent 基于历史信息规划）
    print("\n### 规划下一步任务（Agent 自主规划）###")
    response = maintainer.plan_next_steps()
    response = maintainer.run(
        "请基于我们的分析，创建一个详细的本周重构计划。"
        "计划应该包括：目标、具体任务清单、时间安排和风险。"
        "请使用 NoteTool 创建一个 task_state 类型的笔记来记录这个计划。"
    )
    time.sleep(1)

    # 笔记摘要
    print("\n### 获取笔记摘要 ###")
    note_summary = maintainer.note_tool.run({"action": "summary"})
    print(json.dumps(note_summary, indent=2, ensure_ascii=False))

    # 生成报告
    print("\n### 生成会话报告 ###")
    report = maintainer.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()