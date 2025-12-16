from dotenv import load_dotenv
load_dotenv()

"""
NoteTool 与 ContextBuilder 集成
1. 长期项目追踪
2. 笔记检索与上下文注入
3. 基于历史笔记的连贯建议

该用例问题:
1. 幻觉问题严重, 编造历史笔记ID和内容
2. 相关笔记检索无效
"""

from myAgent.my_llm import MyLLM
from hello_agents import SimpleAgent
from hello_agents.context import ContextConfig, ContextBuilder, ContextPacket
from hello_agents.tools import NoteTool
from hello_agents.core.message import Message
from datetime import datetime
from typing import List, Dict
import json

SYSTEM_INSTRUCTION = """
你是 {project_name} 项目的长期助手。

你的职责:
1. 基于历史笔记提供连贯的建议
2. 追踪项目进展和待解决问题
3. 在回答时引用相关的历史笔记
4. 提供具体、可操作的下一步建议

引用规则:
1. 若历史笔记为空，或没有相关信息等原因导致未进行引用, 请直接利用你的通用知识回答，且禁止标注来源;
1. 如需引用, 只能引用历史笔记中明确出现的笔记, 禁止编造和引用不存在的历史笔记ID、标题和内容;
2. 如需引用, 引用格式必须和历史笔记的 **ID** 或 **标题** 严格一致，并标注来源;
3. 若依据来自用户刚才说的话，必须标注来自当前用户输入。

注意:
- 优先关注标记为 blocker 的问题
- 保持对项目整体进度的认识
"""

class NoteToolAgent(SimpleAgent):
    def __init__(self, name: str, project_name: str, **kwargs):
        llm = MyLLM()
        super().__init__(name=name, llm=llm, **kwargs)
        self.project_name = project_name
        self.note_tool = NoteTool(workspace=f"./{project_name}_notes")
        self.context_builder = ContextBuilder(
            config=ContextConfig(max_tokens=4000)
        )
        self.conversation_history = []


    def run(self, input_text: str, note_as_action: bool = False, **kwargs) -> str:
        relevant_notes = self._retrieve_relevant_notes(input_text)

        # print("DEBUG: Retrieved Relevant Notes:")
        # if not relevant_notes:
        #     print("No relevant notes found.")
        # else:
        #     for i, note in enumerate(relevant_notes):
        #         print(f"Note #{i+1}:")
        #         print(json.dumps(note, indent=2, ensure_ascii=False))

        note_packets = self._note_to_packets(relevant_notes)
        system_prompt = self._build_system_instructions()
        if not system_prompt:
            system_prompt += "\n(注意：当前没有检索到历史笔记，请仅基于你的专业知识回答，无需引用来源。)"

        optimized_context = self.context_builder.build(
            user_query=input_text,
            conversation_history=self.conversation_history,
            system_instructions=system_prompt,
            additional_packets=note_packets
        )
        messages = [
            {'role': 'system', 'content': optimized_context},
            {'role': 'user', 'content': input_text}
        ]
        response = self.llm.invoke(messages)
        if note_as_action:
            self._save_as_note(input_text, response)
        self._update_history(input_text, response)
        return response


    def _retrieve_relevant_notes(self, query: str, limit: int=3) -> List[Dict]:
        try:
            blockers_raw = self.note_tool.run({
                "action": "list",
                "note_type": "blocker",
                "limit": 2
            })
            search_results_raw = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": limit
            })
            blockers = self._ensure_list_of_dicts(blockers_raw)
            search_results = self._ensure_list_of_dicts(search_results_raw)

            all_notes = {}
            for note in blockers + search_results:
                if not isinstance(note, dict):
                    continue
                note_id = note.get("id") or note.get("note_id")
                if note_id:
                    all_notes[note_id] = note
            return list(all_notes.values())[:limit]
        except Exception as e:
            print(f"[WARNING] Failed to retrieve notes: {e}")
            return []


    @staticmethod
    def _ensure_list_of_dicts(data) -> List[Dict]:
        if data is None:
            return []
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return []
        if isinstance(data, dict):
            if 'items' in data and isinstance(data['items'], list):
                return [item for item in data['items'] if isinstance(item, dict)]
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _note_to_packets(notes: List[Dict]) -> List[ContextPacket]:
        packets = []
        for i, note in notes:
            title = note.get("title") or "Untitled Note"
            body = note.get("content") or note.get("body") or ""
            note_type = note.get("type") or "note"
            note_id = note.get("id") or note.get("note_id") or ""
            # content = f"[笔记:{title}]\n{body}"
            content = (
                f"【参考资料 #{i+1}】\n"
                f"ID: {note_id}\n"
                f"类型: {note_type}\n"
                f"标题: {title}\n"
                f"内容:\n{body}\n"
                f"----------------"
            )

            print(f"[DEBUG] Note Packet Content:\n{content}\n")

            ts = None
            for key in ("updated_at", "updatedAt", "time", "timestamp"):
                if key in note:
                    ts = note.get(key)
                    break
            parsed_ts = None
            if isinstance(ts, (int, float)):
                try:
                    parsed_ts = datetime.fromtimestamp(ts)
                except Exception:
                    parsed_ts = None
            elif isinstance(ts, str):
                try:
                    parsed_ts = datetime.fromisoformat(ts)
                except Exception:
                    parsed_ts = None
            if parsed_ts is None:
                parsed_ts = datetime.now()

            packets.append(ContextPacket(
                content=content,
                timestamp=parsed_ts,
                token_count=len(content) // 4,
                relevance_score=0.75,
                metadata={
                    "type": "note",
                    "note_type": note_type,
                    "note_id": note_id,
                }
            ))
        return packets


    def _save_as_note(self, user_input: str, response: str):
        try:
            if "问题" in user_input or "阻塞" in user_input:
                note_type = "blocker"
            elif "计划" in user_input or "下一步" in user_input:
                note_type = "action"
            else:
                note_type = "conclusion"

            self.note_tool.run({
                "action": "create",
                "title": f"{user_input[:30]}...",
                "content": f"## 问题\n{user_input}\n\n## 分析\n{response}",
                "note_type": note_type,
                "tags": [self.project_name, "auto_generated"]
            })

        except Exception as e:
            print(f"[WARNING] Failed to save note: {e}")

    def _build_system_instructions(self) -> str:
        return SYSTEM_INSTRUCTION.format(project_name=self.project_name)

    def _update_history(self, user_input: str, response: str):
        self.conversation_history.append(
            Message(content=user_input, role="user", timestamp=datetime.now()),
        )
        self.conversation_history.append(
            Message(content=response, role="assistant", timestamp=datetime.now()),
        )
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


def main():
    print("=" * 10 + " NoteToolAgent " + "=" * 10)
    agent = NoteToolAgent(
        name="NoteToolAgent",
        project_name="数据管道优化"
    )
    print("第一次交互: 记录项目状态")
    print(f"Question: \n已完成数据模型层的重构,测试覆盖率达到85%。下一步计划重构业务逻辑层。")
    response = agent.run(
        "已完成数据模型层的重构,测试覆盖率达到85%。下一步计划重构业务逻辑层。",
        note_as_action=True
    )
    print(f"Agent Response: \n{response}\n")

    print("第二次交互: 识别并记录阻塞问题")
    print(f"Question: \n在重构业务逻辑层时,遇到了依赖版本冲突的问题,该如何解决?")
    response = agent.run(
        "在重构业务逻辑层时,遇到了依赖版本冲突的问题,该如何解决?",
        note_as_action=True
    )
    print(f"Agent Response: \n{response}\n")

    print("第三次交互: 基于历史笔记提供建议")
    print(f"Question: \n基于之前的进展,请给出下一步优化数据管道的建议。")
    response = agent.run(
        "基于之前的进展,请给出下一步优化数据管道的建议。",
        note_as_action=True
    )
    print(f"Agent Response: \n{response}\n")

    print("查看笔记摘要:")
    summary = agent.note_tool.run({"action": "summary"})
    print(json.dumps(summary, indent=2, ensure_ascii=False).replace("\\n", "\n"))

    print("=" * 30)


if __name__ == "__main__":
    main()
