"""
智能文档问答助手
- PDF文档上传与解析
- 基于文档内容的问答功能
- 记忆与检索
- 用户UI构建
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import gradio as gr
from hello_agents.tools import MemoryTool, RAGTool


class QAssistant:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.memory_tool = MemoryTool(user_id=self.user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        self.current_document = None

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        加载并处理PDF文档
        :param pdf_path: PDF文件路径
        :return: 加载结果
        """
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件未找到: {pdf_path}"}

        start_time = time.time()
        try:
            # 处理PDF文档, 转markdown, 分块, 向量化存储
            result = self.rag_tool.execute(
                "add_document",
                file_path=pdf_path,
                chunk_size=1000,
                chunk_overlap=200
            )

            process_time = time.time() - start_time

            self.current_document = os.path.basename(pdf_path)
            self.stats['documents_loaded'] += 1

            self.memory_tool.execute(
                "add",
                content=f"已加载文档: {self.current_document}",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id
            )

            return {
                "success": True,
                "message": f"文档 '{self.current_document}' 加载成功，处理时间: {process_time:.2f} 秒",
                "document": self.current_document
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"加载文档时出错: {str(e)}"
            }

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        """
        回答用户关于当前文档的问题
        :param question: 用户提问
        :param use_advanced_search: 是否启用高级检索功能 (MQE + HyDE)
        :return: 回答内容
        """
        if not self.current_document:
            return "⚠️ 请先加载文档! "
        self.memory_tool.execute(
            "add",
            content=f"用户提问: {question}",
            memory_type="working",
            importance=0.8,
            session_id=self.session_id
        )

        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=3,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search
        )

        self.memory_tool.execute(
            "add",
            content=f"关于{question}的回答: {answer}",
            memory_type="episodic",
            importance=0.9,
            event_type="qa_interaction",
            session_id=self.session_id
        )

        self.stats['questions_asked'] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        """
        添加笔记或概念到记忆中
        :param content: 笔记内容
        :param concept: 相关概念
        """
        self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.7,
            concept=concept or "general",
            session_id=self.session_id
        )
        self.stats['concepts_learned'] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        """
        回忆相关记忆
        :param query: 查询内容
        :param limit: 返回结果数量
        :return: 相关记忆
        """
        result = self.memory_tool.execute(
            "search",
            query=query,
            limit=limit,
            session_id=self.session_id
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        获取当前会话统计信息
        :return: 统计信息字典
        """
        duration = (datetime.now() - self.stats['session_start']).total_seconds()
        stats_summary = {
            "会话时长": f"{duration:.2f}秒",
            "加载文档": self.stats['documents_loaded'],
            "提问次数": self.stats['questions_asked'],
            "学习笔记": self.stats['concepts_learned'],
            "当前文档": self.current_document or "无"
        }
        return stats_summary

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        生成当前会话报告
        :param save_to_file: 是否保存为 JSON 文件
        :return: 报告内容
        """
        memory_summary = self.memory_tool.execute("summary")
        rag_stats = self.rag_tool.execute("stats")
        duration = (datetime.now() - self.stats['session_start']).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats['session_start'].isoformat(),
                "duration_seconds": duration
            },
            "learning_metrics": {
                "documents_loaded": self.stats['documents_loaded'],
                "questions_asked": self.stats['questions_asked'],
                "concepts_learned": self.stats['concepts_learned']
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats
        }

        if save_to_file:
            report_file = f"report_{self.session_id}.json"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=4, default=str)
                report['report_file'] = report_file
            except Exception as e:
                report['save_error'] = f"保存报告时出错: {str(e)}"

        return report

def create_gradio_ui():
    """ 创建Gradio用户界面 """
    assistant_state = {"assistant": QAssistant()}

    def init_assistant(user_id):
        if not user_id:
            user_id = "web_user"
        assistant_state["assistant"] = QAssistant(user_id=user_id)
        return f"✅ 助手已初始化 (用户: {user_id})"

    def load_pdf(pdf_file) -> str:
        if assistant_state["assistant"] is None:
            return "⚠️ 请先初始化助手!"
        if pdf_file is None:
            return "⚠️ 请上传PDF文件!"

        pdf_path = pdf_file.name
        result = assistant_state['assistant'].load_document(pdf_path)

        if result["success"]:
            return f"✅ {result['message']}\n📄 文档: {result['document']}"
        else:
            return f"❌ {result['message']}"

    def chat(message: str, history: List) -> Tuple[str, List]:
        if assistant_state["assistant"] is None:
            return "", history + [[message, "❌ 请先初始化助手并加载文档"]]
        if not message.strip():
            return "", history

        if any(keyword in message for keyword in ["之前", "学过", "回顾", "历史", "记得"]):
            response = assistant_state["assistant"].recall(message)
            response = f"🧠 **学习回顾**\n\n{response}"
        else:
            response = assistant_state["assistant"].ask(message)
            response = f"💡 **回答**\n\n{response}"
        history.append([message, response])
        return "", history

    def add_note_ui(note_content: str, concept: str) -> str:
        if assistant_state["assistant"] is None:
            return "⚠️ 请先初始化助手!"
        if not note_content.strip():
            return "⚠️ 笔记内容不能为空!"

        assistant_state["assistant"].add_note(note_content, concept)
        return f"✅ 笔记已保存: {note_content[:50]}..."

    def get_stats_ui() -> str:
        if assistant_state["assistant"] is None:
            return "⚠️ 请先初始化助手!"
        stats = assistant_state["assistant"].get_stats()
        result = "📊 **学习统计**\n\n"
        for key, value, in stats.items():
            result += f"- **{key}**: {value}\n"
        return result

    def generate_report_ui() -> str:
        if assistant_state["assistant"] is None:
            return "⚠️ 请先初始化助手!"
        report = assistant_state["assistant"].generate_report(save_to_file=True)

        result = "📝 **学习报告**\n\n"
        for section, content in report.items():
            if section == "report_file":
                result += f"-💾 报告已保存到文件: {content}\n"
            else:
                result += f"- **{section}**: {content}\n"
        return result

    with gr.Blocks(title="智能文档问答助手") as demo:
        gr.Markdown("# 📚 智能文档问答助手")

        with gr.Row():
            user_id_input = gr.Textbox(label="用户ID", placeholder="输入用户ID (默认: web_user)")
            init_button = gr.Button("初始化助手")

        init_output = gr.Textbox(label="初始化状态", interactive=False)

        with gr.Row():
            pdf_upload = gr.File(label="上传PDF文档", file_types=[".pdf"])
            load_button = gr.Button("加载文档")

        load_output = gr.Textbox(label="加载状态", interactive=False)

        chat_history = gr.Chatbot(label="问答历史")
        message_input = gr.Textbox(label="输入您的问题", placeholder="请输入您的问题...")
        send_button = gr.Button("发送")

        note_content = gr.Textbox(label="添加笔记", placeholder="输入笔记内容...")
        concept_input = gr.Textbox(label="相关概念 (可选)", placeholder="输入相关概念...")
        add_note_button = gr.Button("保存笔记")
        note_output = gr.Textbox(label="笔记状态", interactive=False)

        stats_button = gr.Button("查看学习统计")
        stats_output = gr.Textbox(label="学习统计", interactive=False)

        report_button = gr.Button("生成学习报告")
        report_output = gr.Textbox(label="学习报告", interactive=False)

        init_button.click(init_assistant, inputs=[user_id_input], outputs=[init_output])
        load_button.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])
        send_button.click(chat, inputs=[message_input, chat_history], outputs=[message_input, chat_history])
        add_note_button.click(add_note_ui, inputs=[note_content, concept_input], outputs=[note_output])
        stats_button.click(get_stats_ui, outputs=[stats_output])
        report_button.click(generate_report_ui, outputs=[report_output])

    return demo


def main():
    print("启动智能文档问答助手...")
    ui = create_gradio_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

if __name__ == "__main__":
    main()


