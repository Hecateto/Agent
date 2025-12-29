from hello_agents.protocols import A2AClient, A2AServer
import time, threading

researcher = A2AServer("researcher", "研究员")

@researcher.skill("research")
def do_research(text: str) -> str:
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text.strip()
    return str({'topic': topic, 'findings': f"Detailed research findings on {topic}."})

writer = A2AServer("writer", "撰写者")
@writer.skill("write")
def do_write(text: str) -> str:
    import re
    match = re.search(r'write\s+(.+)', text, re.IGNORECASE)
    content = match.group(1).strip() if match else text.strip()
    try:
        data = eval(content)
        topic = data.get('topic', 'unknown topic')
        findings = data.get('findings', '')
    except:
        topic = 'unknown topic'
        findings = content

    return f"Article on {topic}:\n{findings}\n\nThis article provides an in-depth look into {topic} based on the research findings."

editor = A2AServer("editor", "编辑")
@editor.skill("edit")
def do_edit(text: str) -> str:
    import re
    match = re.search(r'edit\s+(.+)', text, re.IGNORECASE)
    content = match.group(1).strip() if match else text.strip()
    result = {
        'article': content + "\n\nEdited for clarity and conciseness.",
        'feedback': "Well-structured article.",
        'approved': True
    }
    return str(result)

threading.Thread(target=lambda: researcher.run(port=5000), daemon=True).start()
threading.Thread(target=lambda: writer.run(port=5001), daemon=True).start()
threading.Thread(target=lambda: editor.run(port=5002), daemon=True).start()
time.sleep(1)  # Give servers time to start

researcher_client = A2AClient("http://localhost:5000")
writer_client = A2AClient("http://localhost:5001")
editor_client = A2AClient("http://localhost:5002")

def collaborative_article_creation(topic: str) -> str:
    research = researcher_client.execute_skill('research', f'research {topic}')
    print(f"\nResearch结果：\n{research}")
    research_data = research.get('result', '')

    article = writer_client.execute_skill('write', f'write {research_data}')
    print(f"\nWrite结果：\n{article}")
    article_content = article.get('result', '')

    edit = editor_client.execute_skill('edit', f'edit {article_content}')
    print(f"\nEdit结果：\n{edit}")
    edit_data = edit.get('result', '')

    return edit_data

if __name__ == "__main__":
    result = collaborative_article_creation("AI在医疗领域的应用")
    print(f"\n最终结果：\n{result}")