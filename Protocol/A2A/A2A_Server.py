from hello_agents.protocols import A2AServer
import threading
import time

researcher = A2AServer(
    name="Researcher",
    description="An agent that conducts research and shares findings with other agents.",
    version="1.0",
)

@researcher.skill("research")
def handle_research(text: str) -> str:
    """
    Conduct research on a given topic and return findings.
    """
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text.strip()
    result = {
        'topic': topic,
        'findings': f"Comprehensive research findings on '{topic}'.",
        'sources': ['Source A', 'Source B', 'Source C']
    }
    return str(result)

def start_server():
    researcher.run(host='localhost', port=5000)

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("✅ Researcher A2A Server is running...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down the Researcher A2A Server...")
