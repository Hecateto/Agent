from dotenv import load_dotenv

from Protocol.A2A.A2A_Client import client

load_dotenv()
from hello_agents.protocols import A2AClient, A2AServer
import time, threading

agent1 = A2AServer("agent1", "Agent 1")
@agent1.skill("propose")
def handle_proposal(text: str) -> str:
    print(f"[Agent 1] Received proposal: \n{text}\n")

    import re
    # 解析提案
    match = re.search(r'propose\s+(.+)', text, re.IGNORECASE)
    proposal_str = match.group(1).strip() if match else text
    try:
        proposal = eval(proposal_str)
        task = proposal.get("task", "unknown task")
        deadline = proposal.get("deadline", "unknown deadline")
        if deadline >= 7:
            result = {"accepted": True, "message": "接受提案"}
        else:
            result = {"accepted": False, "message": "截止日期太紧，无法接受", "counter_proposal": {"task": task, "deadline": 7}}
        return str(result)
    except Exception as e:
        return str({"accepted": False, "message": f"提案格式错误: {e}"})

agent2 = A2AServer("agent2", "Agent 2")
@agent2.skill("negotiate")
def negotiate(text: str) -> str:
    print(f"[Agent 2] Starting negotiation with: \n{text}\n")

    import re
    # 解析协商请求
    match = re.search(r'negotiate\s+task:(.+?)\s+deadline:(\d+)', text, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        deadline = int(match.group(2).strip())
        proposal = {"task": task, "deadline": deadline}
        return str({"status": "negotiating", "proposal": proposal})
    else:
        return str({"status": "error", "message": "协商请求格式错误"})


if __name__ == "__main__":
    threading.Thread(target=lambda: agent1.run(port=7000), daemon=True).start()
    threading.Thread(target=lambda: agent2.run(port=7001), daemon=True).start()
    time.sleep(2)
    print()

    client1 = A2AClient("http://localhost:7000")
    client2 = A2AClient("http://localhost:7001")

    negotiation = client2.execute_skill(
        "negotiate",
        "negotiate task:开发新功能 deadline:5"
    )
    print(f"协商请求：\n{negotiation.get('result')}\n")

    proposal = client1.execute_skill(
        "propose",
        "propose {'task': '开发新功能', 'deadline': 5}"
    )
    print(f"提案评估：\n{proposal.get('result')}\n")

    print("协商完成")