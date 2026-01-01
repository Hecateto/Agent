"""
MCP Weather Agent
"""

import os
from dotenv import load_dotenv
load_dotenv()
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool
from datetime import datetime

def create_weather_agent():
    """
    Create a weather agent that can provide weather information.
    """
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    llm = HelloAgentsLLM()
    assistant = SimpleAgent(
        name="WeatherAgent",
        llm=llm,
        system_prompt=f"""
                你是天气助手，可以查询城市天气。
                当前系统时间是：{current_time}。
                如果用户询问日期或时间，请直接使用上述系统时间，不要编造。
                使用 get_weather 工具查询天气，支持中文城市名。
                """
    )

    server_script = os.path.join(os.path.dirname(__file__), "weather_server.py")
    weather_tool = MCPTool(server_command=['python', server_script])
    assistant.add_tool(weather_tool)

    return assistant


def demo():
    assistant = create_weather_agent()
    print("\n查询上海天气：")
    response = assistant.run("上海今天天气怎么样？")
    print(f"回答: {response}\n")


def interactive():
    assistant = create_weather_agent()
    print("欢迎使用天气助手！输入 '退出' 结束对话。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("再见！")
            break
        response = assistant.run(user_input)
        print(f"天气助手: {response}\n")


if __name__ == "__main__":
    import sys
    # python weather_agent.py demo
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    # python weather_agent.py
    else:
        interactive()
