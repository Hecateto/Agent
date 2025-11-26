import os
import time
import requests
from tavily import TavilyClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==========================================
# 颜色与配置
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_colored(role, text):
    if role == "User":
        print(f"{Colors.HEADER}👤 [用户]: {text}{Colors.ENDC}")
    elif role == "Thought":
        print(f"{Colors.YELLOW}🤔 [思考]: {text}{Colors.ENDC}")
    elif role == "Action":
        print(f"{Colors.BLUE}🛠️ [行动]: {text}{Colors.ENDC}")
    elif role == "Observation":
        print(f"{Colors.GREEN}👁️ [观察]: {text}{Colors.ENDC}")
    elif role == "System":
        print(f"{Colors.RED}⚠️ [系统]: {text}{Colors.ENDC}")
    elif role == "Answer":
        print(f"{Colors.BOLD}✅ [最终答案]: {text}{Colors.ENDC}")


# ==========================================
# 增强型工具库
# ==========================================

def get_weather(city):
    """查询天气，内置重试机制"""
    print(f"   (正在连接天气服务查询 {city}...)")
    url = f"https://wttr.in/{city}?format=j1"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            cur = data['current_condition'][0]
            weather_desc = cur['weatherDesc'][0]['value']
            temp_c = cur['temp_C']
            humidity = cur['humidity']
            return f"【{city}天气】: {weather_desc}, 温度 {temp_c}℃, 湿度 {humidity}%"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return f"Error: 天气查询服务暂时不可用 (已重试{max_retries}次)。请告知用户稍后再试或根据一般经验回答。"


def get_attraction(city, weather):
    """搜索景点，结果清洗和截断"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: 未配置 TAVILY_API_KEY"

    print(f"   (正在搜索适合 {weather} 的 {city} 景点...)")
    tavily = TavilyClient(api_key=api_key)
    query = f"推荐适合在{city}旅游的景点，当前天气{weather}，排除广告"

    try:
        response = tavily.search(query=query, search_depth='basic', max_results=3)

        results = []
        for res in response.get('results', []):
            title = res.get('title', '未知')
            content = res.get('content', '')
            clean_content = content[:150].replace('\n', ' ') + "..."
            results.append(f"- {title}: {clean_content}")

        if not results:
            return "未找到具体景点信息，请尝试更通用的推荐。"

        return "\n".join(results)
    except Exception as e:
        return f"Error: 搜索服务出错: {str(e)}"


# 工具映射表
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# ==========================================
# Prompt
# ==========================================

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。必须严格遵循以下 ReAct 流程。

# 工具库:
- `get_weather`: 参数 `{"city": "城市名"}`
- `get_attraction`: 参数 `{"city": "城市名", "weather": "天气状况"}`
- `finish`: 当你得到答案时调用，参数 `{"answer": "最终回复给用户的话"}`

# 输出协议:
你的每一次回复必须包含且仅包含一个 Thought 和一个 Action (以 JSON 格式)。

格式示例:
Thought: 我需要查询天气。
Action: ```json
{
    "name": "get_weather",
    "args": {
        "city": "Beijing"
    }
}
"""