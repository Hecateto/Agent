"""
A simple MCP server that provides weather information for a given city.
"""
import json
import time
import urllib.parse
from datetime import datetime
from typing import Any

import requests
from hello_agents.protocols import MCPServer

CITY_MAP = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "纽约": "New York",
    "伦敦": "London",
    "巴黎": "Paris",
    "东京": "Tokyo"
}

# 创建 MCP 服务器
weather_server = MCPServer(name="天气信息服务", description="提供城市天气信息的服务")

def get_weather_data(city: str) -> dict[str, float | str | int | Any] | None:
    """Fetch weather data from a public API."""
    city_en = CITY_MAP.get(city, city)
    safe_city = urllib.parse.quote(city_en)
    url = f"https://wttr.in/{safe_city}?format=j1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data['current_condition'][0]
            return {
                "city": city,
                "temperature": current['temp_C'],
                "feels_like": float(current["FeelsLikeC"]),
                "humidity": int(current["humidity"]),
                "condition": current["weatherDesc"][0]["value"],
                "wind_speed": round(float(current["windspeedKmph"]) / 3.6, 1),
                "visibility": float(current["visibility"]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise e


def get_weather(city: str) -> str:
    """MCP method to get weather information for a given city."""
    try:
        weather_info = get_weather_data(city)
        return json.dumps(weather_info, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def list_supported_cities() -> str:
    """MCP method to list supported cities."""
    result = {'cities': list(CITY_MAP.keys()), 'count': len(CITY_MAP)}
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_server_info() -> str:
    """MCP method to get server information."""
    info = {
        "name": weather_server.name,
        "description": weather_server.description,
        "version": "1.0.0",
        "tools": ["get_weather", "list_supported_cities", "get_server_info"]
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


# 注册 MCP 方法
weather_server.add_tool(get_weather)
weather_server.add_tool(list_supported_cities)
weather_server.add_tool(get_server_info)


if __name__ == "__main__":
    weather_server.run()