"""
测试天气查询MCP服务器的功能
"""

import asyncio
import json
import os
from hello_agents.protocols import MCPClient

async def test_weather_server():
    server_script = os.path.join(os.path.dirname(__file__), "weather_server.py")
    client = MCPClient(["python", server_script])

    try:
        async with client:
            # 获取服务器信息
            info = json.loads(await client.call_tool("get_server_info", {}))
            print(f"服务器: {info['name']} v{info['version']}")

            # 列出支持的城市
            cities = json.loads(await client.call_tool("list_supported_cities", {}))
            print(f"支持城市: {cities['count']} 个")

            # 查询天气
            test_cities = ["上海", "伦敦", "巴黎", "东京"]
            for city in test_cities:
                weather = json.loads(await client.call_tool("get_weather", {"city": city}))
                if "error" in weather:
                    print(f"{city} 查询失败: {weather['error']}")
                else:
                    print(f"{city} 天气: {weather['temperature']}°C, {weather['condition']}")
                await asyncio.sleep(1)

            print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_weather_server())
