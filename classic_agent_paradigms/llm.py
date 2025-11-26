import os
from typing import List, Dict, Optional, Callable, Union, Any
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError

load_dotenv()

class LLM:
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = None):
        """
        初始化 LLM 客户端
        """
        self.model = model or os.getenv("MODEL")
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.timeout = timeout or int(os.getenv("TIMEOUT", 60))

        if not all([self.model, self.api_key, self.base_url]):
            raise ValueError("Critical Config Missing: MODEL, API_KEY, and BASE_URL must be provided.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def think(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            stream: bool = True,
            json_mode: bool = False,
            on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        核心推理方法
        :param messages: 对话历史
        :param temperature: 随机度
        :param stream: 是否流式传输
        :param json_mode: 是否强制输出 JSON (需要模型支持)
        :param on_token: 回调函数，每接收到一个 token 时调用 (仅在 stream=True 时有效)
        :return: 完整的响应文本
        """

        # 构造请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**params)

            if stream:
                collected_content = []
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        collected_content.append(content)
                        # 如果提供了回调函数，则调用它（实现 UI 更新或日志记录）
                        if on_token:
                            on_token(content)
                        else:
                            # 默认行为：打印到控制台 (保持原有逻辑作为默认)
                            print(content, end="", flush=True)

                # 流式结束后的换行（仅在默认打印模式下）
                if not on_token:
                    print()

                return "".join(collected_content)
            else:
                # 非流式直接返回
                return response.choices[0].message.content

        except APIConnectionError as e:
            print(f"Server connection error: {e}")
            raise  # 重新抛出，让上层 Agent 决定是否重试
        except APIStatusError as e:
            print(f"API status error: {e.status_code} - {e.response}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    # 自定义回调函数，模拟 Agent 在 Web 界面或不同位置的输出
    def custom_logger(token):
        print(f"{token}", end="", flush=True)


    try:
        llm = LLM()

        example_messages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user",
             "content": "写一个快速排序算法"}
        ]

        final_response = llm.think(
            example_messages,
            temperature=0.7,
            json_mode=False,
            on_token=custom_logger
        )

        print("\n\n--- Final Memory Stored ---")
        print(final_response)

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"Runtime Error: {e}")