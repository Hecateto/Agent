import os
from typing import Optional, List, Dict, Iterator
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model or os.getenv("MODEL")
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")

        if not self.api_key:
            raise ValueError("API_KEY is required")
        if not self.model:
            raise ValueError("MODEL is required")

        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.frequency_penalty = kwargs.get("frequency_penalty", 1.05)
        self.presence_penalty = kwargs.get("presence_penalty", 0.5)
        self.top_p = kwargs.get("top_p", 0.9)
        self.timeout = kwargs.get("timeout", 60)

        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_p=self.top_p,
                extra_body={"repetition_penalty": 1.1},
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用"""
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                extra_body={"repetition_penalty": 1.1},
                **kwargs
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            raise RuntimeError(f"LLM stream invocation failed: {e}")