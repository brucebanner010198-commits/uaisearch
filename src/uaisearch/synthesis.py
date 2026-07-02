import json
from collections.abc import AsyncIterator

import httpx


class LLMClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client or httpx.AsyncClient(timeout=60.0)

    async def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = await self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        async with self._http.stream(
            "POST", f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model, "messages": messages,
                "temperature": temperature, "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                content = delta.get("content")
                if content:
                    yield content
