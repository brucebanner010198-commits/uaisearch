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
