import json
import os

import aiohttp


class GenAIClient:
    def __init__(self, api_key: str = "sk-232cb6a155564c27839633098a0904e1"):
        self.url = "https://genai.rcac.purdue.edu/api/chat/completions"
        env_api_key = os.environ.get("GENAI_API_KEY")
        if env_api_key is not None:
            api_key = env_api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, message: str, model: str = "llama3.1:latest") -> str:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers=self.headers, json=body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    raise Exception(f"Error: {response.status}, {error}")

    async def get_performance_claims(self, readme_text: str) -> dict:
        with open("src/api/performance_claims_ai_prompt.txt", "r") as f:
            prompt = f.read()
        prompt += readme_text
        response = await self.chat(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse GenAI response as JSON: {response}") from e
