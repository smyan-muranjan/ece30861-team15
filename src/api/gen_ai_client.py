import requests
import os

class GenAIClient:
    def __init__(self, api_key: str = "sk-232cb6a155564c27839633098a0904e1"):
        self.url = "https://genai.rcac.purdue.edu/api/chat/completions"
        api_key = os.environ.get("GENAI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, message: str, model: str = "llama3.1:latest") -> str:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        response = requests.post(self.url, headers=self.headers, json=body)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
