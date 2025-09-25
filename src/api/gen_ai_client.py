import json
import os
import re
import ssl

import aiohttp


class GenAIClient:
    def __init__(self):
        self.url = "https://genai.rcac.purdue.edu/api/chat/completions"
        env_api_key = os.environ.get("GENAI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {env_api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, message: str, model: str = "llama3.3:70b") -> str:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        # Create SSL context that doesn't verify certificates for servers
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
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
        # Stage 1: Extract relevant information
        with open(
            "src/api/performance_claims_extraction_prompt.txt", "r"
        ) as f:
            extraction_prompt = f.read()
        extraction_prompt += readme_text
        extraction_response = await self.chat(extraction_prompt)

        # Stage 2: Convert to JSON format
        with open(
            "src/api/performance_claims_conversion_prompt.txt", "r"
        ) as f:
            conversion_prompt = f.read()
        conversion_prompt += "\n" + extraction_response
        json_response = await self.chat(conversion_prompt)

        # Extract JSON object from response (handles markdown code blocks)
        match = re.search(r'\{[^}]*\}', json_response)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Failed to parse extracted JSON: {json_str}"
                ) from e
        else:
            # Try parsing the entire response as fallback
            try:
                return json.loads(json_response)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Failed to parse GenAI response as JSON: {json_response}."
                    f"Extraction response was: {extraction_response}"
                ) from e

    async def get_readme_clarity(self, readme_text: str) -> float:
        with open("src/api/readme_clarity_ai_prompt.txt", "r") as f:
            prompt = f.read()
        prompt += readme_text
        response = await self.chat(prompt)

        # Try to extract a floating point number from the response
        # Handle various possible formats from LLM

        # First, try to parse the response directly as a float
        try:
            return float(response.strip())
        except ValueError:
            pass

        # Try to find a number in the response using regex
        # Look for patterns like "0.6", "0.85", "1.0", etc.
        number_match = re.search(r'\b(?:0?\.\d+|1\.0+|0\.0+|1)\b', response)
        if number_match:
            try:
                value = float(number_match.group(0))
                # Ensure the value is within expected range [0, 1]
                return max(0.0, min(1.0, value))
            except ValueError:
                pass

        # Try to find any decimal number in the response
        decimal_match = re.search(r'\d*\.?\d+', response)
        if decimal_match:
            try:
                value = float(decimal_match.group(0))
                # Ensure the value is within expected range [0, 1]
                return max(0.0, min(1.0, value))
            except ValueError:
                pass

        # If all parsing attempts fail, raise an exception
        raise Exception(
            f"Failed to extract a valid float from GenAI response: {response}"
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        client = GenAIClient()
        readme_text = (
            "This is a sample README file for a machine learning model. "
            "It includes performance metrics such as accuracy and F1-score. "
            "The model achieves 92% accuracy on the test set and has been "
            "benchmarked against several baselines."
        )
        performance_claims = await client.get_performance_claims(readme_text)
        print("Performance Claims:", performance_claims)

        clarity_score = await client.get_readme_clarity(readme_text)
        print("Readme Clarity Score:", clarity_score)

    asyncio.run(main())
