from openai import OpenAI

class DeepSeekLLM:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content