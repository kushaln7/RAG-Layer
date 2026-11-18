"""Test script verifying connectivity to the DeepSeek LLM API."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def test_deepseek_connection() -> str:
    """Invoke DeepSeek LLM to test API connectivity."""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/")
    model = os.getenv("LLM_MODEL", "deepseek-flash")

    if not api_key:
        print("Skipping DeepSeek connection: No API key found in environment.")
        return ""

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    response = llm.invoke("Say 'DeepSeek API is connected successfully!'")
    output = str(response.content)
    print(output)
    return output


if __name__ == "__main__":
    print("Testing DeepSeek LLM connection...")
    test_deepseek_connection()
