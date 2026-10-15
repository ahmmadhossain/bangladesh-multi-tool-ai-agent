
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Project root: parent folder of src
ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

# Load the .env file from the project root
load_dotenv(ENV_FILE, override=True)

# Validate required configuration
api_key = os.getenv("API_KEY")
base_url = os.getenv(
    "BASE_URL",
    "https://api.cheaperinference.com/v1",
)
model = os.getenv("MODEL", "glm-5.3")

if not api_key:
    raise RuntimeError(
        f"API_KEY not found. Please check your .env file: {ENV_FILE}"
    )

print(f"Environment file found: {ENV_FILE}")
print(f"Model: {model}")
print(f"Base URL: {base_url}")

# Connect to CheaperInference
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60,
    max_retries=2,
)

# Test the model
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with: GLM-5.3 API connection successful."
                ),
            }
        ],
    )

    print("\nAPI connection successful!")
    print("Model response:", response.choices[0].message.content)

except Exception as exc:
    print(f"\nAPI request failed: {exc}")
