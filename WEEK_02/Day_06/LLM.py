import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

prompt = "Explain what an API is in two simple sentences."

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)

print("Response:")
print(response.text)

print("\nToken Usage:")
print("Input tokens:", response.usage_metadata.prompt_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
print("Total tokens:", response.usage_metadata.total_token_count)