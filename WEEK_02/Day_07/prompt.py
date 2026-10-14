import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Same task for every prompting technique
task = "Explain FastAPI to a beginner in simple words."


# 1. Zero-shot prompting
zero_shot = task

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=zero_shot
)

print("\n===== ZERO-SHOT =====")
print(response.text)


# 2. One-shot prompting
one_shot = """
Example:
Question: What is Python?
Answer: Python is a programming language used to build applications.

Now answer:
Explain FastAPI to a beginner in simple words.
"""

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=one_shot
)

print("\n===== ONE-SHOT =====")
print(response.text)


# 3. Few-shot prompting
few_shot = """
Examples:

Question: What is Python?
Answer: Python is a programming language used to build applications.

Question: What is Git?
Answer: Git is a version control system used to track changes in code.

Question: What is an API?
Answer: An API allows different software applications to communicate.

Now answer:
Explain FastAPI to a beginner in simple words.
"""

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=few_shot
)

print("\n===== FEW-SHOT =====")
print(response.text)


# 4. Chain-of-Thought style
cot_prompt = """
Explain FastAPI to a beginner.

Before giving the final answer, work through the key concepts
needed to explain it correctly. Then provide the final answer
with the key points in a clear and concise format.
"""

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=cot_prompt
)

print("\n===== COT-STYLE =====")
print(response.text)


# 5. Improved prompt
improved_prompt = """
You are a Python instructor.

Explain FastAPI to a beginner who already knows basic Python.

Requirements:
- Give a one-sentence definition.
- Explain why FastAPI is used.
- Give one simple example.
- Use simple language.
- Keep the answer under 150 words.
- Use bullet points where appropriate.
"""

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=improved_prompt
)

print("\n===== IMPROVED PROMPT =====")
print(response.text)