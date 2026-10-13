import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ddgs import DDGS

# Loading the gemini api key 
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")

client = genai.Client(api_key=api_key)

print("AI Agent setup successful!")

# Caluclator Tool 
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""
    print(f"[TOOL CALLED] calculator({expression!r})")

    import ast
    import operator

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )

        raise ValueError("Unsupported expression")

    try:
        return str(evaluate(ast.parse(expression, mode="eval")))
    except Exception as error:
        return f"Calculation error: {error}"

# Web_Search Tool 
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=5))

        if not results:
            return "No search results found."

        return "\n\n".join(
            f"Title: {r.get('title', '')}\n"
            f"Summary: {r.get('body', '')}\n"
            f"URL: {r.get('href', '')}"
            for r in results
        )

    except Exception as error:
        return f"Search error: {error}"

# Running the Agent to choose itself a tool and perform the function 
def run_agent(question: str) -> str:
    try:
        chat = client.chats.create(
            model="gemini-3.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful AI assistant. "
                    "Use calculator for arithmetic questions. "
                    "Use web_search for current or external information. "
                    "Explain results clearly."
                ),
                tools=[calculator, web_search],
            ),
        )

        response = chat.send_message(question)
        return response.text or "No response generated."

    except Exception as error:
        return f"Agent error: {type(error).__name__}: {error}"



if __name__ == "__main__":
    print("AI Research Agent started!")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            print("\nAgent:", run_agent(question))
        except Exception as error:
            print("Error:", error)