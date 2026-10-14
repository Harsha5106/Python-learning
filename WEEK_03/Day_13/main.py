import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from the project root .env file.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Read the Groq API key from the environment.
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing! Add it to the .env file.")

# Create the Groq client that all worker agents will use.
client = Groq(api_key=api_key)


def extract_text(response) -> str:
    """Safely read the content from a Groq response."""
    try:
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception:
        return ""


# This is a generic worker agent.
# It receives a role and a task, then asks Groq to do that task.
def worker_agent(role: str, task: str) -> str:
    """Run one sub-task for a specific role."""
    prompt = f"""
    You are a {role}.

    Complete the following task carefully and clearly.
    Task: {task}

    Instructions:
    - Be factual and practical.
    - Keep the answer concise but useful.
    - Do not invent facts.
    - Use clear structure with headings if needed.
    """

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please provide your answer in a clear format."},
            ],
            temperature=0.7,
            max_tokens=800,
        )

        result = extract_text(response)
        if not result:
            raise ValueError("Groq returned an empty response.")

        return result

    except Exception as error:
        return f"{role} failed: {type(error).__name__}: {error}"


# This is the orchestrator.
# It assigns two independent tasks to two worker agents and then combines their results.
def orchestrate(topic: str) -> dict:
    """Run the worker-agents flow and merge their output."""

    # Worker 1: Research/analysis task
    research_task = (
        f"Research the topic '{topic}' and provide: "
        "1) top 3 key facts, 2) 2 examples, and 3) why the topic matters."
    )
    research_output = worker_agent("Research Analyst", research_task)

    # Worker 2: Risk/critique task
    risk_task = (
        f"Analyze the topic '{topic}' and list: "
        "1) 3 risks, 2) 3 limitations, and 3) common misconceptions."
    )
    risk_output = worker_agent("Risk Analyst", risk_task)

    # Orchestrator: combine both worker outputs into a single final answer.
    final_prompt = f"""
    You are the orchestrator agent.

    Your job is to combine the work of two specialist workers.

    Worker 1 output:
    {research_output}

    Worker 2 output:
    {risk_output}

    Create a final response with:
    1. Executive Summary
    2. Key Findings
    3. Risks and Limitations
    4. Final Conclusion

    Make the result clear, balanced, and easy to understand.
    """

    final_output = worker_agent("Orchestrator", final_prompt)

    return {
        "topic": topic,
        "research_output": research_output,
        "risk_output": risk_output,
        "final_output": final_output,
    }


if __name__ == "__main__":
    print("AI Orchestrator Agent — Day 13")
    print("Type 'exit' to stop.\n")

    while True:
        topic = input("Enter a topic: ").strip()

        if topic.lower() == "exit":
            print("Goodbye!")
            break

        if not topic:
            print("Please enter a valid topic.")
            continue

        try:
            result = orchestrate(topic)

            print("\n=== RESEARCH WORKER OUTPUT ===")
            print(result["research_output"])

            print("\n=== RISK WORKER OUTPUT ===")
            print(result["risk_output"])

            print("\n=== FINAL ORCHESTRATED OUTPUT ===")
            print(result["final_output"])

        except Exception as error:
            print(f"\nSomething went wrong: {type(error).__name__}: {error}")
