import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY_11 = PROJECT_ROOT / "WEEK_03" / "Day_11"
sys.path.insert(0, str(DAY_11))

import importlib.util

day11_file = DAY_11 / "main.py"

spec = importlib.util.spec_from_file_location("day11_agent", day11_file)
day11_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day11_agent)

web_search = day11_agent.web_search

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")

client = genai.Client(api_key=api_key)

# Configure workflow logging
LOG_FILE = Path(__file__).parent / "agent.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

print("Day 12 multi-step agent setup successful!")


def research_topic(topic):
    """Search the web for information about a topic."""
    logging.info("STEP 1 STARTED: Researching topic: %s", topic)
    print("\n[STEP 1] Researching topic...")

    results = web_search(topic)

    if not results or results.startswith("Search error:"):
        logging.error("Research failed: %s", results)
        raise RuntimeError(f"Research failed: {results}")

    logging.info("STEP 1 COMPLETED: Research results collected")
    print("[STEP 1] Research completed!")

    return results


def summarize_research(topic, research):
    """Use Gemini to summarize the collected research."""

    logging.info("STEP 2 STARTED: Summarizing research")
    print("\n[STEP 2] Summarizing research with Gemini...")

    prompt = f"""
    You are a research assistant.

    Topic: {topic}

    Research results:
    {research}

    Instructions:
    - Write a clear, concise summary.
    - Identify the most important facts.
    - Use simple language.
    - Do not invent facts that are not supported by the research.
    - Mention any uncertainty in the search results.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        summary = response.text

        if not summary:
            raise ValueError("Gemini returned an empty summary.")

        logging.info("STEP 2 COMPLETED: Summary generated")
        print("[STEP 2] Summary completed!")

        return summary

    except Exception as error:
        logging.exception("STEP 2 FAILED: Summarization error")
        raise RuntimeError(
            f"Could not summarize research: {error}"
        ) from error


def save_report(topic, summary):
    """Save the research summary to a text file."""

    report_file = Path(__file__).parent / "research_report.txt"

    logging.info("STEP 3 STARTED: Saving research report")
    print("\n[STEP 3] Saving report...")

    try:
        with open(report_file, "w", encoding="utf-8") as file:
            file.write("AI RESEARCH REPORT\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Topic: {topic}\n\n")
            file.write("SUMMARY\n")
            file.write("-" * 50 + "\n")
            file.write(summary)
            file.write("\n")

        logging.info("STEP 3 COMPLETED: Report saved to %s", report_file)
        print(f"[STEP 3] Report saved: {report_file}")

        return report_file

    except OSError:
        logging.exception("STEP 3 FAILED: Could not save report")
        raise


def run_workflow(topic):
    """Run the complete research workflow."""

    print("\n========== AI RESEARCH WORKFLOW ==========")
    logging.info("WORKFLOW STARTED: Topic=%s", topic)

    try:
        # Step 1: Research
        research = research_topic(topic)

        # Step 2: Summarize
        summary = summarize_research(topic, research)

        # Step 3: Save
        report_file = save_report(topic, summary)

        logging.info("WORKFLOW COMPLETED SUCCESSFULLY")
        print("\n========== WORKFLOW COMPLETED ==========")
        print("\nFINAL SUMMARY:\n")
        print(summary)
        print(f"\nReport location: {report_file}")

        return summary

    except Exception as error:
        logging.exception("WORKFLOW FAILED")
        print(f"\nWorkflow failed: {error}")
        return None


if __name__ == "__main__":
    print("AI Research Agent — Day 12")
    print("Type 'exit' to stop.")

    while True:
        topic = input("\nEnter a research topic: ").strip()

        if topic.lower() == "exit":
            print("Goodbye!")
            break

        if not topic:
            print("Please enter a topic.")
            continue

        run_workflow(topic)
