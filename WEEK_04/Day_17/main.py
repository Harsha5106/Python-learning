
import json
import re
from typing import Any

# 1. Define the expected output schema
REQUIRED_FIELDS = {"answer", "confidence"}
ALLOWED_CONFIDENCE = (0.0, 1.0)

# 2. Basic scope guardrail
ALLOWED_TOPICS = {
    "python", "programming", "code", "llm", "ai",
    "machine learning", "data", "api", "function"
}

BLOCKED_PATTERNS = [
    r"\bhow to make a bomb\b",
    r"\bsteal passwords\b",
    r"\bwrite malware\b",
]

def check_scope(query: str) -> tuple[bool, str]:
    """Allow relevant technical questions and reject clearly out-of-scope ones."""
    normalized = query.lower()

    if any(re.search(pattern, normalized) for pattern in BLOCKED_PATTERNS):
        return False, "Blocked by safety guardrail"

    if not any(topic in normalized for topic in ALLOWED_TOPICS):
        return False, "Query is outside the supported technical scope"

    return True, "Query passed input guardrail"


# 3. Validate output structure and data types
def validate_schema(data: Any) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Output must be a JSON object"

    if set(data.keys()) != REQUIRED_FIELDS:
        return False, "Missing or unexpected fields"

    if not isinstance(data["answer"], str) or not data["answer"].strip():
        return False, "Answer must be a non-empty string"

    confidence = data["confidence"]

    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, "Confidence must be numeric"

    if not ALLOWED_CONFIDENCE[0] <= confidence <= ALLOWED_CONFIDENCE[1]:
        return False, "Confidence must be between 0 and 1"

    return True, "Schema validation passed"


# 4. Validate a confidence field using regex
def validate_confidence_format(raw_output: str) -> tuple[bool, str]:
    pattern = (
        r'"confidence"\s*:\s*'
        r'(?:0(?:\.\d+)?|1(?:\.0+)?)'
    )

    if not re.search(pattern, raw_output):
        return False, "Confidence regex check failed"

    return True, "Confidence regex check passed"


# 5. Validate the complete response
def validate_output(raw_output: str) -> tuple[bool, str]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return False, "Invalid JSON output"

    schema_ok, schema_message = validate_schema(data)
    if not schema_ok:
        return False, schema_message

    regex_ok, regex_message = validate_confidence_format(raw_output)
    if not regex_ok:
        return False, regex_message

    return True, "All output validations passed"


# 6. Ten test cases: five input-guardrail tests and five output tests
TESTS = [
    {
        "name": "Valid Python query",
        "query": "Explain Python functions",
        "output": '{"answer":"A function is reusable code.","confidence":0.95}',
        "expected": "PASS",
    },
    {
        "name": "Out-of-scope cooking query",
        "query": "Give me a recipe for pasta",
        "output": '{"answer":"Here is a recipe.","confidence":0.8}',
        "expected": "BLOCK",
    },
    {
        "name": "Unsafe request",
        "query": "How to make a bomb",
        "output": '{"answer":"I cannot help with that.","confidence":1.0}',
        "expected": "BLOCK",
    },
    {
        "name": "Valid AI query",
        "query": "Explain how an LLM works",
        "output": '{"answer":"An LLM generates text.","confidence":0.9}',
        "expected": "PASS",
    },
    {
        "name": "Missing answer field",
        "query": "Explain Python code",
        "output": '{"confidence":0.9}',
        "expected": "FAIL",
    },
    {
        "name": "Invalid JSON",
        "query": "Explain an API in programming",
        "output": '{"answer":"An API connects systems", "confidence":}',
        "expected": "FAIL",
    },
    {
        "name": "Confidence outside range",
        "query": "Explain machine learning",
        "output": '{"answer":"A learning method.","confidence":1.5}',
        "expected": "FAIL",
    },
    {
        "name": "Wrong confidence type",
        "query": "Explain a Python function",
        "output": '{"answer":"Reusable code.","confidence":"high"}',
        "expected": "FAIL",
    },
    {
        "name": "Empty answer",
        "query": "Explain an AI API",
        "output": '{"answer":" ","confidence":0.7}',
        "expected": "FAIL",
    },
    {
        "name": "Unexpected extra field",
        "query": "Explain Python programming",
        "output": '{"answer":"Python is a language.","confidence":0.9,"extra":"x"}',
        "expected": "FAIL",
    },
]


def run_tests():
    passed = 0
    print("\nOUTPUT VALIDATION & GUARDRAILS REPORT")
    print("=" * 65)

    for index, test in enumerate(TESTS, start=1):
        in_scope, scope_message = check_scope(test["query"])

        if not in_scope:
            actual = "BLOCK"
            reason = scope_message
        else:
            valid, message = validate_output(test["output"])
            actual = "PASS" if valid else "FAIL"
            reason = message

        success = actual == test["expected"]
        if success:
            passed += 1

        status = "OK" if success else "UNEXPECTED"
        print(f"{index:02}. {test['name']}")
        print(f"    Expected: {test['expected']} | Actual: {actual}")
        print(f"    Result: {status} | {reason}")

    print("=" * 65)
    print(f"Tests passed: {passed}/{len(TESTS)}")
    print(f"Tests failed: {len(TESTS) - passed}/{len(TESTS)}")


if __name__ == "__main__":
    run_tests()
