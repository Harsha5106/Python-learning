# Day 7 – Prompt Engineering

## Task

Explain FastAPI to a beginner using different prompting techniques.

## 1. Zero-shot Prompting

### Prompt
Explain FastAPI to a beginner in simple words.

### Observation
The model generated a detailed explanation with an analogy, examples,
code, and reasons for using FastAPI. Since no examples or constraints
were provided, the model decided the response structure itself.

---

## 2. One-shot Prompting

### Prompt
One example was provided before asking the model to explain FastAPI.

### Observation
The model followed the example's concise style and produced a short
explanation.

---

## 3. Few-shot Prompting

### Prompt
Multiple examples were provided before the FastAPI question.

### Observation
The model followed the demonstrated format very closely and produced
a concise answer. This showed that examples can strongly influence
the response format and level of detail.

---

## 4. Chain-of-Thought Style Prompting

### Prompt
The model was asked to work through the key concepts before giving
the final explanation.

### Observation
The response was highly structured and covered API concepts,
frameworks, async programming, type hints, Pydantic, and automatic
documentation.

---

## 5. Improved Prompt

### Prompt
The model was given a role, audience, requirements, example,
constraints, and output length.

### Observation
The response was clear, focused, structured, and stayed within the
requested format.

---

# Comparison

| Technique | Output | Observation |
|---|---|---|
| Zero-shot | Detailed | Model decides structure |
| One-shot | Short | Follows one example |
| Few-shot | Very concise | Strongly follows examples |
| CoT-style | Very detailed | More structured explanation |
| Improved | Focused and structured | Best control over output |

## Best Technique

For this task, the Improved Prompt performed best because it provided
clear instructions, audience, constraints, and output requirements.

CoT-style prompting also produced a detailed and structured answer,
but it was more detailed than necessary for this particular task.

## Conclusion

Prompt engineering allows us to control the quality, structure,
relevance, and format of LLM responses. Zero-shot is useful for simple
tasks, while one-shot and few-shot prompting provide examples to guide
the model. More detailed prompts with clear requirements provide
greater control over the final output.