# LangSmith – Complete Engineering Guide for LLM Apps

> **A full, end-to-end, production-grade explanation of LangSmith concepts**
> Covers: Tracing, Monitoring, Alerting, Evaluation, Datasets, Annotation, Prompt Experimentation, and RAG lifecycle integration.

---

## Table of Contents

1. Why LLM Apps Are Hard
2. What LangSmith Solves
3. Core Mental Model
4. Tracing (Runs & Traces)
5. Monitoring (System Health)
6. Alerting (Automatic Warnings)
7. Evaluation (Answer Quality)
8. Datasets & Annotation
9. Prompt Experimentation
10. RAG Lifecycle with LangSmith
11. Automatic Tracing vs `@traceable`
12. Best Practices & Production Checklist
13. Final Mental Model

---

## 1. Why LLM Apps Are Hard

### Traditional Software

```
Input → Code → Output
```

* Deterministic behavior
* Same input = same output
* Bugs are reproducible
* Logs are sufficient

---

### LLM-Based Applications

```
User Query
  ↓
Prompt Template
  ↓
Retriever (Vector DB)
  ↓
Retrieved Documents
  ↓
LLM
  ↓
Answer
```

### Core Problems

* Same input → different outputs
* Small prompt changes affect global behavior
* Retriever may fetch wrong context
* LLM can hallucinate
* Failures appear as **patterns**, not single errors

Traditional logging cannot explain *why* answers are wrong.

---

## 2. What LangSmith Solves

**LangSmith turns LLM systems from guesswork into engineering.**

It provides:

* Observability (see what happened)
* Evaluation (measure quality)
* Experimentation (compare versions)
* Monitoring (track production health)

---

## 3. Core Mental Model (Very Important)

LangSmith does **not** only track outputs.

> **It tracks execution.**

```
One user request
  ↓
Multiple internal steps (runs)
  ↓
Grouped together as a trace
```

Everything in LangSmith is built on this idea.

---

## 4. Tracing (Runs & Traces)

### 4.1 Run

A **Run** = one atomic operation.

Examples:

* One LLM call
* One retriever search
* One tool invocation
* One chain execution

```python
llm.invoke("What is RAG?")
```

Creates **one LLM run**.

---

### 4.2 Trace

A **Trace** = all runs triggered by a **single user request**.

Example Trace:

```
Trace
 └─ Chain run
     ├─ Prompt formatting run
     ├─ Retriever run
     ├─ LLM run
     └─ Output parser run
```

### Why Tracing Matters

* Identify hallucination sources
* Detect retriever failures
* Pinpoint latency bottlenecks
* Debug agents and tool chains

---

## 5. Monitoring (System Health)

### What Monitoring Means

**Monitoring = analyzing many traces together over time.**

Not one request — hundreds or thousands.

---

### Metrics LangSmith Monitors

| Metric                | Meaning            |
| --------------------- | ------------------ |
| Latency (P50/P95/P99) | Response speed     |
| Token Usage           | LLM consumption    |
| Cost                  | Spend tracking     |
| Error Rate            | Failed requests    |
| Success Rate          | Completed requests |

---

### Latency Explained

* **P50** – median response time
* **P95** – slow users (critical)
* **P99** – worst-case latency

Example:

* P50 = 1s
* P95 = 5s

---

## 6. Alerting (Automatic Warnings)

### What Alerting Is

**Alerting = automatic notifications when metrics cross thresholds.**

Examples:

* Error rate > 5%
* P95 latency spike
* Sudden cost increase

### Why Alerting Matters

* Detect issues before users complain
* Prevent silent quality degradation
* Essential for production reliability

---

## 7. Evaluation (Answer Quality)

### Key Distinction

| Concept    | Question               |
| ---------- | ---------------------- |
| Monitoring | Is the system healthy? |
| Evaluation | Are answers good?      |

Evaluation measures **quality**, not speed or cost.

---

### Common Evaluation Metrics

| Metric       | Meaning              |
| ------------ | -------------------- |
| Correctness  | Factually correct    |
| Relevance    | Answers the question |
| Faithfulness | Grounded in context  |
| Completeness | Covers all aspects   |

---

### Evaluation Methods

1. **Gold-standard datasets**
2. **LLM-as-a-Judge**
3. **Custom Python evaluators** (JSON validity, citations, business rules)

---

## 8. Datasets & Annotation

### What Is a Dataset?

A dataset contains:

* Inputs (questions)
* Expected outputs (optional)
* Metadata

Example:

| Question     | Expected Answer                |
| ------------ | ------------------------------ |
| What is RAG? | Retrieval-Augmented Generation |

---

### Annotation Explained

**Annotation = labeling outputs with quality judgments.**

Examples:

* Correct: Yes / No
* Hallucinated: Yes / No
* Faithful: Yes / No

Annotations are usually **human-provided**.

---

### Why Dataset Versioning Matters

LangSmith:

* Versions datasets
* Tracks changes
* Enables reproducible experiments

---

## 9. Prompt Experimentation

### Definition

**Prompt experimentation = testing multiple prompt versions using data and metrics.**

---

### How LangSmith Enables This

```
Same Dataset
   ↓
Prompt v1 | Prompt v2 | Prompt v3
   ↓
Same RAG Pipeline
   ↓
Evaluators
   ↓
Comparison Dashboard
```

You get **data-backed decisions**, not gut feelings.

---

## 10. RAG Lifecycle with LangSmith

```
Dataset
  ↓
Retriever
  ↓
LLM Response
  ↓
Annotation
  ↓
Evaluation Metrics
  ↓
Prompt / Model Experiments
  ↓
Production Monitoring
  ↓
Alerts
```

Creates a **closed feedback loop**.

---

## 11. Automatic Tracing vs `@traceable`

### Automatic Tracing

* Enabled via environment variables
* Zero code changes
* Traces LangChain primitives only

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
```

### What It Records Automatically

* Inputs / outputs
* Prompt text
* Tokens
* Latency
* Errors

---

### `@traceable`

Used for **custom Python functions**.

```python
from langsmith import traceable

@traceable
def split_documents(docs):
    ...
```

Records:

* Function inputs / outputs
* Execution time
* Errors

---

### Comparison

| Feature              | Automatic Tracing | `@traceable` |
| -------------------- | ----------------- | ------------ |
| Zero config          | Yes               | No           |
| Traces LangChain ops | Yes               | Yes          |
| Traces custom Python | No                | Yes          |

---

## 12. Best Practices & Production Checklist

* Enable automatic tracing early
* Add `@traceable` selectively
* Track P95 latency
* Maintain evaluation datasets
* Run prompt experiments before production changes
* Enable alerts for cost and errors

---

## 13. Final Mental Model (Memorize This)

* **Tracing** → What happened in one request?
* **Monitoring** → Is the system healthy overall?
* **Evaluation** → Are answers good?
* **Prompt Experimentation** → Which version is best?
* **Datasets & Annotation** → What are we measuring against?

---

## Final One-Line Summary

> **LangSmith transforms LLM applications from fragile, opaque systems into measurable, testable, and production-ready engineering systems.**

---

## License

MIT
