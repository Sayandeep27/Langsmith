# LangSmith Tracing Explained (Automatic vs `@traceable`)

---

## Overview

LangSmith provides **observability** for LangChain applications. One of its most powerful features is **automatic tracing**, which works with **zero extra code**. However, this often creates confusion around **when and why `@traceable` is needed**.

This README explains:

* What *automatic tracing* really means
* What LangSmith records automatically
* What it cannot see
* The exact role of `@traceable`
* When you should (and should not) use it

---

## 1. What Is Automatic Tracing?

Automatic tracing means:

> LangSmith automatically records **LangChain operations** without you writing logging or tracing code.

Once enabled, LangSmith hooks into **LangChain primitives** such as:

* LLMs
* Chains
* Retrievers
* Vector stores
* Runnables

You do **not** need decorators, callbacks, or manual logs.

---

## 2. One-Time Setup (Required)

Enable tracing using environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=rag-dev
```

This setup is enough to activate **automatic tracing**.

---

## 3. Example: Automatic Tracing in Action

### Normal LangChain Code

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Explain RAG")
```

### What LangSmith Automatically Records

| Category             | Captured Automatically |
| -------------------- | ---------------------- |
| Input                | "Explain RAG"          |
| Prompt sent to model | Yes                    |
| Model name           | Yes                    |
| Tokens used          | Yes                    |
| Latency              | Yes                    |
| Output               | Yes                    |
| Errors               | Yes                    |

**No extra code was written.**

This is why it is called the **killer feature**.

---

## 4. What Automatic Tracing Does NOT Capture

Automatic tracing **does not see your custom Python functions**.

### Example

```python
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()
```

LangSmith does **not** know:

* When this function ran
* How long it took
* What it returned
* Whether it failed internally

Reason:

> This function is **pure Python**, not a LangChain component.

---

## 5. What LangSmith Hooks Into Automatically

LangSmith traces only **LangChain primitives**.

| Component                       | Automatically Traced |     |
| ------------------------------- | -------------------- | --- |
| LLMs (`ChatOpenAI`, `ChatGroq`) | Yes                  |     |
| Chains (`prompt                 | llm`)                | Yes |
| Retrievers                      | Yes                  |     |
| Vector store search             | Yes                  |     |
| Runnables                       | Yes                  |     |
| Plain Python functions          | No                   |     |

---

## 6. Role of `@traceable`

`@traceable` is used to trace **your own Python functions**.

### Example

```python
from langsmith import traceable

@traceable
def split_documents(docs):
    return splitter.split_documents(docs)
```

Now LangSmith records:

* Function inputs
* Function outputs
* Execution time
* Errors

Without `@traceable`, this function is invisible to LangSmith.

---

## 7. Automatic Tracing vs `@traceable`

### Side-by-Side Comparison

| Feature                        | Automatic Tracing | `@traceable` |
| ------------------------------ | ----------------- | ------------ |
| Zero code required             | Yes               | No           |
| Traces LangChain components    | Yes               | Yes          |
| Traces custom Python functions | No                | Yes          |
| Needed for basic RAG           | No                | No           |
| Needed for debugging internals | Limited           | Yes          |

They **complement each other**, they do not replace each other.

---

## 8. Why Docs Say “Zero Extra Code”

When LangSmith documentation says:

> "Zero extra code"

It specifically means:

* No logging
* No callbacks
* No handlers

**For LangChain components only**.

It does **not** mean:

> All Python code is automatically traced

---

## 9. Real-World Analogy

* **Automatic tracing**: Tracks the flight (takeoff, landing, duration)
* **`@traceable`**: Tracks baggage handling inside the airport

Both are useful, but they observe **different layers**.

---

## 10. When You Should Use What

| Scenario                   | Automatic Tracing | `@traceable` |
| -------------------------- | ----------------- | ------------ |
| Learning LangChain         | Yes               | No           |
| Simple RAG demo            | Yes               | No           |
| Debugging retrieval issues | Yes               | Yes          |
| Production RAG system      | Yes               | Selectively  |
| Performance optimization   | Yes               | Yes          |

---

## 11. Recommended Best Practice

1. Start with **automatic tracing only**
2. Add `@traceable` **only where insight is needed**
3. Avoid over-instrumentation

---

## 12. Final Takeaway

> **Automatic tracing watches LangChain.
> `@traceable` watches your glue code.**

Both together provide full observability — but neither is mandatory for your RAG to work.

---

## License

MIT (use freely in personal or commercial projects)
