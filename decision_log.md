```markdown
# Decision Log: Local RAG Pipeline


## Machine
- **Hardware:** 
CPU - i7-6700K 4GHz, RAM-16GB, GTX 1070, native Ollama.
- **`ollama ps` with my main model loaded:** 
NAME                 ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL              
qwen3.5:4b-q4_K_M    d8b0f5e9760c    3.4 GB    100% GPU     8192       llamacpp    4 minutes from now


## Model Selection 
- **Chosen LLM:** [exact model tag]
- **Thinking:** [on or off, and why]
- **Why:** [Reference benchmark_results.json. What did you compare? What made you choose this one?]
- **What I considered but rejected:** [Other models and why they didn't work]


## Embedding Model
- **Chosen:** [model name, vector dimensions]
- **Why:** [Did you compare? What was the tradeoff?]


## Parsing
- **Extractor:** [which, and what one page with a table looked like after extraction]


## Chunking Strategy
- **Chunk size:** [value and unit: characters or tokens]
- **Overlap:** [value and unit]
- **Why:** [Did you experiment? What happened with different values?]


## Retrieval Configuration
- **Top-K:** [value]
- **Context window:** [value from `ollama ps`; prompt_tokens you observed at this K]
- **Why:** [Did you try different values? What was the impact?]


## Observations
- **What worked well:** [Specific examples from your test questions, with question ids from pipeline_outputs.json]
- **What failed:** [Specific examples: which questions did the system get wrong and why?]
- **Traps I hit:** [What you saw, what you did]
- **Local vs cloud expectations:** [How do you think this would compare to a cloud model on the same task?]


## If I Had More Time / Better Hardware
- [What would you change?]
```