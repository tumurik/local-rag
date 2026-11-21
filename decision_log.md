```markdown
# Decision Log: Local RAG Pipeline


## Machine
- **Hardware:** 
CPU - i7-6700K 4GHz, RAM - 16GB, GPU - GTX 1070 (8 GB VRAM), native Ollama, Python 3.12 - uv.
- **`ollama ps` with my main model loaded:** 
NAME          ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL                   
gemma4:e4b    dc35e8d9c606    353 MB    100% GPU     8192       llamacpp    About a minute from now 


## Model Selection 
- **Chosen LLM:** gemma4:e4b
- **Thinking:** low, max tokens = 1024, temperature = 0.3
- **Why:** With same benchmark (low vs none reasoning) all models performed reasonably well. With thinking on - qwen 3.5(4b) reached 1024 token cap on 3/8 prompts, qwen 3.5(2b) reached token cap 6/8, gemma4 - 0/8. So it wasn't reasonable to use qwen with thinking effort, it's only added latency and cut off asnwer. Gemma4 with thinking did not invent answers like qwen3.5(4b) without thinking and in my opinion gave better answers. 

- **What I considered but rejected:** qwen3.5:2b was fastest on short prompts, but worst on answers, qwen3.5:4b without thinking is faster than gemma (4.8 s vs 9.6 s median) with similar level of answers on recursive chunks, k=5, but hallucinated on numbers. If speed becomes a problem, qwen3.5:4b might be the choice.


## Embedding Model
- **Chosen:** nomic-embed-text, 768 dimensions
with search_document/search_query prefixes.
- **Why:** I did not compare with qwen3:0.6B, 1024 dimensions. I decided to go with more compact model. Cosine space; top-1 scores in the section output run range from ~0.66 (q17) to ~0.85 (q13, q20).



## Parsing
- **Extractor:**
Pypdf and cleanup in ingest.py. 
1. rebuild_layout_table - cells wrapped text joined and empty columns separate cells. 
2. flattened pipe tables - replace || with |\n|, to address meeting notes file tables. 
3. heading detection for section strategy.
I put tables in extraction_samples. In plain mode table rows were lost, in layout rows cells remained side by side, but words stick to each other("toHR").


## Chunking Strategy
- **Chunk size:** sections - one chunk per heading section (heading + paragraphs + lists + tables). Splitted above 1500 chars, merged below 200 chars, 50 char overlap when split needed. "Table of Contents" skipped.
ingest_stats_sections.json:
  "chunk_length_chars": {
    "min": 235,
    "median": 688,
    "max": 1658
  }
- **Overlap:** 50 characters
- **Why:** 
The pass counts look similar, but the answers are clearly more complete with sections:
q13 (v3.2 release): both dates (Feb 12 and Jan 28) reported - no recursive run did this.
Cost: prompts are twice longer, but generation only ~0.8s slower.
I think sections retained more context that was not captured by recursive by page chunking.


## Retrieval Configuration
- **Top-K:** 10
- **Context window:** 8192, 
at k=5 were
"prompt_tokens": {
    "min": ~600,
    "median": ~800,
    "max": ~1200
  }
at k=10
"prompt_tokens": {
    "min": 1465,
    "median": 2202,
    "max": 3067
  }
from run info files
- **Why:** I run twice k=5, and amazingly it get 18/20 and 20/20, because of temperature and different answers. But witk k=10, it got 19/20, I prefer k=10, beacuse i still got enough context window(in worst case 3067/8192), generation time does not change. But i will fix more retrieval errors like on q08 - chunks with other phases were given at rank 6.


## Observations
- **What worked well:** 
1. Refusals worked every time (q19,q20).
2. q15 false premise was corrected in all gemma runs
3. Factual questions are also mostly done correct, where information was provided
- **What failed:** 
1. Some conflictiong sources were the problem -> q14, nowhere was mentioned other price 59 eur in gemma runs, though qwen3.5:4b got it.
2. q8 still got problem with not all phases, section with phases was splitted poorly and even with k=10 it struggles to deliver needed context.
- **Traps I hit:**
1. I tried qwen with thinking(just wanted to see outputs), it return a lot of empty answers, because wasted too much tokens on thinking
2. Rerunning k=5 showed that because of temperature, i could get everything correct/partly correct because of temperature, need to run more times to see mean pass rate.
3. MAX_TOKENS was barely enough for q17 for thinking and answering, better t0 go with 2048.
4. Recursive splitting created 1 char chunk. 
5. pypdf plain mode lost table rows and layout mode stick words together
- **Local vs cloud expectations:** 
I think cloud will do better on conflicting sources and probably be faster.


## If I Had More Time / Better Hardware
1. Raise MAX_TOKENS if i will use thinking.
2. Repeat tests with same configurations to see variance in answers.
3. Look for more tricky questions, because who knows what i've missed, maybe model will be disaster for real life question formulations and other questions types. 
4. Implementation of reranker or hybrid search would've helped in q3.
5. Larger models will reduce omissions and larger context will allow larger top k.
6. Compare  different embedding models.
