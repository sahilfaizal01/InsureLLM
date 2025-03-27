# InsureLLM
LLM-RAG solutions for Financial Services

## Tools Used:
* Python (Programming Language)
* LangChain (Framework)
* OpenAI GPT 4o Mini (LLM)
* Gradio (UI)
* Chroma DB (Vector Database)

## Error Resolution in RAG:
Evaluate Chunking Strategy is the key
* Send entire documents as a context
* Have larger/smaller chunk sizes
* Increase/Decrease overlap

## 1. Simple RAG

Here, I was able to resolve the retrieval error by changing the number of retrieved chunks to 25 compared to the default.
### Output using Default Config
![image](https://github.com/user-attachments/assets/83fe16f5-665e-4436-a994-e20142f4a0b7)

### Output after Refinement
![image](https://github.com/user-attachments/assets/1d9ca377-b498-4b6b-9fa5-14dbbda5a586)
