# AI Research Assistant

An intelligent research assistant that helps users explore academic papers through natural language conversations. This system leverages Retrieval-Augmented Generation (RAG) to provide accurate, citation-backed responses to research queries.

## Overview

The AI Research Assistant is designed to streamline the academic research process by allowing users to interact with research papers through a conversational interface. The system retrieves relevant papers based on keywords, processes them, and enables users to ask questions about the content with responses grounded in the source material.

## Features

- **Paper Retrieval**: Search and fetch papers from ArXiv based on keywords
- **Intelligent Conversations**: Ask questions about papers with context-aware responses
- **Citation Support**: All responses include citations to source materials
- **Evaluation Framework**: Built-in metrics to evaluate system performance
- **User-Friendly Interface**: Clean Streamlit interface for seamless interaction

## System Architecture

The system is built on the following technologies:
- **Frontend**: Streamlit
- **LLM Integration**: Cohere's command-r-plus model
- **Embedding**: Cohere Embeddings
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Data Source**: ArXiv API

## Installation
# Clone the repository
git clone https://github.com/sahilfaizal01/RagHub.git
cd RAG_Research_Assistant

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## Usage
1. Set up your API keys in a `.env` file: COHERE_API_KEY=your_cohere_api_key
2. Run the application: streamlit run app.py
3. In the web interface:
   - Enter research keywords to fetch relevant papers
   - Ask questions about the papers
   - View responses with citations
   - Evaluate system performance

## Evaluation

The system includes a built-in evaluation framework using RAGAS metrics:
- **Faithfulness**: Measures if responses are factually consistent with source documents
- **Relevancy**: Evaluates if responses address the query
- **Context Precision/Recall**: Assesses the quality of retrieved contexts

## Requirements

- Python 3.8+
- Cohere API key
- Internet connection for ArXiv API access

## Future Improvements

- Support for additional academic databases
- Enhanced document processing for tables and figures
- Multi-modal capabilities for diagrams and visual content
- Collaborative research sessions
- Export functionality for research findings

## License

[MIT License](LICENSE)

## Acknowledgments

- [Cohere](https://cohere.com/) for LLM capabilities
- [LangChain](https://www.langchain.com/) for the RAG framework
- [ArXiv](https://arxiv.org/) for the research paper database
- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation metrics





