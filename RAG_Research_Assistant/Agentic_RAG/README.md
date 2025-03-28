# 🔬 Research Assistant
A Streamlit-based application that helps researchers find and understand scholarly information using LangChain, Semantic Scholar API, and vector search.

## Overview
This Research Assistant application provides an interactive chat interface where users can ask research questions and receive comprehensive answers with citations from academic papers. The system leverages the Semantic Scholar API to search for relevant papers and stores them in a vector database for efficient retrieval.

## Features
* Interactive Chat Interface: Ask research questions in natural language
* Academic Paper Search: Searches Semantic Scholar for relevant papers
* Citation Management: Automatically formats responses with proper citations
* Vector Search: Stores papers in a vector database for efficient retrieval
* Conversation Memory: Maintains context throughout the research session

## Components
The application consists of four main components:
* Streamlit Interface (app.py): Provides the web interface for interacting with the research assistant
* Research Agent (agent.py): Handles queries using LangChain and the Semantic Scholar API
* Vector Store (vector_store.py): Manages the storage and retrieval of academic papers
* Requirements (requirements.txt): Lists all necessary dependencies

## Installation
* Clone this repository
* Install the required dependencies: pip install -r requirements.txt
* Create a .env file with your OpenAI API key: OPENAI_API_KEY=your_api_key_here

## Usage
* Run the Streamlit application: streamlit run app.py
* Then open your browser and navigate to the provided URL

## Dependencies
* langchain & langchain-community
* streamlit
* python-dotenv
* openai
* semanticscholar
* chromadb
* sentence-transformers
* langchain_openai
* transformers

## How It Works
* User enters a research query in the chat interface
* The Research Assistant processes the query using GPT-4
* Semantic Scholar API is used to find relevant academic papers
* Papers are stored in a ChromaDB vector database
* Response is formatted with proper citations and references 
* Conversation history is maintained for context



