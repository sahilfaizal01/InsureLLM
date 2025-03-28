from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from prompt_templates import get_history_prompt, get_main_prompt
from langchain_core.prompts import PromptTemplate

def get_session_history(session_id):
    """Manage session chat history."""
    return ChatMessageHistory()

def create_conversation_chain(retriever, api_key):
    llm = ChatCohere(
        api_key=api_key,
        model="command-r-plus-08-2024",
        max_tokens=300,
        temperature=0.6
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, get_history_prompt()
    )
    
    # Create a prompt that includes citation formatting
    document_prompt = PromptTemplate.from_template(
    "{page_content}\nCitation: {citation_id} {title} by {authors} ({year})"
    )
    
    document_chain = create_stuff_documents_chain(
        llm, 
        get_main_prompt(),
        document_prompt=document_prompt,
        document_separator="\n\n"
    )
    
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, document_chain)
    
    conversational_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_chain
