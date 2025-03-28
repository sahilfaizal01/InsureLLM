from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

def get_history_prompt():
    """Create prompt for history-aware retrieval."""
    return ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question considering chat history context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def get_main_prompt():
    """Create main conversation prompt with context and citation instructions."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI research paper assistant. "
         "Provide insights from the research papers. "
         "Below are the relevant documents: \n\n{context}\n\n"
         "Use the context to inform your response. "
         "If information is not available in the context, "
         "acknowledge that transparently. "
         "When referencing information from papers, include the citation ID "
         "at the end of the sentence in square brackets. "
         "At the end of your response, include a 'References' section with a numbered list of all cited papers in this format:\n"
         "1. [citation_id] Title by Authors (Year)\n"
         "2. [citation_id] Title by Authors (Year)\n"
         "Make sure each reference is on a new line and properly numbered."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
