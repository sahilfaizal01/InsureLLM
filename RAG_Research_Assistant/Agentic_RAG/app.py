import streamlit as st
from agent import ResearchAssistant
from vector_store import PaperVectorStore

def main():
    st.title("🔬 Research Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'context_loaded' not in st.session_state:
        st.session_state.context_loaded = False
    
    # Initialize vector store and research assistant
    vector_store = PaperVectorStore()
    assistant = ResearchAssistant(vector_store)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What would you like to research?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                response_data = assistant.query(prompt)
                
                # Format response with citations and references
                formatted_response = assistant.format_response_with_citations(
                    response_data['output'], 
                    response_data['papers']
                )
                
                st.markdown(formatted_response)
          
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")
    main()