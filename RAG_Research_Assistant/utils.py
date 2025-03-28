import streamlit as st

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    session_state_keys = {
        "messages": [],
        "llm_chain": None,
        "keywords": [],
        "research_papers": None,
        "session_config": None
    }
    
    for key, default_value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value