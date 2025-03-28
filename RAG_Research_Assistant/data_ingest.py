# data_ingest.py
import streamlit as st
import feedparser
from urllib.parse import quote
from langchain.schema import Document

def manage_keywords():
    """Manage keyword input and display in sidebar."""
    st.sidebar.header("🔍 Research Paper Search")
    
    # Keyword input
    new_keyword = st.sidebar.text_input(
        "Enter research keywords:",
        placeholder='E.g., machine learning transformers',
        key="keyword_input"
    )
    
    # Add keyword button (implicit)
    if new_keyword and new_keyword not in st.session_state.keywords:
        st.session_state.keywords.append(new_keyword)
    
    # Display current keywords
    if st.session_state.keywords:
        st.sidebar.write("#### Selected Keywords:")
        for keyword in st.session_state.keywords:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(f"- {keyword}")
            with col2:
                if st.button(f"❌", key=f"remove_{keyword}"):
                    st.session_state.keywords.remove(keyword)
                    st.experimental_rerun()

def fetch_research_papers(keywords):
    """Fetch research papers from ArXiv based on keywords."""
    if not keywords:
        st.warning("Please enter at least one keyword.")
        return None
    
    quoted_keywords = [quote(kw) for kw in keywords]
    query = "+AND+".join([f"abs:{quote(keyword)}" for keyword in quoted_keywords])
    
    arxiv_url = (
        f'http://export.arxiv.org/api/query?search_query={query}'
        '&start=0&max_results=50&sortBy=lastUpdatedDate&sortOrder=descending'
    )
    
    feed = feedparser.parse(arxiv_url)
    return feed.entries

def transform_papers_to_documents(papers):
    """Transform raw paper data into LangChain Documents with citation information."""
    if not papers:
        return []
    
    return [
        Document(
            page_content=f"Title: {paper.title}\nAbstract: {paper.summary}".lower(),
            metadata={
                "link": paper.link,
                "title": paper.title,
                "authors": ', '.join([author.get('name', '') for author in getattr(paper, 'authors', [])]),
                "year": getattr(paper, 'published', '').split('-')[0] if hasattr(paper, 'published') else '',
                "citation_id": f"[{i+1}]"
            }
        ) for i, paper in enumerate(papers)
    ]
