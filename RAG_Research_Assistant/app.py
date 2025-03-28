import streamlit as st
from data_ingest import fetch_research_papers, manage_keywords
from retriever import prepare_document_retrieval
from chain_builder import create_conversation_chain
from utils import initialize_session_state
from dotenv import load_dotenv
import os
import pandas as pd
from evaluation import run_ragas_evaluation, display_evaluation_results, save_evaluation_data

load_dotenv()

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Research Paper Insights", page_icon="📚")
    st.title("🔬 AI Research Assistant")
    
    initialize_session_state()
    
    # API Key Input
    api_key = os.getenv("COHERE_API_KEY")
    
    # Keyword Management
    manage_keywords()
    
    # Fetch and Process Papers
    col1, col2 = st.columns(2)
    with col1:
        if st.sidebar.button("Fetch Papers", type="primary"):
            papers = fetch_research_papers(st.session_state.keywords)
            retriever = prepare_document_retrieval(papers)
            
            if retriever:
                st.session_state.llm_chain = create_conversation_chain(retriever, api_key)
                st.session_state.research_papers = papers
                st.sidebar.success(f"Fetched {len(papers)} research papers!")
    
    with col2:
        if st.sidebar.button("App Reset", type="secondary"):
            st.session_state.keywords = []
            st.session_state.llm_chain = None
            st.rerun()
    
    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the research papers..."):
        if not st.session_state.llm_chain:
            st.warning("Fetch papers first!")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Analyzing papers..."):
            # Configure session for conversation
            config = {"configurable": {"session_id": "0"}}
            
            # Invoke with full input dictionary
            response = st.session_state.llm_chain.invoke(
                {"input": prompt},
                config=config
            )["answer"]

        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
    
    # Add a tab for evaluation
    tab1, tab2 = st.tabs(["Research Paper Q&A Assistant", "System Evaluation"])
    
    with tab1:
        # Move your existing Q&A interface here
        st.title("Research Paper Q&A Assistant")
    
    with tab2:
        st.title("RAG System Evaluation")
        
        # Initialize session state for evaluation
        if "eval_questions" not in st.session_state:
            st.session_state.eval_questions = []
            st.session_state.eval_answers = []
            st.session_state.eval_contexts = []
            st.session_state.eval_ground_truths = []
        
        # Evaluation options
        eval_mode = st.radio(
            "Evaluation Mode",
            ["Manual Entry", "Upload Test Set"]
        )
        
        if eval_mode == "Manual Entry":
            with st.form("manual_eval_form"):
                question = st.text_area("Question")
                ground_truth = st.text_area("Ground Truth (Optional)")
                
                submitted = st.form_submit_button("Add to Evaluation Set")
                if submitted and question:
                    # Process the question through your RAG system
                    if "llm_chain" in st.session_state and st.session_state.llm_chain:
                        with st.spinner("Generating answer..."):
                            response = st.session_state.llm_chain.invoke(
    {"input": question},
    config={"configurable": {"session_id": "evaluation"}}
)

                            
                            answer = response["answer"]
                            contexts = [doc.page_content for doc in response.get("context", [])]
                            
                            # Add to evaluation set
                            st.session_state.eval_questions.append(question)
                            st.session_state.eval_answers.append(answer)
                            st.session_state.eval_contexts.append(contexts)
                            if ground_truth:
                                st.session_state.eval_ground_truths.append(ground_truth)
                            else:
                                st.session_state.eval_ground_truths.append(None)
                            
                            st.success("Added to evaluation set!")
                    else:
                        st.error("RAG system not initialized. Please set up the system first.")
        
        elif eval_mode == "Upload Test Set":
            uploaded_file = st.file_uploader("Upload evaluation dataset (CSV)", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if "question" in df.columns:
                    if st.button("Process Test Set"):
                        with st.spinner("Processing test set..."):
                            for _, row in df.iterrows():
                                question = row["question"]
                                ground_truth = row.get("ground_truth", None)
                                
                                # Process through RAG system
                                if "llm_chain" in st.session_state and st.session_state.llm_chain:
                                    response = st.session_state.llm_chain.invoke({
                                        "input": question,
                                        "session_id": "evaluation"
                                    })
                                    
                                    answer = response["answer"]
                                    contexts = [doc.page_content for doc in response.get("context", [])]
                                    
                                    # Add to evaluation set
                                    st.session_state.eval_questions.append(question)
                                    st.session_state.eval_answers.append(answer)
                                    st.session_state.eval_contexts.append(contexts)
                                    st.session_state.eval_ground_truths.append(ground_truth)
                            
                            st.success(f"Processed {len(df)} questions!")
                else:
                    st.error("CSV must contain a 'question' column")
        
        # Display evaluation set
        if st.session_state.eval_questions:
            st.subheader(f"Evaluation Set ({len(st.session_state.eval_questions)} questions)")
            
            for i, (q, a, c) in enumerate(zip(
                st.session_state.eval_questions,
                st.session_state.eval_answers,
                st.session_state.eval_contexts
            )):
                with st.expander(f"Q{i+1}: {q[:50]}..."):
                    st.write("**Question:**", q)
                    st.write("**Answer:**", a)
                    st.write("**Contexts:**")
                    for j, context in enumerate(c):
                        st.text(f"Context {j+1}: {context[:100]}...")
            
            # Run evaluation
            if st.button("Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    # Check if we have ground truths for all questions
                    if all(gt is not None for gt in st.session_state.eval_ground_truths) and len(st.session_state.eval_ground_truths) == len(st.session_state.eval_questions):
                        scores = run_ragas_evaluation(
                            st.session_state.eval_questions,
                            st.session_state.eval_answers,
                            st.session_state.eval_contexts,
                            st.session_state.eval_ground_truths
                        )
                    else:
                        # Run without ground truths
                        scores = run_ragas_evaluation(
                            st.session_state.eval_questions,
                            st.session_state.eval_answers,
                            st.session_state.eval_contexts
                        )
                    
                    display_evaluation_results(scores)
                    
                    # Option to save evaluation data
                    if st.button("Save Evaluation Data"):
                        save_evaluation_data(
                            st.session_state.eval_questions,
                            st.session_state.eval_answers,
                            st.session_state.eval_contexts,
                            st.session_state.eval_ground_truths if all(gt is not None for gt in st.session_state.eval_ground_truths) else None
                        )

if __name__ == "__main__":
    main()
