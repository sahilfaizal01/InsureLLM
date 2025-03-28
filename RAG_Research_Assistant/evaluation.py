import os
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas import evaluate
import pandas as pd
import streamlit as st

def run_ragas_evaluation(questions, answers, contexts, ground_truths=None):
    """
    Run RAGAS evaluation on the RAG system outputs.
    
    Args:
        questions (list): List of user questions
        answers (list): List of generated answers from the RAG system
        contexts (list): List of lists containing retrieved contexts for each question
        ground_truths (list, optional): List of correct answers for reference
    
    Returns:
        dict: Dictionary containing evaluation scores
    """
    # Prepare dataset
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    # Add ground truths if available
    if ground_truths:
        eval_data["ground_truth"] = ground_truths
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    else:
        # Use metrics that don't require ground truth
        metrics = [faithfulness, answer_relevancy, context_precision]
    
    # Convert to Dataset format
    dataset = Dataset.from_dict(eval_data)
    
    # Run evaluation
    scores = evaluate(
        dataset,
        metrics=metrics,
    )
    
    return scores

def display_evaluation_results(scores):
    st.subheader("RAG System Evaluation Results")
    
    # Convert EvaluationResult to DataFrame
    scores_df = scores.to_pandas()
    
    # Calculate averages only for numeric columns
    avg_scores = {}
    for col in scores_df.columns:
        if col not in ['question', 'answer', 'contexts', 'ground_truth']:
            try:
                numeric_values = pd.to_numeric(scores_df[col], errors='coerce')
                avg_scores[col] = numeric_values.mean()
            except:
                st.warning(f"Could not calculate average for column '{col}' - contains non-numeric values")

    # Create a DataFrame for display
    if avg_scores:
        avg_scores_df = pd.DataFrame({metric: [score] for metric, score in avg_scores.items()})
        
        # Display scores
        st.dataframe(avg_scores_df)
        
        # Create bar chart
        chart_data = pd.DataFrame({
            'Metric': list(avg_scores.keys()),
            'Score': list(avg_scores.values())
        })
        
        st.subheader("Metrics Visualization")
        st.bar_chart(chart_data.set_index('Metric'))
    else:
        st.warning("No numeric metrics available to display")
    
    # Show detailed results
    st.subheader("Detailed Results")
    st.dataframe(scores_df)



def save_evaluation_data(questions, answers, contexts, ground_truths=None):
    data = {
        "question": questions,
        "answer": answers,
    }
    
    # Add contexts as separate columns
    max_contexts = max(len(ctx_list) for ctx_list in contexts)
    for i in range(max_contexts):
        data[f"context_{i+1}"] = [
            ctx_list[i] if i < len(ctx_list) else "" 
            for ctx_list in contexts
        ]
    
    if ground_truths:
        data["ground_truth"] = ground_truths
    
    df = pd.DataFrame(data)
    df.to_csv("rag_evaluation_data.csv", index=False)
    st.success("Evaluation data saved to rag_evaluation_data.csv")
