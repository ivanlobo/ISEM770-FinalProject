import os
import uuid
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import torch
from bert_score import score as bertscore_score

# Import from the hybrid RAG pipeline
from hybrid_pipelines import (
    hybrid_rag,
    get_reference_answer,
    update_results_csv,
    # plot_graph_rag # Uncomment if you update graph RAG to return the graph object!
)

st.set_page_config(page_title="FEMA Emergency Planning Assistant (Hybrid)")

st.title("⚡ FEMA Emergency Planning Assistant (Hybrid)")
st.markdown("Ask any FEMA-related question to get a response from a Hybrid RAG strategy.")

folder_path = "data"
query = st.text_input("🔍 Ask a FEMA-related question:")

if st.button("💬 Get Answer") and query:
    with st.spinner("🔄 Processing your query... Please wait."):
        try:
            trace_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            # Call the hybrid RAG function
            rag_result = hybrid_rag(query, folder_path)
            
            # Display the single response from the hybrid RAG pipeline
            if rag_result and "response" in rag_result:
                st.subheader("Hybrid RAG Answer:")
                st.write(rag_result["response"])

                # Get the reference answer
                reference_answer = get_reference_answer(query)
                
                # If a reference answer exists, calculate and display BERTScore
                if reference_answer:
                    st.markdown("---")
                    st.subheader("🔎 Validation Scores:")
                    
                    # Calculate BERTScore
                    P, R, F1 = bertscore_score(
                        [rag_result["response"]],
                        [reference_answer],
                        lang="en",
                        verbose=True
                    )
                    
                    st.write(f"**BERTScore F1:** {F1.item():.3f}")

                    # Format the results for the CSV update
                    # This dictionary structure mirrors what compare_all_rag_fast would produce
                    results = {
                        "hybrid": rag_result,
                        "validation": {
                            "hybrid": {
                                "bert_f1": F1.item(),
                                # Other metrics can be added here if needed
                            }
                        }
                    }
                    
                    st.success("✅ Response generated and validated.")
                else:
                    st.warning("⚠️ No reference answer found for this query. BERTScore could not be calculated.")
                    results = {"hybrid": rag_result}
            else:
                st.warning("⚠️ The hybrid RAG pipeline did not return a response.")
                results = {"hybrid": None}

            end_time = datetime.now()

            # Pass the structured results to the CSV update function
            if not os.path.exists("reference/fema_evaluation_hybrid.csv"):
                st.warning("⚠️ CSV file not found at 'reference/fema_evaluation_hybrid.csv'")
            else:
                update_results_csv(
                    question=query,
                    results=results,
                    csv_path="reference/fema_evaluation_hybrid.csv",
                    trace_id=trace_id,
                    start_time=start_time,
                    end_time=end_time
                )
            
            st.success("✅ All responses generated.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

if st.button("🛑 Shutdown App"):
    st.write("Shutting down...")
    os._exit(0)
