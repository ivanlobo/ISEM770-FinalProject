import os
import uuid
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt

# Import from the Gemini-specific RAG pipeline
from rag_pipelines import (
    compare_all_rag_fast,
    plot_bertscore_heatmap,
    get_reference_answer,
    update_results_csv,
    # plot_graph_rag # Uncomment if you update graph RAG to return the graph object!
)

st.set_page_config(page_title="FEMA Emergency Planning Assistant (OpenAI)")

st.title("⚡ FEMA Emergency Planning Assistant (OpenAI-powered)")
st.markdown("Ask any FEMA-related question to compare how different RAG strategies respond, now powered by OpenAI.")

folder_path = "data"
query = st.text_input("🔍 Ask a FEMA-related question:")

if st.button("💬 Get Answer") and query:
    with st.spinner("🔄 Processing your query... Please wait."):
        try:
            trace_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            # This will now call the OpenAI-powered RAG functions
            results = compare_all_rag_fast(query, folder_path)

            # Find the RAG strategy with the highest BERTScore F1
            best_rag_strategy = None
            highest_bert_score = -1.0
            
            if "validation" in results and results["validation"]:
                for strat in ["standard", "hierarchical", "graph"]:
                    val = results["validation"].get(strat)
                    if val and val['bert_f1'] > highest_bert_score:
                        highest_bert_score = val['bert_f1']
                        best_rag_strategy = strat

            # Display only the best answer
            if best_rag_strategy:
                st.subheader(f"🏆 Best Answer (based on BERTScore): {best_rag_strategy.capitalize()} RAG")
                st.write(results[best_rag_strategy]["response"])
            else:
                st.warning("⚠️ Could not determine the best answer based on BERTScore.")
                # You might want to fall back to displaying all answers or a default one here.
                # For this update, we will just show a warning.

            # Show confidence/validation scores
            if "validation" in results and results["validation"]:
                st.markdown("### 🔎 Confidence scores (compared to reference answer):")
                validation = results["validation"]
                for strat in ["standard", "hierarchical", "graph"]:
                    val = validation.get(strat, None)
                    if val:
                        st.write(
                            f"**{strat.capitalize()} RAG:** "
                            f"Embedding Sim={val['embedding']:.3f}, "
                            f"BERTScore F1={val['bert_f1']:.3f}"
                        )
                if best_rag_strategy:
                    st.success(f"**Best match:** {best_rag_strategy.capitalize()} RAG (by BERTScore F1).")

            # ---- BERTScore Heatmap ----
            reference_answer = get_reference_answer(query)
            if reference_answer:
                st.markdown("### BERTScore Similarity Heatmap")
                plot_bertscore_heatmap(
                    [reference_answer],
                    [results.get("standard", {}).get("response", "")],
                    [results.get("hierarchical", {}).get("response", "")],
                    [results.get("graph", {}).get("response", "")],
                )
                st.pyplot(plt.gcf())
                plt.clf()

            # ---- Graph RAG Visualization (Optional) ----
            # If your graph RAG returns the actual `graph` object (e.g., in results["graph"]["graph"])
            # from rag_pipelines import plot_graph_rag
            # if "graph" in results.get("graph", {}):
            #    st.markdown("### Graph RAG Knowledge Network")
            #    plot_graph_rag(results["graph"]["graph"])
            #    st.pyplot(plt.gcf())
            #    plt.clf()

            end_time = datetime.now()

            if not os.path.exists("reference/fema_evaluation.csv"):
                st.warning("⚠️ CSV file not found at 'reference/fema_evaluation.csv'")
            else:
                update_results_csv(
                    question=query,
                    results=results,
                    csv_path="reference/fema_evaluation.csv",
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