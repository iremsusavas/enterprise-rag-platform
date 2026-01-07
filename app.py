"""
Streamlit Demo Application
Simple UI for the Enterprise RAG Platform
"""
import streamlit as st
import os
from rag_engine import RAGEngine
from ingestion.document_loader import DocumentLoader
import config

# Page config
st.set_page_config(
    page_title="Enterprise RAG Platform",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "rag_engine" not in st.session_state:
    try:
        st.session_state.rag_engine = RAGEngine()
        st.session_state.engine_ready = True
    except Exception as e:
        st.session_state.engine_ready = False
        st.session_state.engine_error = str(e)

if "query_history" not in st.session_state:
    st.session_state.query_history = []


def main():
    st.title("🧠 Enterprise RAG Platform")
    st.markdown("**Multi-Index RAG with Intelligent Query Routing & Evaluation**")
    
    # Sidebar for document ingestion
    with st.sidebar:
        st.header("📄 Document Ingestion")
        
        doc_type = st.selectbox(
            "Document Type",
            ["policy", "legal", "technical"],
            help="Select the type of documents you're uploading"
        )
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True
        )
        
        if st.button("Ingest Documents") and uploaded_files:
            if not st.session_state.engine_ready:
                st.error("RAG engine not initialized. Check API key.")
            else:
                with st.spinner("Processing documents..."):
                    loader = DocumentLoader()
                    documents = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_path = f"/tmp/{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            doc = loader.load_document(temp_path, doc_type)
                            documents.append(doc)
                            os.remove(temp_path)
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                    
                    if documents:
                        try:
                            num_chunks = st.session_state.rag_engine.ingest_documents(documents, doc_type)
                            st.success(f"✅ Ingested {len(documents)} documents ({num_chunks} chunks) into {doc_type} index")
                        except Exception as e:
                            st.error(f"Error ingesting documents: {e}")
        
        st.divider()
        st.markdown("### 📊 Index Statistics")
        
        if st.session_state.engine_ready:
            for index_name in ["policy", "legal", "technical"]:
                stats = st.session_state.rag_engine.vector_stores[index_name].get_stats()
                st.metric(
                    label=f"{index_name.capitalize()} Index",
                    value=f"{stats['total_vectors']} vectors"
                )
    
    # Main content area
    if not st.session_state.engine_ready:
        st.error("⚠️ RAG Engine not initialized")
        st.error(f"Error: {st.session_state.get('engine_error', 'Unknown error')}")
        st.info("💡 Make sure to set OPENAI_API_KEY in your environment or .env file")
        return
    
    # Query input
    st.header("💬 Ask a Question")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'What is our company's vacation policy?' or 'What are the API endpoints for user authentication?'",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        evaluate = st.checkbox("Run Evaluation", value=True)
    
    with col2:
        k = st.slider("Number of chunks to retrieve", 3, 10, 5)
    
    if st.button("🔍 Query", type="primary") and query:
        with st.spinner("Processing query..."):
            try:
                result = st.session_state.rag_engine.query(query, k=k, evaluate=evaluate)
                
                # Display routing decision
                st.subheader("🎯 Query Routing")
                routing = result["routing"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Selected Index", routing["selected_index"].upper())
                with col2:
                    st.metric("Confidence", f"{routing.get('confidence', 0.0):.2f}")
                with col3:
                    st.write("**Reason:**", routing.get("reason", "N/A"))
                
                # Display answer
                st.subheader("📝 Answer")
                st.write(result["answer"])
                
                # Display sources
                with st.expander("📚 Sources Used", expanded=False):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Chunk {i}** (Score: {source['score']:.4f})")
                        st.caption(f"Source: {source['source']}")
                        st.caption(f"ID: {source['chunk_id']}")
                
                # Display evaluation
                if result["evaluation"]:
                    st.subheader("⚖️ LLM-Judge Evaluation")
                    eval_result = result["evaluation"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Faithfulness", f"{eval_result['faithfulness']:.2f}/5.0")
                    with col2:
                        st.metric("Completeness", f"{eval_result['completeness']:.2f}/5.0")
                    with col3:
                        st.metric("Hallucination", f"{eval_result['hallucination']:.2f}/5.0")
                    with col4:
                        overall = eval_result['overall_score']
                        st.metric("Overall Score", f"{overall:.2f}/5.0")
                    
                    if overall >= 4.0:
                        st.success("✅ High quality response")
                    elif overall >= 3.0:
                        st.warning("⚠️ Acceptable response")
                    else:
                        st.error("❌ Low quality response - review needed")
                    
                    with st.expander("📋 Evaluation Reasoning", expanded=False):
                        st.write(eval_result.get("reasoning", "No reasoning provided"))
                
                # Save to history
                st.session_state.query_history.append({
                    "query": query,
                    "result": result
                })
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)
    
    # Query history
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Query History")
        
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {entry['query'][:50]}..."):
                st.write("**Answer:**", entry["result"]["answer"][:200] + "...")
                if entry["result"]["evaluation"]:
                    st.write("**Score:**", f"{entry['result']['evaluation']['overall_score']:.2f}/5.0")


if __name__ == "__main__":
    main()

