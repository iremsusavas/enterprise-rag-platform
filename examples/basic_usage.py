"""
Basic usage examples for Enterprise RAG Platform
"""
from rag_engine import RAGEngine
from ingestion.document_loader import DocumentLoader
import os


def ingest_documents():
    """Example: Ingest documents into the system"""
    print("📄 Document Ingestion Example")
    print("-" * 50)
    
    engine = RAGEngine()
    loader = DocumentLoader()
    
    # Load policy documents
    if os.path.exists("data/policy"):
        print("Loading policy documents...")
        documents = loader.load_directory("data/policy", doc_type="policy")
        print(f"Loaded {len(documents)} documents")
        
        num_chunks = engine.ingest_documents(documents, doc_type="policy")
        print(f"✅ Ingested {num_chunks} chunks into policy index")
    
    # Load legal documents
    if os.path.exists("data/legal"):
        print("\nLoading legal documents...")
        documents = loader.load_directory("data/legal", doc_type="legal")
        print(f"Loaded {len(documents)} documents")
        
        num_chunks = engine.ingest_documents(documents, doc_type="legal")
        print(f"✅ Ingested {num_chunks} chunks into legal index")
    
    # Load technical documents
    if os.path.exists("data/technical"):
        print("\nLoading technical documents...")
        documents = loader.load_directory("data/technical", doc_type="technical")
        print(f"Loaded {len(documents)} documents")
        
        num_chunks = engine.ingest_documents(documents, doc_type="technical")
        print(f"✅ Ingested {num_chunks} chunks into technical index")


def query_example():
    """Example: Query the RAG system"""
    print("\n💬 Query Example")
    print("-" * 50)
    
    engine = RAGEngine()
    
    queries = [
        "What is our company's vacation policy?",
        "What are the terms of the service agreement?",
        "How do I authenticate using the API?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        result = engine.query(query, k=5, evaluate=True)
        
        # Display routing
        routing = result["routing"]
        print(f"📍 Routed to: {routing['selected_index']} index")
        print(f"   Reason: {routing.get('reason', 'N/A')}")
        print(f"   Confidence: {routing.get('confidence', 0.0):.2f}")
        
        # Display answer
        print(f"\n📝 Answer:")
        print(result["answer"])
        
        # Display evaluation
        if result["evaluation"]:
            eval_result = result["evaluation"]
            print(f"\n⚖️ Evaluation:")
            print(f"   Faithfulness: {eval_result['faithfulness']:.2f}/5.0")
            print(f"   Completeness: {eval_result['completeness']:.2f}/5.0")
            print(f"   Hallucination: {eval_result['hallucination']:.2f}/5.0")
            print(f"   Overall Score: {eval_result['overall_score']:.2f}/5.0")
        
        # Display sources
        print(f"\n📚 Sources ({len(result['sources'])} chunks):")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"   {i}. {source['source']} (score: {source['score']:.4f})")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    print("🧠 Enterprise RAG Platform - Example Usage\n")
    
    try:
        # Uncomment to run ingestion (requires data directory)
        # ingest_documents()
        
        # Run query example
        query_example()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

