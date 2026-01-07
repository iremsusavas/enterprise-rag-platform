# 🧠 Enterprise RAG Platform

> **Production-grade RAG system with multi-index routing, domain-aware embeddings, and automated quality evaluation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Enterprise RAG Platform is a sophisticated retrieval-augmented generation system designed to solve real-world enterprise challenges:

- **Multi-document type support** (policy, legal, technical) with specialized handling
- **Intelligent query routing** using LLM agents for index selection
- **Domain-aware embeddings** with separate models per document type
- **Hallucination-resistant prompting** with strict context adherence
- **Automated quality evaluation** via LLM-Judge pipeline

### Key Differentiators

Unlike traditional RAG systems that dump everything into a single index, this platform:

1. **Routes queries intelligently** to domain-specific indexes
2. **Uses optimal embeddings** for each document type
3. **Prevents hallucination** through strict prompting and evaluation
4. **Measures quality** automatically on every response

## 🏗️ Architecture

```
User Query
    ↓
Query Router Agent (LLM-based decision)
    ↓
┌───────────────┬─────────────────┬──────────────────┐
│ Policy Index  │ Legal Index     │ Technical Index  │
│ (embed A)     │ (embed B)       │ (embed C)        │
└───────────────┴─────────────────┴──────────────────┘
    ↓
Retrieved Chunks (FAISS)
    ↓
Strict RAG Prompt (hallucination-resistant)
    ↓
LLM Generation (Open-source)
    ↓
LLM-Judge Evaluation
    ↓
Final Answer + Quality Score
```

### Design Principles

1. **Index-Aware Routing**: Query router selects appropriate index before retrieval
2. **Multi-Embedding Strategy**: Each index uses its optimal embedding model
3. **Semantic Coordinate Alignment**: Query embedded using selected index's model
4. **Strict Prompting**: Explicit rules prevent hallucination
5. **Automated Evaluation**: Every response scored for quality

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/enterprise-rag-platform.git
cd enterprise-rag-platform
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
# Optional: Override default LLM model
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

### Basic Usage

```python
from rag_engine import RAGEngine
from ingestion.document_loader import DocumentLoader

# Initialize engine
engine = RAGEngine()

# Load and ingest documents
loader = DocumentLoader()
documents = loader.load_directory("data/policy", doc_type="policy")
engine.ingest_documents(documents, doc_type="policy")

# Query
result = engine.query("What is our vacation policy?", k=5, evaluate=True)
print(result["answer"])
print(f"Quality Score: {result['evaluation']['overall_score']}/5.0")
```

### Demo UI

```bash
streamlit run app.py
```

## 📦 Project Structure

```
enterprise-rag-platform/
├── ingestion/          # Document loading (PDF, DOCX, TXT, MD)
├── chunking/           # Domain-aware chunkers
│   ├── policy_chunker.py
│   ├── legal_chunker.py
│   └── technical_chunker.py
├── embeddings/         # Multi-embedding strategy
├── vector_db/          # FAISS vector store
├── agents/             # Query router agent
├── llm/                # Open-source LLM client
├── prompts/            # Strict RAG prompts
├── evaluation/         # LLM-Judge pipeline
├── app.py              # Streamlit demo
├── rag_engine.py       # Main orchestration
└── config.py           # Configuration
```

## 🧩 Core Modules

### 1. Intelligent Query Router

Uses an LLM agent to analyze queries and route them to the correct index (policy, legal, or technical).

**Key Insight**: Using LLM not for generation, but for intelligent decision-making.

```python
{
  "selected_index": "legal",
  "reason": "The query refers to contractual obligations and compliance terms.",
  "confidence": 0.92
}
```

### 2. Advanced Chunking Engine

Document-aware chunking strategies:

- **Policy Docs**: Section-based chunking (500 chars, 50 overlap)
- **Legal Docs**: Semantic + larger chunks (1000 chars, 200 overlap)
- **Technical Docs**: Function/heading-based (800 chars, 100 overlap)

### 3. Multi-Embedding Strategy

| Index | Embedding Model | Rationale |
|-------|----------------|-----------|
| Policy | `all-mpnet-base-v2` | General purpose, strong semantic |
| Legal | `multi-qa-mpnet-base-dot-v1` | QA-optimized for legal Q&A |
| Technical | `all-mpnet-base-v2` | Good for technical/API text |

**Critical Rule**: Query is embedded using the selected index's embedding model for semantic coordinate alignment.

### 4. Vector Database Layer

- **FAISS** (local, fast, free) - Default implementation
- **Metadata filtering** - Filter by doc_type, language, source
- **Hybrid search ready** - Architecture supports Weaviate/Pinecone

### 5. Strict RAG Prompt

Hallucination-resistant prompting with explicit rules:

```
CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't know"
3. Do NOT make up information
4. Cite which chunks you used
```

### 6. LLM-Judge Evaluation Pipeline

Automated quality scoring on every response:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is answer faithful to context? (1-5) |
| **Completeness** | Does answer fully address query? (1-5) |
| **Hallucination** | Contains fabricated info? (1-5) |

## 🔧 Configuration

Edit `config.py` to customize:

- **Embedding models** per index
- **Chunk sizes** and overlap
- **LLM model** and temperature
- **Vector DB** type
- **Evaluation thresholds**

## 📊 Evaluation Metrics

The system automatically evaluates every response:

- **Faithfulness ≥ 4.0**: Answer is faithful to context
- **Completeness ≥ 4.0**: Query fully addressed
- **Hallucination ≤ 2.0**: Minimal fabricated content
- **Overall ≥ 4.0**: High quality response

## 🎓 Key Design Decisions

### Why Multiple Indexes?

**Problem**: Mixing policy, legal, and technical docs in one index causes:
- Semantic confusion (similar words, different contexts)
- Suboptimal chunking (legal needs larger chunks)
- Wrong embedding models (legal needs higher capacity)

**Solution**: Separate indexes with specialized chunking and embeddings.

### Why Agent-Based Routing?

**Problem**: Keyword matching fails for semantic queries like "What are my obligations?"

**Solution**: LLM agent understands intent and routes intelligently.

### Why Strict Prompting?

**Problem**: LLMs hallucinate when context is insufficient.

**Solution**: Explicit instructions + "I don't know" option + source citation.

### Why LLM-Judge Evaluation?

**Problem**: RAG quality is subjective and hard to measure.

**Solution**: Automated evaluation on every response with explainable scores.

## ⚖️ Trade-offs

### What We Gained

✅ **Accuracy**: Index-aware routing improves retrieval precision  
✅ **Quality Control**: Automated evaluation catches bad responses  
✅ **Scalability**: Separate indexes can scale independently  
✅ **Maintainability**: Clear separation of concerns

### What We Trade

❌ **Complexity**: More moving parts than simple RAG  
❌ **Cost**: Multiple embedding models + LLM calls for routing/evaluation  
❌ **Latency**: Routing + evaluation add ~1-2 seconds  
❌ **Storage**: Multiple indexes require more disk space

**Verdict**: For enterprise use, the accuracy and quality gains justify the complexity.

## 🔮 Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Multi-language support with language detection
- [ ] Re-ranking with cross-encoder models
- [ ] Query expansion and reformulation
- [ ] Weaviate/Pinecone integration for production
- [ ] Batch evaluation and A/B testing framework
- [ ] Fine-tuned embedding models per domain

## 🛠️ Technology Stack

- **Python 3.8+**
- **sentence-transformers** - Embeddings
- **transformers** - Open-source LLM
- **FAISS** - Vector search
- **Streamlit** - Demo UI

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

---

**🚀 This project focuses on RAG correctness, not just generation — the hardest part in real-world LLM systems.**
