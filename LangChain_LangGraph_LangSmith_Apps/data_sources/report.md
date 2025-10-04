Quarterly Research & Development Report

Author: Internal AI Systems Team
Date: 2025-Q3
Confidentiality: Internal Use Only

1. Executive Summary

This quarter focused on:

Expanding the Universal Retrieval-Augmented Generation (RAG) framework.

Integrating multi-format ingestion pipelines for web, PDF, Markdown, CSV, and DOCX.

Benchmarking embedding models for semantic retrieval accuracy.

Deploying persistent FAISS vector stores for large-scale corpora.

Key outcomes:

+37% retrieval precision in benchmark datasets.

Reduced ingestion latency from 12.4s → 4.8s per 100 documents.

Improved chunking strategy reduced context window waste by 22%.

2. System Architecture Overview
2.1 High-Level Diagram
[ Data Sources ] → [ Loaders ] → [ Chunker ] → [ Embeddings ] → [ Vector Store ] 
                                                      ↓
                                                 [ Retriever ]
                                                      ↓
                                              [ LLM Generator ]

2.2 Components

Loaders: WebBaseLoader, PyPDFLoader, CSVLoader, UnstructuredMarkdownLoader.

Chunker: RecursiveCharacterTextSplitter with adaptive chunk size.

Embeddings: sentence-transformers/all-mpnet-base-v2.

Vector Store: FAISS with optional disk persistence.

LLM: Gemini 2.0 Flash for low-latency generation.

3. Performance Benchmarks
Model Name	Top-K Accuracy	Avg. Latency (ms)	Notes
all-mpnet-base-v2	0.87	42	Best overall balance
multi-qa-mpnet-base-dot-v1	0.85	39	Slightly faster, lower recall
all-MiniLM-L6-v2	0.81	28	Good for lightweight deployments
4. Retrieval Experiments
4.1 Query Types Tested

Factoid: “What is the maximum throughput of the ingestion pipeline?”

Analytical: “Compare the trade-offs between FAISS and Chroma for persistent storage.”

Procedural: “List the steps to deploy the RAG API on AWS EC2.”

4.2 Observations

Factoid queries benefit most from tight chunk overlap.

Analytical queries require semantic diversity in retrieved chunks.

Procedural queries perform best when step-by-step formatting is preserved in source docs.

5. Deployment Notes
5.1 AWS EC2 Setup
# Install dependencies
sudo apt update && sudo apt install -y python3-pip
pip install fastapi uvicorn langchain faiss-cpu

# Run API
uvicorn app:app --host 0.0.0.0 --port 8000

5.2 Docker Build
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

6. Risk Assessment
Risk ID	Description	Likelihood	Impact	Mitigation
R-101	Embedding model drift after retrain	Medium	High	Version-lock embeddings
R-102	Vector store corruption on disk	Low	High	Enable periodic backups
R-103	LLM API latency spikes	Medium	Medium	Implement async batching
7. Recommendations

Persist vector stores to disk for faster restarts.

Add monitoring hooks for retrieval quality over time.

Experiment with hybrid search (BM25 + embeddings).

Automate ingestion from RSS feeds for continuous updates.

8. Appendix
8.1 Sample Prompt Template
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:

8.2 Glossary

RAG: Retrieval-Augmented Generation.

Chunking: Splitting documents into smaller, semantically coherent parts.

Embedding: Vector representation of text for similarity search.

End of Report