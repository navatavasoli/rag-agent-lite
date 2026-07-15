# rag-agent-lite
Ollama-supported RAG agent.

# RAG Agent Lite
Building RAG and agentic RAG from scratch in stages, using LangChain, LangGraph, Qdrant, HuggingFace, and Ollama.

Adapted in part from [agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies).

## Stages
- Stage 1 - Basic RAG (embed -> store -> retrieve -> generate)
- Stage 2 - Hierarchical chunking + hybrid search
- Stage 3 - Tool calling + manual agent loop
- Stage 4 - LangGraph agent with self-correction
- Stage 5 - Multi-agent system with memory and clarification

## Notes and Known Issues
There are some known issues with the current version, particularly including a lot of frequent hallucinations with the model. This occurs both with the agentic and non-agentic pipeline versions. 

## Library
THe library of existing content in the model for retrieval right now consists of the following:
- Into Thin Air - Jon Krakauer
- Obsession (screenplay) - Curry Barker
- Romeo and Juliet - William Shakespeare
- The 48 Laws of Power - Robert Greene
- The Myth of Sisyphus - Albert Camus