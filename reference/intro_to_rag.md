# Introduction to RAG and Agentic RAG

Personal reference notes.

## Natural Language Vectorization & Semantic Understanding
- Natural language is vectorized, in this project into 768 dimensions to be compatible with the Ollama [qwen3:4b-instruct-2507-q4_K_M] model

The embedding model (e.g. `sentence-transformers/all-mpnet-base-v2`) is a neural network trained specifically for this transform trained on huge numbers of sentence pairs labeled "similar" or "dissimilar" adjusting its internal weights until semantically similar sentences land close together and dissimilar ones land far apart. This is **not** the same kind of model as an LLM since it doesn't generate language, it only maps text. This is what supports semantic understanding.

