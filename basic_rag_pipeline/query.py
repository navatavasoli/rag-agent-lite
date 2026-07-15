from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama 

# not creating anything just reading from the collection from ingest.py
# reconnecting to the existing store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
client = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "library"

vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)


# retrieval
query = "how did romeo die"
results = vector_store.similarity_search(query, k = 3)

for i, doc in enumerate(results):
    print(doc.page_content)
    print()


llm = ChatOllama(model = "qwen3:4b-instruct-2507-q4_K_M")
context = "\n\n".join(doc.page_content for doc in results)
prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, come up with the next best answer that you can.
Context: {context}

Question: {query}

Answer:"""

response = llm.invoke(prompt)
print("\nANSWER:\n")
print(response.content + "\n")
#print(doc.metadata["source"])
client.close()