import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path

# text extraction
BOOKS = ["docs/sisyphus.pdf", "docs/romeo_and_juliet.pdf", "docs/the_48_laws.pdf"]

#markdown_text = pymupdf4llm.to_markdown(BOOKS)

all_chunks = []
#all_metadatas = []

# chunking text
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
for pdf_path in BOOKS:
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    chunks = splitter.split_text(markdown_text)
    all_chunks.extend(chunks)
    #all_metadatas.extend([{"source": pdf_path}] * len(chunks))
    print(f"Processing {pdf_path}...")
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    chunks = splitter.split_text(markdown_text)
    all_chunks.extend(chunks)
    print(f"  -> {len(chunks)} chunks")

# embed
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
client = QdrantClient(path = "qdrant_db")

COLLECTION_NAME = "library"
# all-mpnet-base-v2 produces 768-dim vectors
#client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=768, distance=Distance.COSINE))

# don't make a new collection each time - otherwise if there is no existing collection then we make a new one
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# write the chunks of text into the store
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
vector_store.add_texts(all_chunks) #all_metadatas)

client.close()