from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama 
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    question: str
    messages: list[BaseMessage]
    answer: str
    retry_count: int

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
client = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "library"
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# using the decorator means retrieve can't be called like a normal function anymore 
@tool
def retrieve(query: str) -> str:
    """Search the document library, return relevant passages for the given query."""
    results = vector_store.similarity_search(query, k = 3)
    return "\n\n".join(doc.page_content for doc in results)

#result = retrieve.invoke("who is the cousin of juliet capulet")
#print(result)


llm = ChatOllama(model = "qwen3:4b-instruct-2507-q4_K_M")
llm_with_tools = llm.bind_tools([retrieve])
#prompt = "who is the cousin of juliet capulet? is it tybalt?"
#response = llm.invoke(prompt)
response = llm_with_tools.invoke("was nikki in love with bear?")

question = "was nikki in love with bear?"
messages = [HumanMessage(content = question), response]

def agent_note(state: GraphState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


for call in response.tool_calls:
    tool_result = retrieve.invoke(call["args"])
    messages.append(ToolMessage(content = tool_result, tool_call_id=call["id"]))

final_response = llm_with_tools.invoke(messages)
print(final_response.content)

#print(response)
#print(response.tool_calls)
client.close()