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
    grade: str

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

llm = ChatOllama(model = "qwen3:4b-instruct-2507-q4_K_M")
llm_with_tools = llm.bind_tools([retrieve])
'''
response = llm_with_tools.invoke("who is juliet capulet's cousin?")

question = "who is juliet capulet's cousin"
messages = [HumanMessage(content = question), response]

for call in response.tool_calls:
    tool_result = retrieve.invoke(call["args"])
    messages.append(ToolMessage(content = tool_result, tool_call_id=call["id"]))

#final_response = llm_with_tools.invoke(messages)
#print(final_response.content)
'''

def agent_node(state: GraphState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def retrieve_node(state: GraphState) -> dict:
    last_message = state["messages"][-1] # last message
    new_messages= []
    for call in last_message.tool_calls:
        tool_result = retrieve.invoke(call["args"])
        new_messages.append(ToolMessage(content = tool_result, tool_call_id = call["id"]))
    return {"messages": state["messages"] + new_messages}

def generate_node(state: GraphState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"answer": response.content}

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
#graph_builder.add_node("grade", grade_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", "retrieve")
graph_builder.add_edge("retrieve", "generate")
#graph_builder.add_edge("generate", "grade")
graph_builder.add_edge("generate", END)   # temporary — go straight to END for this test

#graph_builder.add_conditional_edges(
#    "grade",
#    decide_next_step,
#    {"retry": "agent", "done": END},
#)

graph = graph_builder.compile()

initial_state = {
    "question": "who is juliet capulet's cousin",
    "messages": [HumanMessage(content="who is juliet capulet's cousin")],
    "answer": "",
    "retry_count": 0,
    "grade": "",
}

result = graph.invoke(initial_state)
print(result["answer"])

graph = graph_builder.compile()
client.close()