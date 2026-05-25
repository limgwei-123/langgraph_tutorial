from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
  model="gpt-4o", temperature=0
)

embeddings = OpenAIEmbeddings(
  model = "text-embedding-3-small"
)

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
  raise FileNotFoundError(f"PDF file not found:{pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
  pages = pdf_loader.load()
  print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
  print(f"Error loading PDF: {e}")
  raise


text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap =200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"C:\Users\User\Desktop\projects\langgraph_tutorial\AI_Agent\chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
  os.makedirs(persist_directory)

has_existing_db = os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))

try:
  if has_existing_db:
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    print("already got liao")
  else:
    vectorstore = Chroma.from_documents(
      documents=pages_split,
      embedding=embeddings,
      persist_directory=persist_directory,
      collection_name=collection_name
    )

    print(f"Created ChromaDB vectore store")
except Exception as e:
  print(f"Error setting up ChromaDB: {str(e)}")
  raise

retriever = vectorstore.as_retriever(
  search_type ="similarity",
  search_kwargs={"k":5}
)

@tool
def retriever_tool(query: str) -> str:

  """
  This tool searches and returns the information from the Stock Market Performance 2024 document.
  """
  print(f"[retriever_tool] received query: {query}", flush=True)

  try:
      print("[retriever_tool] before retriever.invoke", flush=True)
      docs = retriever.invoke(query)
      print(f"[retriever_tool] after retriever.invoke, docs count: {len(docs)}", flush=True)
  except Exception as e:
      print(f"[retriever_tool] ERROR: {type(e).__name__}: {e}", flush=True)
      return f"Retriever error: {type(e).__name__}: {e}"

  if not docs:
      return "I found no relevant information in the Stock Market Performance 2024 document"

  results = []

  for i, doc in enumerate(docs):
      results.append(f"Document {i+1}:\n{doc.page_content}")

  print("[retriever_tool] finished formatting results", flush=True)
  return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools=tools)

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
  """Check if the last message contains tool calls."""
  result = state['messages'][-1]
  return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState):
  """Function to call the LLM with the current state"""
  messages = list(state['messages'])
  messages = [SystemMessage(content=system_prompt)] + messages
  message = llm.invoke(messages)
  return {'messages': [message]}

def take_action(state: AgentState):
  """Execute tool calls from the LLM's response."""

  tool_calls = state['messages'][-1].tool_calls
  results = []
  for t in tool_calls:
    print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

    if not t['name'] in tools_dict: # Checks if a valid tool is present
      print(f"\nTool: {t['name']} does not exist.")
      result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

    else:
      result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
      print(f"Result length: {len(str(result))}")

    results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

  print("Tools Execution Complete. Back to the model!")
  return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action) # this is tool node but written by me

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()