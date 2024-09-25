from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain.tools import tool
from langchain.schema import Document  # Added forpo Document structure
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Error-handling function to make API requests safe
def safe_api_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return {"type": "error", "description": str(e)}

# Function to extract medical terms from user input
def extract_medical_terms(user_input):
    # Example set of symptoms/medical terms for demo; can be expanded
    symptoms = ['headache', 'fever', 'cough', 'nausea', 'sore throat']
    extracted_terms = [term for term in symptoms if term in user_input.lower()]
    return ' '.join(extracted_terms)

# Wrap the file data in a document object for LangChain processing
file_path = 'Data.txt'  # Make sure this file contains disease and treatment information
with open(file_path, 'r', encoding='utf-8') as file:
    file_data = file.read()

documents = [Document(page_content=file_data)]

# Split the document into chunks for embedding
split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

# Use FAISS to create a vector store for document retrieval
vector = FAISS.from_documents(split_docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Create the retriever tool to search the document
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(retriever, "disease_data_search", "Search for information about diseases and treatments from the predefined document.")

@tool
def add_numbers_tool(input_data: str) -> str:
    """Addition of two numbers."""
    try:
        numbers = input_data.split(',')
        num1, num2 = int(numbers[0]), int(numbers[1])
        result = num1 + num2
        return f"The Sum of {num1} and {num2} is {result}"
    except Exception as e:
        return f"Error in addition: {e}"

# Define tools for API querying
@tool
def query_pubmed(input_data: str) -> str:
    """Query the PubMed API for medical information."""
    terms = extract_medical_terms(input_data)
    if not terms:
        return "No relevant medical terms found."
    
    pubmed_url = f"https://api.pubmed.gov/search/?query={terms}&apikey={os.getenv('PUBMED_API_KEY')}"
    pubmed_result = safe_api_request(pubmed_url)
    
    if pubmed_result.get("type") == "error":
        return f"PubMed API Error: {pubmed_result.get('description')}"
    
    return f"PubMed Result: {pubmed_result}"

# Add tools to the agent
tools = [retriever_tool, query_pubmed]

# Pull an existing prompt template for the agent
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor to manage the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Manage chat history for sessions
message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,  # session-based message history
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Start a loop to interact with the user
while True:
    user_input = input("How can I help you today? : ")
    
    # If input contains irrelevant queries, ignore them
    filter_response = extract_medical_terms(user_input)
    if not filter_response:
        print("Irrelevant question detected. Please ask medical-related questions.")
        continue
    
    # Call the agent with chat history
    result = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "test123"}},
    )
    
    print(result)