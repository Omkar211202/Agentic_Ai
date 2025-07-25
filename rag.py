
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set it in .env file.")

# Step 1: Load PDF and process it
pdf_path = "T.pdf"  # Replace with actual PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Step 2: Split text into chunks for embedding storage
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Step 3: Create ChromaDB Vector Store
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=OpenAIEmbeddings()  # Using OpenAI's embedding model
)

# Step 4: Initialize OpenAI Chat Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Step 5: Chat loop to query the knowledge base
print("\nInstitute GPT is ready! Ask questions based on the PDF (type 'exit' to quit).")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    # Retrieve top matching document chunks
    search_results = vector_store.similarity_search(query, k=3)  # Get top 3 matches
    retrieved_texts = "\n\n".join([doc.page_content for doc in search_results])

    # Format the prompt
    prompt = f"Use the following institute-related information to answer the question:\n\n{retrieved_texts}\n\nQuestion: {query}\n\nAnswer:"

    # Get response from GPT
    response = llm.invoke(prompt)
    print("\nAnswer:", response.content)
