import chromadb

# Initialize ChromaDB in-memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = chroma_client.get_or_create_collection(name="test_collection")

# Add a sample document
collection.add(
    ids=["1"],
    documents=["Hello, this is a test document stored in ChromaDB."]
)

# Query ChromaDB
results = collection.query(query_texts=["test"], n_results=1)
print("Query Results:", results)
