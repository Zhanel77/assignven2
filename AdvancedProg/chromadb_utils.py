import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("document_embeddings")

def save_embeddings(documents, embeddings):
    for idx, embedding in enumerate(embeddings):
        collection.add(
            documents=[documents[idx]],
            embeddings=[embedding.tolist()],
            ids=[str(idx)]  # Unique ID for each document
        )
def retrieve_context(query):
    results = collection.query(query_texts=[query], n_results=3)
    return results["documents"]

def ask_ollama(question, context):
    response = ollama.ask(f"Context: {context}\nQuestion: {question}")
    return response
