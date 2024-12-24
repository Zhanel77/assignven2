import chromadb
from chromadb.config import Settings
from embeddings import generate_embeddings

# Инициализация ChromaDB
client = chromadb.Client(Settings(persist_directory="./chroma_storage"))
collection = client.get_or_create_collection("documents")

# Функция для добавления документа в ChromaDB
def add_document_to_chromadb(doc_id, text):
    embedding = generate_embeddings(text)
    collection.add(
        documents=[text],
        metadatas=[{"source": "user"}],
        ids=[doc_id],
        embeddings=[embedding]
    )

# Функция для получения контекста из ChromaDB
def retrieve_context(query, top_k=3):
    query_embedding = generate_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
