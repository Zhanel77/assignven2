from sentence_transformers import SentenceTransformer

def generate_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a pre-trained model
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings

