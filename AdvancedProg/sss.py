import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

# Setup ChromaDB storage path
chroma_storage_path = os.path.join(os.getcwd(), "chroma_db")

chroma_settings = Settings(
    persist_directory=chroma_storage_path
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=chroma_storage_path)

# SentenceTransformer model name
model_name = "all-MiniLM-L6-v2"
embedded_func = SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

# Initialize ChromaDB collection
collection_name = "chat_data"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedded_func
)

# Streamlit interface setup
st.set_page_config(
    page_title="Enhanced Chat with Llama3.2",
    page_icon="🤖",
    layout="wide",
)

st.title("Enhanced Chat with Llama3.2")
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
    }
    .stTextInput input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 12px;
    }
    .stTextArea textarea {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("Menu", ["Add Data", "Show Data", "Chat Bot"])
st.sidebar.markdown("Select an action from the menu above")

if menu == "Add Data":
    st.subheader("Add Data to the Bot")
    input_text = st.text_area("Enter data", placeholder="Type your data here...")
    if st.button("Save Data", key="save_data"):
        if input_text.strip():
            embedding = SentenceTransformer(model_name).encode(input_text).tolist()
            doc_id = f"doc_{len(collection.get()['ids']) + 1}"
            collection.add(documents=[input_text], embeddings=[embedding], ids=[doc_id])
            st.success(f"Added document: {doc_id}")
        else:
            st.error("Data is empty! Please enter valid text.")

elif menu == "Show Data":
    st.subheader("Stored Documents")
    documents = collection.get().get("documents", [])
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc}")
    else:
        st.info("No data available! Add some documents to view here.")

elif menu == "Chat Bot":
    st.subheader("Chat with the Bot")
    user_input = st.text_input("Enter your question:", placeholder="Type your question here...")
    if st.button("Send", key="send_question"):
        if user_input.strip():
            query_vector = SentenceTransformer(model_name).encode(user_input).tolist()

            results = collection.query(
                query_embeddings=query_vector, n_results=1
            )

            # Check if results are relevant
            relevant_docs = results.get("documents", [[]])[0]
            if relevant_docs:
                # Use the first result for simplicity
                retrieved_doc = relevant_docs[0]
                similarity_score = results.get("distances", [[0]])[0][0]

                # Threshold for relevance (adjustable)
                relevance_threshold = 0.5

                if similarity_score <= relevance_threshold:
                    st.write("### Answer from the database:")
                    st.write(f"- {retrieved_doc}")

                    messages = [
                        ChatMessage(role="user", content=f"Relevant context: {retrieved_doc}"),
                        ChatMessage(role="user", content=user_input)
                    ]

                    llm = Ollama(model="llama3.2", request_timeout=120.0)

                    st.write("### Ollama's response:")
                    try:
                        response = ""
                        response_placeholder = st.empty()
                        response_stream = llm.stream_chat(messages=messages)

                        for chunk in response_stream:
                            response += chunk.delta
                            response_placeholder.write(response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                else:
                    st.write("### The database result is not relevant enough. Generating response with AI...")

                    llm = Ollama(model="llama3.2", request_timeout=120.0)
                    messages = [ChatMessage(role="user", content=user_input)]

                    try:
                        response = ""
                        response_placeholder = st.empty()
                        response_stream = llm.stream_chat(messages=messages)

                        for chunk in response_stream:
                            response += chunk.delta
                            response_placeholder.write(response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.write("### No relevant matches found in the database. Generating response with AI...")

                llm = Ollama(model="llama3.2", request_timeout=120.0)
                messages = [ChatMessage(role="user", content=user_input)]

                try:
                    response = ""
                    response_placeholder = st.empty()
                    response_stream = llm.stream_chat(messages=messages)

                    for chunk in response_stream:
                        response += chunk.delta
                        response_placeholder.write(response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Input cannot be empty. Please enter a valid question.")