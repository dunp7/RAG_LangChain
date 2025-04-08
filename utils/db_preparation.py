from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def split_pdf_into_chunks(uploaded_file):
    """Save uploaded PDF, load it, and split into text chunks."""
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", ".", "!", "?"]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def get_embedding_model(embedding_type="huggingface", model_name=None, openai_api_key=None):
    """Return the embedding model based on the specified type."""
    if embedding_type == "openai":
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    elif embedding_type == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
        )

    else:
        raise ValueError(f"Embedding type '{embedding_type}' not supported.")


def create_faiss_index(chunks, embedding_type="huggingface", model_name=None, openai_api_key=None):
    """Convert chunks to embeddings and store in FAISS."""
    embedding_model = get_embedding_model(
        embedding_type=embedding_type,
        model_name=model_name,
        openai_api_key=openai_api_key,
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

def retrieve_relevant_chunks(vectorstore, query, top_k=3):
    """Perform similarity search to retrieve top-k chunks."""
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])
