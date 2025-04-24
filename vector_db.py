import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# Function to handle different file types (PDF, DOCX, TXT)
def load_file(uploaded_file):
    """Load document based on its type (PDF, DOCX, or TXT)."""
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the content based on file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
        pages = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return pages

# Function to split document content into chunks
def split_document_into_chunks(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", ".", "!", "?"]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Function to get the embedding model
from sentence_transformers import SentenceTransformer

def get_embedding_model(embedding_type="huggingface", model_name=None, device="cpu"):
    """Return the embedding model based on the specified type."""
    if embedding_type == "huggingface":
        # Load SentenceTransformer model with `trust_remote_code`
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device= device
        )
        # Pass the loaded model to HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(client=model)
    else:
        raise ValueError(f"Embedding type '{embedding_type}' not supported.")



# Function to create FAISS index
def create_faiss_index(chunks, embedding_type="huggingface", model_name=None, device = 'cpu'):
    """Convert chunks to embeddings and store in FAISS."""
    embedding_model = get_embedding_model(embedding_type=embedding_type, model_name=model_name, device= device)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


# Function to retrieve relevant chunks from FAISS index
def retrieve_relevant_chunks(vectorstore, query, top_k=3):
    """Perform similarity search to retrieve top-k chunks."""
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])


# Function to process uploaded file and create FAISS index
def process_file_and_create_index(uploaded_file, model_name):
    """Process the file, split into chunks, and create a FAISS index."""
    # Load file and split it into chunks
    documents = load_file(uploaded_file)
    chunks = split_document_into_chunks(documents)

    # Create FAISS index
    vectorstore = create_faiss_index(chunks= chunks, model_name=model_name)

    return vectorstore
