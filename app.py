import streamlit as st
# Streamlit app configuration
st.set_page_config(page_title="Embedding Visualizer", layout="wide")

from embedding import *
from utils import *
from vector_db import  *
import numpy as np
import time



# Sidebar menu
st.sidebar.title("Embedding Operations")
option = st.sidebar.radio(
    "Choose an operation:",
    ("Test Embedding Model","Embedding with FAISS", "Demo ChatBot")
)

def clear_all_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Clear all session state when the user selects a new option
if option:
    clear_all_session_state()


# Select model
use_custom_model = st.checkbox("Use a custom model")
if use_custom_model:
    model_name = st.text_input("Enter the custom model name:")
else:
    model_name = st.selectbox(
        "Choose an embedding model:",
        ("All-MiniLM-L6-v2 - 22.7M", 'BAAI -109M', 'all-mpnet-base-v2 - 109M',
         'gte-multilingual-base - 305M', "vietnamese-embedding - 135M", 'vietnamese-document-embedding - 305M')
    )

device_option = st.radio(
    "Choose a device:",
    ("CPU", "GPU")
)


if option == 'Test Embedding Model':
    st.title("Test Embedding")

    # Input 
    input_text = st.text_area(
        "Enter a list of sentences (separated by new lines):",
        placeholder="Type each sentence on a new line."
    )



    if input_text.strip():
        sentences = input_text.strip().split("\n")
        st.write("Input sentences:")
        for idx, sentence in enumerate(sentences, start=1):
            st.write(f"{idx}. {sentence}")
        
        num_clusters = st.number_input(
            "Enter the number of clusters (k):",
            min_value=0,
            max_value=len(sentences),
            value=1,
            step=1
        )

        # Single button for generating embeddings and clustering
        if st.button("Generate Embeddings & Run Clustering"):
            # Start processing time
            start_time = time.time()

            # Device configuration
            device = "cuda" if device_option == "GPU" else "cpu"

            # # Resource usage before processing
            # if device == "cuda":
            #     initial_gpu_usage = get_gpu_usage()
            # else:
            #     initial_cpu_usage = get_cpu_usage()

            # Load the model
            st.write(f"Loading {model_name} model on {device}...")
            model = SentenceTransformer(load_model(model_name), device=device, trust_remote_code=True)

            st.write("Generating embeddings...")
            vector_embeddings = model.encode(sentences)

            # # Resource usage after processing
            # if device == "cuda":
            #     final_gpu_usage = get_gpu_usage()
            # else:
            #     final_cpu_usage = get_cpu_usage()

            end_time = time.time()
            st.write(get_processing_time(start_time))

            # # Display resource usage
            # if device == "cuda":
            #     st.write("GPU Usage During Processing:")
            #     st.write(f"  Memory Used: {final_gpu_usage['memory_used'] - initial_gpu_usage['memory_used']:.2f} MB")
            #     st.write(f"  GPU Utilization: {final_gpu_usage['gpu_utilization']}%")
            # else:
            #     st.write(f"CPU Usage During Processing: {final_cpu_usage}%")

            # Display generated embeddings
            st.write("Generated Embeddings:")
            for idx, embedding in enumerate(vector_embeddings, start=1):
                st.write(f"Sentence {idx}: Dimension {len(embedding)}")

            # Perform k-means clustering
            if vector_embeddings is not None:
                st.write("Performing k-means clustering...")
                clustered_sentences = perform_kmeans_clustering(vector_embeddings, sentences, num_clusters)
                st.write("Cluster Assignments:")
                st.write(clustered_sentences)

    else:
        st.write("Please enter sentences.")



# Add Embedding Section

elif option == "Embedding with FAISS":

    st.title("Add Embedding")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        # Process the uploaded file and create the FAISS index
        vectorstore = process_file_and_create_index(uploaded_file, load_model(model_name), device_option)
        
        # Query input
        query = st.text_area("Enter your query:")
        
        if query:
            # Retrieve relevant chunks from FAISS index
            relevant_chunks = retrieve_relevant_chunks(vectorstore, query)
            
            # Display the retrieved chunks
            st.write("Relevant Chunks:")
            st.write(relevant_chunks)
        

# Demo Small Chatbot to answer things in Uploaded File
elif option == "Demo ChatBot":

    st.title("Demo ChatBot")

    # File uploader for Chatbot
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file to chat with", type=["pdf", "docx", "txt"])

    if uploaded_file:
        # Process the uploaded file and create the FAISS index
        vectorstore = process_file_and_create_index(uploaded_file,load_model(model_name), device_option)


    # Chat input for the chatbot
    user_input = st.text_area(
        "Enter a list of questions (separated by new lines):",
        placeholder="Type each sentence on a new line."
    )
    if user_input.strip():
        # Split the input into multiple questions
        questions = user_input.strip().split("\n")     
        try:
            # Iterate through each question to generate responses
            st.write("### ChatBot Answers:")
            for idx, question in enumerate(questions, start=1):
                # Retrieve relevant chunks for the current question
                context = retrieve_relevant_chunks(vectorstore, question)
                time.sleep(2)
                # Generate chatbot response for the current question
                answer = chatbot_with_gemini(query=question, context=context)
                
                # Display the question and its corresponding answer
                st.markdown(f"**Question {idx}:** {question}")
                st.markdown(f"**Answer {idx}:** {answer}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
