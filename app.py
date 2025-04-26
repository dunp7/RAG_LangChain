import streamlit as st
# Streamlit app configuration
st.set_page_config(page_title="Embedding Visualizer", layout="wide")

from embedding import *
from utils import *
from vector_db import  *
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch

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

    # Input sentences and labels
    input_text = st.text_area(
        "Enter a list of sentences with labels (format: 'sentence|label' per line):",
        placeholder="Type each sentence and its label on a new line, separated by '|'. Example:\nHello world|0\nHow are you?|1"
    )

    if input_text.strip():
        # Parse input sentences and labels
        sentences_and_labels = input_text.strip().split("\n")
        sentences = []
        labels = []

        for item in sentences_and_labels:
            if "|" in item:
                sentence, label = item.rsplit("|", 1)
                sentences.append(sentence.strip())
                labels.append(int(label.strip()))
            else:
                st.error("Invalid format. Ensure each line contains 'sentence|label'.")

        st.write("Parsed Input:")
        for idx, (sentence, label) in enumerate(zip(sentences, labels), start=1):
            st.write(f"{idx}. {sentence} (Label: {label})")

        # Number of clusters (k)
        num_clusters = st.number_input(
            "Enter the number of clusters (k):",
            min_value=1,
            max_value=len(set(labels)),
            value=len(set(labels)),
            step=1
        )

        # Button to generate embeddings and run clustering
        if st.button("Generate Embeddings & Run Clustering"):
            # Start processing time
            start_time = time.time()

            # Device configuration
            device = "cuda" if device_option == "GPU" else "cpu"
            st.write(f"Loading {model_name} model on {device}...")
            model = SentenceTransformer(load_model(model_name), device=device, trust_remote_code=True)

            # Generate embeddings
            st.write("Generating embeddings...")
            vector_embeddings = model.encode(sentences)

            # Perform k-means clustering
            st.write("Performing k-means clustering...")
            clustered_sentences = perform_kmeans_clustering(vector_embeddings, sentences, num_clusters)

            # Map clustered sentences to labels
            predicted_labels = clustered_sentences['Cluster']
            true_labels = labels


            # Nhãn từ K-means (Cluster assignments)
            predicted_labels = clustered_sentences['Cluster'].to_numpy()

            # Nhãn thật (True labels)
            true_labels = np.array(labels)

            # Xây dựng confusion matrix
            confusion = confusion_matrix(true_labels, predicted_labels)

            # Sử dụng Hungarian Algorithm để ánh xạ nhãn
            row_ind, col_ind = linear_sum_assignment(-confusion)

            # Tạo nhãn ánh xạ
            mapped_labels = np.zeros_like(predicted_labels)
            for cluster_id, true_label in zip(col_ind, row_ind):
                mapped_labels[predicted_labels == cluster_id] = true_label

            # Tính accuracy
            clustering_accuracy = accuracy_score(true_labels, mapped_labels)
            print(f"Clustering Accuracy: {clustering_accuracy:.2f}")

            # Display results
            end_time = time.time()
            st.write(get_processing_time(start_time))

            st.write("Cluster Assignments:")
            st.write(clustered_sentences)

            st.write(f"Clustering Accuracy: {clustering_accuracy:.2f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        st.write("Please enter sentences and labels.")



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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
