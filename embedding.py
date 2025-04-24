"""
Embedding Model 
"""
import streamlit as st

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
def load_model(model_name):
    # Define a dictionary with model names and corresponding pre-trained models
    models = {
        "All-MiniLM-L6-v2 - 22.7M": "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI -109M": 'BAAI/llm-embedder',
        "all-mpnet-base-v2 - 109M": "sentence-transformers/all-mpnet-base-v2",
        "gte-multilingual-base - 305M": "Alibaba-NLP/gte-multilingual-base",
        "vietnamese-embedding - 135M": "dangvantuan/vietnamese-embedding",
        "vietnamese-document-embedding - 305M":"dangvantuan/vietnamese-document-embedding"
    }
    return models.get(model_name, model_name)

def generate_embedding(text, model):
    """Generate an embedding for a given text."""
    return model.encode(text)



def perform_kmeans_clustering(embeddings, sentences, num_clusters=3):
    """
    Performs k-means clustering 
    """
    # Ensure embeddings are 2D
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array.")
    
    pca = PCA(n_components=10)  
    embeddings_pca = pca.fit_transform(embeddings)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings_pca)

    # Create a DataFrame for displaying results
    clustered_sentences = pd.DataFrame({
        "Sentence": sentences,
        "Cluster": cluster_assignments
    })

    # Visualization using Streamlit
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=embeddings[:, 0], y=embeddings[:, 1],
        hue=cluster_assignments, palette="viridis", s=100
    )
    plt.title(f"K-means Clustering with {num_clusters} Clusters")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.legend(title="Cluster")
    
    # Display the plot in Streamlit
    st.pyplot(plt)

    return clustered_sentences