import streamlit as st
from utils.db_preparation import split_pdf_into_chunks, create_faiss_index
from utils.qa_generator import generate_questions_answers

# === CONFIGURATION ===
EMBEDDING_TYPE = "huggingface"  # Options: "huggingface", "openai"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For huggingface
OPENAI_API_KEY = "***"  # Only used if embedding_type == "openai"

# === STREAMLIT UI ===
st.title("📘 PDF-Based Question & Answer Generator with FAISS")
st.write("Upload a PDF and generate AI-powered questions with answers.")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
grade = st.selectbox("Select Grade", [str(i) for i in range(1, 13)])
question_type = st.selectbox("Type of Questions", ["MCQ", "One Word", "Logical Reasoning", "Fill in the Blanks"])
num_questions = st.slider("Number of Questions", 1, 50, 20)
query = st.text_input("Enter a topic or keyword to focus on (optional)")

if st.button("Generate Questions & Answers"):
    if pdf_file:
        with st.spinner("🔍 Processing the PDF..."):
            chunks = split_pdf_into_chunks(pdf_file)
            vectorstore = create_faiss_index(
                chunks,
                embedding_type=EMBEDDING_TYPE,
                model_name=MODEL_NAME,
                openai_api_key=OPENAI_API_KEY
            )

        with st.spinner("🤖 Generating questions and answers..."):
            questions_answers = generate_questions_answers(
                query=query,
                grade=grade,
                question_type=question_type,
                num_questions=num_questions,
                vectorstore=vectorstore
            )

        st.success("✅ Questions & Answers generated successfully!")
        st.text_area("Generated Questions & Answers", value=questions_answers, height=500)
    else:
        st.warning("⚠️ Please upload a PDF.")
