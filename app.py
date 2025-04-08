import streamlit as st
from utils.db_preparation import split_pdf_into_chunks, create_faiss_index
from utils.qa_generator import generate_questions_answers

# === STREAMLIT UI ===
st.title("📘 PDF-Based Question & Answer Generator with FAISS")
st.write("Upload a PDF and generate AI-powered questions with answers.")

# File upload
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Options for question generation
grade = st.selectbox("Select Grade", [str(i) for i in range(1, 13)])
question_type = st.selectbox("Type of Questions", ["MCQ", "One Word", "Logical Reasoning", "Fill in the Blanks"])
num_questions = st.slider("Number of Questions", 1, 50, 20)
query = st.text_input("Enter a topic or keyword to focus on (optional)")

# Model configuration
gen_model_type = st.selectbox("Select Generation Mode", ["huggingface", "openai"])
if gen_model_type == "huggingface":
    gen_model_name = st.text_input("Enter Hugging Face Model Name", value="facebook/opt-350m")
    hf_token = st.text_input("Enter Hugging Face Token", type="password")  # Secure input for HF token
else:
    gen_model_name = "gpt-3.5-turbo"  # Default OpenAI model
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")  # Secure input for OpenAI key

# Button to generate questions
if st.button("Generate Questions & Answers"):
    if pdf_file:
        with st.spinner("🔍 Processing the PDF..."):
            # Process the PDF
            chunks = split_pdf_into_chunks(pdf_file)
            vectorstore = create_faiss_index(
                chunks)

        with st.spinner("🤖 Generating questions and answers..."):
            # Generate questions and answers
            questions_answers = generate_questions_answers(
                query=query,
                grade=grade,
                question_type=question_type,
                num_questions=num_questions,
                vectorstore=vectorstore,
                model_type=gen_model_type,
                model_name=gen_model_name,
                openai_api_key=openai_api_key if gen_model_type == "openai" else None,
                hf_token=hf_token if gen_model_type == "huggingface" else None
            )

        # Display results
        st.success("✅ Questions & Answers generated successfully!")
        st.text_area("Generated Questions & Answers", value=questions_answers, height=500)
    else:
        st.warning("⚠️ Please upload a PDF.")
