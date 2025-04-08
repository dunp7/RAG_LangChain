from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import openai
from .db_preparation import retrieve_relevant_chunks
import functools


@functools.lru_cache(maxsize=2)
def load_huggingface_model(model_name="facebook/opt-350m", device_map="auto", torch_dtype="auto", hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, use_auth_token=hf_token, torch_dtype=torch_dtype)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_questions_answers(
    query: str,
    grade: str,
    question_type: str,
    num_questions: int,
    vectorstore,
    model_type: str = "huggingface",
    model_name: str = "facebook/opt-350m",
    openai_api_key: str = None,
    hf_token: str = None
):
    # Step 1: Get context from FAISS

    context = retrieve_relevant_chunks(vectorstore, query)

    # Step 2: Build prompt
    prompt = f"""
        You are a helpful teacher assistant. Based on the following content, generate {num_questions} {question_type} questions with answers.

        Context:
        {context}

        Format:
        1. Question?
        a) Option A
        b) Option B
        c) Option C
        d) Option D
        Answer: b) Option B
            """.strip()

    # Step 3: Route to appropriate model
    if model_type == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI model.")
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        response = llm([HumanMessage(content=prompt)])
        return response.content.strip()

    elif model_type in ["huggingface"]:
        generator = load_huggingface_model(model_name=model_name, hf_token=hf_token)
        result = generator(prompt, max_new_tokens=2048, temperature=0.7)[0]['generated_text']
        cleaned_result = result[len(prompt):].strip() if result.startswith(prompt) else result.strip()
        return cleaned_result

    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
