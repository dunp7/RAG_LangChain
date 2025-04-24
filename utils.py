import os
import time
import psutil
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google import genai
# import pynvml
# Load API Key securely
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDM_PdW0cFCJV1eMYMCimME4UxKrBhK1AQ")  # Replace with your actual API key or use env variables
client = genai.Client(api_key=API_KEY)

# pynvml.nvmlInit()
# Utility Functions
def get_memory_usage():
    """Get memory usage as a percentage."""
    memory = psutil.virtual_memory()
    return f"Memory Usage: {memory.percent}%"

def get_cpu_usage():
    """Get CPU usage as a percentage."""
    return psutil.cpu_percent(interval=0.1)

def get_processing_time(start_time):
    """Calculate elapsed time since start_time."""
    elapsed_time = time.time() - start_time
    return f"Time Taken: {elapsed_time:.4f} seconds"

# def get_gpu_usage():
#     """Returns current GPU usage (memory and utilization)."""
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
#     memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
#     return {
#         "memory_used": memory_info.used / 1024**2,  # In MB
#         "memory_total": memory_info.total / 1024**2,  # In MB
#         "gpu_utilization": utilization.gpu,  # In percentage
#     }



# Gemini API Interaction for Chatbot
def chatbot_with_gemini(query: str, context: str):
    """Use Gemini API to answer a query based on the given context in Vietnamese."""
    try:
        prompt = f"""
        Bạn là một trợ lý thông minh và hữu ích. Dựa trên nội dung dưới đây, hãy trả lời câu hỏi một cách tốt nhất có thể bằng tiếng Việt.

        Ngữ cảnh:
        {context}

        Câu hỏi: {query}
        Câu trả lời:
        """.strip()
        
        # Use Gemini API for text generation
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return f"Error: {e}"