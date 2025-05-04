from src.model.graph_llm import GraphLLM

llama_model_path = {
    "3b" : "meta-llama/Llama-3.2-3B",
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}

# Define the model loading dictionary
load_model = {
    "graph_llm": GraphLLM,
    # Add other model types as needed
}