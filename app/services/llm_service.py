# app/services/llm_service.py
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
try:
    from langchain_community.chat_models import ChatOpenAI
except Exception:
    ChatOpenAI = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

def init_llm(backend: str = "auto", model_name: str = "llama3"):
    backend = backend.lower()
    api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY loaded:", api_key is not None)

    if backend == "auto":
        if api_key and ChatOpenAI is not None:
            backend = "openai"
        else:
            backend = "ollama"

    if backend == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("ChatOpenAI not available in your environment.")
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )
    elif backend == "ollama":
        return Ollama(model=model_name or "llama3")
    else:
        raise ValueError(f"Unsupported llm_backend: {backend}")

def init_embedding(backend: str = "huggingface", model_name: str = "all-MiniLM-L6-v2"):
    backend = backend.lower()
    if backend == "openai":
        return OpenAIEmbeddings()
    elif backend == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embed_backend: {backend}")
