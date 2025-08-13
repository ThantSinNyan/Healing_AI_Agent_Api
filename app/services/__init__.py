# app/services/__init__.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .qa_service import QAAccuracyTester

_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vectorstore = FAISS.load_local(
    "app/vector_db/pdf_book",
    _embeddings,
    allow_dangerous_deserialization=True
)

qa_accuracy_tester = QAAccuracyTester(
    _vectorstore,
    llm_backend="openai",      # force OpenAI usage
    llm_model="gpt-4o",   # or "gpt-4o", "gpt-4", etc. depending on your subscription
    embed_backend="openai",    # use OpenAI embeddings for best compatibility (optional)
    embed_model=None         # embedding model ignored if embed_backend="openai"
)