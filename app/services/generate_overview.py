import os
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from sklearn.metrics.pairwise import cosine_similarity
try:
    from langchain_community.chat_models import ChatOpenAI
except Exception:
    ChatOpenAI = None

@dataclass
class TestResult:
    query: str
    ground_truth: str
    answer: str
    raw_score: float
    score: float

class QAAccuracyTester:
    def __init__(
        self,
        vectorstore,
        llm_backend: str = "auto",
        llm_model: str = "llama3",
        embed_backend: str = "huggingface",
        embed_model: str = "all-MiniLM-L6-v2",
        chain_type: str = "stuff",
    ):
        self.vectorstore = vectorstore
        self.llm = self._init_llm(llm_backend, llm_model)
        self.qa_chain = load_qa_chain(self.llm, chain_type=chain_type)
        self.embedding = self._init_embedding(embed_backend, embed_model)

        self.chiron_prompt_template = """
              You are a knowledgeable astrologer and compassionate counselor. Using the provided documents, create a detailed and insightful analysis of the astrological placement: **Chiron in {sign} in the {house}**.

              Please organize your response according to the following outline to maintain clarity and depth:

              ---

              **Placement**: Chiron in {sign} ({house})

              **Core Wounded Themes**:
              - Describe the fundamental emotional or psychological wounds associated with both the {sign} and the {house}.
              - Explain how the qualities of {sign} and the life areas ruled by the {house} combine and influence this placement.

              **Summary Overview**:
              - Provide a concise but meaningful paragraph summarizing what Chiron in {sign} in the {house} symbolizes on a soul level or life-lesson perspective.

              **Wounded Keywords**:
              - List the most meaningful keywords reflecting the core wounds linked to this placement, separated by commas.

              **Healing Keywords**:
              - List key terms that capture the primary modes and qualities of healing for this placement, separated by commas.

              **Primary Challenges**:
              - Use bullet points to outline common internal conflicts, recurring patterns, or experiences typical for individuals with this placement.

              **Path to Healing**:
              - Use bullet points to describe practical and emotional strategies for healing.
              - Emphasize growth, self-compassion, trust-building, and transforming wounds into strengths or gifts.

              ---

              Please synthesize information from all relevant documents about Chiron, the {sign}, the {house}, and their combination. Avoid simply repeating phrases; instead, rephrase and integrate the content to produce a cohesive, original explanation.

              Now, answer the question: What does it mean to have Chiron in {sign} in the {house}?
              """

        self.prompt = PromptTemplate(
            input_variables=["sign", "house"],
            template=self.chiron_prompt_template
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    def _init_llm(self, backend: str, model_name: str):
        load_dotenv()
        backend = backend.lower()
        print("open ai api key-->",os.getenv("OPENAI_API_KEY"))
      
        if backend == "auto":
            print("open ai api key-->",os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
                backend = "openai"
            else:
                backend = "ollama"

        if backend == "openai":
            if ChatOpenAI is None:
                raise RuntimeError("ChatOpenAI not available in your environment.")
            return ChatOpenAI(
                  model=model_name or "gpt-4o-mini",
                  temperature=0,
                  openai_api_key=os.getenv("OPENAI_API_KEY")  # ✅ Pass explicitly
              )
        elif backend == "ollama":
            return Ollama(model=model_name or "llama3")
        else:
            raise ValueError(f"Unsupported llm_backend: {backend}")

    def _init_embedding(self, backend: str, model_name: str):
        backend = backend.lower()
        if backend == "auto":
            if os.getenv("OPENAI_API_KEY"):
                from langchain.embeddings import OpenAIEmbeddings
                return OpenAIEmbeddings()
            else:
                backend = "huggingface"

        if backend == "openai":
            from langchain.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings()
        elif backend == "huggingface":
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embed_backend: {backend}")

    def retrieve_docs(self, query: str, k: int = 5, show_preview: bool = False) -> List[Any]:
        try:
            if hasattr(self.vectorstore, "as_retriever"):
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                docs = retriever.get_relevant_documents(query)
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
        except Exception:
            docs = self.vectorstore.similarity_search(query, k=k)

        if show_preview:
            for i, doc in enumerate(docs):
                print(f"\n=== Chunk {i+1} ===\n{doc.page_content[:1000]}")
        return docs

    def get_answer(self, query: str, docs: List[Any]) -> str:
        try:
            return self.qa_chain.run(input_documents=docs, question=query)
        except Exception as exc:
            print("Error while running QA chain:", exc)
            return ""

    def compute_accuracy(self, model_answer: str, ground_truth: str) -> Tuple[float, float]:
        gt_emb = self.embedding.embed_query(ground_truth)
        ans_emb = self.embedding.embed_query(model_answer)
        raw = cosine_similarity([gt_emb], [ans_emb])[0][0]
        normalized = (raw + 1.0) / 2.0
        normalized = max(0.0, min(1.0, normalized))
        return float(raw), float(normalized)

    def run_single_test(self, query: str, ground_truth: str, k: int = 5, show_preview: bool = False) -> Dict[str, Any]:
        docs = self.retrieve_docs(query, k=k, show_preview=show_preview)
        answer = self.get_answer(query, docs)
        raw, normalized = self.compute_accuracy(answer, ground_truth)
        return {
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "raw_score": raw,
            "score": normalized,
        }

    def run_batch_tests(self, test_cases: List[Tuple[str, str]], k: int = 5, show_preview: bool = False):
        results = []
        for q, gt in test_cases:
            res = self.run_single_test(q, gt, k=k, show_preview=show_preview)
            results.append(res)
        avg_score = sum(r["score"] for r in results) / max(1, len(results))
        return results, avg_score

    def retrieve_and_combine_docs(self, query_sign, query_house, k=5, show_preview=False):
        docs_sign = self.vectorstore.similarity_search(query_sign, k=k)
        docs_house = self.vectorstore.similarity_search(query_house, k=k)
        combined_docs = docs_sign + docs_house
        if show_preview:
            print("=== Sign Docs Preview ===")
            for i, doc in enumerate(docs_sign):
                print(f"\n--- Sign Chunk {i+1} ---\n{doc.page_content[:500]}")
            print("\n=== House Docs Preview ===")
            for i, doc in enumerate(docs_house):
                print(f"\n--- House Chunk {i+1} ---\n{doc.page_content[:500]}")
        return combined_docs

    def get_combined_chiron_answer(self, sign, house, k=5, show_preview=False):
        query_sign = f"What does Chiron in {sign} mean?"
        query_house = f"What does Chiron in the {house} house mean?"
        combined_docs = self.retrieve_and_combine_docs(query_sign, query_house, k=k, show_preview=show_preview)
        answer = self.qa_chain.run(
            sign=sign,
            house=house,
            input_documents=combined_docs
        )
        return answer


# Initialize vectorstore and QAAccuracyTester instance for reuse
print("Loading embeddings and vectorstore...")
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vectorstore = FAISS.load_local("app/vector_db/pdf_book", _embeddings,  allow_dangerous_deserialization=True)

qa_accuracy_tester =QAAccuracyTester(
    _vectorstore,
    llm_backend="openai",      # force OpenAI usage
    llm_model="gpt-4o-mini",   # or "gpt-4o", "gpt-4", etc. depending on your subscription
    embed_backend="openai",    # use OpenAI embeddings for best compatibility (optional)
    embed_model=None           # embedding model ignored if embed_backend="openai"
)
