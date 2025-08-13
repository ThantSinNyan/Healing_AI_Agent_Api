# app/services/qa_service.py
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from .llm_service import init_llm, init_embedding
from langchain_huggingface import HuggingFaceEmbeddings
from app.prompts.chiron_overview_prompt import CHIRON_PROMPT
from app.models.healing_model import ChironAnalysisResponse
from app.utils.chiron_parser import parse_chiron_text
from concurrent.futures import ThreadPoolExecutor

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
        self.llm = init_llm(llm_backend, llm_model)
        self.qa_chain = load_qa_chain(self.llm, chain_type=chain_type)
        self.embedding = init_embedding(embed_backend, embed_model)

        self.prompt = PromptTemplate(
            input_variables=["sign", "house"],
            template=CHIRON_PROMPT
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.prompt)

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
      combined_docs = self.retrieve_and_combine_docs(
          query_sign, query_house, k=k, show_preview=show_preview
      )

      raw_answer = self.qa_chain.run(
          sign=sign,
          house=house,
          input_documents=combined_docs
      )
      formatedAnswer=parse_chiron_text(raw_answer)
      print("formatedAnswer-->",formatedAnswer)

      return formatedAnswer
    
    def get_combined_chiron_answer(self, sign, house, k=5):
      # Build the two separate queries
      query_sign = f"Chiron in {sign}"
      query_house = f"Chiron in the {house} house"

      # Parallel retrieval
      with ThreadPoolExecutor() as executor:
          future_sign = executor.submit(self.vectorstore.similarity_search, query_sign, k=max(1, k // 2))
          future_house = executor.submit(self.vectorstore.similarity_search, query_house, k=max(1, k // 2))
          docs_sign = future_sign.result()
          docs_house = future_house.result()

      # Combine retrieved documents
      combined_docs = docs_sign + docs_house

      # Trim document content to limit token usage (max ~2000 chars total)
      max_chars = 2000
      trimmed_docs = []
      char_count = 0
      for doc in combined_docs:
          if char_count + len(doc.page_content) <= max_chars:
              trimmed_docs.append(doc)
              char_count += len(doc.page_content)
      combined_docs = trimmed_docs

      # Run QA chain (prompt remains exactly the same)
      raw_answer = self.qa_chain.run(input_documents=combined_docs, sign=sign, house=house)
      formatedAnswer=parse_chiron_text(raw_answer)
  
      return formatedAnswer