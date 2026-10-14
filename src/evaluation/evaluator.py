import os
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

from src.indexing.vector_store import VideoVectorStore
from src.retrieval.generator import RAGGenerator

load_dotenv()


class RAGEvaluator:
    def __init__(self, generator: RAGGenerator, store: VideoVectorStore):
        self.generator = generator
        self.store = store
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)

    def evaluate_retrieval(self, query: str, expected_keywords: List[str], top_k: int = 3) -> Dict[str, Any]:
        results = self.store.search(query, top_k=top_k)
        retrieved_texts = " ".join([r["text"].lower() for r in results])

        hits = [kw for kw in expected_keywords if kw.lower() in retrieved_texts]
        recall_score = len(hits) / len(expected_keywords) if expected_keywords else 0.0

        return {
            "query": query,
            "hits": hits,
            "misses": [kw for kw in expected_keywords if kw not in hits],
            "keyword_recall": round(recall_score, 2),
            "chunks_retrieved": len(results)
        }

    def evaluate_generation_faithfulness(self, query: str, context: str, answer: str, max_retries: int = 4) -> Dict[str, Any]:
        judge_prompt = f"""
You are an unbiased AI evaluator.
Evaluate the following generated answer against the provided context.

Context:
{context}

Question:
{query}

Answer to evaluate:
{answer}

Evaluate based on two criteria:
1. Groundedness (1-5): Are all claims in the answer supported by the context without hallucination?
2. Relevance (1-5): Does the answer directly answer the user's question?

Respond in valid JSON with this exact structure:
{{
  "groundedness_score": <int 1-5>,
  "relevance_score": <int 1-5>,
  "rationale": "<brief explanation>"
}}
"""
        delay = 10.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=judge_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                try:
                    return json.loads(response.text)
                except Exception:
                    return {"raw_evaluation": response.text}
            except (errors.ServerError, errors.APIError, errors.ClientError) as e:
                err_msg = str(e).lower()
                if "503" in err_msg or "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
                    print(f"\n[Judge Rate Limit Hit] Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                    delay += 15.0
                else:
                    raise e

        return {
            "groundedness_score": 0,
            "relevance_score": 0,
            "rationale": "Evaluation failed due to upstream quota/capacity limits."
        }

    def run_benchmark(self, benchmark_file: str) -> List[Dict[str, Any]]:
        with open(benchmark_file, "r") as f:
            benchmarks = json.load(f)

        reports = []
        for i, item in enumerate(benchmarks, start=1):
            q = item["question"]
            keywords = item.get("expected_keywords", [])
            print(f"\n--- Evaluating Test Case {i}/{len(benchmarks)}: '{q}' ---")

            # 1. Retrieval
            retrieval_metrics = self.evaluate_retrieval(q, keywords)

            # 2. Generation
            rag_output = self.generator.generate_answer(q, top_k=3)
            retrieved_context = "\n".join([f"[{s['time_display']}] {s.get('text', '')}" for s in rag_output["sources"]])

            # Rate limit cooldown between generation and evaluation call
            time.sleep(12)

            # 3. LLM Judge
            llm_metrics = self.evaluate_generation_faithfulness(
                query=q,
                context=retrieved_context,
                answer=rag_output["answer"]
            )

            reports.append({
                "question": q,
                "retrieval_evaluation": retrieval_metrics,
                "llm_judge_evaluation": llm_metrics,
                "generated_answer": rag_output["answer"]
            })

            # Cooldown before the next test case
            time.sleep(12)

        return reports


if __name__ == "__main__":
    store = VideoVectorStore()
    gen = RAGGenerator(vector_store=store)
    evaluator = RAGEvaluator(generator=gen, store=store)

    benchmark_path = str(ROOT_DIR / "src" / "evaluation" / "test_benchmark.json")
    print(f"Running automated evaluation on benchmark: {benchmark_path}...")

    results = evaluator.run_benchmark(benchmark_path)
    print("\n================ FINAL BENCHMARK RESULTS ================")
    print(json.dumps(results, indent=2))