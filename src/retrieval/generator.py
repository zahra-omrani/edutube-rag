import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

from src.indexing.vector_store import VideoVectorStore

load_dotenv()


import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

from src.indexing.vector_store import VideoVectorStore

load_dotenv()


class RAGGenerator:
    """
    Combines semantic retrieval from ChromaDB with Gemini generation
    to produce grounded, timestamped answers.
    """
    def __init__(self, vector_store: VideoVectorStore, model_name: str = "gemini-3.8-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

        self.client = genai.Client(api_key=api_key)
        self.vector_store = vector_store
        self.model_name = model_name

    def _build_context_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk["metadata"]
            block = (
                f"[Source {i}]\n"
                f"Video ID: {meta.get('video_id')}\n"
                f"Timestamp: {meta.get('time_display')}\n"
                f"Jump URL: {meta.get('jump_url')}\n"
                f"Transcript Text: {chunk.get('text')}\n"
            )
            context_blocks.append(block)

        combined_context = "\n---\n".join(context_blocks)
        return (
            f"Retrieved Transcript Segments:\n"
            f"{combined_context}\n\n"
            f"User Question: {query}\n\n"
            f"Please answer the question following the grounding and citation rules."
        )

    def _call_gemini_with_retry(self, user_content: str, system_instruction: str, max_retries: int = 4) -> str:
        delay = 10.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    )
                )
                return response.text
            except (errors.ServerError, errors.APIError, errors.ClientError) as e:
                err_msg = str(e).lower()
                if "503" in err_msg or "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
                    print(f"\n[Rate Limit/Capacity Hit] Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                    delay += 15.0
                else:
                    raise e

        return (
            "⚠️ Google's Gemini servers are currently busy or rate limits were exceeded. "
            "Please try again in a moment."
        )

    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs retrieval and grounded generation."""
        retrieved = self.vector_store.search(query, top_k=top_k)
        if not retrieved:
            return {
                "answer": "I could not find any relevant segments in the indexed lectures for this question.",
                "sources": []
            }

        system_instruction = (
            "You are an expert technical AI tutor specializing in educational video lectures. "
            "Your task is to answer the user's question using ONLY the provided transcript segments. "
            "Strict Guidelines:\n"
            "1. Grounding: Do not invent details not present in the transcripts.\n"
            "2. Citations: Every key point or claim must be accompanied by a markdown citation linking directly "
            "to the video timestamp, formatted exactly as: [Watch at MM:SS](Jump URL).\n"
            "3. Conciseness: Provide a direct, well-structured explanation using bullet points or numbered lists where appropriate.\n"
            "4. If the context does not contain enough details, clearly state that."
        )

        user_content = self._build_context_prompt(query, retrieved)
        answer_text = self._call_gemini_with_retry(user_content, system_instruction)

        sources_with_text = []
        for chunk in retrieved:
            item = dict(chunk["metadata"])
            item["text"] = chunk["text"]
            sources_with_text.append(item)

        return {
            "answer": answer_text,
            "sources": sources_with_text
        }

def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved = self.vector_store.search(query, top_k=top_k)
        if not retrieved:
            return {
                "answer": "I could not find any relevant segments in the indexed lectures for this question.",
                "sources": []
            }

        system_instruction = (
            "You are an expert technical AI tutor specializing in educational video lectures. "
            "Your task is to answer the user's question using ONLY the provided transcript segments. "
            "Strict Guidelines:\n"
            "1. Grounding: Do not invent details not present in the transcripts.\n"
            "2. Citations: Every key point or claim must be accompanied by a markdown citation linking directly "
            "to the video timestamp, formatted exactly as: [Watch at MM:SS](Jump URL).\n"
            "3. Conciseness: Provide a direct, well-structured explanation using bullet points or numbered lists where appropriate.\n"
            "4. If the context does not contain enough details, clearly state that."
        )

        user_content = self._build_context_prompt(query, retrieved)
        answer_text = self._call_gemini_with_retry(user_content, system_instruction)

        # Include text along with metadata
        sources_with_text = []
        for chunk in retrieved:
            item = dict(chunk["metadata"])
            item["text"] = chunk["text"]
            sources_with_text.append(item)

        return {
            "answer": answer_text,
            "sources": sources_with_text
        }