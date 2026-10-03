import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.indexing.vector_store import VideoVectorStore

# Load environment variables (.env)
load_dotenv()


class RAGGenerator:
    """
    Combines semantic retrieval from ChromaDB with Gemini LLM generation
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
        """
        Formats retrieved video chunks into a structured prompt with clear boundaries.
        """
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

        user_content = (
            f"Retrieved Transcript Segments:\n"
            f"{combined_context}\n\n"
            f"User Question: {query}\n\n"
            f"Please answer the question following the grounding and citation rules."
        )
        return user_content

    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves context and generates a grounded response.
        """
        # 1. Retrieve most relevant video segments
        retrieved = self.vector_store.search(query, top_k=top_k)
        if not retrieved:
            return {
                "answer": "I could not find any relevant segments in the indexed lectures for this question.",
                "sources": []
            }

        # 2. System instructions for anti-hallucination and citation format
        system_instruction = (
            "You are an expert technical AI tutor specializing in educational video lectures. "
            "Your task is to answer the user's question using ONLY the provided transcript segments. "
            "Strict Guidelines:\n"
            "1. Grounding: Do not invent details not present in the transcripts.\n"
            "2. Citations: Every key point or claim must be accompanied by a markdown citation linking directly "
            "to the video timestamp, formatted exactly as: [Watch at MM:SS](Jump URL).\n"
            "3. Conciseness: Provide a direct, well-structured explanation using bullet points or numbered lists where appropriate.\n"
            "4. If the provided context does not contain enough information to answer the question, state clearly that "
            "the lecture does not cover that specific detail."
        )

        user_content = self._build_context_prompt(query, retrieved)

        # 3. Call Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,  # Low temperature for high factual grounding
            )
        )

        return {
            "answer": response.text,
            "sources": [chunk["metadata"] for chunk in retrieved]
        }


if __name__ == "__main__":
    # Test end-to-end retrieval + generation
    store = VideoVectorStore()
    rag = RAGGenerator(vector_store=store)

    test_query = "How should I structure monitoring and feedback loops for an ML project?"
    print(f"Query: {test_query}\n")
    print("Generating grounded response from Gemini...\n")

    result = rag.generate_answer(test_query, top_k=2)
    print("=== Generated Answer ===")
    print(result["answer"])
    print("\n=== Sources Referenced ===")
    for src in result["sources"]:
        print(f"- {src['time_display']} -> {src['jump_url']}")