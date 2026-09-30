import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class VideoVectorStore:
    """
    Manages indexing and similarity search for timestamped video chunks
    using ChromaDB and Sentence Transformers.
    """
    def __init__(self, collection_name: str = "edutube_chunks", persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use an efficient, high-performance open-source embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Timestamped YouTube technical video chunks"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embeds and stores video chunks with rich metadata.
        """
        if not chunks:
            return

        documents = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "video_id": c["video_id"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "time_display": c["time_display"],
                "jump_url": c["jump_url"]
            }
            for c in chunks
        ]

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(chunks)} chunks into vector store.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes semantic search against indexed lecture chunks.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        retrieved_items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [None] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                retrieved_items.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return retrieved_items


if __name__ == "__main__":
    from src.ingestion.transcript_loader import YouTubeTranscriptLoader
    from src.processing.temporal_chunker import TemporalChunker

    video_id = "glLO8cnwj6s"
    
    # 1. Fetch
    loader = YouTubeTranscriptLoader()
    raw = loader.get_transcript(video_id)
    
    # 2. Chunk
    chunker = TemporalChunker(window_seconds=75.0, overlap_seconds=15.0)
    chunks = chunker.chunk_transcript(raw, video_id=video_id)
    
    # 3. Index into vector store
    store = VideoVectorStore()
    store.add_chunks(chunks)

    # 4. Test Semantic Query
    test_query = "How should I structure monitoring and feedback loops for an ML project?"
    print(f"\nSearching for: '{test_query}'\n")
    top_matches = store.search(test_query, top_k=2)

    for idx, match in enumerate(top_matches, 1):
        meta = match["metadata"]
        print(f"--- Result {idx} ---")
        print(f"Timestamp: {meta['time_display']} | Jump URL: {meta['jump_url']}")
        print(f"Context: {match['text'][:200]}...\n")