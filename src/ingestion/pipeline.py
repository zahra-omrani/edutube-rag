import sys
from pathlib import Path
from typing import List, Dict, Any

from src.ingestion.transcript_loader import YouTubeTranscriptLoader
from src.ingestion.youtube_search import YouTubeSearcher
from src.processing.temporal_chunker import TemporalChunker
from src.indexing.vector_store import VideoVectorStore

class IngestionPipeline:
    """
    Coordinates discovery, transcript fetching, temporal chunking,
    and vector store persistence across one or multiple videos.
    """
    def __init__(self, vector_store: VideoVectorStore):
        self.vector_store = vector_store
        self.loader = YouTubeTranscriptLoader()
        self.chunker = TemporalChunker(window_seconds=75.0, overlap_seconds=15.0)
        self.searcher = YouTubeSearcher()

    def index_single_video(self, video_url_or_id: str) -> Dict[str, Any]:
        """Fetches and indexes a single video by URL or ID."""
        video_id = self.loader.extract_video_id(video_url_or_id)
        raw_snippets = self.loader.get_transcript(video_id)
        
        if not raw_snippets:
            return {"status": "error", "message": f"No captions found for {video_id}", "chunks": 0}

        chunks = self.chunker.chunk_transcript(raw_snippets, video_id=video_id)
        self.vector_store.add_chunks(chunks)
        
        return {
            "status": "success",
            "video_id": video_id,
            "chunks_count": len(chunks)
        }

    def discover_and_index_topic(self, topic_query: str, max_videos: int = 3) -> Dict[str, Any]:
        """
        Discovers the top educational videos for a topic and indexes their transcripts.
        """
        videos = self.searcher.search_videos(topic_query, max_results=max_videos)
        indexed_videos = []
        total_chunks = 0

        for vid in videos:
            v_id = vid["video_id"]
            raw_snippets = self.loader.get_transcript(v_id)
            if raw_snippets:
                chunks = self.chunker.chunk_transcript(raw_snippets, video_id=v_id)
                self.vector_store.add_chunks(chunks)
                total_chunks += len(chunks)
                indexed_videos.append({
                    "video_id": v_id,
                    "title": vid["title"],
                    "channel": vid["channel"],
                    "chunks": len(chunks)
                })

        return {
            "status": "success",
            "indexed_videos": indexed_videos,
            "total_chunks": total_chunks
        }