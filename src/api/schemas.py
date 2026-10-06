from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SingleVideoIndexRequest(BaseModel):
    video_url_or_id: str = Field(..., description="YouTube video URL or 11-character video ID")

class TopicIndexRequest(BaseModel):
    topic: str = Field(..., description="Search query/topic for educational video discovery")
    max_videos: int = Field(default=3, ge=1, le=5, description="Number of videos to discover and index")

class VideoIndexResponse(BaseModel):
    status: str
    message: Optional[str] = None
    video_id: Optional[str] = None
    chunks_count: Optional[int] = 0
    indexed_videos: Optional[List[Dict[str, Any]]] = None
    total_chunks: Optional[int] = 0

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to answer using indexed video transcripts")
    top_k: int = Field(default=3, ge=1, le=6, description="Number of context segments to retrieve")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]