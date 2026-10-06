import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.indexing.vector_store import VideoVectorStore
from src.retrieval.generator import RAGGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.api.schemas import (
    SingleVideoIndexRequest,
    TopicIndexRequest,
    VideoIndexResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="EduTube-RAG API",
    description="REST backend for indexing educational YouTube lectures and answering questions with timestamped jump-links.",
    version="1.0.0",
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service singletons
store = VideoVectorStore()
generator = RAGGenerator(vector_store=store)
pipeline = IngestionPipeline(vector_store=store)


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend service status."""
    return {"status": "ok", "service": "EduTube-RAG"}


@app.post("/index/video", response_model=VideoIndexResponse)
def index_single_video(payload: SingleVideoIndexRequest):
    """Fetches, chunks, and indexes a single YouTube video by URL or ID."""
    try:
        result = pipeline.index_single_video(payload.video_url_or_id)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/topic", response_model=VideoIndexResponse)
def index_topic(payload: TopicIndexRequest):
    """Discovers top educational YouTube videos on a topic and indexes their transcripts."""
    try:
        result = pipeline.discover_and_index_topic(payload.topic, max_videos=payload.max_videos)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def answer_query(payload: QueryRequest):
    """Retrieves relevant transcript chunks and generates a grounded response with Gemini."""
    try:
        res = generator.generate_answer(query=payload.query, top_k=payload.top_k)
        return {
            "query": payload.query,
            "answer": res["answer"],
            "sources": res["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))