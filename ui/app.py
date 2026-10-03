import sys
from pathlib import Path

# Add project root to sys.path so 'src' is discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.indexing.vector_store import VideoVectorStore
from src.retrieval.generator import RAGGenerator
from src.ingestion.pipeline import IngestionPipeline

st.set_page_config(
    page_title="EduTube-RAG",
    page_icon="🎓",
    layout="wide"
)

# Initialize and cache services
@st.cache_resource
def get_services():
    store = VideoVectorStore()
    generator = RAGGenerator(vector_store=store)
    pipeline = IngestionPipeline(vector_store=store)
    return store, generator, pipeline

store, generator, pipeline = get_services()

# Session State Initialization
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = "glLO8cnwj6s"
if "video_start_time" not in st.session_state:
    st.session_state.video_start_time = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Indexing Controls ---
with st.sidebar:
    st.title("🎓 Knowledge Base")
    mode = st.radio("Ingestion Mode", ["Single Video URL", "Search & Index YouTube Topic"])

    if mode == "Single Video URL":
        input_url = st.text_input("YouTube URL or ID", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("Index Video", use_container_width=True):
            if input_url.strip():
                with st.spinner("Fetching transcript and generating embeddings..."):
                    res = pipeline.index_single_video(input_url.strip())
                    if res["status"] == "success":
                        st.session_state.current_video_id = res["video_id"]
                        st.session_state.video_start_time = 0
                        st.success(f"Indexed {res['chunks_count']} chunks for `{res['video_id']}`!")
                    else:
                        st.error(res["message"])

    elif mode == "Search & Index YouTube Topic":
        topic = st.text_input("Enter Topic Query", placeholder="e.g., Transformer self-attention mechanism")
        video_count = st.slider("Number of videos to index", min_value=1, max_value=5, value=3)
        if st.button("Search & Ingest from YouTube", use_container_width=True):
            if topic.strip():
                with st.spinner(f"Searching YouTube and indexing top {video_count} videos..."):
                    res = pipeline.discover_and_index_topic(topic.strip(), max_videos=video_count)
                    if res["indexed_videos"]:
                        st.success(f"Indexed {res['total_chunks']} total chunks across {len(res['indexed_videos'])} videos!")
                        st.session_state.current_video_id = res["indexed_videos"][0]["video_id"]
                        for v in res["indexed_videos"]:
                            st.write(f"- **{v['title']}** ({v['chunks']} chunks)")
                    else:
                        st.warning("No videos with available transcripts found.")

    st.markdown("---")
    st.caption(f"Active Player Video ID: `{st.session_state.current_video_id}`")
    top_k = st.slider("Context chunks to retrieve (Top K)", min_value=1, max_value=6, value=3)

# --- Main Layout: Video Player + Chat ---
st.title("🎓 EduTube-RAG")
st.subheader("Grounded semantic search with direct timestamp jump-links")

col_player, col_chat = st.columns([1, 1], gap="medium")

with col_player:
    st.markdown("### Lecture Player")
    base_youtube_url = f"https://www.youtube.com/watch?v={st.session_state.current_video_id}"
    st.video(base_youtube_url, start_time=st.session_state.video_start_time)
    
    if st.session_state.video_start_time > 0:
        minutes = st.session_state.video_start_time // 60
        seconds = st.session_state.video_start_time % 60
        st.info(f"Cued to: **{minutes:02d}:{seconds:02d}** ({st.session_state.video_start_time}s)")

with col_chat:
    st.markdown("### Ask a Question")
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    user_query = st.chat_input("Ask a question about any indexed video...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching video context and generating answer..."):
                result = generator.generate_answer(user_query, top_k=top_k)
                answer_text = result["answer"]
                sources = result["sources"]

                st.markdown(answer_text)

                if sources:
                    st.markdown("##### Jump Directly in Player:")
                    cols = st.columns(min(len(sources), 4))
                    for i, src in enumerate(sources[:4]):
                        start_sec = int(src.get("start_time", 0))
                        time_lbl = src.get("time_display", f"{start_sec}s")
                        target_vid = src.get("video_id", st.session_state.current_video_id)
                        
                        if cols[i].button(f"▶ {time_lbl}", key=f"jump_{target_vid}_{start_sec}_{i}"):
                            st.session_state.video_start_time = start_sec
                            st.session_state.current_video_id = target_vid
                            st.rerun()

        st.session_state.chat_history.append({
            "query": user_query,
            "answer": answer_text,
            "sources": sources
        })