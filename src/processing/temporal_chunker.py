from typing import List, Dict, Any

class TemporalChunker:
    """
    Aggregates granular transcript snippets into temporal windows 
    while preserving timestamp metadata and generating deep-link URLs.
    """
    def __init__(self, window_seconds: float = 75.0, overlap_seconds: float = 15.0):
        """
        :param window_seconds: Target temporal duration for each chunk.
        :param overlap_seconds: Overlap duration between consecutive chunks to preserve context.
        """
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds

    def chunk_transcript(self, raw_snippets: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """
        Transforms raw snippets [{'text': ..., 'start': ..., 'duration': ...}]
        into structured chunks with start/end bounds and YouTube jump URLs.
        """
        if not raw_snippets:
            return []

        chunks = []
        n = len(raw_snippets)
        i = 0

        while i < n:
            start_time = raw_snippets[i]["start"]
            target_end_time = start_time + self.window_seconds
            
            chunk_texts = []
            j = i
            end_time = start_time

            while j < n and raw_snippets[j]["start"] < target_end_time:
                chunk_texts.append(raw_snippets[j]["text"].strip())
                end_time = raw_snippets[j]["start"] + raw_snippets[j]["duration"]
                j += 1

            combined_text = " ".join(chunk_texts).replace("\n", " ").strip()
            
            # Format seconds to integer for URL anchor
            start_int = int(start_time)
            jump_url = f"https://www.youtube.com/watch?v={video_id}&t={start_int}s"

            # Compute human-readable mm:ss format
            minutes = start_int // 60
            seconds = start_int % 60
            time_display = f"{minutes:02d}:{seconds:02d}"

            chunks.append({
                "chunk_id": f"{video_id}_{start_int}",
                "video_id": video_id,
                "text": combined_text,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "time_display": time_display,
                "jump_url": jump_url
            })

            # Advance index considering overlap
            if j >= n:
                break
            
            # Find next starting snippet after (end_time - overlap_seconds)
            next_start_target = end_time - self.overlap_seconds
            next_i = i + 1
            while next_i < j and raw_snippets[next_i]["start"] < next_start_target:
                next_i += 1
            
            i = next_i if next_i > i else i + 1

        return chunks


if __name__ == "__main__":
    from src.ingestion.transcript_loader import YouTubeTranscriptLoader

    sample_id = "glLO8cnwj6s"
    loader = YouTubeTranscriptLoader()
    raw_snippets = loader.get_transcript(sample_id)

    chunker = TemporalChunker(window_seconds=75.0, overlap_seconds=15.0)
    chunks = chunker.chunk_transcript(raw_snippets, video_id=sample_id)

    print(f"Aggregated {len(raw_snippets)} raw snippets into {len(chunks)} temporal chunks.")
    if chunks:
        print("\n--- First Chunk Preview ---")
        first = chunks[0]
        print(f"Time: {first['time_display']} ({first['start_time']}s -> {first['end_time']}s)")
        print(f"Jump Link: {first['jump_url']}")
        print(f"Text snippet: {first['text'][:150]}...")