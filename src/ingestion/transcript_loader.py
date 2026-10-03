from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Any
import urllib.parse as urlparse

class YouTubeTranscriptLoader:
    """
    Extracts timestamped transcripts from YouTube videos using video IDs or URLs.
    Supports both modern (.fetch()) and legacy (.get_transcript()) library APIs.
    """
    def __init__(self):
        self.api = YouTubeTranscriptApi()

    @staticmethod
    def extract_video_id(url_or_id: str) -> str:
        """Parses a YouTube URL to extract the 11-character video ID."""
        if len(url_or_id) == 11 and "/" not in url_or_id:
            return url_or_id
        
        parsed_url = urlparse.urlparse(url_or_id)
        if parsed_url.hostname in ('youtu.be',):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                query_params = urlparse.parse_qs(parsed_url.query)
                return query_params.get('v', [None])[0]
        raise ValueError(f"Invalid YouTube URL or ID: {url_or_id}")

    def get_transcript(self, video_url_or_id: str, languages=('en',)) -> List[Dict[str, Any]]:
        """
        Retrieves raw transcript segments.
        Always returns a standardized list of dicts:
        [{'text': str, 'start': float, 'duration': float}, ...]
        """
        video_id = self.extract_video_id(video_url_or_id)
        lang_list = list(languages)
        
        try:
            # 1. Try modern API: instance.fetch()
            if hasattr(self.api, "fetch"):
                snippets = self.api.fetch(video_id, languages=lang_list)
            # 2. Try class-level fetch() if available
            elif hasattr(YouTubeTranscriptApi, "fetch"):
                snippets = YouTubeTranscriptApi.fetch(video_id, languages=lang_list)
            # 3. Fallback for older versions: get_transcript()
            elif hasattr(YouTubeTranscriptApi, "get_transcript"):
                snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_list)
            else:
                raise AttributeError("No supported fetch/get_transcript method found on YouTubeTranscriptApi")

            # Standardize output to plain dictionary records
            standardized = []
            for item in snippets:
                if isinstance(item, dict):
                    standardized.append({
                        "text": item.get("text", ""),
                        "start": float(item.get("start", 0.0)),
                        "duration": float(item.get("duration", 0.0))
                    })
                else:
                    standardized.append({
                        "text": getattr(item, "text", ""),
                        "start": float(getattr(item, "start", 0.0)),
                        "duration": float(getattr(item, "duration", 0.0))
                    })
            return standardized

        except Exception as e:
            print(f"Error fetching transcript for video {video_id}: {e}")
            return []

if __name__ == "__main__":
    sample_url = "https://www.youtube.com/watch?v=glLO8cnwj6s"
    loader = YouTubeTranscriptLoader()
    raw_segments = loader.get_transcript(sample_url)
    
    print(f"Total raw transcript snippets fetched: {len(raw_segments)}")
    if raw_segments:
        print("First snippet sample:")
        print(raw_segments[0])