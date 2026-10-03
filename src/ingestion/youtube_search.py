import os
from typing import List, Dict, Any
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

class YouTubeSearcher:
    """
    Queries the official YouTube Data API v3 to discover relevant educational videos.
    """
    def __init__(self):
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY not found in .env file.")
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def search_videos(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Searches YouTube for videos matching a topic query with captions/subtitles enabled.
        """
        request = self.youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            videoCaption="closedCaption",  # ensures transcript exists
            relevanceLanguage="en",
            maxResults=max_results
        )
        response = request.execute()

        results = []
        for item in response.get("items", []):
            results.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            })
        return results

if __name__ == "__main__":
    searcher = YouTubeSearcher()
    query = "PyTorch loss functions explained"
    print(f"Searching YouTube for: '{query}'...")
    videos = searcher.search_videos(query, max_results=3)
    for v in videos:
        print(f"- [{v['video_id']}] {v['title']} ({v['channel']})")