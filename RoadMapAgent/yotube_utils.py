import langchain
from googleapiclient.discovery import build
from langchain.tools import BaseTool
from langchain_community.document_loaders import YoutubeLoader


class YoutubeUtilTool:

    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube1 = build('youtube', 'v3', developerKey=self.api_key)

    def search_video(self, query: str, max_results: int = 2):
        search_response = self.youtube1.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results
        ).execute()
        videos = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                video_id = search_result['id']['videoId']
                print(video_id)
                video_data = {
                    'title': search_result['snippet']['title'],
                    'description': search_result['snippet']['description'],
                    'videoId': video_id,
                    'url': 'https://www.youtube.com/watch?v='+video_id
                }
                videos.append(video_data)
        return videos
    
    def run(self, query: str):
        result = self.search_video(query)
        return result
    
    def extractTranscript(self, url: str):
        print(url)
        loder = YoutubeLoader.from_youtube_url(url)
        transcript = loder.load()
        return transcript 
    
        
