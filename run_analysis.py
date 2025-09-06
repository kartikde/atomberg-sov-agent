# run_analysis.py

import os
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
# Load the API key securely from the .env file
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Check if the API key was loaded successfully
if not YOUTUBE_API_KEY:
    raise ValueError("YouTube API key not found. Please create a .env file and add YOUTUBE_API_KEY='your_key'")

# Define brands to track for the "smart fan" query
TARGET_BRAND = "atomberg"
COMPETITOR_BRANDS = ["orient", "crompton", "havells", "usha"] # Example competitors
ALL_BRANDS = [TARGET_BRAND] + COMPETITOR_BRANDS

# --- 2. SENTIMENT ANALYSIS SETUP ---
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded.")

def analyze_sentiment(text):
    """Analyzes sentiment and returns 1 for POSITIVE, -1 for NEGATIVE."""
    try:
        # Truncate text to the model's max input size to avoid errors
        result = sentiment_pipeline(text[:512])[0]
        return 1 if result['label'] == 'POSITIVE' else -1
    except Exception:
        return 0 # Return neutral (0) if analysis fails

# --- 3. YOUTUBE DATA FETCHER ---
try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    print(f"Error building YouTube client: {e}")
    youtube = None

def search_youtube_videos(query, max_results=20):
    """Searches YouTube for a query and returns video details."""
    print(f"\nSearching for top {max_results} videos for query: '{query}'...")
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response.get('items', [])]

    # Get video statistics (views, likes) in a single call for efficiency
    video_details_request = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    )
    video_details_response = video_details_request.execute()
    return video_details_response.get('items', [])

def get_video_comments(video_id, max_comments=50):
    """Fetches top comments for a given video ID."""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            order="relevance"
        )
        response = request.execute()
        return response.get('items', [])
    except Exception:
        # Comments can be disabled, so we handle this gracefully
        return []

# --- 4. SOV ANALYSIS AGENT ---
def analyze_sov(search_query="smart fan", num_videos=20):
    """Main function to run the SoV analysis."""
    if not youtube:
        print("YouTube service is not available. Check your API key or network.")
        return None, None

    videos = search_youtube_videos(query=search_query, max_results=num_videos)
    analysis_data = []

    print(f"Analyzing {len(videos)} videos...")
    for i, video in enumerate(videos):
        video_id = video['id']
        title = video['snippet']['title']
        description = video['snippet']['description']
        view_count = int(video.get('statistics', {}).get('viewCount', '0'))
        
        print(f"  - Processing video {i+1}/{len(videos)}: {title[:50]}...")

        video_text_content = (title + " " + description).lower()
        
        # Analyze mentions in video title/description
        for brand in ALL_BRANDS:
            if brand in video_text_content:
                sentiment = analyze_sentiment(title)
                score = sentiment * view_count # Weighted by views
                analysis_data.append({
                    "brand": brand,
                    "source": "Video",
                    "text": title,
                    "sentiment": "Positive" if sentiment == 1 else "Negative",
                    "engagement": view_count,
                    "wess_score": score
                })

        # Analyze mentions in comments
        comments = get_video_comments(video_id)
        for comment_thread in comments:
            comment = comment_thread['snippet']['topLevelComment']['snippet']
            comment_text = comment['textDisplay'].lower()
            comment_likes = int(comment.get('likeCount', 0))

            for brand in ALL_BRANDS:
                if brand in comment_text:
                    sentiment = analyze_sentiment(comment_text)
                    score = sentiment * (comment_likes + 1) # Weighted by likes (+1 to avoid multiplying by zero)
                    analysis_data.append({
                        "brand": brand,
                        "source": "Comment",
                        "text": comment['textDisplay'],
                        "sentiment": "Positive" if sentiment == 1 else "Negative",
                        "engagement": comment_likes,
                        "wess_score": score
                    })

    if not analysis_data:
        print("No brand mentions found.")
        return None, None

    # --- 5. REPORTING ---
    df = pd.DataFrame(analysis_data)
    sov_report = df.groupby('brand')['wess_score'].sum().sort_values(ascending=False).astype(int)
    
    return df, sov_report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Run the analysis
    results_df, report = analyze_sov(search_query="smart fan", num_videos=20)
    
    if report is not None:
        print("\n" + "="*40)
        print("  SHARE OF VOICE (SoV) FINAL REPORT")
        print("="*40)
        print(report)
        print("="*40)
        
        # Save results to CSV files for your 2-pager document
        results_df.to_csv("sov_analysis_raw_data.csv", index=False)
        report.to_csv("sov_report_summary.csv")
        print("\nSuccessfully saved 'sov_analysis_raw_data.csv' and 'sov_report_summary.csv'.")
        print("You can use these files to create charts and findings for your report.")