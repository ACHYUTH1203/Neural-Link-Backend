import subprocess
import os
import re
import webvtt
from pymongo import MongoClient

# -------------------------
# MongoDB Setup
# -------------------------
def get_collection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Elon"]
    return db["podcasts"]

# -------------------------
# Extract Video ID
# -------------------------
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# -------------------------
# Download Auto Subtitles
# -------------------------
def download_subtitles(url, video_id):
    command = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--skip-download",
        "-o", f"{video_id}.%(ext)s",
        url
    ]
    subprocess.run(command, check=True)

# -------------------------
# Convert VTT → Speaker Format
# -------------------------
def parse_vtt_to_conversation(video_id, interviewer_name="Nikhil"):
    vtt_file = f"{video_id}.en.vtt"

    if not os.path.exists(vtt_file):
        raise FileNotFoundError("Subtitle file not found")

    conversation = []
    turn = 0  # alternate speakers

    for caption in webvtt.read(vtt_file):
        text = caption.text.strip()
        if not text:
            continue

        speaker = interviewer_name if turn % 2 == 0 else "Elon Musk"

        conversation.append({
            "speaker": speaker,
            "text": text
        })

        turn += 1

    return conversation

# -------------------------
# Main Pipeline
# -------------------------
def scrape_youtube_to_mongo(url, interviewer_name="Nikhil"):
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return

    print(f"🎬 Fetching transcript for Video ID: {video_id}")

    try:
        download_subtitles(url, video_id)
        conversation = parse_vtt_to_conversation(video_id, interviewer_name)

        document = {
            "title": f"Transcript for {video_id}",
            "url": url,
            "content": conversation
        }

        collection = get_collection()
        collection.insert_one(document)

        print("✅ Transcript stored EXACTLY in required format")

    except Exception as e:
        print(f"❌ Error: {e}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    target_url = "https://youtu.be/Rni7Fz7208c?si=SJG_PGYNGiLQ-lER"
    scrape_youtube_to_mongo(target_url, interviewer_name="Nikhil")
