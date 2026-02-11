from pymongo import MongoClient
from youtube_transcript_api._transcripts import TranscriptListFetcher
from youtube_transcript_api._http_client import RequestsClient


YOUTUBE_URL = "https://youtu.be/BYXbuik3dgA"


def extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    else:
        raise ValueError("Invalid YouTube URL")


def fetch_transcript(video_id: str) -> str:
    """
    Works with OLD youtube-transcript-api versions
    """
    http_client = RequestsClient()
    fetcher = TranscriptListFetcher(http_client, None)

    transcript_list = fetcher.fetch(video_id)

    # Prefer English transcript
    if "en" in transcript_list:
        transcript = transcript_list["en"]
    else:
        transcript = next(iter(transcript_list.values()))

    transcript_data = transcript.fetch()

    return " ".join(item["text"] for item in transcript_data)


def store_in_mongo(video_id: str, transcript_text: str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Elon"]
    collection = db["podcasts"]

    doc = {
        "video_id": video_id,
        "transcript": transcript_text
    }

    result = collection.insert_one(doc)
    print(f"✅ Stored in MongoDB with _id: {result.inserted_id}")


def main():
    print("🚀 Starting transcript fetch...")

    video_id = extract_video_id(YOUTUBE_URL)
    print(f"🎥 Video ID: {video_id}")

    transcript_text = fetch_transcript(video_id)
    print(f"📝 Transcript length: {len(transcript_text)} characters")

    store_in_mongo(video_id, transcript_text)

    print("🎉 Done!")


if __name__ == "__main__":
    main()
