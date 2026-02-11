# import requests
# from bs4 import BeautifulSoup
# import re
# from pymongo import MongoClient

# # 1. MongoDB Connection
# def get_podcasts_collection():
#     # Update URI if using MongoDB Atlas
#     client = MongoClient("mongodb://localhost:27017/") 
#     db = client["Elon"] # Using your existing DB name from the image
#     return db["podcasts"]

# # 2. Cleaning Function
# def clean_transcript_text(raw_text):
#     # Remove timestamps like (00:00:00)
#     text = re.sub(r'\(\d{2}:\d{2}:\d{2}\)', '', raw_text)
#     # Remove extra whitespace and newlines
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # 3. Main Scraper
# def scrape_lex_podcast(url):
#     print(f"Fetching: {url}")
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     response = requests.get(url, headers=headers)
    
#     if response.status_code != 200:
#         print("Failed to retrieve page")
#         return

#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Extract Title
#     title = soup.find('h1', class_='entry-title').get_text(strip=True)
    
#     # Extract Transcript Segments
#     segments = []
#     # Lex's site uses 'ts-segment' for each speaker block
#     for block in soup.find_all('div', class_='ts-segment'):
#         speaker = block.find('span', class_='ts-name').get_text(strip=True)
#         raw_content = block.find('span', class_='ts-text').get_text(strip=True)
        
#         segments.append({
#             "speaker": speaker,
#             "text": clean_transcript_text(raw_content)
#         })

#     # Prepare Data for MongoDB
#     podcast_data = {
#         "title": title,
#         "url": url,
#         "content": segments,
#         "full_text_blob": " ".join([s['text'] for s in segments])
#     }

#     # Save to MongoDB
#     collection = get_podcasts_collection()
#     collection.insert_one(podcast_data)
#     print(f"Successfully saved '{title}' to podcasts collection.")

# # Run it
# if __name__ == "__main__":
#     url = "https://lexfridman.com/elon-musk-4-transcript/"
#     scrape_lex_podcast(url)

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
from pymongo import MongoClient

# 1. MongoDB Setup
def get_collection(name):
    # Connects to your existing local 'Elon' database
    client = MongoClient("mongodb://localhost:27017/")
    return client["Elon"][name]

# 2. Advanced Cleaning Function
def clean_page_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Strip non-text elements to save space in DB
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    
    # Target the primary article or body content
    content = soup.find('article') or soup.find(id='content') or soup.body
    text = content.get_text(separator='\n', strip=True) if content else ""
    # Remove excessive newlines for cleaner AI context
    return re.sub(r'\n+', '\n', text).strip()

# 3. Execution Logic
async def scrape_elon_intelligence():
    sources = [
        {"title": "5-Step Design Process", "url": "https://modelthinkers.com/mental-model/musks-5-step-design-process", "coll": "frameworks"},
        {"title": "Algorithm to Cut Bureaucracy", "url": "https://www.corporate-rebels.com/blog/musks-algorithm-to-cut-bureaucracy", "coll": "frameworks"},
        {"title": "Founders Podcast: How Elon Works", "url": "https://youtu.be/aStHTTPxlis", "coll": "biographies"},
        {"title": "Nikhil Kamath Podcast", "url": "https://youtu.be/Rni7Fz7208c", "coll": "podcasts"}
    ]

    async with async_playwright() as p:
        # Launching with a real user-agent to bypass the "Access Denied" blocks
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )

        for item in sources:
            print(f"Scraping: {item['title']}...")
            page = await context.new_page()
            try:
                # Wait for the page to be fully loaded (networkidle)
                await page.goto(item['url'], wait_until="networkidle", timeout=60000)
                html = await page.content()
                
                cleaned_text = clean_page_content(html)
                
                # Update DB (Upsert ensures no duplicates)
                get_collection(item['coll']).update_one(
                    {"title": item['title']},
                    {"$set": {
                        "content": cleaned_text, 
                        "url": item['url'], 
                        "timestamp": "2026-02-10"
                    }},
                    upsert=True
                )
                print(f"  Successfully stored in '{item['coll']}'")
            except Exception as e:
                print(f"  Error scraping {item['title']}: {e}")
            finally:
                await page.close()

        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_elon_intelligence())