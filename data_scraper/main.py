
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import pytesseract
import json
import os

# --- Configuration ---
URL = "https://waitbutwhy.com/2015/05/elon-musk-introduction.html"
OUTPUT_DIR = "images"
OUTPUT_JSON = "output.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helper: download image
# -------------------------
def download_image(url, idx):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    ext = img.format.lower() if img.format else "png"
    path = f"{OUTPUT_DIR}/img_{idx}.{ext}"
    img.save(path)

    return path, img

# -------------------------
# Helper: decide OCR or not
# -------------------------
def should_ocr(img_url, alt_text):
    keywords = ["text", "diagram", "chart", "note", "drawing"]
    # Wait But Why uses many hand-drawn notes; OCR is often useful here
    return (
        not alt_text or 
        any(k in img_url.lower() for k in keywords) or
        any(k in alt_text.lower() for k in keywords)
    )

# -------------------------
# OCR image
# -------------------------
def ocr_image(img: Image.Image) -> str:
    # Adding config to improve accuracy for handwritten-style text if needed
    return pytesseract.image_to_string(img).strip()

# -------------------------
# Main scraper
# -------------------------
def scrape():
    print(f"🚀 Starting scrape of: {URL}")
    try:
        response = requests.get(URL, timeout=20)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"❌ Failed to fetch URL: {e}")
        return []

    soup = BeautifulSoup(html, "lxml")

    # WBW content is usually inside 'article' or 'entry-content'
    article = soup.find("article") or soup.find(class_="entry-content")
    if not article:
        raise Exception("Could not find the main article container.")

    content_blocks = []
    img_count = 0

    # Expanded tag list to capture all text types: headers, paragraphs, lists, and quotes
    tags_to_track = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "img"]

    for elem in article.find_all(tags_to_track, recursive=True):
        
        # -------- IMAGE BLOCK --------
        if elem.name == "img":
            src = elem.get("src")
            if not src or "ad-server" in src or "pixel" in src:
                continue

            img_url = urljoin(URL, src)
            alt = elem.get("alt", "")

            try:
                img_path, img_obj = download_image(img_url, img_count)
                
                block = {
                    "type": "image",
                    "src": img_url,
                    "local_path": img_path,
                    "alt": alt
                }

                if should_ocr(img_url, alt):
                    ocr_text = ocr_image(img_obj)
                    if ocr_text:
                        block["ocr_text"] = ocr_text

                content_blocks.append(block)
                img_count += 1
                print(f"📸 Saved image {img_count}")
            except Exception as e:
                print(f"⚠️ Skipping image {img_url}: {e}")

        # -------- TEXT BLOCK --------
        else:
            text = elem.get_text(strip=True)
            if text:
                # Deduplication: BeautifulSoup find_all can sometimes grab a tag 
                # and its parent if both match. This check prevents double-saving.
                if content_blocks and content_blocks[-1].get("content") == text:
                    continue
                    
                content_blocks.append({
                    "type": "text",
                    "tag": elem.name,
                    "content": text
                })

    return content_blocks

# -------------------------
# Run scraper
# -------------------------
if __name__ == "__main__":
    data = scrape()

    if data:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Success! Saved {len(data)} items to {OUTPUT_JSON}")
    else:
        print("❌ No data was scraped.")