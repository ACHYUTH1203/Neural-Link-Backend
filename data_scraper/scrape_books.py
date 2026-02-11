from pypdf import PdfReader
from pymongo import MongoClient

# -------------------------
# MongoDB Setup
# -------------------------
def get_books_collection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Elon"]
    return db["books"]

# -------------------------
# PDF Text Extraction
# -------------------------
def extract_text_from_pdf(file_path):
    print(f"📖 Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    full_text = ""
    
    # Loop through each page to extract text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    return full_text

# -------------------------
# Main Logic
# -------------------------
def add_book_to_db():
    pdf_path = r"C:\Users\achyu\Desktop\elonmusk_book.pdf"
    
    try:
        book_content = extract_text_from_pdf(pdf_path)
        
        # Prepare the document as requested
        book_data = {
            "book name": "Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future",
            "author": "Ashlee Vance",
            "content": book_content,
            "type": "biography"
        }
        
        # Save to MongoDB
        collection = get_books_collection()
        collection.update_one(
            {"book name": book_data["book name"]}, 
            {"$set": book_data}, 
            upsert=True
        )
        
        print(f"✅ Successfully added '{book_data['book name']}' to books collection.")
        
    except Exception as e:
        print(f"❌ Error adding book: {e}")

if __name__ == "__main__":
    add_book_to_db()