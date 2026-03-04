"""
Diagnostic script to check what's in the PDF and vector database
"""
import os
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


pdf_path = "data/exchange.pdf"
if os.path.exists(pdf_path):
    print(f"✓ PDF found: {pdf_path}")
    reader = PdfReader(pdf_path)
    print(f"  Total pages: {len(reader.pages)}")
    print("\n--- First 500 chars of page 1 ---")
    text = reader.pages[0].extract_text()
    print(text[:500])
else:
    print(f"✗ PDF NOT found: {pdf_path}")


alt_pdf = "data/exchange.pdf"
if os.path.exists(alt_pdf):
    print(f"\n✓ Alternative PDF found: {alt_pdf}")
    reader = PdfReader(alt_pdf)
    print(f"  Total pages: {len(reader.pages)}")
    print("\n--- First 500 chars of page 1 ---")
    text = reader.pages[0].extract_text()
    print(text[:500])


persist_path = "data/chroma_db"
if os.path.exists(persist_path):
    print(f"\n✓ Vector DB exists at {persist_path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_path)
    

    collection = vector_store._collection
    count = collection.count()
    print(f"  Documents in DB: {count}")
    
  
    print("\n--- Testing retrieval for 'hematology' ---")
    results = vector_store.similarity_search("what does hematology study?", k=3)
    for i, doc in enumerate(results):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200])
else:
    print(f"\n✗ Vector DB NOT found at {persist_path}")

print("\n--- Cache Analysis ---")
import json
cache_file = "data/answer_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    print(f"Cached questions: {len(cache)}")
    for qhash, answer in list(cache.items())[:3]:
        print(f"\n  Answer: {answer[:80]}...")
