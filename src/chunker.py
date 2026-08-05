import os
import json
import glob

RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "chunks")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.jsonl")

# Chunk Settings
CHUNK_SIZE = 80     
CHUNK_OVERLAP = 20   
MIN_CHUNK_WORDS = 15 

# Split a list of words into overlapping chunks
def chunk_words(words, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []                   
    start = 0                     
    step = chunk_size - overlap   

    while start < len(words):      
        end = start + chunk_size               
        chunks.append(words[start:end])        
        start += step                          

    return chunks              

# Merge trailing chunk if too small 
def merge_small_trailing_chunk(chunks):
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_WORDS:  
        chunks[-2] = chunks[-2] + chunks[-1]  
        chunks.pop()                         

    return chunks   

# Create list of chunk dictionaries
def chunk_document(doc):
    words = doc["text"].split()                                 
    word_chunks = merge_small_trailing_chunk(chunk_words(words))  

    chunk_records = []                          
    for i, w in enumerate(word_chunks):         
        chunk_records.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{i:03d}",  
            "doc_id": doc["doc_id"],                      
            "chunk_index": i,                               
            "title": doc.get("title", ""),                  
            "url": doc.get("url", ""),                      
            "text": " ".join(w),                           
            "num_words": len(w),                           
        })

    return chunk_records                        

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))

    if not doc_paths:
        print(f"No documents found in {RAW_DIR}/ -- run the scraper first.")
        return

    total_chunks = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for path in doc_paths:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)

            for chunk in chunk_document(doc):
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Chunked {len(doc_paths)} documents into {total_chunks} chunks.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()