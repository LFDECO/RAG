# test_pipeline.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from chunking import extract_text_from_pdf, chunk_text
from vector_store import SimpleVectorStore

def main():
    print("1. Extracting text from 1706.03762v7.pdf...")
    text = extract_text_from_pdf('1706.03762v7.pdf')
    print(f"   Extracted {len(text):,} characters.")

    print("\n2. Chunking text (size=1000, overlap=200)...")
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"   Created {len(chunks)} chunks.")

    print("\n3. Building Vector Store index...")
    vector_store = SimpleVectorStore()
    vector_store.add_texts(chunks)

    queries = [
        "What is the core architecture proposed in the paper?",
        "What is scaled dot product attention formula?",
        "What optimizer and learning rate schedule was used during training?"
    ]

    for q in queries:
        print("\n" + "="*60)
        print(f"QUERY: \"{q}\"")
        print("="*60)
        results = vector_store.search(q, top_k=2)
        for rank, (chunk_text_content, score) in enumerate(results, 1):
            print(f"\n--- [Rank {rank}] (Similarity Score: {score:.4f}) ---")
            preview = chunk_text_content.replace('\n', ' ')[:300]
            print(f"{preview}...")

if __name__ == '__main__':
    main()
