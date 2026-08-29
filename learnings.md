# Engineering Learnings: RAG from Scratch

A comprehensive engineering reference capturing first-principles intuition, mathematical formulas, system architecture diagrams, and empirical findings from building a baseline RAG pipeline.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph INDEXING ["STAGE 1: INDEXING (Offline Data Preparation)"]
        A["1. Raw PDF Document"] --> B["2. Page Text Extraction (pypdf)"]
        B --> C["3. Fixed-Size Sliding Window Chunking"]
        C --> D["4. SentenceTransformer ('all-MiniLM-L6-v2')"]
        D --> E["5. NumPy Vector Matrix (50 x 384)"]
    end

    subgraph RETRIEVAL ["STAGE 2: RETRIEVAL (Runtime Search)"]
        F["6. User Natural Language Query"] --> G["7. Query Embedding Vector (1 x 384)"]
        E & G --> H["8. NumPy Cosine Similarity Search"]
        H --> I["9. Top-K Index Extraction: np.argsort(scores)[::-1][:top_k]"]
    end

    subgraph GENERATION ["STAGE 3: GENERATION (Answer Synthesis)"]
        I --> J["10. Prompt Context Assembly"]
        J --> K["11. Gemini API (gemini-3.6-flash)"]
        K --> L["12. Grounded Final Answer"]
    end

    style INDEXING fill:#1a365d,stroke:#2b6cb0,color:#fff
    style RETRIEVAL fill:#2c5282,stroke:#3182ce,color:#fff
    style GENERATION fill:#234e52,stroke:#319795,color:#fff
```

---

## 2. Document Processing & Sliding Window Chunking

### Core Intuition
Rather than embedding an entire 15-page PDF into a single vector (which causes **semantic dilution**), we segment the text into focused passages (chunks).

```mermaid
flowchart LR
    subgraph SLIDING_WINDOW ["Sliding Window Mechanics"]
        direction LR
        W1["Window 1: Chars 0..1000"]
        W2["Window 2: Chars 800..1800"]
        W3["Window 3: Chars 1600..2600"]
        
        W1 -. "Overlap (200 chars)" .-> W2
        W2 -. "Overlap (200 chars)" .-> W3
    end
```

### Mathematical Formula
For a given `chunk_size` and `chunk_overlap`:
$$\text{step} = \text{chunk\_size} - \text{chunk\_overlap}$$

> [!WARNING]
> **Empirical Failure Mode: Fixed Character Slicing**
> Slicing text strictly at character index `1000` disregards word and sentence boundaries. 
> * **Observed Output:** `"...dispensing with recurrence and convolutions entirely. Exper"` in Chunk 0, and `"iments on two machine..."` in Chunk 1.
> * **Impact:** Severing `"Experiments"` into `"Exper"` and `"iments"` degrades tokenizer vocabulary matching, drops vector cosine similarity scores, and confuses the LLM context window.

---

## 3. Dense Vector Embeddings & Similarity Search (Pure NumPy)

Embedding models transform raw text into $D$-dimensional dense floating-point arrays ($\mathbb{R}^{384}$).

```mermaid
flowchart TD
    A["Matrix of Document Embeddings M: (50 x 384)"]
    B["Query Vector q: (1 x 384)"]
    
    A & B --> C["1. Dot Product: np.dot(M, q) -> Shape (50,)"]
    C --> D["2. Matrix Row Norms: np.linalg.norm(M, axis=1)"]
    D --> E["3. Query Norm: np.linalg.norm(q)"]
    E --> F["4. Cosine Similarity Score Array (50,)"]
    F --> G["5. Top-K Indices: np.argsort(scores)[::-1][:top_k]"]
```

### Mathematical Formulations

1. **Dot Product:**
   $$\text{dot\_p} = M \cdot \vec{q} = \sum_{j=1}^{D} M_{i, j} \cdot q_j$$

2. **Euclidean Norm (Magnitude):**
   $$\|M_i\|_2 = \sqrt{\sum_{j=1}^{D} M_{i, j}^2}$$

3. **Cosine Similarity:**
   $$\text{sim}(M_i, \vec{q}) = \frac{M_i \cdot \vec{q}}{\|M_i\|_2 \|\vec{q}\|_2}$$

4. **Top-K Indexing Chain:**
   ```python
   top_indices = np.argsort(scores)[::-1][:top_k]
   ```
   * `np.argsort(scores)` $\to$ Returns indices sorted ascending (lowest score first).
   * `[::-1]` $\to$ Reverses array to descending order (highest score first).
   * `[:top_k]` $\to$ Slices the top $K$ best matching index positions.

---

## 4. RAG Prompt Assembly & LLM Generation

To prevent hallucinations, the retrieved top $K$ passages are structured into an explicit system prompt.

```mermaid
sequenceDiagram
    participant User
    participant VectorStore
    participant RAGChain
    participant GeminiAPI

    User->>RAGChain: answer_question("What is scaled dot product attention?")
    RAGChain->>VectorStore: search(query, top_k=3)
    VectorStore-->>RAGChain: Return Top-3 Chunks
    Note over RAGChain: Format CONTEXT string + Strict System Rules
    RAGChain->>GeminiAPI: generate_content(prompt)
    GeminiAPI-->>RAGChain: Grounded Text Answer
    RAGChain-->>User: Display Final Output
```

> [!IMPORTANT]
> **Prompt Grounding Rule:**
> Always instruct the LLM to rely *strictly* on the provided `CONTEXT` block and to reply with *"I cannot find the answer in the provided document"* if the context lacks the required information.

---

## 5. Repository File Map

| Component | File Link | Description |
| :--- | :--- | :--- |
| **Chunker** | [chunking.py](file:///c:/Users/91801/Documents/GitHub/RAG/chunking.py) | PDF text extraction (`pypdf`) + sliding window chunking loop. |
| **Vector Store** | [vector_store.py](file:///c:/Users/91801/Documents/GitHub/RAG/vector_store.py) | `SentenceTransformer` embeddings + pure NumPy Cosine Similarity engine. |
| **RAG Chain** | [rag_chain.py](file:///c:/Users/91801/Documents/GitHub/RAG/rag_chain.py) | End-to-end RAG pipeline connecting vector search to Gemini 3.6 Flash. |
| **Test Harness** | [test_pipeline.py](file:///c:/Users/91801/Documents/GitHub/RAG/test_pipeline.py) | Verification script for indexing & vector retrieval. |

---

## 6. Verified End-to-End Execution Benchmark

* **Source Document:** *Attention Is All You Need* (`1706.03762v7.pdf`)
* **Question:** *"What is the formula for Scaled Dot-Product Attention, and why is scaling by sqrt(d_k) necessary?"*
* **Vector Match Score:** `0.5772` (Top Rank)
* **Generated Answer:**

> * **Formula for Scaled Dot-Product Attention:**  
>   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
> 
> * **Why scaling by $\sqrt{d_k}$ is necessary:**  
>   For large values of $d_k$, the dot products grow large in magnitude, which pushes the softmax function into regions that have extremely small gradients. Scaling the dot products by $\frac{1}{\sqrt{d_k}}$ counteracts this effect.
