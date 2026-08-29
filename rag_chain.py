# rag_chain.py
import os
from google import genai
from chunking import extract_text_from_pdf, chunk_text
from vector_store import SimpleVectorStore


class RAGPipeline:
    def __init__(self, pdf_path: str, model_name: str = "gemini-3.6-flash"):
        """
        Initializes the PDF Indexing and Gemini LLM Client.
        """
        self.client = genai.Client()
        self.model_name = model_name

        print(f"--- 1. Indexing Document: {pdf_path} ---")
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text, chunk_size=1000, chunk_overlap=200)

        self.vector_store = SimpleVectorStore()
        self.vector_store.add_texts(chunks)
        print("--- Indexing Complete! ---\n")

    def answer_question(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves top_k relevant chunks, formats a context-grounded prompt, and generates an answer.
        """
        # Step 1: Retrieve top_k relevant chunks
        search_results = self.vector_store.search(query, top_k=top_k)

        # TODO: Implement the RAG Generation logic!
        #
        # 1. Combine retrieved chunk texts into a context block:
        #    context_text = "\n\n".join([f"--- Chunk {i+1} ---\n{text}" for i, (text, score) in enumerate(search_results)])
        #
        # 2. Construct the prompt string:
        #    prompt = f"""
        #    You are an expert AI researcher. Answer the user's question using ONLY the provided context passages.
        #    If the context does not contain enough information to answer, state clearly "I cannot find the answer in the provided document."

        #    CONTEXT:
        #    {context_text}

        #    QUESTION:
        #    {query}
        #    """
        #
        # 3. Call Gemini API:
        #    response = self.client.models.generate_content(
        #        model=self.model_name,
        #        contents=prompt
        #    )
        #    return response.text

        context_str= "\n\n".join([f"--- Chunk {i+1} ---\n{text}" for i, (text, score) in enumerate(search_results)])
        prompt=f""" You are an expert AI researcher. Answer the user's question using ONLY the provided context passages.
           If the context does not contain enough information to answer, state clearly "I cannot find the answer in the provided document."

            CONTEXT:
            {context_str}

            QUESTION:
            {query}
            """

        #calling api
        reply=self.client.models.generate_content(model=self.model_name,contents=prompt)

        return reply.text


if __name__ == "__main__":
    # Test our full RAG pipeline on Attention Is All You Need paper
    pipeline = RAGPipeline("1706.03762v7.pdf")
    
    question = "What is the formula for Scaled Dot-Product Attention, and why is scaling by sqrt(d_k) necessary?"
    print(f"QUESTION: {question}\n")
    
    answer = pipeline.answer_question(question, top_k=3)
    print("ANSWER:")
    print(answer)
