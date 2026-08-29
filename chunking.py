# src/chunking.py
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text content from all pages of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits a single raw text string into overlapping chunks using a sliding window.
    
    Args:
        text: The full raw text string.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
        
    Returns:
        List of chunk strings.
    """
    # 1. Validation guard: chunk_overlap must be strictly less than chunk_size
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    chunks = []
    # TODO: Implement the sliding window loop here!
    # Hint:
    # Calculate step size: step = chunk_size - chunk_overlap
    # Use a start pointer `i` from 0 to len(text), advancing by `step`
    #initial Window
    step=chunk_size-chunk_overlap
    start_idx=0
    while start_idx<len(text):
        passage=text[start_idx:start_idx+chunk_size]
        chunks.append(passage)
        start_idx+=step



    
    return chunks