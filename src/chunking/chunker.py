import re
from typing import List
from base_chunker import BaseChunker

class Chunker(BaseChunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 500):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        tokens = text.strip().split()
        if not tokens:
            return []
        
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks
    
