from typing import List, Dict
import json, glob

def load_all_chunks(glob_pattern: str) -> List[Dict]:
    chunks = []
    for path in glob.glob(glob_pattern, recursive=True):
        data = json.load(open(path, 'r', encoding='utf-8'))
        for rec in data:
            if "doc_id" not in rec or "chunk_id" not in rec or "text" not in rec:
                raise ValueError(f"Missing required keys in chunk record: {rec}")
            chunks.append(rec)
    return chunks

