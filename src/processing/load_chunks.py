from typing import List, Dict
import json, glob

def load_all_chunks(glob_pattern: str) -> List[Dict]:
    all_chunks = []
    for path in glob.glob(glob_pattern, recursive=True):
        data = json.load(open(path, encoding="utf-8"))
        all_chunks.extend(data)
    return all_chunks
