import argparse
import os
import json
from typing import List, Dict
from chunker import Chunker

def chunk_file(file_path: str, chunker: Chunker) -> List[str]:
    doc_id = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    raw_chunks = chunker.split_text(text)
    return [{ "doc_id": doc_id, "chunk_id": f"{i}", "text": chunk } for i, chunk in enumerate(raw_chunks)]

def save_chunks(original_file: str, chunks: List[Dict], input_root: str, output_dir: str):
    rel_path = os.path.relpath(original_file, input_root)
    base_name = os.path.splitext(rel_path)[0] + ".json"
    out_path = os.path.join(output_dir, base_name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out_f:
        json.dump(chunks, out_f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {out_path}")

def process_input(input_path: str, output_dir: str, chunker: Chunker):
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.lower().endswith('.txt'):
                    full_path = os.path.join(root, fname)
                    chunks = chunk_file(full_path, chunker)
                    save_chunks(full_path, chunks, input_path, output_dir)
    elif os.path.isfile(input_path):
        if input_path.lower().endswith('.txt'):
            chunks = chunk_file(input_path, chunker)
            save_chunks(input_path, chunks, os.path.dirname(input_path), output_dir)
        else:
            print(f"Skipping non-.txt file: {input_path}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Chunk plain-text documents using your BaseChunker implementation."
    )
    parser.add_argument(
        "input", help="Path to a .txt file or a directory of .txt files to chunk"
    )
    parser.add_argument(
        "output", help="Directory where chunked JSON files will be written"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Number of tokens per chunk (default: 200)"
    )
    parser.add_argument(
        "--overlap", type=int, default=50,
    )

    args = parser.parse_args()
    chunker = Chunker(chunk_size=args.chunk_size, overlap=args.overlap)
    process_input(args.input, args.output, chunker)

if __name__ == "__main__":
    main()