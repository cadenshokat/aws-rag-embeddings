import os, json, glob, time, re
from typing import List, Dict
from dotenv import load_dotenv
import argparse
from load_chunks import load_all_chunks
from openai import OpenAI

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NUM_QUESTIONS = 4
SLEEP = 5
model = "gpt-4o-mini"

def make_prompt(chunk_text: str) -> str:
    return f"""
                Generate according to the above rules. Return **only** json. **All** string fields must be valid JSON strings wrapped in double quotes.
                
                Here is the text chunk:\n\n\"\"\"\n{chunk_text}\n\"\"\"\n\n
            """

def generate(model: str, prompt: str) -> List[Dict]:
    while True:
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    { "role": "system", "content":        
                    """
                        You are an expert at generating reading-comprehension questions in **strict JSON** form. 
                        Given the user’s chunk, you will output **only** a JSON array of objects—no commentary, no extra text.
                        Each object must have:
                        - question    : the question text
                        - answer_span : the exact sentence from the chunk that answers this question
                        Output exactly 4 questions.
                    """ 
                    },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.2,
                max_tokens=NUM_QUESTIONS * 100
            )
            text = resp.choices[0].message.content.strip()
            print(text)
            raw = text
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"```$", "", raw).strip()

            m = re.search(r"\[.*\]", raw, flags=re.S)
            if m:
                raw = m.group(0)

            arr = json.loads(raw)

            print(arr)
            return arr
        except json.JSONDecodeError as e:
            print("Failed to parse JSON, retrying...", e)
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from chunk JSON files via GPT-4"
    )
    parser.add_argument("chunks_glob",
                        help="Glob pattern for chunk JSON files (e.g. 'chunks/**/*.json')")
    parser.add_argument("output",
                        help="Output JSONL file for QA pairs")
    parser.add_argument("--model", default=model,
                        help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between requests (default: 0.5)")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        parser.error("Please set OPENAI_API_KEY environment variable")

    chunks = load_all_chunks(args.chunks_glob)
    print(f"Loaded {len(chunks)} chunks.")

    with open(args.output, "w", encoding="utf-8") as out_f:
        total = 0
        for rec in chunks:
            qas = generate(args.model, make_prompt(rec["text"]))
            i = 0
            for qa in qas:
                i += 1
                out = {
                    "global_id": total,
                    "doc_id":      rec["doc_id"],
                    "chunk_id":    rec["chunk_id"],
                    "question_id": i,
                    "question":    qa["question"],
                    "answer_span": qa["answer_span"],
                    "chunk": rec.get('text')
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1
            time.sleep(args.sleep)

    print(f"Done — generated {total} questions across {len(chunks)} chunks into '{args.output}'.")

if __name__ == "__main__":
    main()