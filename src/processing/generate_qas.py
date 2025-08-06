import os, json, glob, time
from typing import List, Dict
from dotenv import load_dotenv
from load_chunks import load_all_chunks
from openai import OpenAI

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NUM_QUESTIONS = 4
SLEEP = 5
MODEL = "gpt-4o-mini"

def make_prompt(chunk_text: str) -> str:
    return f"""
            You are creating reading-comprehension questions. 
            Given only the text below, generate exactly {NUM_QUESTIONS} short, factual questions
            that can be answered *only* from that text.  
            Return your output as a JSON array of objects with fields:
            - id:       integer question number (1..{NUM_QUESTIONS})
            - question: the question text

            Text:
                \"\"\"
                {chunk_text}
                \"\"\"
             """

def generate(prompt: str) -> List[Dict]:
    while True:
        try:
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    { "role": "system", "content": "You output JSON only, no extra text." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.7,
                max_tokens=NUM_QUESTIONS * 50
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            print("Failted to parse JSON, retrying...", e)
            time.sleep(1)

def main():
    chunks = load_all_chunks("../../../dataset/chunks/*.json")

    output_records = []
    for chunk in chunks:
        qas = generate(make_prompt(chunk["chunk"]))

        for qa in qas:
            output_records.append({
                "doc_id":    chunk.get("doc_id"),
                "section":   chunk.get("section"),
                "chunk_id":  chunk["chunk_id"],
                "question_id": qa["id"],
                "question":  qa["question"]
            })

        time.sleep(0.5)

        with open("synthetic_qa_pairs.jsonl", "w", encoding="utf-8") as out:
            for rec in output_records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Done — generated {len(output_records)} questions across {len(chunks)} chunks.")

if __name__ == "__main__":
    main()