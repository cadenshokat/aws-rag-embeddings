from datasets import load_dataset
from src.utils.paths import TRAIN_JSON, TEST_JSON
from src.utils.seed import set_seed

REMOVE_COLS = ["chunk_id", "doc_id", "question_id", "answer_span"]

def main():
    set_seed(42)

    ds = load_dataset("CadenShokat/aws-rag-qa-positives", split="train")

    ds = ds.rename_column("question", "anchor")
    ds = ds.rename_column("chunk", "positive")
    ds = ds.remove_columns(REMOVE_COLS)

    ds = ds.add_column("id", list(range(len(ds))))

    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.1, seed=42)

    split["train"].to_json(str(TRAIN_JSON), orient="records")
    split["test"].to_json(str(TEST_JSON), orient="records")

    print(f"Wrote:\n- {TRAIN_JSON}\n- {TEST_JSON}")

if __name__ == "__main__":
    main()
