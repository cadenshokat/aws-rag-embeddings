from datasets import load_dataset, concatenate_datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from src.utils.paths import TRAIN_JSON, TEST_JSON

def build_eval(matryoshka_dims: list[int] | tuple[int, ...]):
    test_dataset  = load_dataset("json", data_files=str(TEST_JSON),  split="train")
    train_dataset = load_dataset("json", data_files=str(TRAIN_JSON), split="train")

    aws_dataset = concatenate_datasets([train_dataset, test_dataset])

    corpus = dict(zip(aws_dataset["id"], aws_dataset["positive"]))

    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))

    relevant_docs: dict[int, list[int]] = {}
    g2c = {}
    for cid, g in zip(aws_dataset["id"], aws_dataset["global_id"]):
        g2c.setdefault(g, []).append(cid)

    for qid, g in zip(test_dataset["id"], test_dataset["global_id"]):
        relevant_docs[qid] = g2c.get(g, [])

    evaluators = []
    for dim in matryoshka_dims:
        ir = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        evaluators.append(ir)

    return SequentialEvaluator(evaluators)
