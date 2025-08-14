import torch
from sentence_transformers import SentenceTransformer
from src.utils.config import CFG
from src.eval.ir_eval import build_eval
from src.eval.log_metrics import print_results_table

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CFG.model_id, device=device)

    evaluator = build_eval(CFG.matryoshka_dims)
    base_results = evaluator(model)

    print_results_table("Base Model Evaluation Results", base_results, CFG.matryoshka_dims)

if __name__ == "__main__":
    main()
