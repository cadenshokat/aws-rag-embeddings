import torch
from sentence_transformers import SentenceTransformer
from src.utils.config import CFG
from src.eval.ir_eval import build_eval
from src.eval.log_metrics import print_results_table

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CFG.output_dir, device=device)

    evaluator = build_eval(CFG.matryoshka_dims)
    ft_results = evaluator(model)

    print_results_table("Fine Tuned Model Evaluation Results", ft_results, CFG.matryoshka_dims)

if __name__ == "__main__":
    main()
