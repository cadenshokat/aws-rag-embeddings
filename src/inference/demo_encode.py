import torch
from sentence_transformers import SentenceTransformer
from src.utils.config import CFG

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CFG.output_dir, device=device, truncate_dim=256)

    sentences = [
        
    ]

    emb = model.encode(sentences)
    print("Embeddings shape:", emb.shape)

    sims = model.similarity(emb, emb)
    print("Similarity row[0]:", sims[0])

if __name__ == "__main__":
    main()
