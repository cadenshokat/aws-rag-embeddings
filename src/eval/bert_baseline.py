from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from scipy.stats import spearmanr
from sklearn.datasets import fetch_20newsgroups
import torch
import numpy as np


if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

def embed_texts(texts, model, tokenizer, device=device):
    ins = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**ins).last_hidden_state

    vecs = out.mean(dim=1)
    return normalize(vecs, dim=-1).cpu().numpy()

def spearman_eval(model_name="bert-base-uncased", split="validation"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    ds = load_dataset("glue", "stsb", split=split)

    sims, gold = [], []
    for ex in ds:
        u = embed_texts([ex["sentence1"]], model, tokenizer)[0]
        v = embed_texts([ex["sentence2"]], model, tokenizer)[0]

        sims.append(float(np.dot(u, v)))
        gold.append(ex["label"] / 5.0)

    corr, _ = spearmanr(sims, gold)
    print(f"BERT Baseline Spearman: {corr:.4f}")


def embed_in_batches(texts, model, tokenizer, batch_size=100):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = embed_texts(batch, model, tokenizer)
        all_vecs.append(vecs)
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.vstack(all_vecs)


def clustering_purity(model_name="bert-base-uncased", sample_size=2000, batch_size=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).eval().to(device)

    ds     = load_dataset("SetFit/20_newsgroups", split="train")
    texts  = ds["text"][:sample_size]
    labels = np.array(ds["label"][:sample_size])

    vecs = embed_in_batches(texts, model, tokenizer, batch_size)

    clusters = KMeans(n_clusters=len(set(labels)),
                      random_state=0).fit_predict(vecs)
    purity = (clusters == labels).sum() / len(labels)
    print(f"Purity (N={sample_size}): {purity:.4f}")



if __name__ == "__main__":
    spearman_eval()
    clustering_purity()

    