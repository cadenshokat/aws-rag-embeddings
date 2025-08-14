def print_results_table(title: str, results: dict, dims: list[int] | tuple[int, ...]):
    print(f"\n{title}")
    print("-" * 85)
    header = f"{'Metric':15} " + " ".join([f"{d:>12}d" for d in dims])
    print(header)
    print("-" * 85)

    metrics = [
        "ndcg@10", "mrr@10", "map@100",
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
    ]

    for m in metrics:
        row = [f"{'=='+m+'==' if m=='ndcg@10' else m:15}"]
        for d in dims:
            key = f"dim_{d}_cosine_{m}"
            row.append(f"{results[key]:12.4f}")
        print(" ".join(row))
    print("-" * 85)
    print(f"seq_score: {results['sequential_score']:1f}")
