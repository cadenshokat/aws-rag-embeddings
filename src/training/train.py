import os, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.losses import MatryoshkaLoss

from src.utils.config import CFG
from src.utils.paths import TRAIN_JSON, TEST_JSON
from src.eval.ir_eval import build_eval

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    # base model with SDPA
    model = SentenceTransformer(
        CFG.model_id,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Embed AWS Docs",
        ),
    )

    train_dataset = load_dataset("json", data_files=str(TRAIN_JSON), split="train")
    test_dataset  = load_dataset("json", data_files=str(TEST_JSON),  split="train")

    evaluator = build_eval(CFG.matryoshka_dims)

    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=list(CFG.matryoshka_dims))

    args = SentenceTransformerTrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,  
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["positive", "anchor"]),
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()
    trainer.save_model()

    if os.getenv("HUGGINGFACE_HUB_TOKEN"):
        trainer.model.push_to_hub(CFG.output_dir)

if __name__ == "__main__":
    main()
