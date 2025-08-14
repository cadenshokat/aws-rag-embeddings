from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    model_id: str = "nomic-ai/modernbert-embed-base"
    matryoshka_dims: tuple[int, ...] = (768, 512, 256, 128, 64)
    output_dir: str = "modernbert-embed-aws"

CFG = Config()
