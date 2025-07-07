from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenizer(model_name="bert-base-uncased", max_len=128):
    token = AutoTokenizer.from_pretrained(model_name)
    token.model_max_length = max_len
    return token

def load(tokenizer, split="validation"):
    ds = load_dataset("glue", "stsb", split=split)

    

