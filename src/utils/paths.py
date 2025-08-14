from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = PROC_DIR / "train_dataset.json"
TEST_JSON  = PROC_DIR / "test_dataset.json"
