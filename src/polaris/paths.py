from pathlib import Path


DATA_ROOT_PATH = Path(__file__).resolve().parents[3]


ROOT_PATH = Path(__file__).resolve().parents[2]
PRETRAIN_SPLIT_DIR = DATA_ROOT_PATH / "datasets/pretrain"
FINETUNE_SPLIT_DIR = DATA_ROOT_PATH / "datasets/preprocessed-1d"
PROMPT_PATH = ROOT_PATH / "src/polaris/prompt/CKEPE_prompt.json"
DATASET_LABELS_PATH = ROOT_PATH / "src/polaris/prompt/dataset_class_names.json"
RESULTS_PATH = ROOT_PATH / "logs/polaris/results"
ECGFM_PATH = ""