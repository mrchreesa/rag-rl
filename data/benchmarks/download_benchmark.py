import os
from datasets import load_dataset

# 1. Define where you want the data to go
output_dir = "data/benchmarks/hotpotqa"
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading HotpotQA to {output_dir}...")

# 2. Download the 'hotpotqa' subset specifically
ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa")

# 3. Save it as JSONL (which FlashRAG expects)
# You only strictly need 'dev' for evaluation, but 'train' is good to have.
ds['dev'].to_json(f"{output_dir}/dev.jsonl")
ds['train'].to_json(f"{output_dir}/train.jsonl")

print("✅ Question dataset downloaded successfully!")