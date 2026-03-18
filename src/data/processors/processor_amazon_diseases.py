"""
Amazon Diseases Dataset Processor.
Loads dataset from HuggingFace and generates splits for QA benchmark.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from datasets import load_dataset

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42
HF_DATASET = "juniofreitas/dataset-chatbot-doencas_negligenciadas"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_hf_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from HuggingFace."""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        return list(dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def prepare_records(dataset: List[Dict]) -> List[Dict]:
    """Prepare records by adding task and ID."""
    records = []
    for i, ex in enumerate(dataset):
        ex["task"] = "qa"
        ex["id"] = f"{i:05d}"
        records.append(ex)
    return records


def split_dataset(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train, dev and test."""
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    
    n = len(records)
    train_idx = int(TRAIN_SPLIT * n)
    dev_idx = int(DEV_SPLIT * n)
    
    return records[:train_idx], records[train_idx:dev_idx], records[dev_idx:]


def save_jsonl(records: List[Dict], path: Path) -> None:
    """Save records in JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Saved: {path} ({len(records)} records)")


def save_ids(ids: List[str], path: Path) -> None:
    """Save list of IDs to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


def main():
    """Main function."""
    # Validate arguments
    if len(sys.argv) != 2:
        print("Usage: python process_amazon_diseases.py <output_base_dir>")
        print("Example: python process_amazon_diseases.py /path/to/benchmark")
        sys.exit(1)
    
    output_base_dir = Path(sys.argv[1])
    output_dir = output_base_dir / "amazon_diseases"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    dataset = load_hf_dataset(HF_DATASET)
    records = prepare_records(dataset)
    train, dev, test = split_dataset(records)
    
    # Save datasets
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(dev, output_dir / "dev.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")
    save_jsonl(records, output_dir / "full.jsonl")
    
    # Save IDs for reproducibility
    save_ids([r["id"] for r in train], output_dir / "train_ids.json")
    save_ids([r["id"] for r in dev], output_dir / "dev_ids.json")
    save_ids([r["id"] for r in test], output_dir / "test_ids.json")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total records: {len(records)}")
    logger.info(f"Train: {len(train)} ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"Dev: {len(dev)} ({(DEV_SPLIT-TRAIN_SPLIT)*100:.0f}%)")
    logger.info(f"Test: {len(test)} ({(1-DEV_SPLIT)*100:.0f}%)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()