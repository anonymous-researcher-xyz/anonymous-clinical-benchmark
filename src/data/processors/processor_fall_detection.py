"""
Fall Detection Dataset Processor.
Processes clinical notes for patiente fall classification.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv(input_path: Path) -> pd.DataFrame:
    """Load CSV file."""
    try:
        logger.info(f"Loading CSV: {input_path}")
        return pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        sys.exit(1)


def prepare_records(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of records with IDs."""
    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"{i+1:05d}",
            "Evolucao": row["Evolucao"],
            "Target": row["Target"],
            "task": "classification"
        })
    return records


def split_dataset(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train, dev and test."""
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    
    n = len(records)
    train_idx = int(TRAIN_SPLIT * n)
    dev_idx = int(DEV_SPLIT * n)
    
    return (
        records[:train_idx],
        records[train_idx:dev_idx],
        records[dev_idx:]
    )


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
    if len(sys.argv) != 3:
        print("Usage: python process_fall_detection.py <input_csv> <output_base_dir>")
        print("Example: python process_fall_detection.py data/fall_data.csv /path/to/benchmark")
        sys.exit(1)
    
    input_csv_path = Path(sys.argv[1])
    output_base_dir = Path(sys.argv[2])
    output_dir = output_base_dir / "fall_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_csv_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    df = load_csv(input_csv_path)
    records = prepare_records(df)
    train_records, dev_records, test_records = split_dataset(records)
    
    # Save datasets
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(dev_records, output_dir / "dev.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")
    save_jsonl(records, output_dir / "full.jsonl")
    
    # Save IDs for reproducibility
    save_ids([r["id"] for r in train_records], output_dir / "train_ids.json")
    save_ids([r["id"] for r in dev_records], output_dir / "dev_ids.json")
    save_ids([r["id"] for r in test_records], output_dir / "test_ids.json")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total records: {len(records)}")
    logger.info(f"Train: {len(train_records)} ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"Dev: {len(dev_records)} ({(DEV_SPLIT-TRAIN_SPLIT)*100:.0f}%)")
    logger.info(f"Test: {len(test_records)} ({(1-DEV_SPLIT)*100:.0f}%)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()