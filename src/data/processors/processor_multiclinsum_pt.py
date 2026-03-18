"""
MultiClinSum PT Dataset Processor.
Processes clinical text summarization data in Portuguese.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

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


def read_file(path: Path) -> str:
    """Read text file content."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


def load_data(fulltext_dir: Path, summary_dir: Path) -> List[Dict]:
    """Load fulltext and summary pairs."""
    if not fulltext_dir.exists():
        logger.error(f"Fulltext directory not found: {fulltext_dir}")
        sys.exit(1)
    
    if not summary_dir.exists():
        logger.error(f"Summary directory not found: {summary_dir}")
        sys.exit(1)
    
    records = []
    files = sorted([f for f in fulltext_dir.iterdir() if f.suffix == ".txt"])
    
    logger.info(f"Found {len(files)} fulltext files")
    
    for text_file in files:
        base_name = text_file.stem
        summary_file = summary_dir / f"{base_name}_sum.txt"
        
        if not summary_file.exists():
            logger.warning(f"Summary not found for {text_file.name}, skipping...")
            continue
        
        try:
            text = read_file(text_file)
            summary = read_file(summary_file)
            
            records.append({
                "id": base_name,
                "text": text,
                "summary": summary,
                "task": "summarization"
            })
        except Exception as e:
            logger.warning(f"Error processing {text_file.name}: {e}")
            continue
    
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
        print("Usage: python process_multiclinsum_pt.py <input_base_dir> <output_base_dir>")
        print("Example: python process_multiclinsum_pt.py /path/to/data /path/to/benchmark")
        sys.exit(1)
    
    input_base_dir = Path(sys.argv[1])
    output_base_dir = Path(sys.argv[2])
    output_dir = output_base_dir / "multiclinsum_pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define data directories
    data_dir = input_base_dir / "multiclinsum_gs_train_pt"
    fulltext_dir = data_dir / "fulltext"
    summary_dir = data_dir / "summaries"
    
    logger.info(f"Input directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    records = load_data(fulltext_dir, summary_dir)
    
    if not records:
        logger.error("No records loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(records)} records")
    
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