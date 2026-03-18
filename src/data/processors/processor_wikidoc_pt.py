"""
WikiDoc PT Dataset Processor.
Processes medical QA data in Portuguese from WikiDoc, extracting questions and answers
from instruction-formatted text.
"""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42
HF_DATASET = "rhaymison/medicine-medical_meadow_wikidoc_pt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_question_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract question and answer from instruction-formatted text.
    Removes <<SYS>>...<</SYS>> block and extracts:
    - question (between [INST] and [/INST])
    - answer (after [/INST])
    """
    if not isinstance(text, str):
        return None, None

    # Remove <<SYS>> ... <</SYS>>
    text = re.sub(r"<<SYS>>.*?<</SYS>", "", text, flags=re.DOTALL).strip()

    # Extract question and answer
    match = re.search(r"\[INST\](.*?)\[/INST\](.*)", text, flags=re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer

    return None, None


def load_hf_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from HuggingFace."""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        return list(dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def prepare_records(dataset: List[Dict]) -> List[Dict]:
    """Prepare records by extracting questions and answers."""
    records = []
    skipped = 0
    
    for i, ex in enumerate(dataset):
        # Get the field containing the text (usually 'question')
        q_field = "question" if "question" in ex else list(ex.keys())[0]
        raw_text = ex[q_field]

        question, answer = extract_question_answer(raw_text)

        if question and answer:
            records.append({
                "id": f"{i:05d}",
                "question": question,
                "answer": answer,
                "task": "qa"
            })
        else:
            skipped += 1
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} examples without valid question/answer pairs")
    
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
    if len(sys.argv) != 2:
        print("Usage: python process_wikidoc_pt.py <output_base_dir>")
        print("Example: python process_wikidoc_pt.py /path/to/benchmark")
        sys.exit(1)
    
    output_base_dir = Path(sys.argv[1])
    output_dir = output_base_dir / "wikidoc_pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    dataset = load_hf_dataset(HF_DATASET)
    records = prepare_records(dataset)
    
    if not records:
        logger.error("No valid records extracted. Exiting.")
        sys.exit(1)
    
    logger.info(f"Extracted {len(records)} valid QA pairs")
    
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