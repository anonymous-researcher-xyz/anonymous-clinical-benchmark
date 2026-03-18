"""
DrBodeBench Dataset Processor.
Processes medical multiple-choice questions with stratified splitting and answer balancing.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from copy import deepcopy

from datasets import load_dataset

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
HF_DATASET = "recogna-nlp/drbodebench"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed globally
random.seed(RANDOM_SEED)


def load_hf_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from HuggingFace."""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        return list(dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def remove_extra_alternative(example: Dict, seed: int = RANDOM_SEED) -> Dict:
    """Remove one incorrect alternative if there are 5 alternatives."""
    alternatives = example["alternativas"]
    correct_answer = example["resposta"]

    # Remove empty alternatives
    for key in list(alternatives.keys()):
        if alternatives[key] in [None, "", " "]:
            del alternatives[key]

    valid_alts = [k for k, v in alternatives.items() if v not in [None, "", " "]]

    if len(valid_alts) == 5:
        # Get incorrect alternatives
        incorrect = [alt for alt in valid_alts if alt != correct_answer]
        random.seed(seed)
        to_remove = random.choice(incorrect)
        if to_remove in alternatives:
            del alternatives[to_remove]
    
    return example


def balance_answers_perfectly(data: List[Dict], seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Shuffle alternatives to achieve perfect balance of correct answer letters.
    Updates both alternatives and answer field.
    """
    # Shuffle original order
    random.seed(seed)
    random.shuffle(data)

    new_data = []
    possible_letters = ['A', 'B', 'C', 'D']
    num_alternatives = len(possible_letters)

    for i, record in enumerate(data):
        # Determine target letter for correct answer in cycle (A, B, C, D, A, ...)
        target_letter = possible_letters[i % num_alternatives]

        # Separate correct and incorrect texts
        correct_text = record["alternativas"][record["resposta"]]
        incorrect_texts = [
            text for key, text in record["alternativas"].items()
            if key != record["resposta"]
        ]

        # Shuffle incorrect alternatives
        random.seed(seed)
        random.shuffle(incorrect_texts)

        # Build new alternatives dictionary
        new_alternatives = {}
        new_alternatives[target_letter] = correct_text

        # Fill other letters with incorrect alternatives
        incorrect_idx = 0
        for letter in possible_letters:
            if letter != target_letter:
                new_alternatives[letter] = incorrect_texts[incorrect_idx]
                incorrect_idx += 1

        # Create copy and update
        new_record = deepcopy(record)
        new_record["alternativas"] = dict(sorted(new_alternatives.items()))
        new_record["resposta"] = target_letter
        new_data.append(new_record)

    # Final shuffle to avoid A,B,C,D pattern
    random.seed(seed)
    random.shuffle(new_data)

    return new_data


def split_dataset_stratified(records: List[Dict], seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset with stratification by answer class."""
    random.seed(seed)
    random.shuffle(records)

    # Group by answer class
    buckets = defaultdict(list)
    for record in records:
        buckets[record["resposta"]].append(record)

    train, dev, test = [], [], []

    # Split each class proportionally
    for answer, items in buckets.items():
        n = len(items)
        n_dev = int(n * DEV_SPLIT)
        n_test = int(n * TEST_SPLIT)
        n_train = n - n_dev - n_test

        random.seed(seed)
        random.shuffle(items)

        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    # Shuffle final sets
    random.seed(seed)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def print_answer_distribution(examples: List[Dict], split_name: str = "") -> None:
    """Print distribution of correct answers."""
    counts = Counter(ex["resposta"] for ex in examples)
    total = sum(counts.values())
    
    if split_name:
        logger.info(f"\nAnswer distribution - {split_name}:")
    else:
        logger.info("\nAnswer distribution:")
    
    for letter in sorted(counts):
        count = counts[letter]
        percentage = (count / total) * 100
        logger.info(f"  {letter}: {count} ({percentage:.2f}%)")


def prepare_records(dataset: List[Dict]) -> List[Dict]:
    """Prepare records by adding task, ID, and processing alternatives."""
    records = []
    for i, ex in enumerate(dataset):
        ex["task"] = "multiple_choice"
        ex["id"] = f"{i:05d}"
        ex = remove_extra_alternative(ex)
        records.append(ex)
    return records


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
        print("Usage: python process_drbodebench.py <output_base_dir>")
        print("Example: python process_drbodebench.py /path/to/benchmark")
        sys.exit(1)
    
    output_base_dir = Path(sys.argv[1])
    output_dir = output_base_dir / "drbodebench"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    dataset = load_hf_dataset(HF_DATASET)
    records = prepare_records(dataset)
    records = balance_answers_perfectly(records)
    
    print_answer_distribution(records, "full")
    
    # Stratified split
    train, dev, test = split_dataset_stratified(records)
    
    print_answer_distribution(train, "train")
    print_answer_distribution(dev, "dev")
    print_answer_distribution(test, "test")
    
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
    logger.info(f"Dev: {len(dev)} ({DEV_SPLIT*100:.0f}%)")
    logger.info(f"Test: {len(test)} ({TEST_SPLIT*100:.0f}%)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()