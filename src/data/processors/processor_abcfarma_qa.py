"""
ABCFarma QA Dataset Processor.
Generates question-answer pairs about medications and active ingredients.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42

TEMPLATES = [
    "A que princípio ativo pertence o medicamento '{brand}'?",
    "Qual é o princípio ativo de '{brand}'?",
    "De que substância é feito o medicamento '{brand}'?",
    "A substância ativa de '{brand}' é qual?",
    "Qual a droga presente no medicamento '{brand}'?",
    "Você sabe qual é o princípio ativo de '{brand}'?",
    "O medicamento '{brand}' contém qual princípio ativo?",
    "Qual é a substância base de '{brand}'?"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_corpus(input_file: Path) -> Dict[str, List[str]]:
    """Load medication corpus."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON: {input_file}")
        sys.exit(1)


def generate_qa_pairs(data: Dict[str, List[str]]) -> List[Dict]:
    """Generate question-answer pairs."""
    qa_pairs = []
    current_id = 1
    
    for generic, brands in data.items():
        for brand in brands:
            brand = brand.strip()
            template = random.choice(TEMPLATES)
            question = template.format(brand=brand)
            
            qa_pairs.append({
                "id": current_id,
                "question": question,
                "answer": generic,
                "task": "qa"
            })
            current_id += 1
    
    return qa_pairs


def split_dataset(qa_pairs: List[Dict]) -> tuple:
    """Split dataset into train, dev and test."""
    random.seed(RANDOM_SEED)
    random.shuffle(qa_pairs)
    
    total = len(qa_pairs)
    train_idx = int(TRAIN_SPLIT * total)
    dev_idx = int(DEV_SPLIT * total)
    
    return (
        qa_pairs[:train_idx],
        qa_pairs[train_idx:dev_idx],
        qa_pairs[dev_idx:]
    )


def save_jsonl(records: List[Dict], filename: Path) -> None:
    """Save records in JSONL format."""
    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Saved: {filename} ({len(records)} records)")


def save_ids(ids: List[int], filename: Path) -> None:
    """Save list of IDs to JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


def main():
    """Main function."""
    # Validate arguments
    if len(sys.argv) != 3:
        print("Usage: python process_abcfarma_qa.py <input_file> <output_base_dir>")
        print("Example: python process_abcfarma_qa.py corpus/data/abcfarma_corpus.json /path/to/benchmark")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    base_output_dir = Path(sys.argv[2])
    output_dir = base_output_dir / "abcfarma_qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    data = load_corpus(input_file)
    qa_pairs = generate_qa_pairs(data)
    train_records, dev_records, test_records = split_dataset(qa_pairs)
    
    # Save datasets
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(dev_records, output_dir / "dev.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")
    save_jsonl(qa_pairs, output_dir / "full.jsonl")
    
    # Save IDs for reproducibility
    save_ids([r["id"] for r in train_records], output_dir / "train_ids.json")
    save_ids([r["id"] for r in dev_records], output_dir / "dev_ids.json")
    save_ids([r["id"] for r in test_records], output_dir / "test_ids.json")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Total records: {len(qa_pairs)}")
    logger.info(f"Train: {len(train_records)} ({TRAIN_SPLIT*100:.0f}%)")
    logger.info(f"Dev: {len(dev_records)} ({(DEV_SPLIT-TRAIN_SPLIT)*100:.0f}%)")
    logger.info(f"Test: {len(test_records)} ({(1-DEV_SPLIT)*100:.0f}%)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()