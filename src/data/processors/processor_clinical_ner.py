"""
Clinical NER Labels Dataset Processor.
Processes clinical Named Entity Recognition data from Excel files with IOB tagging,
converting entity tags to full names and extracting entities by category.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42

# Entity mapping from abbreviations to full names
ENTITY_MAPPING = {
    "C": "Condição",
    "AS": "Anatomia",
    "THER": "Terapia",
    "T": "Procedimento",
    "EV": "Evolução",
    "G": "Genética",
    "N": "Negação",
    "OBS": "Observação",
    "R": "Resultado",
    "DT": "Tempo",
    "V": "Valor",
    "RA": "ViaAdmin",
    "CH": "Característica",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def to_iob2(tags: List[str]) -> List[str]:
    """Convert any IOB* format to IOB2 format."""
    iob2 = []
    prev_ent = "O"
    
    for tag in tags:
        if tag == "O" or not isinstance(tag, str):
            iob2.append("O")
            prev_ent = "O"
            continue

        prefix, ent = tag.split("-", 1)

        # If starts with I- but not in proper sequence, force B-
        if prefix == "I" and prev_ent == "O":
            iob2.append(f"B-{ent}")
        else:
            iob2.append(f"{'B' if prefix == 'B' and prev_ent != ent else 'I'}-{ent}")

        prev_ent = ent
    
    return iob2


def get_entities(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    """Extract entities from a sequence of IOB2 tags."""
    entities = defaultdict(list)
    current_entity = []
    current_tag = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # End previous entity if exists
            if current_entity:
                mapped_tag = ENTITY_MAPPING.get(current_tag)
                if mapped_tag:
                    full_entity = " ".join(current_entity)
                    entities[mapped_tag].append(full_entity)
            
            current_tag = tag.split("-", 1)[1]
            current_entity = [token]
            
        elif tag.startswith("I-") and current_tag and tag.split("-", 1)[1] == current_tag:
            current_entity.append(token)
            
        else:
            # End previous entity if exists
            if current_entity:
                mapped_tag = ENTITY_MAPPING.get(current_tag)
                if mapped_tag:
                    full_entity = " ".join(current_entity)
                    entities[mapped_tag].append(full_entity)
            
            current_entity = []
            current_tag = None

    # Add last entity if sentence ends with one
    if current_entity:
        mapped_tag = ENTITY_MAPPING.get(current_tag)
        if mapped_tag:
            full_entity = " ".join(current_entity)
            entities[mapped_tag].append(full_entity)
        
    return dict(entities)


def process_file(file_path: Path, global_id_start: int) -> Tuple[List[Dict], int]:
    """Process a single Excel file and extract records."""
    try:
        df = pd.read_excel(
            file_path, 
            sheet_name=0, 
            header=None,
            names=["token", "pos", "original", "tag"]
        )
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return [], global_id_start

    # Split into sentences (separated by empty rows)
    sentences, current = [], []
    for _, row in df.iterrows():
        if pd.isna(row["token"]):
            if current:
                sentences.append(current)
                current = []
        else:
            current.append((str(row["token"]), row["tag"]))
    
    if current:
        sentences.append(current)

    # Process each sentence
    records = []
    for idx, sent in enumerate(sentences):
        tokens = [tok for tok, _ in sent]
        raw_tags = [tag if isinstance(tag, str) else "O" for _, tag in sent]
        tags = to_iob2(raw_tags)
        
        entities = get_entities(tokens, tags)
        text = " ".join(tokens)
        
        records.append({
            "id": f"{global_id_start + idx:05d}",
            "text": text,
            "tags": entities,
            "task": "ner"
        })
    
    return records, global_id_start + len(records)


def load_all_excel_files(input_dirs: List[Path]) -> List[Dict]:
    """Load and process all Excel files from input directories."""
    all_records = []
    global_id = 0
    total_files = 0

    for folder_path in input_dirs:
        if not folder_path.exists():
            logger.warning(f"Directory not found: {folder_path}, skipping...")
            continue
        
        xlsx_files = sorted([f for f in folder_path.iterdir() if f.suffix == ".xlsx"])
        logger.info(f"Found {len(xlsx_files)} Excel files in {folder_path}")
        
        for file_path in xlsx_files:
            records, global_id = process_file(file_path, global_id)
            all_records.extend(records)
            total_files += 1
    
    logger.info(f"Processed {total_files} files total")
    return all_records


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
    if len(sys.argv) != 4:
        print("Usage: python process_clinical_ner.py <input_dir1> <input_dir2> <output_base_dir>")
        print("Example: python process_clinical_ner.py /path/to/dir1 /path/to/dir2 /path/to/benchmark")
        sys.exit(1)
    
    input_dirs = [Path(sys.argv[1]), Path(sys.argv[2])]
    output_base_dir = Path(sys.argv[3])
    output_dir = output_base_dir / "clinical_ner"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directories: {input_dirs}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load and process
    records = load_all_excel_files(input_dirs)
    
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