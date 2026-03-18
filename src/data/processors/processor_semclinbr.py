"""
SemClinBr NER Dataset Processor.
Processes clinical Named Entity Recognition data from XML format,
grouping entities by semantic categories.
"""

import json
import logging
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Configuration
TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.9
RANDOM_SEED = 42

# Semantic groups mapping
SEMANTIC_GROUPS = {
    'Disorder': [
        'Finding',
        'Sign or Symptom',
        'Disease or Syndrome',
    ],
    'Procedures': [
        'Therapeutic or Preventive Procedure',
        'Health Care Activity',
        'Diagnostic Procedure',
    ],
    'Chemicals and Drugs': [
        'Pharmacologic Substance',
        'Organic Chemical',
        'Hormone',
    ],
    'Abbreviation': [
        'Abbreviation'
    ]
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_tag_to_group_mapping(semantic_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """Build lookup dictionary from tag to semantic group."""
    tag_to_group = {}
    for group, tags in semantic_groups.items():
        for tag in tags:
            tag_to_group[tag] = group
    return tag_to_group


def load_annotations(xml_file: Path, tag_to_group: Dict[str, str]) -> Tuple[str, Dict[str, List[str]]]:
    """Load text and annotations from XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text = root.find('TEXT').text.strip()
        
        entities = defaultdict(list)
        
        for ann in root.findall('.//annotation'):
            ent_text = ann.attrib['text']
            tags = ann.attrib['tag'].split('|')
            
            for tag in tags:
                group = tag_to_group.get(tag)
                if group and ent_text not in entities[group]:
                    entities[group].append(ent_text)
        
        return text, dict(entities)
    
    except Exception as e:
        logger.error(f"Error loading {xml_file}: {e}")
        raise


def convert_to_record(text: str, annotations: Dict[str, List[str]], record_id: str) -> Dict:
    """Convert text and annotations to record format."""
    return {
        "id": record_id,
        "text": text,
        "tags": annotations,
        "task": "ner"
    }


def load_all_xml_files(xml_dir: Path, tag_to_group: Dict[str, str]) -> List[Dict]:
    """Load and process all XML files from directory."""
    if not xml_dir.exists():
        logger.error(f"XML directory not found: {xml_dir}")
        sys.exit(1)
    
    xml_files = sorted([f for f in xml_dir.iterdir() if f.suffix == '.xml'])
    
    if not xml_files:
        logger.error(f"No XML files found in {xml_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(xml_files)} XML files")
    
    records = []
    for xml_file in xml_files:
        try:
            record_id = xml_file.stem
            text, annotations = load_annotations(xml_file, tag_to_group)
            record = convert_to_record(text, annotations, record_id)
            records.append(record)
        except Exception as e:
            logger.warning(f"Skipping {xml_file.name}: {e}")
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
        print("Usage: python process_semclinbr.py <xml_base_dir> <output_base_dir>")
        print("Example: python process_semclinbr.py /path/to/xml /path/to/benchmark")
        sys.exit(1)
    
    xml_base_dir = Path(sys.argv[1])
    output_base_dir = Path(sys.argv[2])
    
    # Define paths
    xml_dir = xml_base_dir / 'SemClinBr-xml-public-v1'
    output_dir = output_base_dir / "semclinbr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directory: {xml_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Build tag mapping
    tag_to_group = build_tag_to_group_mapping(SEMANTIC_GROUPS)
    
    # Load and process
    records = load_all_xml_files(xml_dir, tag_to_group)
    
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