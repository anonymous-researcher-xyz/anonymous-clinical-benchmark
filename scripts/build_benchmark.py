"""
Clinical Benchmark Dataset Processor
Main script to process all datasets for the clinical benchmark.
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def require_path(path: Path, description: str) -> bool:
    """
    Ensure a required directory or file exists.
    If not, show a clear message asking the user to download it manually.
    """
    if path.exists():
        logger.info(f"Found: {description} → {path}")
        return True
    else:
        logger.error(f"""
            ❌ Missing required data: {description}
            Expected at: {path}

            ➡️ Please download/clone manually and place it in this directory:

                {path.parent}

            Then run the script again.
            """)
        return False

# Dataset configurations
DATASETS = {
    'abcfarma_qa': {
        'script': 'processor_abcfarma_qa.py',
        'description': 'ABCFarma QA - Medication to active ingredient questions',
        'setup': lambda raw: require_path(
            raw / 'corpus/data/abcfarma_corpus.json',
            "ABCFarma corpus JSON"
        ),
        'args': lambda raw, output: [
            str(raw / 'corpus/data/abcfarma_corpus.json'),
            str(output)
        ]
    },

    'amazon_diseases': {
        'script': 'processor_amazon_diseases.py',
        'description': 'Amazon Diseases - Diseases QA from HuggingFace',
        'setup': lambda raw: True,
        'args': lambda raw, output: [str(output)]
    },

    'drbodebench': {
        'script': 'processor_drbodebench.py',
        'description': 'DrBodeBench - Medical multiple choice questions',
        'setup': lambda raw: True,
        'args': lambda raw, output: [str(output)]
    },

    'fall_detection': {
        'script': 'processor_fall_detection.py',
        'description': 'Fall Detection - Clinical notes classification',
        'setup': lambda raw: require_path(
            raw / 'fall-detection/training_data.csv',
            "Fall Detection training_data.csv"
        ),
        'args': lambda raw, output: [
            str(raw / 'fall-detection/training_data.csv'),
            str(output)
        ]
    },

    'multiclinsum_pt': {
        'script': 'processor_multiclinsum_pt.py',
        'description': 'MultiClinSum PT - Clinical text summarization',
        'setup': lambda raw: require_path(
            raw / 'multiclinsum_gs_train_pt',
            "MultiClinSum extracted folder"
        ),
        'args': lambda raw, output: [
            str(raw / 'multiclinsum_gs_train_pt'),
            str(output)
        ]
    },

    'wikidoc_pt': {
        'script': 'processor_wikidoc_pt.py',
        'description': 'WikiDoc PT - Medical QA from HuggingFace',
        'setup': lambda raw: True,
        'args': lambda raw, output: [str(output)]
    },

    'semclinbr': {
        'script': 'processor_semclinbr.py',
        'description': 'SemClinBr - Clinical NER with semantic groups',
        'setup': lambda raw: require_path(
            raw,
            "SemClinBr extracted corpus"
        ),
        'args': lambda raw, output: [
            str(raw),
            str(output)
        ]
    },

    'clinical_ner': {
        'script': 'processor_clinical_ner.py',
        'description': 'Clinical NER - NER with entity labels',
        'setup': lambda raw: require_path(
            raw / 'PortugueseClinicalNER',
            "PortugueseClinicalNER cloned repository"
        ),
        'args': lambda raw, output: [
            str(raw / 'PortugueseClinicalNER/Texts SPN 1 Labeled English'),
            str(raw / 'PortugueseClinicalNER/Texts SPN 2 Labeled English'),
            str(output)
        ]
    }
}

def run_dataset_processor(
    dataset_name: str,
    config: Dict,
    output_base_dir: Path,
    raw_dir: Path,
    processors_dir: Path
) -> tuple[bool, float]:
    """
    Run a single dataset processor.
    
    Returns:
        Tuple of (success: bool, duration: float)
    """
    start_time = time.time()
    
    script_path = processors_dir / config['script']
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False, 0.0
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {dataset_name}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"{'='*70}\n")
    
    # Run setup (clone repos, download data, etc.)
    logger.info("Running setup...")
    setup_success = config['setup'](raw_dir)
    if not setup_success:
        logger.error(f"❌ Setup failed for {dataset_name}")
        return False, 0.0
    
    # Get arguments
    args = config['args'](raw_dir, output_base_dir)
    
    # Build command
    cmd = [sys.executable, str(script_path)] + args
    
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        duration = time.time() - start_time
        logger.info(f"✅ Successfully processed: {dataset_name}")
        logger.info(f"⏱️  Time: {duration:.2f} seconds\n")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"❌ Error processing {dataset_name}")
        logger.error(f"Exit code: {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Unexpected error processing {dataset_name}: {e}")
        return False, duration


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process clinical benchmark datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
""" + '\n'.join([f"  - {name}: {cfg['description']}" 
                 for name, cfg in DATASETS.items()])
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()) + ['all'],
        default=['all'],
        help='Datasets to process (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../data/benchmark'),
        help='Base output directory for all datasets (default: ../data/benchmark)'
    )
    
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('../data/raw'),
        help='Raw directory for cloning repos and downloads (default: ../data/raw)'
    )
    
    parser.add_argument(
        '--processors-dir',
        type=Path,
        default=Path('../src/data/processors'),
        help='Directory containing dataset processing scripts (default: ../src/data/processors/)'
    )
    
    args = parser.parse_args()
    
    # Validate datasets directory
    if not args.processors_dir.exists():
        logger.error(f"Processors directory not found: {args.processors_dir}")
        print('erro 1')
        sys.exit(1)
    
    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to process
    if 'all' in args.datasets:
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = args.datasets
    
    logger.info(f"Output directory: {args.output_dir.absolute()}")
    logger.info(f"Workspace directory: {args.raw_dir.absolute()}")
    logger.info(f"Processing {len(datasets_to_process)} dataset(s): {', '.join(datasets_to_process)}")
    
    # Process datasets
    results = {}
    timings = {}
    total_start = time.time()
    
    for dataset_name in datasets_to_process:
        config = DATASETS[dataset_name]
        success, duration = run_dataset_processor(
            dataset_name,
            config,
            args.output_dir,
            args.raw_dir,
            args.processors_dir
        )
        results[dataset_name] = success
        timings[dataset_name] = duration
    
    total_duration = time.time() - total_start
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    logger.info(f"Total datasets: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    if successful:
        logger.info(f"\n✅ Successful datasets:")
        for name in successful:
            logger.info(f"  - {name} ({timings[name]:.2f}s)")
    
    if failed:
        logger.info(f"\n❌ Failed datasets:")
        for name in failed:
            logger.info(f"  - {name} ({timings[name]:.2f}s)")
    
    logger.info(f"\n{'='*70}\n")
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()