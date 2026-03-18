"""
Metrics Computation Script
Computes task-specific metrics for benchmark evaluation.
Supports: QA and summarization (ROUGE+BERTScore), Multiple Choice, Classification, NER.
"""

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from transformers import AutoTokenizer
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics import f1_score as seqeval_f1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BERTScore models for Portuguese
BERTSCORE_MODELS = {
    "biobertpt": "pucpr/biobertpt-all",
    "bertimbau": "neuralmind/bert-base-portuguese-cased"
}

# Entity labels for NER tasks
SEMCLINBR_LABELS = [
    "Disorder", "Procedures", "Chemicals and Drugs", "Abbreviation"
]

CLINICAL_NER_LABELS = [
    "Condição", "Anatomia", "Terapia", "Procedimento", "Evolução",
    "Genética", "Negação", "Observação", "Resultado", "Tempo",
    "Valor", "ViaAdmin", "Característica"
]

# Dataset-specific reference keys mapping
DATASET_REFERENCE_KEYS = {
    "amazon_diseases": "Response",
    "wikidoc_pt": "answer",
    "abcfarma_qa": "answer",
    "multiclinsum_pt": "summary",
    "fall_detection": "Target",
    "drbodebench": "resposta",
    "semclinbr": "tags",
    "clinical_ner": "tags",
}

# ==================== UTILITY FUNCTIONS ====================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_results(results: Dict, predictions_path: Path, output_dir: Path):
    """
    Save metrics results to JSON file.
    """
    pred_name = predictions_path.name

    if pred_name.endswith("_predictions.jsonl"):
        metrics_filename = pred_name.replace("_predictions.jsonl", "_metrics.json")
    else:
        # fallback
        metrics_filename = pred_name + "_metrics.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / metrics_filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {filepath}")


def clean_prefix(text: str) -> str:
    """Remove model/assistant/system prefixes from text."""
    text = text.strip()
    text = re.sub(r"(model|assistant|system)\s*\n?", "", text, flags=re.IGNORECASE)
    return text.strip()

def normalize_text(text: str) -> str:
    """Remove accents and lowercase for robust comparison."""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text

# ==================== QA AND SUMMARIZATION METRICS (ROUGE + BERTScore) ====================

def compute_rouge_metrics(examples: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE metrics for QA/Summarization tasks.
    Based on original baseline logic.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    for i, ex in enumerate(examples):
        ref = ex.get("ground_truth", "")
        pred = ex.get("prediction", "")

        # Clean prediction
        pred_start = pred.find("Resposta:")
        if pred_start != -1:
            pred_start += len("Resposta:")
            pred = pred[pred_start:].strip()
        else:
            pred = pred.strip()
        
        pred = clean_prefix(pred)
        scores = scorer.score(ref, pred)
        results.append(scores)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(examples)} examples")

    # Compute averages
    sums = {}
    n = len(results)
    
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        sums[metric] = {"precision": 0, "recall": 0, "fmeasure": 0}
    
    for res in results:
        for metric, scores in res.items():
            sums[metric]["precision"] += scores.precision
            sums[metric]["recall"] += scores.recall
            sums[metric]["fmeasure"] += scores.fmeasure
    
    averages = {}
    logger.info("=== ROUGE Metrics ===")
    for metric, scores in sums.items():
        p = scores['precision'] / n
        r = scores['recall'] / n
        f = scores['fmeasure'] / n
        averages[metric] = {"precision": p, "recall": r, "fmeasure": f}
        logger.info(f"{metric.upper()}:")
        logger.info(f"  Precision: {p:.4f}")
        logger.info(f"  Recall:    {r:.4f}")
        logger.info(f"  F1 Score:  {f:.4f}")
    
    return averages

def compute_bertscore_metrics(
    examples: List[Dict],
    model_name: str = "pucpr/biobertpt-all",
    max_len: int = 510
) -> Tuple[float, float, float]:
    """
    Compute BERTScore metrics.
    Based on original baseline logic with truncation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preds = []
    refs = []

    for ex in examples:
        pred = ex.get("prediction", "")
        ref = ex.get("ground_truth", "")

        # Clean prediction
        pred_start = pred.find("Resposta:")
        if pred_start != -1:
            pred_start += len("Resposta:")
            pred = pred[pred_start:].strip()
        else:
            pred = pred.strip()
        pred = clean_prefix(pred)

        # Truncate based on tokens
        pred_tokens = tokenizer.tokenize(pred)[:max_len]
        ref_tokens = tokenizer.tokenize(ref)[:max_len]
        
        pred_trunc = tokenizer.convert_tokens_to_string(pred_tokens)
        ref_trunc = tokenizer.convert_tokens_to_string(ref_tokens)
        
        preds.append(pred_trunc)
        refs.append(ref_trunc)
    
    P, R, F1 = bertscore(
        preds,
        refs,
        lang='pt',
        model_type=model_name,
        num_layers=12,
        verbose=True
    )
    
    logger.info(f"=== BERTScore ({model_name}) ===")
    logger.info(f"Precision: {P.mean().item():.4f}")
    logger.info(f"Recall:    {R.mean().item():.4f}")
    logger.info(f"F1:        {F1.mean().item():.4f}")
    
    return float(P.mean()), float(R.mean()), float(F1.mean())

# ==================== CLASSIFICATION METRICS ====================

def compute_classification_metrics(examples: List[Dict]) -> Dict[str, float]:
    """
    Compute binary classification metrics for fall detection.
    Extracts "Queda: 0/1" from generated response.
    """
    y_true = []
    y_pred = []

    for ex in examples:
        target = ex.get("ground_truth")
        generated = ex.get("prediction", "")

        match = re.search(r'Queda: ?([01])', generated)
        if match:
            pred = int(match.group(1))
        else:
            logger.warning(f"Malformed prediction: {generated[:80]}...")
            pred = 0 
        y_true.append(int(target))
        y_pred.append(pred)

    if not y_true or not y_pred:
        logger.error("No valid predictions found for classification.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logger.info("=== Binary Classification (Fall Risk) ===")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==================== MULTIPLE CHOICE METRICS ====================

def extract_answer(text: str) -> str:
    """Extract answer letter from model response."""
    text = clean_prefix(text)
    text = text + ' .'
    pattern = r"[Rr][Ee][Ss][Pp][Oo][Ss][Tt][Aa].*?([A-E])[\)\.\s]"
    match = re.search(pattern, text)
    return match.group(1) if match else None

def compute_multiple_choice_metrics(examples: List[Dict]) -> Dict[str, float]:
    """
    Compute multiple choice classification metrics.
    Extracts letter (A-D) from response and computes macro metrics.
    """
    y_true = []
    y_pred = []

    for ex in examples:
        target = ex.get("ground_truth")
        generated = ex.get("prediction", "")
        predicted_letter = extract_answer(generated)

        if predicted_letter is not None:
            y_true.append(target)
            y_pred.append(predicted_letter)
        else:
            logger.warning(f"Malformed or unrecognized response: {generated}...")
            predicted_letter = 'N'  # Special class for no answer
            y_true.append(target)
            y_pred.append(predicted_letter)

    if not y_true:
        logger.error("No valid predictions found.")
        return {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    logger.info("Classification Report (by letter):\n")
    logger.info(classification_report(
        y_true, y_pred,
        labels=['A', 'B', 'C', 'D', 'N']
    ))

    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(
        y_true, y_pred,
        labels=['A', 'B', 'C', 'D', 'N']
    ))

    # Macro metrics
    precision = precision_score(
        y_true, y_pred,
        average='macro',
        labels=['A', 'B', 'C', 'D', 'N']
    )
    recall = recall_score(
        y_true, y_pred,
        average='macro',
        labels=['A', 'B', 'C', 'D', 'N']
    )
    f1 = f1_score(
        y_true, y_pred,
        average='macro',
        labels=['A', 'B', 'C', 'D', 'N']
    )

    logger.info("\nGlobal Metrics (macro average):")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")

    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

# ==================== NER METRICS ====================

def parse_model_output_entities(text):
    entity_map = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            cat, tokens_str = line.split(":", 1)
            tokens_list = [
                t.strip()
                for t in re.split(r'[;,]', tokens_str) 
                if t.strip()
            ]
            entity_map[cat.strip()] = tokens_list
    return entity_map

def create_bio_labels(text: str, entities: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Convert entities to BIO labels at token level.
    
    Args:
        text: full document text
        entities: dict {class: [list of entities]}
    
    Returns:
        tokens: list of tokens
        labels: list of BIO labels for each token
    """

    # tokenization
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    labels = ["O"] * len(tokens)
    tokens_normalized = [normalize_text(t) for t in tokens]
    
    for entity_class, entity_list in entities.items():
        if not entity_list:
            continue
        
        for ent in entity_list:
            if not ent or not isinstance(ent, str):
                continue
            
            # Remove trailing punctuation
            ent = re.sub(r'[.,;:!?]+$', '', ent.strip())
            if not ent:
                continue
            
            # Tokenize entity
            ent_tokens = re.findall(r'\w+|[^\w\s]', ent, re.UNICODE)
            ent_tokens_normalized = [normalize_text(t) for t in ent_tokens]
            n = len(ent_tokens_normalized)
            
            # Find exact match in text
            for i in range(len(tokens_normalized) - n + 1):
                if tokens_normalized[i:i+n] == ent_tokens_normalized:
                    labels[i] = f"B-{entity_class.upper()}"
                    for j in range(1, n):
                        labels[i+j] = f"I-{entity_class.upper()}"
    return tokens, labels

def compute_ner_metrics(examples: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Compute NER metrics at entity level using BIO format.
    Uses seqeval for proper entity-level evaluation.
    
    Returns:
        macro: dict with precision, recall, F1
        micro: dict with precision, recall, F1
    """
    all_y_true = []
    all_y_pred = []

    for ex in examples:
        text = ex["text"]
        labels_ref = ex["labels_ref"]
        labels_pred = ex["labels_pred"]
       
        # Convert to BIO format
        tokens_ref, bio_ref = create_bio_labels(text, labels_ref)
        tokens_pred, bio_pred = create_bio_labels(text, labels_pred)

        # Ensure tokens match
        assert tokens_ref == tokens_pred
        
        all_y_true.append(bio_ref)
        all_y_pred.append(bio_pred)

    logger.info('all_y_true:', all_y_true)
    logger.info('all_y_pred:', all_y_pred)
    # Compute macro and micro metrics using seqeval
    macro_precision = seqeval_precision(all_y_true, all_y_pred, average="macro")
    macro_recall = seqeval_recall(all_y_true, all_y_pred, average="macro")
    macro_f1 = seqeval_f1(all_y_true, all_y_pred, average="macro")

    micro_precision = seqeval_precision(all_y_true, all_y_pred, average="micro")
    micro_recall = seqeval_recall(all_y_true, all_y_pred, average="micro")
    micro_f1 = seqeval_f1(all_y_true, all_y_pred, average="micro")

    macro = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }

    micro = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1
    }

    logger.info("MACRO METRICS (average per example):")
    logger.info(f"Precision: {macro_precision:.4f}")
    logger.info(f"Recall: {macro_recall:.4f}")
    logger.info(f"F1: {macro_f1:.4f}")
    
    logger.info("\nMICRO METRICS (global TP/FP/FN):")
    logger.info(f"Precision: {micro_precision:.4f}")
    logger.info(f"Recall: {micro_recall:.4f}")
    logger.info(f"F1: {micro_f1:.4f}")

    return macro, micro

# ==================== MAIN FUNCTION ====================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compute metrics for benchmark predictions'
    )
    
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to predictions JSONL file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=False,
        default=Path("../../results/metrics"),
        help='Path to save metrics JSON'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['qa', 'summarization', 'classification', 'multiple_choice', 'ner'],
        help='Task type'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        choices=list(DATASET_REFERENCE_KEYS.keys()),
        help='Dataset name'
    )
    args = parser.parse_args()
    
    # Validate input
    if not args.predictions.exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    # Load predictions
    logger.info(f"Loading predictions from: {args.predictions}")
    examples = load_jsonl(args.predictions)
    logger.info(f"Loaded {len(examples)} examples")
    
    # Compute metrics based on task
    results = {}
    
    if args.task in ['qa', 'summarization']:
        logger.info("Computing ROUGE metrics...")
        rouge_metrics = compute_rouge_metrics(examples)
        results['rouge'] = rouge_metrics
        
        # Compute BERTScore with both models
        for bert_name, bert_model in BERTSCORE_MODELS.items():
            try:
                logger.info(f"Computing BERTScore with {bert_name}...")
                p, r, f1 = compute_bertscore_metrics(examples, bert_model)
                results[f'bertscore_{bert_name}'] = {
                    "precision": p,
                    "recall": r,
                    "f1": f1
                }
            except Exception as e:
                logger.error(f"Error computing BERTScore with {bert_name}: {e}")
    
    elif args.task == 'classification':
        logger.info("Computing classification metrics...")
        results['classification'] = compute_classification_metrics(examples)
    
    elif args.task == 'multiple_choice':
        logger.info("Computing multiple choice metrics...")
        results['multiple_choice'] = compute_multiple_choice_metrics(examples)
    
    elif args.task == 'ner':
        logger.info("Computing NER metrics...")
        
        # Process examples for NER format
        processed_examples = []
        for ex in examples:
            text = ex.get("text", "")  # Text field
            pred_text = ex.get("prediction", "")
            ref_entities = ex.get("ground_truth", {})
            pred_entities = parse_model_output_entities(pred_text)
            processed_examples.append({
                "text": text,
                "labels_ref": ref_entities,
                "labels_pred": pred_entities
            })
        
        macro, micro = compute_ner_metrics(processed_examples)
        results['ner_macro'] = macro
        results['ner_micro'] = micro
    
    # Save results
    save_results(results, args.predictions, args.output)
    
    logger.info("Metrics computation complete!")

if __name__ == "__main__":
    main()