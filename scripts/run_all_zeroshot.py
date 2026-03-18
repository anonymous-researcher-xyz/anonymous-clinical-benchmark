import subprocess
import sys
from pathlib import Path
import time

# ==========================
# CONFIGURATION
# ==========================

MODEL_NAME = "medgemma"
BATCH_SIZE = 8
MAX_TOKENS = 512

DATASETS = [
    ("abcfarma_qa",      "qa"),
    ("amazon_diseases",  "qa"),
    ("drbodebench",      "multiple_choice"),
    ("fall_detection",   "classification"),
    ("multiclinsum_pt",  "summarization"),
    ("wikidoc_pt",       "qa"),
    ("semclinbr",        "ner"),
    ("clinical_ner",     "ner"),
]

BASE_DIR = Path(__file__).resolve().parent.parent
INFER_SCRIPT = BASE_DIR / "src/inference/zeroshot_inference.py"
METRIC_SCRIPT = BASE_DIR / "src/evaluation/evaluation_metrics.py"

BENCHMARK_DIR = BASE_DIR / "data/benchmark"
PRED_DIR = BASE_DIR / "results/predictions"
PRED_DIR.mkdir(exist_ok=True, parents=True)

METRICS_DIR = BASE_DIR / "results/metrics"
METRICS_DIR.mkdir(exist_ok=True, parents=True)

# ==========================
# OPTIONAL: AUTO SWITCH CONDA
# ==========================

def run_in_env(cmd, env_name=None):
    """
    Run a Python command optionally inside a conda env.
    """
    if env_name:
        full_cmd = ["conda", "run", "-n", env_name, "python"] + cmd
    else:
        full_cmd = [sys.executable] + cmd

    print("\n🚀 Running command:", " ".join(str(x) for x in full_cmd))
    return subprocess.run(full_cmd, check=True)

# ==========================
# EXECUTION
# ==========================

def main():
    start = time.time()

    print("\n===================================")
    print("🔥 Running ZERO-SHOT benchmark")
    print("===================================\n")

    for dataset_name, task in DATASETS:
        print(f"\n=== 🧪 DATASET: {dataset_name} | Task: {task} ===")

        test_file = BENCHMARK_DIR / dataset_name / "test.jsonl"
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue

        out_pred = PRED_DIR / f"{dataset_name}_zeroshot_predictions.jsonl"

        # -----------------------------
        # 1) ZERO-SHOT INFERENCE
        # -----------------------------
        infer_cmd = [
            str(INFER_SCRIPT),
            "--dataset-name", dataset_name,
            "--dataset-path", str(test_file),
            "--batch-size", str(BATCH_SIZE),
            "--max-tokens", str(MAX_TOKENS),
            "--output", str(out_pred)
        ]

        run_in_env(infer_cmd, env_name="conda-gpu")   # Make ≈ your GPU env

        # -----------------------------
        # 2) ZERO-SHOT METRICS
        # -----------------------------
        metric_cmd = [
            str(METRIC_SCRIPT),
            "--predictions", str(out_pred),
            "--model-name", MODEL_NAME,
            "--task", task,
            "--dataset-name", dataset_name,
            "--output", str(METRICS_DIR)
        ]

        run_in_env(metric_cmd, env_name="conda-cpu")   # Your CPU evaluation env

    total = time.time() - start
    print(f"\n⏱️ Finished all datasets in {total:.2f}s")

if __name__ == "__main__":
    main()
