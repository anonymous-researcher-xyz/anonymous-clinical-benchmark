# Croissant Metadata Files

This directory contains [Croissant metadata](http://mlcommons.org/croissant/) files for all datasets in the ClinicalBenchPT benchmark. Croissant is an open-source metadata format standard developed by MLCommons for describing machine learning datasets.

## 📋 What is Croissant?

Croissant provides a standardized way to describe datasets, enabling:
- **Discoverability**: Better indexing in dataset search engines (e.g., Google Dataset Search)
- **Interoperability**: Easy loading with standard tools and libraries
- **Reproducibility**: Complete documentation of data structure and preprocessing steps
- **Transparency**: Clear information about licensing, data collection, and ethical considerations

Learn more at: [mlcommons.org/croissant](http://mlcommons.org/croissant/)

## 📁 Available Metadata Files

| Dataset | Croissant File | License | Access |
|---------|---------------|---------|--------|
| ABCFarma-QA | [croissant_abcfarma.jsonld](croissant_abcfarma.jsonld) | MIT | Public |
| AmazonDiseases-QA | [croissant_amazon_diseases.jsonldn](croissant_amazon_diseases.jsonld) | Unknown | Public (HF) |
| Wikidoc-pt-QA | [croissant_wikidoc_pt.jsonld](croissant_wikidoc_pt.jsonld) | Apache-2.0 | Public (HF) |
| DrBodeBench-DRB | [croissant_drbodebench_drb.jsonld](croissant_drbodebench_drb.jsonld) | Apache-2.0 | Public (HF) |
| FallDetection-PT | [croissant_fall_detection_pt.jsonld](croissant_fall_detection_pt.jsonld) | AGPL-3.0 | Public |
| MultiClinSum-PT | [croissant_multiclinsum_pt.jsonld](croissant_multiclinsum_pt.jsonld) | CC-BY-4.0 | Public (Zenodo) |
| PortugueseClinicalNER - ClinPT | [croissant_clinpt.jsonld](croissant_clinpt.jsonld) | Unknown | Public |
| SemClinBr - SEM| [croissant_semclinbr_sem.jsonld](croissant_semclinbr_sem.jsonld) | Unknown | Requires permission |

## 🚀 Loading Datasets with Croissant

### Using mlcroissant (Python)

The official `mlcroissant` library provides easy loading of datasets described by Croissant metadata.

**Installation:**
```bash
pip install mlcroissant
```

**Basic Usage:**
```python
from mlcroissant import Dataset

# Load from local Croissant file
dataset = Dataset("croissant_abcfarma.jsonld")

print("RecordSets:", [rs.id for rs in dataset.metadata.record_sets])

# Iterate through records
for record in dataset.records("questions"):
    print(f"ID: {record['id']}")
    print(f"Question: {record['question']}")
    print(f"Answer: {record['answer']}")
    print("---")
```

## 🔍 Understanding the Metadata Structure

Each Croissant file contains:

### 1. Dataset Information
```json
{
  "@type": "sc:Dataset",
  "name": "ABCFarma-QA",
  "description": "Question answering dataset about medications...",
  "license": "MIT License"
}
```

### 2. Distribution (Files)
```json
{
  "distribution": [
    {
      "@id": "train.jsonl",
      "name": "train.jsonl",
      "encodingFormat": "application/jsonlines"
    }
  ]
}
```

### 3. Record Structure (Fields)
```json
{
  "recordSet": [{
    "field": [
      {
        "@id": "questions/id",
        "name": "id",
        "dataType": "sc:Integer"
      },
      {
        "@id": "questions/question",
        "name": "question",
        "dataType": "sc:Text"
      }
    ]
  }]
}
```

## ⚠️ Important Notes

1. **These metadata files describe the preprocessed datasets**, not the original sources
2. **You must first**:
   - Download original datasets from their sources
   - Obtain necessary permissions (e.g., SemClinBr)
   - Run our preprocessing scripts
   - Then use these Croissant files to understand/load the processed data

3. **Data is NOT distributed** in this repository due to licensing constraints

## 📚 Additional Resources

- [Croissant Format Specification](https://github.com/mlcommons/croissant)
- [mlcroissant Documentation](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant)
- [Google Dataset Search](https://datasetsearch.research.google.com/) (Croissant-compatible)
- [Croissant Editor](https://huggingface.co/spaces/MLCommons/croissant-editor) (Visual editor for Croissant files)

---

For questions or issues, please refer to the main [repository README](../README.md) or open an issue.