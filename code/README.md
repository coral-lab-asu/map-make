# Map&Make: Schema Guided Text to Table Generation

[![Paper](https://img.shields.io/badge/Paper-arxiv.2505.23174-b31b1b.svg)](https://arxiv.org/abs/2505.23174)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-Rotowire--Text--to--Table-yellow)](https://huggingface.co/datasets/McH04/Rotowire-Text-to-Table)

**(ACL 2025)** Map&Make is a schema-guided approach that decomposes text into atomic propositions to infer latent schemas, then generates tables that capture both qualitative nuances and quantitative facts. This repository contains prompts, evaluation scripts, and resources for reproducing and extending the work.

---

## Overview

![Map&Make Overview](../static/images/M&M.jpg)

Map&Make addresses **text-to-table generation**: transforming dense, unstructured text into interpretable tables. The method:

1. **Decomposes** input text into atomic propositions
2. **Infers** latent schemas from these propositions
3. **Generates** tables (player and team statistics in RotoWire) that are faithful to the text

We evaluate on RotoWire (multi-table schema), [Livesum](https://github.com/HKUST-KnowComp/LiveSum/tree/main) (numerical aggregation), and Wiki40 (open-domain extraction). We also release a **corrected RotoWire test set** to address hallucination and missing-information issues in prior releases (see [dataset README](https://huggingface.co/datasets/McH04/Rotowire-Text-to-Table) and paper for details).

---

## Repository Structure

```
map-make/
├── code/
│   ├── evals/          # Evaluation scripts and metrics
│   ├── prompts/        # Prompts for Map&Make and baselines
│   └── README.md       # Code-level documentation
├── static/             # Website assets (images, CSS, JS)
├── index.html          # Project website
└── README.md           # This file
```

---

## Prompts (`code/prompts/`)

Prompts are organized by dataset and by method (Map&Make vs. baselines).

### Map&Make pipeline (RotoWire: `prompts/Rotowire/`)

| File | Description |
|------|-------------|
| **Atomization.txt** | Decompose text into atomic, single-fact statements. |
| **Schema_Extraction.txt** | Infer table schema (headers, structure) from atomic statements. |
| **Table_Generation.txt** | Generate table rows from atomic statements under the inferred schema. |
| **Unified.txt** | Single prompt that chains Atomization → Schema Extraction → Table Generation (Unified CoT). |
| **Unified_CoT.txt** | Unified chain-of-thought variant. |

These implement the pipeline described in the paper: text → atomic propositions → schema → tables.

### Baselines (RotoWire: `prompts/Baselines_Rotowire/`)

| File | Description |
|------|-------------|
| **CoT_0shot.txt** / **CoT_1shot.txt** | Chain-of-thought table generation (zero- and one-shot). |
| **T3_Text_Tuple.txt** | T3-style text-to-tuple formulation. |
| **T3_Tuple_Table.txt** | T3-style tuple-to-table. |
| **T3_Tuple_integrate.txt** | T3 integrated tuple+table. |
| **T3_merged_1shot.txt** | T3 merged one-shot prompt. |

Used for comparison against Map&Make in the paper.

---

## Evaluation (`code/evals/`)

Evaluation is split into **cell-level metrics** (EM, chrF, BERTScore), **entailment-based** metrics, and **QA-based** (AutoQA) metrics.

### Cell-level metrics (`em_chrf_bert/`)

- **em_chrf_bert_eval.py**  
  Evaluates generated tables against gold tables using:
  - **Exact Match (EM)** for headers and cells  
  - **chrF** (character n-gram F-score)  
  - **BERTScore** (e.g. RoBERTa-large) for semantic similarity  

  Supports precision, recall, and F1 over headers and non-header cells, with optional grouping by row/column header.

- **em_chrf_bert_collated.py**  
  Collates and aggregates results across runs or samples.

### Table entailment (`tabeval/`)

- **entailment.py**  
  Uses an NLI model (e.g. RoBERTa-large-MNLI) to check whether each **gold** proposition is entailed by the **predicted** table (unrolled to text). Computes precision, recall, and F1 over propositions (referenceless, table-as-output evaluation).

- **tabunroll.txt**  
  Instructions/template for unrolling tables into natural language statements for entailment checking.

### AutoQA (`autoqa/`)

- **generate_qa.txt**  
  Instructions for generating fact-based questions from the **source text** (coverage of names, events, statistics, etc.).
- **qa_on_tables.txt**  
  Instructions for answering those questions using the **generated tables**.
- **check_answers.txt**  
  Instructions for checking QA answers against the text.

Used to measure whether the generated tables support the same factual questions as the source text (QA-based fidelity).

### Utilities (`utils/`)

- **eval.py**  
  Core table evaluation: parsing model output into tables, matching headers and cells to gold, and computing EM/chrF/BERTScore with configurable options.
- **table_utils.py**  
  Helpers for loading, normalizing, and comparing table structures.
- **utils.py**  
  General I/O and formatting utilities for evals.

---

## Dataset

- **Corrected RotoWire test set (text–table pairs)**  
  [Hugging Face: McH04/Rotowire-Text-to-Table](https://huggingface.co/datasets/McH04/Rotowire-Text-to-Table)  
  Clean benchmark for text-to-table; see the dataset card for motivation, structure, and citation.

- **Livesum**  
  [HKUST-KnowComp/LiveSum](https://github.com/HKUST-KnowComp/LiveSum/tree/main)  
  Used in the paper for settings requiring numerical aggregation.

---

## Citation

If you use Map&Make or the corrected RotoWire dataset, please cite:

```bibtex
@inproceedings{ahuja-etal-2025-map,
    title = "Map{\&}Make: Schema Guided Text to Table Generation",
    author = "Ahuja, Naman and Bardoliya, Fenil and Baral, Chitta and Gupta, Vivek",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1460/",
    doi = "10.18653/v1/2025.acl-long.1460",
    pages = "30249--30262"
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Links

- **Paper:** [ACL 2025](https://aclanthology.org/2025.acl-long.1460/) | [arXiv](https://arxiv.org/abs/2505.23174)
- **Project page:** [https://coral-lab-asu.github.io/map-make](https://coral-lab-asu.github.io/map-make)
- **Dataset:** [Hugging Face – McH04/Rotowire-Text-to-Table](https://huggingface.co/datasets/McH04/Rotowire-Text-to-Table)
