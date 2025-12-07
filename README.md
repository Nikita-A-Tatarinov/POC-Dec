# SORGA: Structured Ontology-Guided Reasoning with Graph-Constrained Aggregation

Integration of three state-of-the-art KG-enhanced reasoning methods (GCR, ReKnoS, ORT) with novel improvements for multi-hop reasoning and semantic clustering.

## Quick Start

```bash
# Install dependencies
uv sync

# Set OpenAI API key
export OPENAI_API_KEY="your-key"

# Run experiments
uv run scripts/run_pipeline.py \
    --config configs/webqsp/full_with_improvements.yaml \
    --output results/webqsp_full_with_improvements.json
```

## Configuration Files

All experiment configurations are in `configs/`:

```
configs/
├── webqsp/
│   ├── gcr_only.yaml              # GCR baseline
│   ├── reknos_only.yaml           # ReKnoS baseline  
│   ├── ort_only.yaml              # ORT baseline
│   ├── gcr_reknos.yaml            # GCR + ReKnoS
│   ├── gcr_ort.yaml               # GCR + ORT
│   ├── reknos_ort.yaml            # ReKnoS + ORT
│   ├── full_integration.yaml     # All three methods
│   └── full_with_improvements.yaml # Full + novel improvements
└── cwq/
    └── [same structure as webqsp]
```

### Configuration Parameters

Example config structure:

```yaml
dataset:
  name: "webqsp"
  split: "test"
  
kg:
  source: "freebase"
  max_hops: 2
  
modules:
  gcr:
    enabled: true
    max_paths: 10
  
  reknos:
    enabled: true
    top_n_super_relations: 3
    use_semantic_clustering: false  # Set true for improvements
    
  ort:
    enabled: true
    use_multihop_backward: false    # Set true for improvements
    max_backward_hops: 3
    
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
```

## Running Experiments

### Single Experiment

```bash
# WebQSP with improvements
uv run scripts/run_pipeline.py \
    --config configs/webqsp/full_with_improvements.yaml \
    --output results/webqsp_full_with_improvements.json

# CWQ baseline
uv run scripts/run_pipeline.py \
    --config configs/cwq/gcr_only.yaml \
    --output results/cwq_gcr_only.json
```

### Command-Line Overrides

Override config parameters directly:

```bash
uv run scripts/run_pipeline.py \
    --config configs/webqsp/full_integration.yaml \
    --output results/test.json \
    --dataset.split dev \
    --llm.temperature 0.3
```

## Project Structure

```
POC-Dec/
├── configs/              # YAML configuration files
├── data/                 # Dataset files (WebQSP, CWQ)
├── results/              # Experiment outputs (JSON)
├── src/
│   ├── modules/          # GCR, ReKnoS, ORT implementations
│   ├── kg_interface.py   # Knowledge graph interface
│   └── evaluator.py      # Evaluation metrics
├── scripts/
│   └── run_pipeline.py   # Main entry point
└── POC_Dec/              # Paper manuscript (ACL format)
```

## Results Format

Output JSON structure:

```json
{
  "metadata": {
    "dataset": "webqsp",
    "config": "full_with_improvements",
    "timestamp": "2025-12-07T10:00:00"
  },
  "metrics": {
    "hit_at_1": 0.40,
    "f1": 0.42,
    "precision": 0.45,
    "recall": 0.39
  },
  "predictions": [
    {
      "question_id": "WebQTest-123",
      "question": "Where did Barack Obama attend college?",
      "predicted_answers": ["Columbia University", "Harvard Law School"],
      "gold_answers": ["Columbia University", "Harvard Law School"],
      "correct": true
    }
  ]
}
```

## Evaluation Metrics

- **Hit@1**: Percentage of questions with at least one correct answer
- **F1**: Harmonic mean of precision and recall
- **Precision**: Correct answers / predicted answers
- **Recall**: Correct answers / gold answers

## Data

Uses pre-extracted Freebase subgraphs:
- **WebQSP**: 1,628 test questions
- **CWQ**: 3,531 test questions

Datasets are automatically downloaded from HuggingFace on first run.

## Requirements

- Python 3.10+
- `uv` package manager
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- ~8GB RAM for in-memory knowledge graphs

## Citation

```bibtex
@inproceedings{tatarinov2025sorga,
  title={SORGA: Structured Ontology-Guided Reasoning with Graph-Constrained Aggregation},
  author={Tatarinov, Nikita and Chava, Sudheer},
  year={2025}
}
```