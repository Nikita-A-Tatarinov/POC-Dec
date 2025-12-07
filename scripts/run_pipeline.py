#!/usr/bin/env python3
"""
SORGA End-to-End Pipeline - Production Version
===============================================

This script runs full-scale SORGA experiments on KGQA datasets.

Features:
- Automatic dataset download from HuggingFace
- In-memory knowledge graph construction (GCR approach)
- Complete SORGA 4-phase reasoning
- Comprehensive evaluation metrics

Usage:
    python scripts/run_pipeline.py --dataset webqsp --split test
    python scripts/run_pipeline.py --dataset cwq --split validation --output results/cwq_val.json
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from src.modules.sorga import SORGA
from src.modules.kg_interface import KGInterface
from src.modules.ontology import OntologyModule
from src.modules.ontology_ort import OntologyModuleORT
from src.modules.reknos import ReKnoSModule
from src.modules.super_relations import SuperRelationModule
from src.modules.llm_interface import MockLLMInterface, OpenAIInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file (default: configs/default_config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if answer is None:
        return ""
    return str(answer).lower().strip()


def evaluate_answer(predicted: str, gold_answers: List[str]) -> Dict[str, float]:
    """
    Evaluate predicted answer against gold answers
    
    Args:
        predicted: Predicted answer from SORGA
        gold_answers: List of acceptable gold answers
        
    Returns:
        Dictionary with Hit@1, precision, recall, F1 scores
    """
    if not predicted or not gold_answers:
        return {"hit_at_1": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pred_norm = normalize_answer(predicted)
    gold_norm = [normalize_answer(g) for g in gold_answers]
    
    # Hit@1 (exact match) - primary metric for paper comparison
    hit_at_1 = 1.0 if pred_norm in gold_norm else 0.0
    
    # Token-level F1 (simple bag-of-words)
    pred_tokens = set(pred_norm.split())
    gold_tokens = set()
    for g in gold_norm:
        gold_tokens.update(g.split())
    
    if not pred_tokens or not gold_tokens:
        return {"hit_at_1": hit_at_1, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    common_tokens = pred_tokens & gold_tokens
    
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common_tokens) / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "hit_at_1": hit_at_1,      # Primary metric (for paper comparison)
        "precision": precision,
        "recall": recall,
        "f1": f1                   # Secondary metric (for partial credit analysis)
    }


def load_hf_dataset(dataset_name: str, split: str) -> Any:
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: 'webqsp' or 'cwq'
        split: 'train', 'validation', or 'test'
        
    Returns:
        Loaded dataset
    """
    dataset_mapping = {
        'webqsp': 'rmanluo/RoG-webqsp',
        'cwq': 'rmanluo/RoG-cwq'
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_mapping.keys())}")
    
    dataset_id = dataset_mapping[dataset_name]
    logger.info(f"Loading dataset: {dataset_id}, split: {split}")
    
    try:
        dataset = load_dataset(dataset_id, split=split)
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def run_pipeline(
    dataset_name: str = None,
    split: str = None,
    output_path: str = None,
    limit: int = None,
    skip_module_building: bool = False,
    mock_llm: bool = None,
    config_path: str = None
) -> Dict[str, Any]:
    """
    Run the complete SORGA pipeline
    
    Args:
        dataset_name: 'webqsp' or 'cwq' (overrides config if provided)
        split: Dataset split to use (overrides config if provided)
        output_path: Path to save results JSON
        limit: Limit number of examples (None = all, overrides config if provided)
        skip_module_building: Skip ontology/super-relation module building
        mock_llm: Use mock LLM (no API calls, overrides config if provided)
        config_path: Path to config file (default: configs/default_config.yaml)
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 80)
    logger.info("SORGA PIPELINE START")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info("\n[Step 0/5] Loading configuration...")
    config = load_config(config_path)
    
    # Use config values if args not provided
    if dataset_name is None:
        dataset_name = config.get('dataset', {}).get('name', 'webqsp')
    if split is None:
        split = config.get('dataset', {}).get('split', 'test')
    if limit is None:
        limit = config.get('dataset', {}).get('limit')
    if mock_llm is None:
        mock_llm = config.get('llm', {}).get('use_mock', True)
    
    logger.info(f"Using dataset: {dataset_name}, split: {split}, limit: {limit}, mock_llm: {mock_llm}")
    
    # Step 1: Load dataset from HuggingFace
    logger.info("\n[Step 1/5] Loading dataset from HuggingFace...")
    dataset = load_hf_dataset(dataset_name, split)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        logger.info(f"Limited to {len(dataset)} examples")
    
    # Step 1.5: Build global KG for ORT ontology (if needed)
    kg_interface = KGInterface()
    
    if config.get('use_ort', False):
        logger.info("\n[Step 1.5/5] Building global ontology for ORT...")
        logger.info("Loading all question graphs to construct ontology...")
        
        for idx, entry in enumerate(tqdm(dataset, desc="Loading graphs for ontology")):
            graph_triples = entry.get('graph', [])
            kg_interface.load_from_dataset(graph_triples, merge=True)
        
        logger.info(f"✓ Global KG loaded: {kg_interface.graph.number_of_nodes()} nodes, "
                   f"{kg_interface.graph.number_of_edges()} edges")
        logger.info("This global graph will be used to build ORT ontology structures")
    
    # Step 2: Initialize components
    logger.info("\n[Step 2/5] Initializing SORGA components...")
    
    # LLM Interface
    if mock_llm:
        llm_interface = MockLLMInterface()
        logger.info("✓ Components initialized (Mock LLM mode)")
    else:
        try:
            llm_model = config.get('llm', {}).get('model_name', 'gpt-4o-mini')
            llm_interface = OpenAIInterface(model_name=llm_model)
            logger.info(f"✓ Components initialized ({llm_model})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            logger.error("Please set OPENAI_API_KEY environment variable")
            raise
    
    # Ontology Module
    ontology_module = OntologyModule(kg_interface)
    
    # ORT Module (optional)
    ontology_ort_module = None
    if config.get('use_ort', False):
        logger.info("✓ Creating ORT module")
        ontology_ort_module = OntologyModuleORT(
            kg_interface=kg_interface,
            llm_interface=llm_interface,
            config=config
        )
    
    # ReKnoS Module (optional)
    reknos_module = None
    if config.get('use_reknos', False):
        logger.info("✓ Creating ReKnoS module")
        reknos_module = ReKnoSModule(
            kg_interface=kg_interface,
            llm_interface=llm_interface,
            config=config
        )
    
    # Super-Relation Module
    super_relation_module = SuperRelationModule(kg_interface)
    
    # SORGA
    sorga = SORGA(
        kg_interface=kg_interface,
        llm_interface=llm_interface,
        ontology_module=ontology_module,
        super_relation_module=super_relation_module,
        ontology_ort_module=ontology_ort_module,
        reknos_module=reknos_module,
        config=config
    )
    
    # Step 3: Process questions
    logger.info(f"\n[Step 3/5] Processing {len(dataset)} questions...")
    
    # Get logging frequency from config (default: log every question)
    log_every_n = config.get('log_every_n', 1)
    if log_every_n > 1:
        logger.info(f"Detailed logging enabled for every {log_every_n} questions")
    
    results = []
    metrics_accumulator = {
        "total": 0,
        "hit_at_1_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "f1_sum": 0.0,
        "errors": 0
    }
    
    for idx, entry in enumerate(tqdm(dataset, desc="Processing questions")):
        # Determine if we should log details for this question
        should_log_details = (idx % log_every_n == 0) or (idx == len(dataset) - 1)
        
        try:
            # Extract data
            question = entry['question']
            q_entities = entry.get('q_entity', [])
            gold_answers = entry.get('answer', [])
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            graph_triples = entry.get('graph', [])
            
            # Conditional logging based on log_every_n
            if should_log_details:
                logger.info(f"\n{'='*80}")
                logger.info(f"Question {idx+1}/{len(dataset)}: {question}")
                logger.info(f"Entities: {q_entities}")
                logger.info(f"Graph size: {len(graph_triples)} triples")
                logger.info(f"{'='*80}")
            else:
                logger.debug(f"\nQuestion {idx+1}: {question}")
                logger.debug(f"Entities: {q_entities}")
                logger.debug(f"Graph size: {len(graph_triples)} triples")
            
            # Load graph for this question
            # Note: If ORT is enabled, we already have a global graph with ontology.
            # For GCR/ReKnoS, we load the question-specific graph.
            # This is a tradeoff: ORT uses global ontology, GCR/ReKnoS use local graphs.
            if not config.get('use_ort', False):
                # No ORT: use question-specific graph (GCR/ReKnoS approach)
                kg_interface.load_from_dataset(graph_triples)
            else:
                # ORT enabled: Keep global graph but note we're processing this question's subgraph
                # The global graph already contains this question's triples from Step 1.5
                # ORT will use the global ontology, GCR/ReKnoS will work with global graph too
                logger.debug(f"Using global graph (contains {kg_interface.graph.number_of_nodes()} total nodes)")
                pass
            
            # Run SORGA
            predicted_answer = sorga.answer_question(question, q_entities)
            
            # Evaluate (extract answer string from result dictionary)
            answer_str = predicted_answer.get('answer', '') if isinstance(predicted_answer, dict) else str(predicted_answer)
            eval_metrics = evaluate_answer(answer_str, gold_answers)
            
            # Store result
            result = {
                "id": entry.get('id', f"q_{idx}"),
                "question": question,
                "entities": q_entities,
                "gold_answers": gold_answers,
                "predicted_answer": predicted_answer,
                "metrics": eval_metrics,
                "graph_size": len(graph_triples)
            }
            results.append(result)
            
            # Update metrics
            metrics_accumulator["total"] += 1
            metrics_accumulator["hit_at_1_sum"] += eval_metrics["hit_at_1"]
            metrics_accumulator["precision_sum"] += eval_metrics["precision"]
            metrics_accumulator["recall_sum"] += eval_metrics["recall"]
            metrics_accumulator["f1_sum"] += eval_metrics["f1"]
            
            logger.debug(f"Predicted: {answer_str}")
            logger.debug(f"Gold: {gold_answers}")
            logger.debug(f"Hit@1: {eval_metrics['hit_at_1']:.3f}, F1: {eval_metrics['f1']:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing question {idx}: {e}")
            metrics_accumulator["errors"] += 1
            results.append({
                "id": entry.get('id', f"q_{idx}"),
                "question": entry.get('question', ''),
                "error": str(e)
            })
    
    # Step 4: Compute aggregate metrics
    logger.info("\n[Step 4/5] Computing aggregate metrics...")
    
    if metrics_accumulator["total"] > 0:
        aggregate_metrics = {
            "total_questions": metrics_accumulator["total"],
            "errors": metrics_accumulator["errors"],
            "hit_at_1": metrics_accumulator["hit_at_1_sum"] / metrics_accumulator["total"],
            "avg_precision": metrics_accumulator["precision_sum"] / metrics_accumulator["total"],
            "avg_recall": metrics_accumulator["recall_sum"] / metrics_accumulator["total"],
            "avg_f1": metrics_accumulator["f1_sum"] / metrics_accumulator["total"]
        }
    else:
        aggregate_metrics = {
            "total_questions": 0,
            "errors": metrics_accumulator["errors"],
            "hit_at_1": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0
        }
    
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE METRICS")
    logger.info("=" * 80)
    logger.info(f"Total Questions: {aggregate_metrics['total_questions']}")
    logger.info(f"Errors: {aggregate_metrics['errors']}")
    logger.info(f"Hit@1 (Primary): {aggregate_metrics['hit_at_1']:.4f}")
    logger.info(f"Average F1 (Secondary): {aggregate_metrics['avg_f1']:.4f}")
    logger.info(f"Average Precision: {aggregate_metrics['avg_precision']:.4f}")
    logger.info(f"Average Recall: {aggregate_metrics['avg_recall']:.4f}")
    logger.info("=" * 80)
    
    # Step 5: Save results
    if output_path:
        logger.info(f"\n[Step 5/5] Saving results to {output_path}...")
        output_data = {
            "metadata": {
                "dataset": dataset_name,
                "split": split,
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(dataset),
                "processed": metrics_accumulator["total"],
                "errors": metrics_accumulator["errors"]
            },
            "aggregate_metrics": aggregate_metrics,
            "results": results
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")
    else:
        logger.info("\n[Step 5/5] Skipping save (no output path specified)")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    return {
        "aggregate_metrics": aggregate_metrics,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="SORGA Production Pipeline - Full-Scale Experiments (GCR Approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults from config file
  python scripts/run_pipeline.py
  
  # Run full WebQSP test set (1,628 questions)
  python scripts/run_pipeline.py --dataset webqsp --split test
  
  # Run CWQ validation set with custom output
  python scripts/run_pipeline.py --dataset cwq --split validation --output results/cwq_val.json
  
  # Run with custom config file
  python scripts/run_pipeline.py --config configs/my_config.yaml
  
  # Run with debug logging
  python scripts/run_pipeline.py --dataset webqsp --split test --debug
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/default_config.yaml)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['webqsp', 'cwq'],
        help='Dataset to use (overrides config if provided)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'validation', 'test'],
        help='Dataset split to use (overrides config if provided)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON (default: results/<dataset>_<split>_<timestamp>.json)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples to process (overrides config if provided)'
    )
    
    parser.add_argument(
        '--skip-module-building',
        action='store_true',
        default=True,
        help='Skip building ontology and super-relation modules (default: True for production)'
    )
    
    parser.add_argument(
        '--mock-llm',
        action='store_true',
        default=None,
        help='Use mock LLM (no API calls, overrides config if provided)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config to get defaults
    config = load_config(args.config)
    
    # Set default output path if not specified
    if args.output is None:
        dataset_name = args.dataset or config.get('dataset', {}).get('name', 'webqsp')
        split = args.split or config.get('dataset', {}).get('split', 'test')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{dataset_name}_{split}_{timestamp}.json"
    
    try:
        run_pipeline(
            dataset_name=args.dataset,
            split=args.split,
            config_path=args.config,
            output_path=args.output,
            limit=args.limit,
            skip_module_building=args.skip_module_building,
            mock_llm=args.mock_llm  # Use config default if not specified
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
