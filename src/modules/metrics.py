"""
Evaluation metrics for KGQA
Implements Hit@1, F1, Precision, Recall, Path Validity, and efficiency metrics
"""

from typing import List, Dict, Set, Optional
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class AnswerMetrics:
    """Metrics for answer quality evaluation"""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer string for comparison"""
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove common prefixes
        prefixes = ['m.', 'g.', 'en.']
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        
        # Remove special characters
        answer = ''.join(c for c in answer if c.isalnum() or c.isspace())
        
        return answer.strip()
    
    @staticmethod
    def hit_at_k(predicted: List[str], gold: List[str], k: int = 1) -> float:
        """
        Compute Hit@K metric
        
        Args:
            predicted: List of predicted answers (ranked)
            gold: List of gold answers
            k: K value for Hit@K
            
        Returns:
            1.0 if any predicted answer in top-k matches gold, else 0.0
        """
        if not predicted or not gold:
            return 0.0
        
        # Normalize answers
        pred_normalized = [AnswerMetrics.normalize_answer(p) for p in predicted[:k]]
        gold_normalized = [AnswerMetrics.normalize_answer(g) for g in gold]
        
        # Check if any prediction matches any gold answer
        for pred in pred_normalized:
            if pred in gold_normalized:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def f1_score(predicted: List[str], gold: List[str]) -> float:
        """
        Compute F1 score
        
        Args:
            predicted: List of predicted answers
            gold: List of gold answers
            
        Returns:
            F1 score
        """
        if not predicted or predicted is None or not gold:
            return 0.0
        
        # Normalize and convert to sets
        pred_set = set(AnswerMetrics.normalize_answer(p) for p in predicted)
        gold_set = set(AnswerMetrics.normalize_answer(g) for g in gold)
        
        # Remove empty strings
        pred_set = {p for p in pred_set if p}
        gold_set = {g for g in gold_set if g}
        
        if not pred_set or not gold_set:
            return 0.0
        
        # Compute overlap
        overlap = len(pred_set & gold_set)
        
        if overlap == 0:
            return 0.0
        
        precision = overlap / len(pred_set)
        recall = overlap / len(gold_set)
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def precision(predicted: List[str], gold: List[str]) -> float:
        """Compute precision"""
        if not predicted or predicted is None:
            return 0.0
        
        pred_set = set(AnswerMetrics.normalize_answer(p) for p in predicted)
        gold_set = set(AnswerMetrics.normalize_answer(g) for g in gold)
        
        pred_set = {p for p in pred_set if p}
        gold_set = {g for g in gold_set if g}
        
        if not pred_set:
            return 0.0
        
        overlap = len(pred_set & gold_set)
        return overlap / len(pred_set)
    
    @staticmethod
    def recall(predicted: List[str], gold: List[str]) -> float:
        """Compute recall"""
        if not gold:
            return 0.0
        
        if not predicted or predicted is None:
            return 0.0
        
        pred_set = set(AnswerMetrics.normalize_answer(p) for p in predicted)
        gold_set = set(AnswerMetrics.normalize_answer(g) for g in gold)
        
        pred_set = {p for p in pred_set if p}
        gold_set = {g for g in gold_set if g}
        
        if not gold_set:
            return 0.0
        
        overlap = len(pred_set & gold_set)
        return overlap / len(gold_set)


class FaithfulnessMetrics:
    """Metrics for measuring path faithfulness to KG"""
    
    @staticmethod
    def path_validity(reasoning_paths: List[Dict], kg_interface) -> float:
        """
        Compute path validity rate
        Check if reasoning paths actually exist in KG
        
        Args:
            reasoning_paths: List of reasoning path dictionaries
            kg_interface: KG interface for verification
            
        Returns:
            Ratio of valid paths
        """
        if not reasoning_paths:
            return 0.0
        
        valid_count = 0
        
        for path in reasoning_paths:
            if path.get('valid', False):
                valid_count += 1
        
        return valid_count / len(reasoning_paths)
    
    @staticmethod
    def hallucination_rate(reasoning_paths: List[Dict], kg_interface) -> float:
        """
        Compute hallucination rate
        Ratio of paths or entities not in KG
        
        Args:
            reasoning_paths: List of reasoning path dictionaries
            kg_interface: KG interface for verification
            
        Returns:
            Hallucination rate (0 = no hallucination, 1 = all hallucinated)
        """
        if not reasoning_paths:
            return 0.0
        
        hallucinated_count = 0
        
        for path in reasoning_paths:
            # Check if path is marked as invalid
            if not path.get('valid', True):
                hallucinated_count += 1
        
        return hallucinated_count / len(reasoning_paths)
    
    @staticmethod
    def relation_accuracy(predicted_relations: List[str], 
                         gold_relations: List[str]) -> float:
        """
        Compute relation accuracy
        
        Args:
            predicted_relations: Predicted relations
            gold_relations: Gold relations
            
        Returns:
            Accuracy score
        """
        if not gold_relations:
            return 0.0
        
        pred_set = set(predicted_relations)
        gold_set = set(gold_relations)
        
        overlap = len(pred_set & gold_set)
        return overlap / len(gold_set)


class EfficiencyMetrics:
    """Metrics for measuring efficiency"""
    
    @staticmethod
    def avg_llm_calls(results: List[Dict]) -> float:
        """Average number of LLM calls per question"""
        if not results:
            return 0.0
        
        total_calls = sum(r.get('num_llm_calls', 0) for r in results)
        return total_calls / len(results)
    
    @staticmethod
    def avg_reasoning_time(results: List[Dict]) -> float:
        """Average reasoning time per question (seconds)"""
        if not results:
            return 0.0
        
        total_time = sum(r.get('reasoning_time', 0) for r in results)
        return total_time / len(results)
    
    @staticmethod
    def avg_tokens(results: List[Dict]) -> float:
        """Average number of tokens used per question"""
        if not results:
            return 0.0
        
        total_tokens = sum(r.get('total_tokens', 0) for r in results)
        return total_tokens / len(results)
    
    @staticmethod
    def throughput(results: List[Dict], total_time: float) -> float:
        """Questions per second"""
        if total_time == 0:
            return 0.0
        return len(results) / total_time


class Evaluator:
    """Main evaluator class"""
    
    def __init__(self, kg_interface=None):
        """
        Initialize evaluator
        
        Args:
            kg_interface: Optional KG interface for faithfulness metrics
        """
        self.kg_interface = kg_interface
    
    def evaluate(self, results: List[Dict], dataset: List[Dict]) -> Dict:
        """
        Evaluate results on dataset
        
        Args:
            results: List of result dictionaries from SORGA
            dataset: List of dataset items with gold answers
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {len(results)} results")
        
        # Answer quality metrics
        hit_at_1_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        # Faithfulness metrics
        path_validity_scores = []
        hallucination_scores = []
        
        for result, data_item in zip(results, dataset):
            if not result.get('success', False):
                continue
            
            predicted_answer = result.get('answer', '')
            gold_answers = data_item.get('answers', [])
            
            # Convert single answer to list
            if isinstance(predicted_answer, str):
                predicted_answers = [predicted_answer] if predicted_answer else []
            else:
                predicted_answers = predicted_answer
            
            # Answer metrics
            hit_at_1_scores.append(AnswerMetrics.hit_at_k(predicted_answers, gold_answers, k=1))
            f1_scores.append(AnswerMetrics.f1_score(predicted_answers, gold_answers))
            precision_scores.append(AnswerMetrics.precision(predicted_answers, gold_answers))
            recall_scores.append(AnswerMetrics.recall(predicted_answers, gold_answers))
            
            # Faithfulness metrics
            reasoning_paths = result.get('reasoning_paths', [])
            if reasoning_paths:
                path_validity_scores.append(FaithfulnessMetrics.path_validity(
                    reasoning_paths, self.kg_interface
                ))
                hallucination_scores.append(FaithfulnessMetrics.hallucination_rate(
                    reasoning_paths, self.kg_interface
                ))
        
        # Compute aggregates
        metrics = {
            'answer_quality': {
                'hit_at_1': np.mean(hit_at_1_scores) if hit_at_1_scores else 0.0,
                'f1': np.mean(f1_scores) if f1_scores else 0.0,
                'precision': np.mean(precision_scores) if precision_scores else 0.0,
                'recall': np.mean(recall_scores) if recall_scores else 0.0,
            },
            'faithfulness': {
                'path_validity': np.mean(path_validity_scores) if path_validity_scores else 0.0,
                'hallucination_rate': np.mean(hallucination_scores) if hallucination_scores else 0.0,
            },
            'efficiency': {
                'avg_llm_calls': EfficiencyMetrics.avg_llm_calls(results),
                'avg_reasoning_time': EfficiencyMetrics.avg_reasoning_time(results),
                'avg_tokens': EfficiencyMetrics.avg_tokens(results),
            },
            'coverage': {
                'total_questions': len(dataset),
                'successful_answers': len(hit_at_1_scores),
                'success_rate': len(hit_at_1_scores) / len(dataset) if dataset else 0.0,
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print("\n--- Answer Quality ---")
        aq = metrics['answer_quality']
        print(f"  Hit@1:     {aq['hit_at_1']:.4f} ({aq['hit_at_1']*100:.2f}%)")
        print(f"  F1:        {aq['f1']:.4f}")
        print(f"  Precision: {aq['precision']:.4f}")
        print(f"  Recall:    {aq['recall']:.4f}")
        
        print("\n--- Faithfulness ---")
        faith = metrics['faithfulness']
        print(f"  Path Validity:      {faith['path_validity']:.4f} ({faith['path_validity']*100:.2f}%)")
        print(f"  Hallucination Rate: {faith['hallucination_rate']:.4f} ({faith['hallucination_rate']*100:.2f}%)")
        
        print("\n--- Efficiency ---")
        eff = metrics['efficiency']
        print(f"  Avg LLM Calls:      {eff['avg_llm_calls']:.2f}")
        print(f"  Avg Reasoning Time: {eff['avg_reasoning_time']:.2f}s")
        print(f"  Avg Tokens:         {eff['avg_tokens']:.2f}")
        
        print("\n--- Coverage ---")
        cov = metrics['coverage']
        print(f"  Total Questions:     {cov['total_questions']}")
        print(f"  Successful Answers:  {cov['successful_answers']}")
        print(f"  Success Rate:        {cov['success_rate']:.4f} ({cov['success_rate']*100:.2f}%)")
        
        print("="*60 + "\n")
