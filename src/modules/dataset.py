"""
Dataset loaders for WebQSP and CWQ
Handles data loading, preprocessing, and batching
"""

from typing import List, Dict, Optional, Tuple
import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class KGQADataset:
    """Base class for KGQA datasets"""
    
    def __init__(self, data_path: str, split: str = "train"):
        """
        Initialize dataset
        
        Args:
            data_path: Path to dataset file
            split: Dataset split ("train", "dev", "test")
        """
        self.data_path = Path(data_path)
        self.split = split
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load dataset from file"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)


class WebQSPDataset(KGQADataset):
    """WebQSP (Web Questions on Freebase) dataset"""
    
    def load_data(self):
        """Load WebQSP data"""
        logger.info(f"Loading WebQSP {self.split} data from {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # WebQSP format
            if 'Questions' in raw_data:
                questions = raw_data['Questions']
            else:
                questions = raw_data
            
            for item in questions:
                # Extract fields
                question_id = item.get('QuestionId', item.get('qid', ''))
                question_text = item.get('ProcessedQuestion', item.get('question', ''))
                
                # Parse topic entities
                parses = item.get('Parses', [])
                topic_entities = []
                if parses:
                    parse = parses[0]
                    if 'TopicEntityMid' in parse:
                        topic_entities = [parse['TopicEntityMid']]
                    elif 'InferentialChain' in parse:
                        # Extract from inferential chain
                        chain = parse['InferentialChain']
                        if chain and len(chain) > 0:
                            topic_entities = [chain[0].split('.')[0]]
                
                # Extract answers
                answers = []
                if parses:
                    parse = parses[0]
                    if 'Answers' in parse:
                        answers = [ans.get('AnswerArgument', ans.get('answer', '')) 
                                  for ans in parse['Answers']]
                
                # Create data entry
                self.data.append({
                    'id': question_id,
                    'question': question_text,
                    'topic_entities': topic_entities,
                    'answers': answers,
                    'split': self.split,
                    'dataset': 'WebQSP'
                })
            
            logger.info(f"Loaded {len(self.data)} questions from WebQSP {self.split}")
        
        except Exception as e:
            logger.error(f"Error loading WebQSP data: {e}")
            self.data = []


class CWQDataset(KGQADataset):
    """ComplexWebQuestions dataset"""
    
    def load_data(self):
        """Load CWQ data"""
        logger.info(f"Loading CWQ {self.split} data from {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for item in raw_data:
                # Extract fields
                question_id = item.get('ID', '')
                question_text = item.get('question', '')
                
                # Parse topic entities
                composition_answer = item.get('composition_answer', '')
                topic_entities = []
                if composition_answer:
                    # Extract mid from composition_answer
                    # Format: "m.0xyz" or list of mids
                    if isinstance(composition_answer, str):
                        topic_entities = [composition_answer]
                    elif isinstance(composition_answer, list):
                        topic_entities = composition_answer
                
                # Also check for explicit topic entity field
                if 'topic_entity' in item:
                    topic_entities = [item['topic_entity']]
                
                # Extract answers
                answers = item.get('answer', [])
                if isinstance(answers, dict):
                    answers = [answers.get('answer_argument', '')]
                elif not isinstance(answers, list):
                    answers = [str(answers)]
                
                # Create data entry
                self.data.append({
                    'id': question_id,
                    'question': question_text,
                    'topic_entities': topic_entities,
                    'answers': answers,
                    'complexity': item.get('complexity', 'unknown'),
                    'compositionality_type': item.get('compositionality_type', 'unknown'),
                    'split': self.split,
                    'dataset': 'CWQ'
                })
            
            logger.info(f"Loaded {len(self.data)} questions from CWQ {self.split}")
        
        except Exception as e:
            logger.error(f"Error loading CWQ data: {e}")
            self.data = []


class GrailQADataset(KGQADataset):
    """GrailQA dataset"""
    
    def load_data(self):
        """Load GrailQA data"""
        logger.info(f"Loading GrailQA {self.split} data from {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for item in raw_data:
                # Extract fields
                question_id = item.get('qid', '')
                question_text = item.get('question', '')
                
                # Parse topic entities
                topic_entities = []
                if 'graph_query' in item and 'nodes' in item['graph_query']:
                    nodes = item['graph_query']['nodes']
                    for node in nodes:
                        if node.get('node_type') == 'entity':
                            topic_entities.append(node.get('id', ''))
                
                # Extract answers
                answers = item.get('answer', [])
                if isinstance(answers, dict):
                    answers = [answers.get('answer_argument', '')]
                elif not isinstance(answers, list):
                    answers = [str(answers)]
                
                # Create data entry
                self.data.append({
                    'id': question_id,
                    'question': question_text,
                    'topic_entities': topic_entities,
                    'answers': answers,
                    'level': item.get('level', 'unknown'),
                    'split': self.split,
                    'dataset': 'GrailQA'
                })
            
            logger.info(f"Loaded {len(self.data)} questions from GrailQA {self.split}")
        
        except Exception as e:
            logger.error(f"Error loading GrailQA data: {e}")
            self.data = []


def load_dataset(dataset_name: str, data_path: str, split: str = "train") -> KGQADataset:
    """
    Factory function to load dataset
    
    Args:
        dataset_name: Name of dataset ("webqsp", "cwq", "grailqa")
        data_path: Path to dataset file
        split: Dataset split
        
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "webqsp":
        return WebQSPDataset(data_path, split)
    elif dataset_name == "cwq":
        return CWQDataset(data_path, split)
    elif dataset_name == "grailqa":
        return GrailQADataset(data_path, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class DatasetStatistics:
    """Calculate and display dataset statistics"""
    
    @staticmethod
    def compute_stats(dataset: KGQADataset) -> Dict:
        """Compute dataset statistics"""
        stats = {
            'total_questions': len(dataset),
            'avg_question_length': 0,
            'questions_with_entities': 0,
            'avg_entities_per_question': 0,
            'questions_with_answers': 0,
            'avg_answers_per_question': 0,
        }
        
        question_lengths = []
        entity_counts = []
        answer_counts = []
        
        for item in dataset:
            # Question length
            question_lengths.append(len(item['question'].split()))
            
            # Entity count
            entity_count = len(item.get('topic_entities', []))
            entity_counts.append(entity_count)
            if entity_count > 0:
                stats['questions_with_entities'] += 1
            
            # Answer count
            answer_count = len(item.get('answers', []))
            answer_counts.append(answer_count)
            if answer_count > 0:
                stats['questions_with_answers'] += 1
        
        # Compute averages
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if entity_counts:
            stats['avg_entities_per_question'] = sum(entity_counts) / len(entity_counts)
        if answer_counts:
            stats['avg_answers_per_question'] = sum(answer_counts) / len(answer_counts)
        
        return stats
    
    @staticmethod
    def print_stats(dataset: KGQADataset):
        """Print dataset statistics"""
        stats = DatasetStatistics.compute_stats(dataset)
        
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.data[0]['dataset'] if dataset.data else 'Unknown'} ({dataset.split})")
        print(f"{'='*50}")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Average question length: {stats['avg_question_length']:.2f} words")
        print(f"Questions with entities: {stats['questions_with_entities']} ({stats['questions_with_entities']/stats['total_questions']*100:.1f}%)")
        print(f"Average entities per question: {stats['avg_entities_per_question']:.2f}")
        print(f"Questions with answers: {stats['questions_with_answers']} ({stats['questions_with_answers']/stats['total_questions']*100:.1f}%)")
        print(f"Average answers per question: {stats['avg_answers_per_question']:.2f}")
        print(f"{'='*50}\n")
