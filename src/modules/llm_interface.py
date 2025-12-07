"""
LLM Interface for SORGA
Supports multiple LLM backends (GPT-4, Llama, etc.) with prompt templates
"""

from typing import List, Dict, Optional, Union
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PromptTemplates:
    """Prompt templates for different reasoning stages"""
    
    # Phase 1: Ontology-guided planning
    ABSTRACT_PATH_RANKING = """Given a question and several type-level reasoning paths, rank them by relevance.

Question: {question}

Type-level paths:
{paths}

Rank these paths from most to least relevant (provide indices):"""
    
    # Phase 2: Super-relation selection
    SUPER_RELATION_SELECTION = """Given a question and groups of similar relations (super-relations), select the most relevant ones.

Question: {question}

Super-relation groups:
{super_relations}

Select top {top_k} most relevant super-relation groups (provide indices):"""
    
    # Phase 3: Path verification
    PATH_VERIFICATION = """Verify if the following reasoning path correctly answers the question.

Question: {question}

Reasoning path: {path}

Is this path valid and relevant? (Yes/No):"""
    
    # Phase 4: Answer synthesis
    ANSWER_SYNTHESIS = """Given a question and multiple reasoning paths with evidence, synthesize the final answer.

Question: {question}

Reasoning paths and evidence:
{evidence}

Based on the evidence, what is the most accurate answer? Provide just the answer:"""
    
    # Relation selection
    RELATION_SELECTION = """Given a question, an entity, and possible relations from that entity, select the most relevant relations.

Question: {question}
Entity: {entity}
Relations: {relations}

Select the top {top_k} most relevant relations:"""


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def generate_with_constraints(self, prompt: str, allowed_tokens_fn,
                                  **kwargs) -> str:
        """Generate with token-level constraints"""
        pass
    
    def reset_stats(self):
        """Reset call statistics"""
        self.call_count = 0
        self.total_tokens = 0
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'model_name': self.model_name
        }


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI models (GPT-4, GPT-4o, GPT-3.5-turbo)"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None,
                temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from openai import OpenAI
            import os
            
            # Get API key from parameter or environment
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {model_name}")
            
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate response using OpenAI API"""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            
            self.call_count += 1
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            logger.error(f"Model: {self.model_name}, Prompt length: {len(prompt)} chars")
            raise
    
    def generate_with_constraints(self, prompt: str, allowed_tokens_fn, **kwargs) -> str:
        """OpenAI doesn't support token-level constraints, fallback to regular generation"""
        logger.debug("OpenAI API doesn't support constrained generation, using regular generation")
        return self.generate(prompt, **kwargs)


class HuggingFaceInterface(LLMInterface):
    """Interface for Hugging Face models (Llama, Mistral, etc.)"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf",
                temperature: float = 0.0, max_tokens: int = 512,
                device: str = "cuda"):
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.torch = torch
            
        except ImportError:
            raise ImportError("transformers or torch not installed. Install with: pip install transformers torch")
    
    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate response using Hugging Face model"""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp if temp > 0 else 1.0,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            self.call_count += 1
            self.total_tokens += len(outputs[0])
            
            return response
        
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return ""
    
    def generate_with_constraints(self, prompt: str, allowed_tokens_fn, **kwargs) -> str:
        """Generate with token-level constraints using prefix_allowed_tokens_fn"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    do_sample=self.temperature > 0,
                    prefix_allowed_tokens_fn=allowed_tokens_fn,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            self.call_count += 1
            self.total_tokens += len(outputs[0])
            
            return response
        
        except Exception as e:
            logger.error(f"Constrained generation error: {e}")
            return self.generate(prompt, **kwargs)


class MockLLMInterface(LLMInterface):
    """Mock LLM for testing without actual API calls"""
    
    def __init__(self, model_name: str = "mock", temperature: float = 0.0, max_tokens: int = 512, **kwargs):
        """Initialize mock LLM (accepts and ignores extra kwargs for compatibility)"""
        super().__init__(model_name, temperature, max_tokens)
    
    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, **kwargs) -> str:
        """Return mock response"""
        self.call_count += 1
        self.total_tokens += 100  # Mock token count
        
        # Simple mock responses based on prompt keywords
        if "rank" in prompt.lower():
            return "1, 2, 3"
        elif "select" in prompt.lower():
            return "0, 1, 2"
        elif "verify" in prompt.lower():
            return "Yes"
        elif "answer" in prompt.lower():
            return "Barack Obama"
        else:
            return "Mock response"
    
    def generate_with_constraints(self, prompt: str, allowed_tokens_fn, **kwargs) -> str:
        """Return mock constrained response"""
        return self.generate(prompt, **kwargs)


def create_llm_interface(model_type: str, **kwargs) -> LLMInterface:
    """
    Factory function to create LLM interface
    
    Args:
        model_type: Type of model ("openai", "huggingface", "mock")
        **kwargs: Additional arguments for the interface
        
    Returns:
        LLM interface instance
    """
    if model_type.lower() == "openai":
        return OpenAIInterface(**kwargs)
    elif model_type.lower() == "huggingface":
        return HuggingFaceInterface(**kwargs)
    elif model_type.lower() == "mock":
        return MockLLMInterface(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class PromptBuilder:
    """Helper class to build prompts for different stages"""
    
    @staticmethod
    def build_abstract_path_ranking(question: str, paths: List[List[str]]) -> str:
        """Build prompt for ranking abstract paths"""
        paths_str = "\n".join([
            f"{i}. {' -> '.join(path)}"
            for i, path in enumerate(paths)
        ])
        return PromptTemplates.ABSTRACT_PATH_RANKING.format(
            question=question,
            paths=paths_str
        )
    
    @staticmethod
    def build_super_relation_selection(question: str, super_relations: List[str],
                                       top_k: int = 5) -> str:
        """Build prompt for selecting super-relations"""
        sr_str = "\n".join([
            f"{i}. {sr}"
            for i, sr in enumerate(super_relations)
        ])
        return PromptTemplates.SUPER_RELATION_SELECTION.format(
            question=question,
            super_relations=sr_str,
            top_k=top_k
        )
    
    @staticmethod
    def build_answer_synthesis(question: str, evidence: Dict, paths: List[Dict]) -> str:
        """Build prompt for answer synthesis"""
        evidence_str = ""
        for i, path in enumerate(paths[:10]):  # Limit to top 10
            evidence_str += f"\nPath {i+1}: {path.get('path_string', '')}\n"
        
        return PromptTemplates.ANSWER_SYNTHESIS.format(
            question=question,
            evidence=evidence_str
        )
    
    @staticmethod
    def build_relation_selection(question: str, entity: str, relations: List[str],
                                 top_k: int = 10) -> str:
        """Build prompt for relation selection"""
        rel_str = ", ".join(relations[:50])  # Limit display
        return PromptTemplates.RELATION_SELECTION.format(
            question=question,
            entity=entity,
            relations=rel_str,
            top_k=top_k
        )
