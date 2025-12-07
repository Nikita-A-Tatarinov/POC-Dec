"""
Trie data structure for graph-constrained decoding
Adapted from GCR (Facebook Research) for SORGA framework
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Trie:
    """
    Prefix tree (Trie) data structure for storing valid token sequences
    Used for graph-constrained decoding to ensure generated paths exist in KG
    """
    
    def __init__(self, sequences: List[List[int]] = None):
        """
        Initialize Trie
        
        Args:
            sequences: List of token sequences to add to trie
        """
        self.trie_dict = {}
        self.len = 0
        
        if sequences:
            for sequence in sequences:
                self.add(sequence)
        
        self.append_trie = None
        self.bos_token_id = None
    
    def append(self, trie: 'Trie', bos_token_id: int):
        """
        Append another trie to this one
        
        Args:
            trie: Trie to append
            bos_token_id: Beginning of sequence token ID
        """
        self.append_trie = trie
        self.bos_token_id = bos_token_id
    
    def add(self, sequence: List[int]):
        """
        Add a sequence to the trie
        
        Args:
            sequence: List of token IDs
        """
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1
    
    def get(self, prefix_sequence: List[int]) -> List[int]:
        """
        Get all valid next tokens given a prefix sequence
        
        Args:
            prefix_sequence: Prefix token sequence
            
        Returns:
            List of valid next token IDs
        """
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )
    
    @staticmethod
    def load_from_dict(trie_dict: Dict) -> 'Trie':
        """
        Load trie from dictionary
        
        Args:
            trie_dict: Dictionary representation of trie
            
        Returns:
            Trie instance
        """
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie
    
    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        """
        Recursively add sequence to trie dictionary
        
        Args:
            sequence: Token sequence
            trie_dict: Trie dictionary to add to
        """
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])
    
    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie: Optional['Trie'] = None,
        bos_token_id: Optional[int] = None,
    ) -> List[int]:
        """
        Recursively get valid next tokens from trie
        
        Args:
            prefix_sequence: Prefix sequence
            trie_dict: Trie dictionary
            append_trie: Optional appended trie
            bos_token_id: Beginning of sequence token
            
        Returns:
            List of valid next token IDs
        """
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []
    
    def __iter__(self):
        """Iterate over all sequences in trie"""
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence
        
        return _traverse([], self.trie_dict)
    
    def __len__(self):
        """Get number of sequences in trie"""
        return self.len
    
    def __getitem__(self, value):
        """Get valid next tokens for prefix"""
        return self.get(value)


class GraphConstrainedDecoding:
    """
    Graph-constrained decoding for LLM generation
    Ensures generated tokens follow valid paths in the knowledge graph
    """
    
    def __init__(self, tokenizer, trie: Trie, 
                 start_token_id: Optional[int] = None,
                 end_token_id: Optional[int] = None,
                 enable_constrained_by_default: bool = False):
        """
        Initialize graph-constrained decoder
        
        Args:
            tokenizer: Tokenizer for LLM
            trie: Trie containing valid token sequences
            start_token_id: Token marking start of constrained generation
            end_token_id: Token marking end of constrained generation
            enable_constrained_by_default: Whether to enable constraints from start
        """
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token = start_token_id
        self.end_token = end_token_id
        self.all_tokens = list(range(len(tokenizer)))
        self.constrained_flag = enable_constrained_by_default
        self.L_input = None
        self.last_start_token = None
    
    def check_constrained_flag(self, sent: List[int]) -> tuple:
        """
        Check if we should enable constrained decoding for current sequence
        
        Args:
            sent: Current token sequence
            
        Returns:
            Tuple of (should_constrain, input_length)
        """
        # Convert to list if tensor
        if hasattr(sent, 'tolist'):
            sent = sent.tolist()
        
        # Check for start token
        if self.start_token not in sent:
            return False, len(sent)
        
        # Find last start token position
        last_start_idx = len(sent) - 1 - sent[::-1].index(self.start_token)
        
        # Count end tokens after last start token
        end_token_count = sent[last_start_idx:].count(self.end_token)
        
        # If no end token found, we're in constrained region
        if end_token_count == 0:
            self.last_start_token = last_start_idx
            return True, last_start_idx
        else:
            self.last_start_token = None
            return False, len(sent)
    
    def allowed_tokens_fn(self, batch_id: int, sent) -> List[int]:
        """
        Get list of allowed tokens for next generation step
        This is called by the LLM's constrained generation
        
        Args:
            batch_id: Batch index
            sent: Current sequence (can be tensor or list)
            
        Returns:
            List of allowed token IDs
        """
        # Convert tensor to list if needed
        if hasattr(sent, 'tolist'):
            sent_list = sent.tolist()
        else:
            sent_list = sent
        
        constrained_flag = self.constrained_flag
        
        # Check if we should enter constrained decoding
        if self.start_token is not None and self.end_token is not None:
            constrained_flag, L_input = self.check_constrained_flag(sent_list)
        else:
            # Use full sequence for constraints
            if self.L_input is None:
                self.L_input = len(sent_list)
            L_input = self.L_input
        
        # Get allowed tokens
        allow_tokens = self.all_tokens
        
        if constrained_flag:
            # Get valid next tokens from trie based on generated suffix
            suffix = sent_list[L_input:]
            allow_tokens = self.trie.get(suffix)
            
            # If no valid tokens, allow all (fallback to prevent getting stuck)
            if len(allow_tokens) == 0:
                logger.warning(f"No valid tokens found for suffix {suffix}, allowing all tokens")
                return self.all_tokens
        
        return allow_tokens
    
    def update_trie(self, new_sequences: List[List[int]]):
        """
        Update trie with new valid sequences
        Allows dynamic expansion during reasoning
        
        Args:
            new_sequences: New sequences to add to trie
        """
        for sequence in new_sequences:
            self.trie.add(sequence)
        
        logger.info(f"Added {len(new_sequences)} sequences to trie. Total: {len(self.trie)}")


def build_trie_from_paths(paths: List[str], tokenizer) -> Trie:
    """
    Build trie from list of reasoning path strings
    
    Args:
        paths: List of path strings (e.g., "entity1 -> relation -> entity2")
        tokenizer: Tokenizer to encode paths
        
    Returns:
        Trie containing tokenized paths
    """
    sequences = []
    
    for path in paths:
        # Tokenize path
        tokens = tokenizer.encode(path, add_special_tokens=False)
        sequences.append(tokens)
    
    trie = Trie(sequences)
    logger.info(f"Built trie with {len(trie)} paths")
    
    return trie


def build_trie_from_kg_paths(kg_interface, entities: List[str], 
                              max_hops: int = 3,
                              tokenizer=None) -> Trie:
    """
    Build trie from actual KG paths
    
    Args:
        kg_interface: KGInterface instance
        entities: List of starting entities
        max_hops: Maximum path length
        tokenizer: Tokenizer for encoding
        
    Returns:
        Trie containing valid KG paths
    """
    path_strings = []
    
    for entity in entities[:100]:  # Limit for efficiency
        # Get 1-hop paths
        if max_hops >= 1:
            relations_1hop = kg_interface.get_1hop_relations(entity)
            for rel in relations_1hop[:50]:  # Limit relations
                path = f"{entity} -> {rel}"
                path_strings.append(path)
        
        # Get 2-hop paths
        if max_hops >= 2:
            relations_2hop = kg_interface.get_2hop_relations(entity)
            for rel1, rel2 in relations_2hop[:50]:
                path = f"{entity} -> {rel1} -> {rel2}"
                path_strings.append(path)
        
        # Get 3-hop paths
        if max_hops >= 3:
            relations_3hop = kg_interface.get_3hop_relations(entity)
            for rel1, rel2, rel3 in relations_3hop[:20]:
                path = f"{entity} -> {rel1} -> {rel2} -> {rel3}"
                path_strings.append(path)
    
    logger.info(f"Extracted {len(path_strings)} KG paths")
    
    if tokenizer:
        return build_trie_from_paths(path_strings, tokenizer)
    else:
        # Return trie with string-based encoding (simplified)
        logger.warning("No tokenizer provided, using character-level encoding")
        sequences = [[ord(c) for c in path] for path in path_strings]
        return Trie(sequences)
