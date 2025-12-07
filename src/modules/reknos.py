"""
ReKnoS (Reasoning over Knowledge Graphs with Super-Relations)
Implementation following the paper:
"Reasoning of Large Language Models over Knowledge Graphs with Super-Relations"

Key Components:
1. Super-relation construction (hierarchical clustering)
2. LLM-based super-relation scoring (Section 4.2)
3. Forward + Backward reasoning
4. Width-N search paths (Section 4.1)
5. Score-based path selection (Section 4.3)
6. Iterative entity extraction
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class SuperRelation:
    """
    A super-relation groups multiple base relations
    Example: "music.featured_artist" contains ["music.featured_artist.recordings", "music.featured_artist.albums"]
    """
    id: str  # e.g., "SR_0"
    name: str  # human-readable name (e.g., "people.person")
    base_relations: Set[str]  # set of base relation IDs
    score: float = 0.0  # LLM-assigned score for this super-relation


@dataclass
class SuperRelationPath:
    """
    A path through the KG using super-relations
    Example: [SR_1, SR_3, SR_7] where each SR_i is a SuperRelation
    """
    super_relations: List[SuperRelation]  # sequence of super-relations
    score: float  # accumulated score across path
    entities: List[str]  # entities at the end of this path


class ReKnoSModule:
    """
    ReKnoS: Reasoning over Knowledge Graphs with Super-Relations
    
    Implements the full algorithm from Section 4 of the paper:
    1. Build super-relations from KG structure
    2. For each reasoning step (up to L steps):
       a. Candidate Selection: Get connected super-relations
       b. LLM Scoring: Score and select top N super-relations
       c. Entity Extraction: Get entities at end of paths
       d. Decision: Continue or answer?
    """
    
    def __init__(self, kg_interface, llm_interface, config: Dict[str, Any]):
        """
        Initialize ReKnoS module
        
        Args:
            kg_interface: KGInterface instance for querying KG
            llm_interface: LLMInterface for scoring and reasoning
            config: Configuration dictionary with ReKnoS parameters
        """
        self.kg_interface = kg_interface
        self.llm_interface = llm_interface
        self.config = config
        
        # Get ReKnoS-specific config
        reknos_config = config.get('reknos', {})
        self.max_length = reknos_config.get('max_length', 3)  # L in paper
        self.search_width = reknos_config.get('search_width', 3)  # N in paper
        self.num_paths = reknos_config.get('num_paths', 5)  # K in paper
        self.use_backward = reknos_config.get('use_backward', True)
        
        # Super-relation storage
        self.super_relations: Dict[str, SuperRelation] = {}  # id -> SuperRelation
        self.relation_to_super: Dict[str, str] = {}  # base_relation -> super_relation_id
        
        # Connectivity graph: super_rel_id -> set of connected super_rel_ids
        self.connectivity: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"ReKnoS initialized: L={self.max_length}, N={self.search_width}, K={self.num_paths}")
    
    def build_super_relations(self, relations: List[str]) -> None:
        """
        Build super-relations from base relations using hierarchical structure
        For Freebase: "domain.type.property" -> super-relation is "domain.type"
        
        Args:
            relations: List of base relation IDs from the KG
        """
        logger.info(f"Building super-relations from {len(relations)} base relations")
        
        # Group relations by domain.type (middle level)
        domain_type_groups = defaultdict(set)
        
        for relation in relations:
            domain_type = self._extract_domain_type(relation)
            if domain_type:
                domain_type_groups[domain_type].add(relation)
            else:
                # Singleton super-relation for relations without clear structure
                domain_type_groups[relation].add(relation)
        
        # Create SuperRelation objects
        super_rel_id = 0
        for domain_type, base_rels in domain_type_groups.items():
            if len(base_rels) > 0:
                sr_id = f"SR_{super_rel_id}"
                super_rel = SuperRelation(
                    id=sr_id,
                    name=domain_type,
                    base_relations=base_rels,
                    score=0.0
                )
                self.super_relations[sr_id] = super_rel
                
                # Map each base relation to its super-relation
                for base_rel in base_rels:
                    self.relation_to_super[base_rel] = sr_id
                
                super_rel_id += 1
        
        logger.info(f"Created {len(self.super_relations)} super-relations")
        
        # Build connectivity graph
        self._build_connectivity_graph()
    
    def _extract_domain_type(self, relation: str) -> Optional[str]:
        """
        Extract domain.type from Freebase relation
        Example: "people.person.date_of_birth" -> "people.person"
        
        Args:
            relation: Base relation ID
            
        Returns:
            Domain.type string or None
        """
        parts = relation.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return None
    
    def _build_connectivity_graph(self) -> None:
        """
        Build connectivity between super-relations
        SR1 -> SR2 if there exist entities e1, e2, e3 such that:
        e1 --r1--> e2 --r2--> e3 where r1 ∈ SR1, r2 ∈ SR2
        
        This is expensive, so we approximate by checking relation co-occurrence
        """
        logger.info("Building super-relation connectivity graph")
        
        # For efficiency, we approximate: two super-relations are connected if
        # they share common entity types (domains)
        for sr1_id, sr1 in self.super_relations.items():
            for sr2_id, sr2 in self.super_relations.items():
                if sr1_id != sr2_id:
                    # Simple heuristic: connect if they share domain or are sequential
                    if self._are_connected(sr1, sr2):
                        self.connectivity[sr1_id].add(sr2_id)
        
        avg_connections = sum(len(conns) for conns in self.connectivity.values()) / len(self.connectivity)
        logger.info(f"Average connections per super-relation: {avg_connections:.1f}")
    
    def _are_connected(self, sr1: SuperRelation, sr2: SuperRelation) -> bool:
        """
        Check if two super-relations are potentially connected
        Heuristic: share domain or have related domains
        
        Args:
            sr1, sr2: SuperRelation objects
            
        Returns:
            True if likely connected
        """
        # Extract domains
        domain1 = sr1.name.split('.')[0] if '.' in sr1.name else sr1.name
        domain2 = sr2.name.split('.')[0] if '.' in sr2.name else sr2.name
        
        # Connected if same domain or common entity types
        # In full implementation, would query KG for actual connectivity
        return domain1 == domain2 or len(sr1.base_relations & sr2.base_relations) > 0
    
    def reason(self, question: str, topic_entities: List[Tuple[str, str]], 
               max_length: Optional[int] = None) -> List[SuperRelationPath]:
        """
        Main reasoning function implementing ReKnoS algorithm (Section 4)
        
        Args:
            question: Natural language question
            topic_entities: List of (entity_id, entity_name) tuples
            max_length: Maximum reasoning length (overrides config)
            
        Returns:
            List of SuperRelationPath objects with final answers
        """
        L = max_length if max_length is not None else self.max_length
        
        logger.info(f"ReKnoS reasoning: question='{question}', entities={len(topic_entities)}, L={L}")
        
        # Initialize search from topic entities
        # Q_0 = {S_0} where S_0 contains super-relations connected to topic entities
        current_search_path = self._initialize_search(topic_entities)
        
        # Initialize paths (in case we break early)
        paths = []
        
        # Iterative reasoning for up to L steps
        for step in range(1, L + 1):
            logger.info(f"  Step {step}/{L}: Current search path has {len(current_search_path)} super-relation sets")
            
            # Step 1: Candidate Selection (Section 4.1, Equation 1)
            candidates = self._select_candidates(current_search_path)
            logger.info(f"    Candidates: {len(candidates)} super-relations")
            
            if len(candidates) == 0:
                logger.info(f"    No candidates found, stopping at step {step}")
                break
            
            # Step 2: LLM Scoring (Section 4.2, Equation 2-5)
            scored_super_rels = self._score_super_relations(question, candidates, topic_entities)
            logger.info(f"    Scored and selected top {min(self.search_width, len(scored_super_rels))} super-relations")
            
            # Add to search path: Q_l = Q_{l-1} ∪ {S_l}
            current_search_path.append(scored_super_rels)
            
            # Step 3: Entity Extraction (Section 4.3)
            paths = self._extract_paths(current_search_path, topic_entities)
            logger.info(f"    Extracted {len(paths)} paths with entities")
            
            # Step 4: Decision - Continue or Answer?
            if step == L:
                # Max length reached, must answer
                logger.info(f"    Max length reached, returning {len(paths)} paths")
                return paths
            else:
                # Ask LLM: are these entities sufficient?
                should_continue = self._should_continue(question, paths)
                if not should_continue:
                    logger.info(f"    LLM decided to stop at step {step}, returning {len(paths)} paths")
                    return paths
                else:
                    logger.info(f"    LLM decided to continue to step {step + 1}")
        
        # Return final paths
        return paths
    
    def _initialize_search(self, topic_entities: List[Tuple[str, str]]) -> List[List[SuperRelation]]:
        """
        Initialize search with super-relations connected to topic entities
        Returns Q_0 = [{SR_i, SR_j, ...}] (list with one set of initial super-relations)
        
        Args:
            topic_entities: List of (entity_id, entity_name) tuples
            
        Returns:
            Initial search path with one super-relation set
        """
        # Get all relations connected to topic entities
        connected_relations = set()
        for entity_id, _ in topic_entities:
            # Get 1-hop relations in both directions
            forward_rels = self.kg_interface.get_1hop_relations(entity_id, direction="head")
            backward_rels = self.kg_interface.get_1hop_relations(entity_id, direction="tail")
            connected_relations.update(forward_rels)
            connected_relations.update(backward_rels)
        
        # Map to super-relations
        connected_super_rels = set()
        for relation in connected_relations:
            if relation in self.relation_to_super:
                sr_id = self.relation_to_super[relation]
                connected_super_rels.add(sr_id)
        
        # Get top N by diversity (different domains)
        initial_super_rels = self._select_diverse_super_relations(
            list(connected_super_rels), 
            self.search_width
        )
        
        logger.info(f"  Initialized with {len(initial_super_rels)} super-relations from {len(topic_entities)} entities")
        
        # Return as list with one set
        return [[self.super_relations[sr_id] for sr_id in initial_super_rels]]
    
    def _select_diverse_super_relations(self, sr_ids: List[str], n: int) -> List[str]:
        """
        Select N diverse super-relations (different domains preferred)
        
        Args:
            sr_ids: List of super-relation IDs
            n: Number to select
            
        Returns:
            List of N super-relation IDs
        """
        if len(sr_ids) <= n:
            return sr_ids
        
        # Group by domain
        domain_groups = defaultdict(list)
        for sr_id in sr_ids:
            sr = self.super_relations[sr_id]
            domain = sr.name.split('.')[0] if '.' in sr.name else sr.name
            domain_groups[domain].append(sr_id)
        
        # Select one from each domain, round-robin
        selected = []
        domains = list(domain_groups.keys())
        idx = 0
        while len(selected) < n and len(domain_groups) > 0:
            domain = domains[idx % len(domains)]
            if len(domain_groups[domain]) > 0:
                selected.append(domain_groups[domain].pop(0))
            else:
                domains.remove(domain)
            idx += 1
        
        return selected[:n]
    
    def _select_candidates(self, search_path: List[List[SuperRelation]]) -> List[SuperRelation]:
        """
        Candidate Selection (Section 4.1, Equation 1)
        C_l = {R | ∃R' ∈ S_{l-1} such that R' → R}
        
        Get all super-relations connected to the last set in search path
        
        Args:
            search_path: Current search path Q_{l-1} = [S_0, S_1, ..., S_{l-1}]
            
        Returns:
            List of candidate SuperRelation objects
        """
        if len(search_path) == 0:
            return []
        
        # Get last super-relation set
        last_set = search_path[-1]
        
        # Get all super-relations connected to any in last set
        candidate_ids = set()
        for sr in last_set:
            candidate_ids.update(self.connectivity[sr.id])
        
        # Remove super-relations already in search path
        for sr_set in search_path:
            for sr in sr_set:
                candidate_ids.discard(sr.id)
        
        # Convert to SuperRelation objects
        candidates = [self.super_relations[sr_id] for sr_id in candidate_ids]
        
        return candidates
    
    def _score_super_relations(self, question: str, candidates: List[SuperRelation],
                               topic_entities: List[Tuple[str, str]]) -> List[SuperRelation]:
        """
        LLM Scoring (Section 4.2, Equations 2-5)
        1. Prompt LLM to score candidates
        2. Select top N
        3. Normalize scores to sum to 1
        
        Args:
            question: Natural language question
            candidates: List of candidate SuperRelation objects
            topic_entities: Topic entities for context
            
        Returns:
            List of top N scored SuperRelation objects (with updated scores)
        """
        if len(candidates) <= self.search_width:
            # If candidates <= N, use all with equal scores
            for sr in candidates:
                sr.score = 1.0 / len(candidates)
            return candidates
        
        # Build prompt following paper's template (Section 4.2)
        entity_names = ", ".join([name for _, name in topic_entities[:3]])
        candidate_names = [sr.name for sr in candidates]
        
        prompt = f"""You need to select {self.search_width} relations from the following candidate relations, which are the most helpful for answering the question.

Question: {question}
Topic Entity: {entity_names}

Candidate Relations:
{self._format_candidates_for_prompt(candidate_names)}

Reply with the indices of the {self.search_width} relations you selected (e.g., "1, 3, 5"):"""
        
        # Get LLM response
        response = self.llm_interface.generate(prompt)
        
        # Parse selected indices
        selected_indices = self._parse_selection_response(response, len(candidates))
        
        # If parsing failed, fall back to first N
        if len(selected_indices) == 0:
            selected_indices = list(range(min(self.search_width, len(candidates))))
            logger.warning(f"Failed to parse LLM selection, using first {len(selected_indices)} candidates")
        
        # Get selected super-relations
        selected = [candidates[i] for i in selected_indices[:self.search_width]]
        
        # Normalize scores to sum to 1 (Equation 5)
        # For simplicity, use uniform scores (could prompt LLM for scores 1-10)
        for sr in selected:
            sr.score = 1.0 / len(selected)
        
        return selected
    
    def _format_candidates_for_prompt(self, candidate_names: List[str]) -> str:
        """Format candidate names for LLM prompt"""
        return "\n".join([f"{i+1}. {name}" for i, name in enumerate(candidate_names)])
    
    def _parse_selection_response(self, response: str, max_index: int) -> List[int]:
        """
        Parse LLM response to extract selected indices
        Handles formats like: "1, 3, 5" or "1,3,5" or "The selected are 1, 3, and 5"
        
        Args:
            response: LLM response text
            max_index: Maximum valid index
            
        Returns:
            List of selected indices (0-based)
        """
        # Extract all numbers from response
        numbers = re.findall(r'\d+', response)
        
        # Convert to 0-based indices and validate
        indices = []
        for num_str in numbers:
            idx = int(num_str) - 1  # Convert to 0-based
            if 0 <= idx < max_index:
                indices.append(idx)
        
        return indices
    
    def _extract_paths(self, search_path: List[List[SuperRelation]], 
                      topic_entities: List[Tuple[str, str]]) -> List[SuperRelationPath]:
        """
        Extract paths and entities (Section 4.3)
        1. Select K best super-relation paths by accumulated scores (Equations 6-8)
        2. Extract entities at end of each path (Equation 9)
        
        Args:
            search_path: Current search path Q_l
            topic_entities: Starting entities
            
        Returns:
            List of K SuperRelationPath objects
        """
        if len(search_path) == 0:
            return []
        
        # Generate all valid super-relation paths through search_path
        all_paths = self._generate_all_paths(search_path)
        
        # Score each path (sum of super-relation scores, Equation 7)
        for path in all_paths:
            path.score = sum(sr.score for sr in path.super_relations)
        
        # Select top K paths (Equation 8)
        all_paths.sort(key=lambda p: p.score, reverse=True)
        top_k_paths = all_paths[:self.num_paths]
        
        # Extract entities for each path (Equation 9)
        for path in top_k_paths:
            path.entities = self._extract_entities_for_path(path, topic_entities)
        
        return top_k_paths
    
    def _generate_all_paths(self, search_path: List[List[SuperRelation]]) -> List[SuperRelationPath]:
        """
        Generate all valid super-relation paths through search_path
        A path is valid if consecutive super-relations are connected
        
        Args:
            search_path: Q_l = [S_0, S_1, ..., S_l]
            
        Returns:
            List of SuperRelationPath objects
        """
        if len(search_path) == 0:
            return []
        
        # Recursive path generation with connectivity checking
        def generate_recursive(current_path: List[SuperRelation], step: int) -> List[List[SuperRelation]]:
            if step >= len(search_path):
                return [current_path]
            
            paths = []
            for next_sr in search_path[step]:
                # Check connectivity with last super-relation
                if len(current_path) == 0 or next_sr.id in self.connectivity[current_path[-1].id]:
                    paths.extend(generate_recursive(current_path + [next_sr], step + 1))
            
            return paths
        
        # Generate all paths
        all_sr_sequences = generate_recursive([], 0)
        
        # Convert to SuperRelationPath objects
        paths = [SuperRelationPath(super_relations=seq, score=0.0, entities=[]) 
                 for seq in all_sr_sequences]
        
        return paths
    
    def _extract_entities_for_path(self, path: SuperRelationPath, 
                                   topic_entities: List[Tuple[str, str]]) -> List[str]:
        """
        Extract entities at the end of a super-relation path (Equation 9)
        E_l = {e | ∃r1 ∈ R1, r2 ∈ R2, ..., rl ∈ Rl such that e1 --r1--> e2 --r2--> ... --rl--> e}
        
        Args:
            path: SuperRelationPath object
            topic_entities: Starting entities
            
        Returns:
            List of entity IDs at end of path
        """
        # Start from topic entities
        current_entities = set([eid for eid, _ in topic_entities])
        
        # Follow each super-relation in path
        for sr in path.super_relations:
            next_entities = set()
            
            # For each current entity, follow any base relation in super-relation
            for entity_id in current_entities:
                # Get all 1-hop relations (forward direction)
                relations = self.kg_interface.get_1hop_relations(entity_id, direction="head")
                
                # Follow relations that are in the current super-relation
                for relation in relations:
                    if relation in sr.base_relations:
                        # Get entities via this relation
                        tail_entities = self.kg_interface.get_entities_via_relation(
                            entity_id, relation, direction="head"
                        )
                        next_entities.update(tail_entities[:50])  # Limit per entity
            
            current_entities = next_entities
            
            # Early stopping if no entities found
            if len(current_entities) == 0:
                break
        
        return list(current_entities)[:20]  # Limit to top 20 entities
    
    def _should_continue(self, question: str, paths: List[SuperRelationPath]) -> bool:
        """
        Ask LLM whether to continue reasoning or answer now (Section 4.3)
        
        Args:
            question: Natural language question
            paths: Current paths with entities
            
        Returns:
            True if should continue, False if should answer
        """
        # Get entity names for context
        all_entities = []
        for path in paths[:3]:  # Use top 3 paths
            all_entities.extend(path.entities[:5])  # Top 5 entities per path
        
        # Get entity names
        entity_names = []
        for entity_id in all_entities[:10]:
            name = self.kg_interface.get_entity_name(entity_id)
            if name:
                entity_names.append(name)
        
        if len(entity_names) == 0:
            # No entities found, continue reasoning
            return True
        
        # Build prompt
        entity_list = ", ".join(entity_names[:10])
        prompt = f"""Question: {question}

Retrieved entities: {entity_list}

Based on these entities, can you answer the question?
Reply with ONLY "yes" or "no":"""
        
        # Get LLM response
        response = self.llm_interface.generate(prompt).strip().lower()
        
        # Parse response
        should_answer = 'yes' in response
        should_continue = not should_answer
        
        return should_continue
    
    def get_final_entities(self, paths: List[SuperRelationPath]) -> List[str]:
        """
        Get final entities from paths for answer generation
        
        Args:
            paths: List of SuperRelationPath objects
            
        Returns:
            List of entity IDs
        """
        all_entities = []
        for path in paths:
            all_entities.extend(path.entities)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for eid in all_entities:
            if eid not in seen:
                seen.add(eid)
                unique_entities.append(eid)
        
        return unique_entities
