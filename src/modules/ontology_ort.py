"""
ORT (Ontology-Guided Reverse Thinking) Implementation
Following the ACL 2025 paper as closely as possible

Paper: "Ontology-Guided Reverse Thinking Makes Large Language Models 
        Stronger on Knowledge Graph Question Answering"

Key differences from our previous implementation:
1. Implements BACKWARD search from aims (target) to conditions (source)
2. Includes LLM aim & condition recognition with types
3. Implements 3-stage pruning: conditions, cycles, semantics
4. Uses label reasoning paths to guide KG queries
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity with its label (type)"""
    entity_id: str
    entity_name: str
    labels: List[str]  # Entity types


@dataclass
class LabelReasoningPath:
    """Abstract path at label (type) level"""
    labels: List[str]  # Sequence of type labels
    score: float = 1.0


class OntologyModuleORT:
    """
    Implements ORT algorithm from ACL 2025 paper
    
    Main steps:
    1. Aim and Condition Recognition (with LLM)
    2. Ontology-Guided Reverse Thinking Reasoning
       - Build neighbor label dictionary
       - Construct reverse reasoning tree
       - Prune by conditions
       - Prune cycles
       - Prune by semantics (with LLM)
    3. Guided Answer Mining (use paths to query KG)
    """
    
    def __init__(self, kg_interface, llm_interface, config: Dict):
        """
        Initialize ORT module
        
        Args:
            kg_interface: Interface to knowledge graph
            llm_interface: Interface to LLM for aim recognition and filtering
            config: Configuration dict
        """
        self.kg_interface = kg_interface
        self.llm = llm_interface
        self.config = config
        
        # Ontology structures
        self.label_list: List[str] = []  # All possible labels (types) in KG
        self.neighbor_label_dict: Dict[str, Set[str]] = defaultdict(set)  # Label → neighbor labels
        self.label_descriptions: Dict[str, str] = {}  # Label → description
        
        # Build ontology structures
        self._build_ontology()
    
    def _build_ontology(self):
        """
        Build ontology structures from KG
        
        According to paper Section 2.2.1:
        - Extract all labels (types) from KG
        - Build neighbor label dictionary: for each label, find all labels
          that appear in the same relation-defined triple
        """
        logger.info("Building ontology structures for ORT")
        
        # Get all types from KG
        self.label_list = self.kg_interface.get_all_types()
        logger.info(f"Extracted {len(self.label_list)} labels from KG")
        
        # Build label descriptions (for LLM prompts)
        for label in self.label_list:
            # Extract readable name from Freebase ID
            # e.g., "people.person" → "Person"
            self.label_descriptions[label] = self._label_to_description(label)
        
        # Build neighbor label dictionary
        # For each label, find all labels connected via relations in ontology
        relation_types = self.kg_interface.get_all_relation_types()
        
        for relation, domain_type, range_type in relation_types:
            # If relation connects domain_type to range_type in ontology
            # Then they are neighbors
            if domain_type and range_type:
                self.neighbor_label_dict[domain_type].add(range_type)
                self.neighbor_label_dict[range_type].add(domain_type)  # Bidirectional
        
        logger.info(f"Built neighbor dictionary with {len(self.neighbor_label_dict)} labels")
    
    def _label_to_description(self, label: str) -> str:
        """
        Convert Freebase label to human-readable description
        
        Args:
            label: Freebase type ID (e.g., "people.person")
            
        Returns:
            Human-readable description (e.g., "Person")
        """
        # Simple heuristic: take last part and capitalize
        parts = label.split('.')
        if len(parts) >= 2:
            return parts[-1].replace('_', ' ').title()
        return label.replace('_', ' ').title()
    
    def recognize_aims_and_conditions(self, question: str, 
                                      topic_entities: List[Entity]) -> Tuple[List[Entity], List[Entity]]:
        """
        Step 1: Aim and Condition Recognition
        
        According to paper Section 2.1:
        - Condition entities: known information in the question
        - Aim entities: content user wants to query
        - Use LLM to extract both with their labels (types)
        
        Args:
            question: Natural language question
            topic_entities: Entities mentioned in question
            
        Returns:
            (condition_entities, aim_entities) - both with labels
        """
        logger.info("Step 1: Recognizing aims and conditions")
        
        # Build prompt for LLM (based on paper Figure 3)
        prompt = self._build_aim_condition_prompt(question, topic_entities)
        
        # Query LLM
        response = self.llm.generate(prompt, max_tokens=500, temperature=0)
        
        # Parse response to extract condition and aim entities with labels
        condition_entities, aim_entities = self._parse_aim_condition_response(
            response, topic_entities, question
        )
        
        logger.info(f"Recognized {len(condition_entities)} conditions, {len(aim_entities)} aims")
        return condition_entities, aim_entities
    
    def _build_aim_condition_prompt(self, question: str, 
                                   topic_entities: List[Entity]) -> str:
        """
        Build prompt for aim/condition recognition (paper Figure 3)
        
        Args:
            question: User question
            topic_entities: Known entities
            
        Returns:
            Prompt string
        """
        # Build label description table
        label_table = "\n".join([
            f"- {label} ({self.label_descriptions[label]})"
            for label in self.label_list[:100]  # Limit to top 100 to fit context
        ])
        
        prompt = f"""Your task is to extract conditional entities and their types and target entities and their types from the user's input question.

The user's question is: {question}

Please choose the entity types from the following table:
{label_table}

Each row describes an entity type in the format:
- Entity Type (Description)

Extracting Rules:
- Conditional entities are the known information provided in the question.
- Target entities are the content the user wants to query in the question.

Known entities in the question:
{self._format_entities(topic_entities)}

Please extract:
1. Condition entities (known) with their types
2. Aim entities (targets to find) with their types

Format your response as JSON:
{{
    "conditions": [
        {{"entity": "entity_name", "types": ["type1", "type2"]}}
    ],
    "aims": [
        {{"entity": "aim_description", "types": ["type1", "type2"]}}
    ]
}}
"""
        return prompt
    
    def _format_entities(self, entities: List[Entity]) -> str:
        """Format entities for prompt"""
        return "\n".join([
            f"- {e.entity_name} (ID: {e.entity_id}, Types: {', '.join(e.labels)})"
            for e in entities
        ])
    
    def _parse_aim_condition_response(self, response: str, 
                                     topic_entities: List[Entity],
                                     question: str = "") -> Tuple[List[Entity], List[Entity]]:
        """
        Parse LLM response to extract condition and aim entities
        
        Args:
            response: LLM JSON response
            topic_entities: Original entities
            question: Original question (for type inference fallback)
            
        Returns:
            (condition_entities, aim_entities)
        """
        import json
        import re
        
        try:
            # Strip code blocks if present (```json ... ```)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(1).strip()
            
            data = json.loads(cleaned_response)
            
            # Extract conditions
            condition_entities = []
            for cond in data.get("conditions", []):
                # Match with topic entities
                entity = self._match_entity(cond["entity"], topic_entities)
                if entity:
                    # Priority 1: Try KG types (most reliable)
                    kg_types = self.kg_interface.get_entity_types(entity.entity_id)
                    
                    if kg_types:
                        # KG types are most reliable, use them
                        entity.labels = kg_types
                        logger.debug(f"Using {len(kg_types)} types from KG for {entity.entity_name}: {kg_types[:3]}")
                    else:
                        # Priority 2: Try LLM-provided types
                        llm_types = cond.get("types", [])
                        if llm_types:
                            # Validate LLM types exist in ontology
                            valid_types = [t for t in llm_types if t in self.label_list]
                            if valid_types:
                                entity.labels = valid_types
                                logger.debug(f"Using {len(valid_types)} valid LLM types for {entity.entity_name}: {valid_types}")
                            else:
                                # LLM types not in ontology, use inference
                                entity.labels = self._infer_types_from_entity_name(entity.entity_name)
                                logger.debug(f"LLM types not in ontology, inferred for {entity.entity_name}: {entity.labels}")
                        else:
                            # Priority 3: Infer from entity name
                            entity.labels = entity.labels if entity.labels else self._infer_types_from_entity_name(entity.entity_name)
                            logger.debug(f"Inferred types from name for {entity.entity_name}: {entity.labels}")
                    
                    condition_entities.append(entity)
            
            # Extract aims
            aim_entities = []
            for aim in data.get("aims", []):
                # Get LLM-provided types
                llm_types = aim.get("types", [])
                
                # Validate LLM types exist in ontology
                if llm_types:
                    valid_types = [t for t in llm_types if t in self.label_list]
                    if valid_types:
                        llm_types = valid_types
                        logger.debug(f"Using {len(valid_types)} valid LLM types for aim: {valid_types}")
                    else:
                        # LLM types not in ontology, infer from question
                        llm_types = self._infer_aim_types_from_question(question)
                        logger.debug(f"LLM types not in ontology, inferred from question: {llm_types}")
                else:
                    # If LLM didn't provide types, infer from question
                    llm_types = self._infer_aim_types_from_question(question)
                    logger.debug(f"Inferred {len(llm_types)} aim types from question: {llm_types}")
                
                # Aims may not be in topic_entities (they're what we're looking for)
                aim_entity = Entity(
                    entity_id=f"aim_{len(aim_entities)}",
                    entity_name=aim["entity"],
                    labels=llm_types
                )
                aim_entities.append(aim_entity)
            
            return condition_entities, aim_entities
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {response}")
            # Fallback: treat all topic entities as conditions
            return topic_entities, []
    
    def _match_entity(self, entity_text: str, entities: List[Entity]) -> Optional[Entity]:
        """Match entity text with Entity object"""
        entity_text_lower = entity_text.lower()
        for entity in entities:
            if entity_text_lower in entity.entity_name.lower():
                return entity
        return None
    
    def construct_reverse_reasoning_tree(self, 
                                        condition_labels: List[str],
                                        aim_labels: List[str],
                                        max_hop: int = 3) -> Dict:
        """
        Step 2: Construct Reverse Reasoning Tree
        
        According to paper Section 2.2.2:
        - Start from aim labels (virtual root)
        - Recursively add neighbor labels
        - Build tree BACKWARD from aims toward conditions
        - Stop at max_hop depth
        
        Args:
            condition_labels: Labels of condition entities
            aim_labels: Labels of aim entities
            max_hop: Maximum recursion depth
            
        Returns:
            Tree structure (nested dicts)
        """
        logger.info("Step 2: Constructing reverse reasoning tree")
        
        # Create virtual root
        tree = {
            "label": "VIRTUAL_ROOT",
            "children": []
        }
        
        # Add all aim labels as children of root
        for aim_label in aim_labels:
            child = {
                "label": aim_label,
                "children": [],
                "depth": 0
            }
            tree["children"].append(child)
            # Recursively build tree
            self._build_tree_recursive(child, max_hop, 0)
        
        logger.info(f"Built reverse reasoning tree with depth {max_hop}")
        return tree
    
    def _build_tree_recursive(self, node: Dict, max_hop: int, current_depth: int):
        """
        Recursively build reverse reasoning tree
        
        Args:
            node: Current tree node
            max_hop: Maximum depth
            current_depth: Current recursion depth
        """
        if current_depth >= max_hop:
            return
        
        label = node["label"]
        
        # Query neighbor label dictionary
        neighbors = self.neighbor_label_dict.get(label, set())
        
        # Add each neighbor as child
        for neighbor_label in neighbors:
            child = {
                "label": neighbor_label,
                "children": [],
                "depth": current_depth + 1
            }
            node["children"].append(child)
            
            # Recursive call
            self._build_tree_recursive(child, max_hop, current_depth + 1)
    
    def prune_by_conditions(self, tree: Dict, condition_labels: List[str]) -> Dict:
        """
        Step 3: Prune By Conditions
        
        According to paper Section 2.2.3 and Algorithm 1:
        - DFS through tree to collect all paths
        - Remove paths without condition labels
        - For paths with conditions, keep only up to last condition
        
        Args:
            tree: Reverse reasoning tree
            condition_labels: Labels of condition entities
            
        Returns:
            Pruned tree
        """
        logger.info("Step 3: Pruning by conditions")
        
        # Collect all root-to-leaf paths
        paths = self._collect_paths(tree)
        
        # Filter paths
        filtered_paths = []
        for path in paths:
            # Find condition labels in path
            condition_indices = [
                i for i, label in enumerate(path) 
                if label in condition_labels
            ]
            
            if condition_indices:
                # Keep path up to last condition
                last_condition_idx = condition_indices[-1]
                filtered_path = path[:last_condition_idx + 1]
                filtered_paths.append(filtered_path)
            # else: path without condition, discard
        
        # Rebuild tree from filtered paths
        pruned_tree = self._build_tree_from_paths(filtered_paths)
        
        logger.info(f"Pruned to {len(filtered_paths)} paths with conditions")
        return pruned_tree
    
    def _collect_paths(self, tree: Dict) -> List[List[str]]:
        """
        Collect all root-to-leaf paths from tree using DFS
        
        Args:
            tree: Tree structure
            
        Returns:
            List of paths (each path is list of labels)
        """
        paths = []
        
        def dfs(node: Dict, current_path: List[str]):
            label = node.get("label")
            if label and label != "VIRTUAL_ROOT":
                current_path = current_path + [label]
            
            children = node.get("children", [])
            if not children:
                # Leaf node
                if current_path:
                    paths.append(current_path)
            else:
                # Continue DFS
                for child in children:
                    dfs(child, current_path)
        
        dfs(tree, [])
        return paths
    
    def _build_tree_from_paths(self, paths: List[List[str]]) -> Dict:
        """
        Build tree structure from list of paths
        
        Args:
            paths: List of label paths
            
        Returns:
            Tree structure
        """
        tree = {
            "label": "VIRTUAL_ROOT",
            "children": []
        }
        
        for path in paths:
            # Insert path into tree
            current = tree
            for label in path:
                # Find or create child with this label
                child = None
                for c in current["children"]:
                    if c["label"] == label:
                        child = c
                        break
                
                if not child:
                    child = {
                        "label": label,
                        "children": []
                    }
                    current["children"].append(child)
                
                current = child
        
        return tree
    
    def prune_cycles(self, tree: Dict) -> Dict:
        """
        Step 4: Prune Cycle Sub-paths
        
        According to paper Section 2.2.4:
        - DFS through tree with visited set
        - Remove edges that create cycles
        
        Args:
            tree: Tree with potential cycles
            
        Returns:
            Cycle-free tree
        """
        logger.info("Step 4: Pruning cycles")
        
        def dfs_prune_cycles(node: Dict, visited: Set[str]):
            """DFS to remove cycles"""
            label = node.get("label")
            
            if label in visited:
                # Cycle detected, mark for removal
                return True  # Signal to remove this node
            
            if label and label != "VIRTUAL_ROOT":
                visited.add(label)
            
            # Process children
            children_to_keep = []
            for child in node.get("children", []):
                # Recursive call with copy of visited set
                should_remove = dfs_prune_cycles(child, visited.copy())
                if not should_remove:
                    children_to_keep.append(child)
            
            node["children"] = children_to_keep
            
            return False  # Don't remove this node
        
        dfs_prune_cycles(tree, set())
        
        logger.info("Cycle pruning complete")
        return tree
    
    def prune_by_semantics(self, tree: Dict, question: str) -> List[LabelReasoningPath]:
        """
        Step 5: Prune By Semantics
        
        According to paper Section 2.2.5 and Figure 4:
        - Collect all paths from pruned tree
        - Use LLM to filter paths based on relevance to question
        - Return only LLM-approved paths
        
        Args:
            tree: Tree after condition and cycle pruning
            question: Original question
            
        Returns:
            List of semantically relevant label reasoning paths
        """
        logger.info("Step 5: Pruning by semantics with LLM")
        
        # Collect all paths
        paths = self._collect_paths(tree)
        
        if not paths:
            logger.warning("No paths to filter")
            return []
        
        # Reverse paths to forward direction (aim → condition becomes condition → aim)
        forward_paths = [list(reversed(path)) for path in paths]
        
        # Build prompt for LLM filtering
        prompt = self._build_semantic_filtering_prompt(question, forward_paths)
        
        # Query LLM
        response = self.llm.generate(prompt, max_tokens=1000, temperature=0)
        
        # Parse response to get filtered paths
        filtered_paths = self._parse_filtered_paths(response, forward_paths)
        
        # Convert to LabelReasoningPath objects
        reasoning_paths = [
            LabelReasoningPath(labels=path, score=1.0)
            for path in filtered_paths
        ]
        
        logger.info(f"Filtered to {len(reasoning_paths)} semantically relevant paths")
        return reasoning_paths
    
    def _build_semantic_filtering_prompt(self, question: str, 
                                        paths: List[List[str]]) -> str:
        """
        Build prompt for semantic path filtering (paper Figure 4)
        
        Args:
            question: User question
            paths: Candidate reasoning paths
            
        Returns:
            Prompt string
        """
        # Format paths for display
        path_strings = []
        for i, path in enumerate(paths):
            path_str = " → ".join([self.label_descriptions.get(label, label) for label in path])
            path_strings.append(f"{i+1}. {path_str}")
        
        paths_text = "\n".join(path_strings)
        
        prompt = f"""Please filter the reasoning paths based on the user question and the given possible reasoning paths.

User question: {question}

Possible reasoning paths:
{paths_text}

Please return the indices of the filtered reasoning paths that are most helpful for answering the question.
Return your answer as a JSON list of integers, e.g., [1, 3, 5]
"""
        return prompt
    
    def _parse_filtered_paths(self, response: str, 
                             all_paths: List[List[str]]) -> List[List[str]]:
        """
        Parse LLM response to get filtered path indices
        
        Args:
            response: LLM response with path indices
            all_paths: All candidate paths
            
        Returns:
            Filtered paths
        """
        import json
        import re
        
        try:
            # Strip code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(1).strip()
            
            # Try to parse as JSON list
            indices = json.loads(cleaned_response)
            if isinstance(indices, list):
                filtered = []
                for idx in indices:
                    if isinstance(idx, int) and 1 <= idx <= len(all_paths):
                        filtered.append(all_paths[idx - 1])  # 1-indexed to 0-indexed
                return filtered
        except json.JSONDecodeError:
            # Try to extract numbers from response
            numbers = re.findall(r'\d+', response)
            if numbers:
                filtered = []
                for num_str in numbers:
                    idx = int(num_str)
                    if 1 <= idx <= len(all_paths):
                        filtered.append(all_paths[idx - 1])
                return filtered
        
        # Fallback: return all paths
        logger.warning("Could not parse LLM filtering response, keeping all paths")
        return all_paths
    
    def generate_label_reasoning_paths(self, question: str,
                                      topic_entities: List[Entity],
                                      max_hop: int = 3) -> Tuple[List[LabelReasoningPath], List[Entity]]:
        """
        Main entry point: Generate label reasoning paths using full ORT algorithm
        
        Implements all 5 steps:
        1. Aim and Condition Recognition
        2. Construct Reverse Reasoning Tree
        3. Prune by Conditions
        4. Prune Cycles
        5. Prune by Semantics
        
        Args:
            question: Natural language question
            topic_entities: Entities mentioned in question
            max_hop: Maximum path length
            
        Returns:
            Tuple of (label_reasoning_paths, condition_entities)
            - Returns condition entities for use in guided KG traversal
        """
        logger.info("=" * 80)
        logger.info("ORT: Generating Label Reasoning Paths")
        logger.info("=" * 80)
        
        # Step 1: Recognize aims and conditions
        condition_entities, aim_entities = self.recognize_aims_and_conditions(
            question, topic_entities
        )
        
        if not aim_entities:
            logger.warning("No aim entities recognized, using fallback")
            # Fallback: infer aim types from question
            aim_entities = [Entity(
                entity_id="aim_fallback",
                entity_name="answer",
                labels=self._infer_aim_types_from_question(question)
            )]
        
        # Extract labels
        condition_labels = [label for entity in condition_entities for label in entity.labels]
        aim_labels = [label for entity in aim_entities for label in entity.labels]
        
        logger.info(f"Condition labels: {condition_labels}")
        logger.info(f"Aim labels: {aim_labels}")
        
        # Step 2: Construct reverse reasoning tree
        tree = self.construct_reverse_reasoning_tree(
            condition_labels, aim_labels, max_hop
        )
        
        # Step 3: Prune by conditions
        tree = self.prune_by_conditions(tree, condition_labels)
        
        # Step 4: Prune cycles
        tree = self.prune_cycles(tree)
        
        # Step 5: Prune by semantics
        label_reasoning_paths = self.prune_by_semantics(tree, question)
        
        logger.info(f"Generated {len(label_reasoning_paths)} label reasoning paths")
        
        # Return both paths and condition entities (needed for guided traversal)
        return label_reasoning_paths, condition_entities
    
    def _infer_aim_types_from_question(self, question: str) -> List[str]:
        """
        Fallback: Infer aim types from question keywords
        
        Args:
            question: User question
            
        Returns:
            List of inferred aim types (validated against ontology)
        """
        # Simple keyword matching
        question_lower = question.lower()
        
        candidate_types = []
        
        if "who" in question_lower or "person" in question_lower or "people" in question_lower:
            candidate_types = ["people.person"]
        elif "where" in question_lower or "place" in question_lower or "location" in question_lower:
            candidate_types = ["location.location", "location.country", "location.citytown"]
        elif "when" in question_lower or "date" in question_lower or "time" in question_lower or "year" in question_lower:
            candidate_types = ["type.datetime", "time.event"]
        elif "language" in question_lower or "speak" in question_lower:
            candidate_types = ["language.human_language"]
        elif "disease" in question_lower or "illness" in question_lower or "died" in question_lower:
            candidate_types = ["medicine.disease", "people.cause_of_death"]
        elif "art" in question_lower or "painting" in question_lower or "artwork" in question_lower or "create" in question_lower:
            candidate_types = ["visual_art.artwork", "book.written_work", "music.composition"]
        elif "company" in question_lower or "organization" in question_lower or "corporation" in question_lower:
            candidate_types = ["organization.organization", "business.company"]
        elif "what" in question_lower:
            # Generic entity
            candidate_types = ["common.topic"]
        else:
            candidate_types = ["common.topic"]
        
        # Validate candidates exist in ontology
        valid_types = [t for t in candidate_types if t in self.label_list]
        return valid_types if valid_types else ["common.topic"]
    
    def _infer_types_from_entity_name(self, entity_name: str) -> List[str]:
        """
        Infer entity types from entity name when KG lookup fails
        
        Args:
            entity_name: Human-readable entity name
            
        Returns:
            List of inferred types
        """
        name_lower = entity_name.lower()
        
        # Check for common patterns
        if any(word in name_lower for word in ["country", "nation", "state"]):
            return ["location.country"]
        elif any(word in name_lower for word in ["city", "town", "village"]):
            return ["location.citytown"]
        elif any(word in name_lower for word in ["person", "people"]):
            return ["people.person"]
        elif any(word in name_lower for word in ["company", "corporation", "inc"]):
            return ["organization.organization"]
        elif any(word in name_lower for word in ["language"]):
            return ["language.human_language"]
        else:
            # Generic fallback
            return ["common.topic"]
    
    def guided_kg_traversal(self, 
                           label_paths: List[LabelReasoningPath],
                           condition_entities: List[Entity],
                           max_entities_per_step: int = 20) -> List[Dict]:
        """
        Implement Section 2.3 from ORT paper: Guided Answer Mining
        
        Use label reasoning paths to guide KG traversal with TYPE CONSTRAINTS.
        This is the CRITICAL missing piece - paths must constrain which entities
        we explore, not just suggest where to look.
        
        Algorithm from paper:
        1. Start with condition entities matching first label in path
        2. For each entity, query KG neighbors
        3. FILTER neighbors to only those matching NEXT label in path
        4. Continue iteratively following label sequence
        5. Build tree of type-constrained entity paths
        
        Args:
            label_paths: Abstract label reasoning paths from ORT
            condition_entities: Condition entities to start traversal from
            max_entities_per_step: Limit entities per step (prevent explosion)
            
        Returns:
            List of concrete entity path dictionaries with answer entities
        """
        import random
        
        logger.info("=" * 80)
        logger.info("ORT: Guided KG Traversal (Section 2.3)")
        logger.info("=" * 80)
        
        all_entity_paths = []
        
        for path_idx, label_path in enumerate(label_paths):
            if not label_path.labels:
                continue
            
            logger.debug(f"\nProcessing label path {path_idx+1}/{len(label_paths)}: {' -> '.join(label_path.labels)}")
            
            # Step 1: Find condition entities matching first label
            # According to paper: "For each reasoning path, the first node is a condition node"
            current_level_entities = []
            first_label = label_path.labels[0]
            
            for entity in condition_entities:
                # Check if entity has the first label type
                if first_label in entity.labels:
                    current_level_entities.append({
                        'entity_id': entity.entity_id,
                        'entity_name': entity.entity_name,
                        'path': [entity.entity_id],
                        'labels_matched': [first_label]
                    })
            
            if not current_level_entities:
                logger.debug(f"No condition entities match first label '{first_label}', trying relaxed match")
                # Fallback: Use any condition entity if strict match fails
                for entity in condition_entities:
                    current_level_entities.append({
                        'entity_id': entity.entity_id,
                        'entity_name': entity.entity_name,
                        'path': [entity.entity_id],
                        'labels_matched': [first_label]  # Assume match for traversal
                    })
            
            logger.debug(f"Starting with {len(current_level_entities)} condition entities")
            
            # Step 2: Iteratively expand following label constraints
            for step_idx in range(1, len(label_path.labels)):
                target_label = label_path.labels[step_idx]
                next_level_entities = []
                
                logger.debug(f"Step {step_idx}: Expanding to label '{target_label}'")
                
                for entity_info in current_level_entities:
                    entity_id = entity_info['entity_id']
                    
                    # Get all neighbors from KG
                    try:
                        neighbors = self.kg_interface.get_neighbors(entity_id, max_hops=1)
                    except Exception as e:
                        logger.warning(f"Error getting neighbors for {entity_id}: {e}")
                        continue
                    
                    # CRITICAL: Filter to only neighbors matching target_label
                    filtered_neighbors = []
                    for neighbor_id in neighbors:
                        # Get neighbor's types from KG
                        neighbor_types = self.kg_interface.get_entity_types(neighbor_id)
                        
                        # Check if any type matches target label
                        if target_label in neighbor_types:
                            filtered_neighbors.append(neighbor_id)
                    
                    logger.debug(f"Entity {entity_id}: {len(neighbors)} neighbors -> {len(filtered_neighbors)} match type '{target_label}'")
                    
                    # Limit expansion (prevent explosion)
                    if len(filtered_neighbors) > max_entities_per_step:
                        filtered_neighbors = random.sample(filtered_neighbors, max_entities_per_step)
                        logger.debug(f"Sampled down to {max_entities_per_step} entities")
                    
                    # Add filtered neighbors to next level
                    for neighbor_id in filtered_neighbors:
                        neighbor_name = self.kg_interface.get_entity_name(neighbor_id)
                        next_level_entities.append({
                            'entity_id': neighbor_id,
                            'entity_name': neighbor_name,
                            'path': entity_info['path'] + [neighbor_id],
                            'labels_matched': entity_info['labels_matched'] + [target_label]
                        })
                
                if not next_level_entities:
                    logger.debug(f"No entities found matching '{target_label}', stopping traversal for this path")
                    break
                
                current_level_entities = next_level_entities
                logger.debug(f"Step {step_idx} complete: {len(current_level_entities)} entities at next level")
            
            # Step 3: Collect final entities as answers
            # Last level contains answer entities that followed full label path
            for entity_info in current_level_entities:
                entity_path = {
                    "label_path": label_path.labels,
                    "entity_path": entity_info['path'],
                    "answer_entity": entity_info['entity_id'],
                    "answer_entity_name": entity_info['entity_name'],
                    "labels_matched": entity_info['labels_matched'],
                    "path_type": "ort_guided",
                    "path_score": label_path.score
                }
                all_entity_paths.append(entity_path)
        
        logger.info(f"Guided traversal complete: {len(all_entity_paths)} entity paths found")
        logger.info("=" * 80)
        
        return all_entity_paths
