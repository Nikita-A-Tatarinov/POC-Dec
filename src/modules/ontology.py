"""
Ontology Module for ORT-style Type-level Planning
Implements reverse thinking from abstract types to concrete entities
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class OntologyModule:
    """
    Handles type hierarchy and ontology-guided planning
    Implements ORT-style reverse thinking at type level
    """
    
    def __init__(self, kg_interface):
        """
        Initialize Ontology module
        
        Args:
            kg_interface: KGInterface instance for querying KG
        """
        self.kg_interface = kg_interface
        self.type_hierarchy = {}  # child_type -> parent_types
        self.reverse_hierarchy = {}  # parent_type -> child_types
        self.type_relations = defaultdict(set)  # type -> set of relations
        self.relation_types = defaultdict(set)  # relation -> set of domain/range types
        
    def build_type_hierarchy(self, types: List[str]) -> None:
        """
        Build type hierarchy from list of types
        
        Args:
            types: List of Freebase type IDs
        """
        logger.info(f"Building type hierarchy for {len(types)} types")
        
        # In Freebase, types follow hierarchical structure via domain
        # e.g., people.person is more specific than people
        for type_id in types:
            parts = type_id.split('.')
            
            # Build hierarchy: people.person -> people
            if len(parts) >= 2:
                parent = parts[0]
                if parent not in self.type_hierarchy:
                    self.type_hierarchy[parent] = set()
                
                self.type_hierarchy[type_id] = {parent}
                
                if parent not in self.reverse_hierarchy:
                    self.reverse_hierarchy[parent] = set()
                self.reverse_hierarchy[parent].add(type_id)
            else:
                # Top-level type
                self.type_hierarchy[type_id] = set()
        
        logger.info(f"Built hierarchy with {len(self.type_hierarchy)} type nodes")
    
    def add_relation_type_mapping(self, relation: str, domain_type: str, 
                                   range_type: Optional[str] = None) -> None:
        """
        Add mapping between relation and its domain/range types
        
        Args:
            relation: Relation ID
            domain_type: Domain (subject) type
            range_type: Range (object) type (optional)
        """
        self.type_relations[domain_type].add(relation)
        self.relation_types[relation].add(('domain', domain_type))
        
        if range_type:
            self.relation_types[relation].add(('range', range_type))
    
    def infer_relation_types_from_kg(self, relations: List[str]) -> None:
        """
        Infer type mappings for relations from their structure
        
        Args:
            relations: List of relation IDs
        """
        logger.info(f"Inferring types for {len(relations)} relations")
        
        for relation in relations:
            # Extract domain from relation structure
            # e.g., "people.person.date_of_birth" -> domain: "people.person"
            parts = relation.split('.')
            
            if len(parts) >= 2:
                domain_type = f"{parts[0]}.{parts[1]}"
                self.add_relation_type_mapping(relation, domain_type)
    
    def get_abstract_labels(self, entity_types: List[str], max_depth: int = 2) -> List[str]:
        """
        Get abstract type labels for entity types
        Implements ORT-style abstraction to higher-level types
        
        Args:
            entity_types: List of specific entity types
            max_depth: Maximum depth to traverse in hierarchy
            
        Returns:
            List of abstract type labels (more general types)
        """
        abstract_labels = set()
        
        for entity_type in entity_types:
            # Add the entity type itself
            abstract_labels.add(entity_type)
            
            # Traverse upward in hierarchy
            current_types = {entity_type}
            for _ in range(max_depth):
                next_types = set()
                for t in current_types:
                    if t in self.type_hierarchy:
                        parents = self.type_hierarchy[t]
                        next_types.update(parents)
                        abstract_labels.update(parents)
                
                if not next_types:
                    break
                current_types = next_types
        
        return list(abstract_labels)
    
    def get_concrete_types(self, abstract_type: str, max_depth: int = 2) -> List[str]:
        """
        Get concrete (specific) types from abstract type
        Reverse operation: from general to specific
        
        Args:
            abstract_type: General type
            max_depth: Maximum depth to traverse
            
        Returns:
            List of more specific types
        """
        concrete_types = set()
        
        # Traverse downward in hierarchy
        current_types = {abstract_type}
        for _ in range(max_depth):
            next_types = set()
            for t in current_types:
                if t in self.reverse_hierarchy:
                    children = self.reverse_hierarchy[t]
                    next_types.update(children)
                    concrete_types.update(children)
            
            if not next_types:
                break
            current_types = next_types
        
        return list(concrete_types)
    
    def generate_abstract_path(self, source_types: List[str], 
                               target_types: List[str],
                               max_hops: int = 3) -> List[List[str]]:
        """
        Generate abstract type-level paths from source to target
        Core of ORT-style reverse thinking
        
        Args:
            source_types: Types of source entity
            target_types: Expected types of answer
            max_hops: Maximum path length
            
        Returns:
            List of abstract paths (sequences of type labels)
        """
        # Get abstract labels for both source and target
        source_abstract = self.get_abstract_labels(source_types)
        target_abstract = self.get_abstract_labels(target_types)
        
        abstract_paths = []
        
        # BFS to find type-level paths
        queue = deque([(s, [s], 0) for s in source_abstract])
        visited = set(source_abstract)
        
        while queue:
            current_type, path, depth = queue.popleft()
            
            if depth >= max_hops:
                continue
            
            # Check if we reached target type
            for target_type in target_abstract:
                if self._types_compatible(current_type, target_type):
                    abstract_paths.append(path + [target_type])
            
            # Get relations associated with current type
            if current_type in self.type_relations:
                for relation in self.type_relations[current_type]:
                    # Infer next possible types from relation
                    next_types = self._get_relation_range_types(relation)
                    
                    for next_type in next_types:
                        if next_type not in visited or depth < max_hops - 1:
                            queue.append((next_type, path + [next_type], depth + 1))
                            visited.add(next_type)
        
        # If no paths found, return generic path
        if not abstract_paths and source_abstract and target_abstract:
            abstract_paths.append([source_abstract[0], target_abstract[0]])
        
        logger.debug(f"Generated {len(abstract_paths)} abstract paths")
        return abstract_paths[:10]  # Return top 10 paths
    
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """
        Check if two types are compatible (same or in hierarchy)
        
        Args:
            type1, type2: Type IDs
            
        Returns:
            True if types are compatible
        """
        if type1 == type2:
            return True
        
        # Check if type1 is ancestor of type2
        if type2 in self.type_hierarchy:
            ancestors = self._get_all_ancestors(type2)
            if type1 in ancestors:
                return True
        
        # Check if type2 is ancestor of type1
        if type1 in self.type_hierarchy:
            ancestors = self._get_all_ancestors(type1)
            if type2 in ancestors:
                return True
        
        return False
    
    def _get_all_ancestors(self, type_id: str) -> Set[str]:
        """Get all ancestor types"""
        ancestors = set()
        current = {type_id}
        
        while current:
            next_level = set()
            for t in current:
                if t in self.type_hierarchy:
                    parents = self.type_hierarchy[t]
                    next_level.update(parents)
                    ancestors.update(parents)
            current = next_level
        
        return ancestors
    
    def _get_relation_range_types(self, relation: str) -> List[str]:
        """
        Get range (target) types for a relation
        
        Args:
            relation: Relation ID
            
        Returns:
            List of possible range types
        """
        if relation not in self.relation_types:
            # Infer from relation structure
            parts = relation.split('.')
            if len(parts) >= 2:
                # Heuristic: range type often relates to the property name
                # For now, return domain type and its related types
                domain = f"{parts[0]}.{parts[1]}"
                return [domain]
            return []
        
        range_types = []
        for role, type_id in self.relation_types[relation]:
            if role == 'range':
                range_types.append(type_id)
        
        return range_types if range_types else [self._infer_range_type(relation)]
    
    def _infer_range_type(self, relation: str) -> str:
        """
        Infer range type from relation structure
        
        Args:
            relation: Relation ID
            
        Returns:
            Inferred type
        """
        parts = relation.split('.')
        
        # Heuristic-based inference
        if len(parts) >= 3:
            property_name = parts[-1]
            
            # Common patterns
            if 'date' in property_name or 'time' in property_name:
                return 'type.datetime'
            elif 'place' in property_name or 'location' in property_name:
                return 'location.location'
            elif 'person' in property_name:
                return 'people.person'
            elif 'name' in property_name or 'title' in property_name:
                return 'type.text'
        
        # Default: same domain as relation
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        
        return 'type.object'
    
    def get_type_constrained_relations(self, current_type: str, 
                                       target_types: List[str]) -> List[str]:
        """
        Get relations that can transition from current type toward target types
        Type-guided relation filtering
        
        Args:
            current_type: Current position in type space
            target_types: Goal types
            
        Returns:
            List of promising relations
        """
        if current_type not in self.type_relations:
            return []
        
        all_relations = list(self.type_relations[current_type])
        
        # Score relations by compatibility with target types
        scored_relations = []
        for relation in all_relations:
            range_types = self._get_relation_range_types(relation)
            
            # Check compatibility with any target type
            compatibility_score = 0
            for range_type in range_types:
                for target_type in target_types:
                    if self._types_compatible(range_type, target_type):
                        compatibility_score += 1
            
            if compatibility_score > 0:
                scored_relations.append((relation, compatibility_score))
        
        # Sort by score
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        
        return [rel for rel, score in scored_relations[:20]]  # Top 20
    
    def generate_reverse_abstract_path(self, source_types: List[str],
                                       target_types: List[str],
                                       max_hops: int = 3) -> List[List[str]]:
        """
        Generate abstract type-level paths using REVERSE reasoning (ORT-style)
        Start from target types (answer) and work backward to source types (question)
        
        This implements the core ORT innovation: reverse thinking from aim to condition
        
        Args:
            source_types: Types of source entity (question entities)
            target_types: Expected types of answer
            max_hops: Maximum path length
            
        Returns:
            List of abstract paths (sequences of type labels from target to source)
        """
        # Get abstract labels
        source_abstract = self.get_abstract_labels(source_types)
        target_abstract = self.get_abstract_labels(target_types)
        
        reverse_paths = []
        
        # BFS starting from TARGET types (reverse reasoning)
        queue = deque([(t, [t], 0) for t in target_abstract])
        visited = set(target_abstract)
        
        while queue:
            current_type, path, depth = queue.popleft()
            
            if depth >= max_hops:
                continue
            
            # Check if we reached source type (backward connection)
            for source_type in source_abstract:
                if self._types_compatible(current_type, source_type):
                    # Found a reverse path! Reverse it to get forward direction
                    forward_path = list(reversed(path + [source_type]))
                    reverse_paths.append(forward_path)
            
            # Get inverse relations (what types can point TO current_type)
            # This is the key difference: we traverse backward
            prev_types = self._get_inverse_relation_types(current_type)
            
            for prev_type in prev_types:
                if prev_type not in visited or depth < max_hops - 1:
                    queue.append((prev_type, path + [prev_type], depth + 1))
                    visited.add(prev_type)
        
        # If no paths found via reverse, return generic path
        if not reverse_paths and source_abstract and target_abstract:
            reverse_paths.append([source_abstract[0], target_abstract[0]])
        
        logger.debug(f"Generated {len(reverse_paths)} reverse abstract paths")
        return reverse_paths[:10]  # Return top 10 paths
    
    def _get_inverse_relation_types(self, target_type: str) -> Set[str]:
        """
        Get types that can point TO the target type via relations
        This enables backward traversal for reverse reasoning
        
        Args:
            target_type: The type we want to reach
            
        Returns:
            Set of types that have relations pointing to target_type
        """
        inverse_types = set()
        
        # Search through all relations to find ones with range = target_type
        for relation, type_info in self.relation_types.items():
            for role, rel_type in type_info:
                if role == 'range' and self._types_compatible(rel_type, target_type):
                    # This relation points TO target_type
                    # Get its domain (source) types
                    for role2, domain_type in type_info:
                        if role2 == 'domain':
                            inverse_types.add(domain_type)
        
        # Also consider parent types (more general types)
        if target_type in self.type_hierarchy:
            inverse_types.update(self.type_hierarchy[target_type])
        
        return inverse_types
