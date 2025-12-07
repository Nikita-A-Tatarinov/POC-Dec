"""
Knowledge Graph Interface Module
Following GCR approach: Uses pre-extracted subgraphs from RoG datasets
Provides in-memory graph operations for Freebase subgraphs
"""

from typing import List, Dict, Tuple, Optional
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class KGInterface:
    """Interface for querying Freebase knowledge graph subgraphs in memory"""
    
    def __init__(self):
        """Initialize KG interface with in-memory graph"""
        self.graph: Optional[nx.DiGraph] = None
        self.entity_names: Dict[str, str] = {}  # Store entity ID -> name mappings
    
    def get_entity_name(self, entity_id: str) -> str:
        """
        Get human-readable name for entity from in-memory names
        
        Args:
            entity_id: Freebase entity ID (without ns: prefix)
            
        Returns:
            Entity name or "UnknownEntity" if not found
        """
        if entity_id in self.entity_names:
            return self.entity_names[entity_id]
        return "UnknownEntity"
    
    def get_entity_types(self, entity_id: str) -> List[str]:
        """
        Get types for an entity from in-memory graph
        
        For WebQSP/Freebase: Infers types from relations since explicit
        type.object.type triples are missing in subgraphs.
        
        Args:
            entity_id: Freebase entity ID
            
        Returns:
            List of type IDs
        """
        if self.graph is None or entity_id not in self.graph:
            return []
        
        types = set()
        
        # Method 1: Explicit type relations (if available)
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if data.get('relation') == 'type.object.type':
                types.add(target)
        
        # Method 2: Infer types from outgoing relations
        # Freebase relations have format: domain.type.property
        # If entity has relation "people.person.nationality", it's likely a people.person
        for _, _, data in self.graph.out_edges(entity_id, data=True):
            relation = data.get('relation', '')
            if relation and '.' in relation:
                parts = relation.split('.')
                if len(parts) >= 2:
                    domain_type = f"{parts[0]}.{parts[1]}"
                    types.add(domain_type)
        
        # Method 3: Infer types from incoming relations
        # If entity is target of "location.country.languages_spoken", it's likely a language
        for _, _, data in self.graph.in_edges(entity_id, data=True):
            relation = data.get('relation', '')
            if relation and '.' in relation:
                parts = relation.split('.')
                if len(parts) >= 3:
                    property_name = parts[-1]
                    # Infer type from property semantics
                    if 'language' in property_name:
                        types.add('language.human_language')
                    elif 'location' in property_name or 'place' in property_name or 'country' in property_name:
                        types.add('location.location')
                    elif 'person' in property_name or 'people' in property_name:
                        types.add('people.person')
                    elif 'organization' in property_name or 'company' in property_name:
                        types.add('organization.organization')
                    elif 'team' in property_name:
                        types.add('sports.sports_team')
                    elif 'currency' in property_name:
                        types.add('finance.currency')
        
        # If no types found, add common.topic as fallback
        if not types:
            types.add('common.topic')
        
        return list(types)
    
    def get_neighbors(self, entity_id: str, max_hops: int = 1) -> List[str]:
        """
        Get neighboring entity IDs from entity (1-hop only for now)
        
        Args:
            entity_id: Freebase entity ID  
            max_hops: Maximum hops (currently only supports 1)
            
        Returns:
            List of neighboring entity IDs
        """
        if self.graph is None or entity_id not in self.graph:
            return []
        
        neighbors = set()
        
        # Outgoing edges (this entity -> neighbors)
        for _, target, _ in self.graph.out_edges(entity_id, data=True):
            neighbors.add(target)
        
        # Incoming edges (neighbors -> this entity)
        for source, _, _ in self.graph.in_edges(entity_id, data=True):
            neighbors.add(source)
        
        return list(neighbors)
    
    def get_1hop_relations(self, entity_id: str, direction: str = "head") -> List[str]:
        """
        Get 1-hop relations from entity in in-memory graph
        
        Args:
            entity_id: Freebase entity ID
            direction: "head" or "tail" for outgoing/incoming relations
            
        Returns:
            List of relation IDs
        """
        if self.graph is None:
            return []
        
        relations = self._query_in_memory_1hop(entity_id, direction)
        return [r for r in relations if not self._should_filter_relation(r)]
    
    def get_2hop_relations(self, entity_id: str) -> List[Tuple[str, str]]:
        """
        Get 2-hop relation paths from entity in in-memory graph
        
        Args:
            entity_id: Freebase entity ID
            
        Returns:
            List of (relation1, relation2) tuples
        """
        if self.graph is None:
            return []
        
        paths = self._query_in_memory_2hop(entity_id)
        return [(r1, r2) for r1, r2 in paths 
                if not self._should_filter_relation(r1) and not self._should_filter_relation(r2)]
    
    def get_3hop_relations(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """
        Get 3-hop relation paths from entity in in-memory graph
        
        Args:
            entity_id: Freebase entity ID
            
        Returns:
            List of (relation1, relation2, relation3) tuples
        """
        if self.graph is None or entity_id not in self.graph:
            return []
        
        relation_paths = []
        
        # Get all 3-hop paths
        for e1 in self.graph.successors(entity_id):
            r1 = self.graph[entity_id][e1].get('relation', '')
            if not r1 or self._should_filter_relation(r1):
                continue
            
            for e2 in self.graph.successors(e1):
                r2 = self.graph[e1][e2].get('relation', '')
                if not r2 or self._should_filter_relation(r2):
                    continue
                
                for e3 in self.graph.successors(e2):
                    r3 = self.graph[e2][e3].get('relation', '')
                    if r3 and not self._should_filter_relation(r3):
                        relation_paths.append((r1, r2, r3))
        
        return relation_paths
    
    def get_entities_via_relation(self, entity_id: str, relation: str, 
                                   direction: str = "head") -> List[str]:
        """
        Get entities connected via a relation in in-memory graph
        
        Args:
            entity_id: Source entity ID
            relation: Relation ID
            direction: "head" (outgoing) or "tail" (incoming)
            
        Returns:
            List of connected entity IDs
        """
        if self.graph is None:
            return []
        
        return self._query_in_memory_entities(entity_id, relation, direction)
    
    def get_2hop_entities(self, entity_id: str, rel1: str, rel2: str) -> List[Tuple[str, str]]:
        """
        Get entities via 2-hop path in in-memory graph
        
        Args:
            entity_id: Source entity
            rel1: First relation
            rel2: Second relation
            
        Returns:
            List of (intermediate_entity, final_entity) tuples
        """
        if self.graph is None:
            return []
        
        entity_pairs = []
        # First hop: entity_id --rel1--> intermediate_entities
        intermediate_entities = self._query_in_memory_entities(entity_id, rel1, direction="head")
        
        # Second hop: each intermediate --rel2--> final_entities
        for intermediate in intermediate_entities:
            final_entities = self._query_in_memory_entities(intermediate, rel2, direction="head")
            for final in final_entities:
                entity_pairs.append((intermediate, final))
        
        return entity_pairs
    
    def get_3hop_entities(self, entity_id: str, rel1: str, rel2: str, rel3: str) -> List[Tuple[str, str, str]]:
        """
        Get entities via 3-hop path in in-memory graph
        
        Args:
            entity_id: Source entity
            rel1, rel2, rel3: Relations in path
            
        Returns:
            List of (entity1, entity2, entity3) tuples
        """
        if self.graph is None:
            return []
        
        entity_triples = []
        # First hop: entity_id --rel1--> e1
        first_hop = self._query_in_memory_entities(entity_id, rel1, direction="head")
        
        # Second and third hops
        for e1 in first_hop:
            second_hop = self._query_in_memory_entities(e1, rel2, direction="head")
            for e2 in second_hop:
                third_hop = self._query_in_memory_entities(e2, rel3, direction="head")
                for e3 in third_hop:
                    entity_triples.append((e1, e2, e3))
        
        return entity_triples
    
    def _should_filter_relation(self, relation: str) -> bool:
        """
        Check if relation should be filtered out
        
        Args:
            relation: Relation ID
            
        Returns:
            True if relation should be filtered
        """
        filter_prefixes = [
            "type.object",
            "type.type",
            "common.",
            "freebase.",
            "kg.",
            "base."
        ]
        
        for prefix in filter_prefixes:
            if relation.startswith(prefix):
                return True
        
        # Filter relations ending with metadata-like suffixes
        filter_suffixes = ["_id", "_code", "_number", "webpage", "url"]
        for suffix in filter_suffixes:
            if relation.endswith(suffix):
                return True
        
        return False
    
    def extract_domain(self, relation: str) -> str:
        """
        Extract domain from relation (for super-relation grouping)
        
        Args:
            relation: Relation ID (e.g., "people.person.date_of_birth")
            
        Returns:
            Domain string (e.g., "people.person")
        """
        parts = relation.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return parts[0] if parts else ""
    
    def load_from_dataset(self, graph_triples: List[List[str]], merge: bool = False) -> None:
        """
        Load knowledge graph from dataset's 'graph' field (GCR approach)
        
        Args:
            graph_triples: List of triples [subject, predicate, object] from dataset
            merge: If True, merge with existing graph. If False, replace graph (default).
        
        Example:
            >>> kg = KGInterface()
            >>> dataset = load_dataset("rmanluo/RoG-webqsp", split="train[:1]")
            >>> kg.load_from_dataset(dataset[0]['graph'])
            
            >>> # Or build global graph from multiple questions
            >>> for entry in dataset:
            >>>     kg.load_from_dataset(entry['graph'], merge=True)
        """
        if not merge:
            # Fresh graph (default behavior)
            self.graph = nx.DiGraph()
            self.entity_names = {}  # Store entity names
        elif self.graph is None:
            # First load in merge mode - initialize
            self.graph = nx.DiGraph()
            self.entity_names = {}
        
        logger.debug(f"Loading {len(graph_triples)} triples into in-memory graph...")
        
        for triple in graph_triples:
            if len(triple) != 3:
                logger.warning(f"Invalid triple format: {triple}")
                continue
            
            subject, predicate, obj = triple
            # Normalize by stripping whitespace
            subject = subject.strip()
            predicate = predicate.strip()
            obj = obj.strip()
            
            # Store human-readable names for entities
            # If entity ID starts with 'm.' or 'g.' (Freebase mid), it's an ID
            # Otherwise, it's likely a human-readable name
            if not subject.startswith(('m.', 'g.')) and subject not in self.entity_names:
                self.entity_names[subject] = subject  # The string itself is the name
            if not obj.startswith(('m.', 'g.')) and obj not in self.entity_names:
                self.entity_names[obj] = obj  # The string itself is the name
            
            # Add edge with relation as attribute
            self.graph.add_edge(subject, obj, relation=predicate)
        
        logger.debug(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges, {len(self.entity_names)} named entities")
    
    def _query_in_memory_1hop(self, entity: str, direction: str = "head") -> List[str]:
        """Query 1-hop relations from in-memory graph"""
        if self.graph is None:
            return []
        
        relations = set()
        
        if direction == "head":
            # Outgoing edges from entity
            if entity in self.graph:
                for _, target, data in self.graph.out_edges(entity, data=True):
                    if 'relation' in data:
                        relations.add(data['relation'])
        else:  # tail
            # Incoming edges to entity
            if entity in self.graph:
                for source, _, data in self.graph.in_edges(entity, data=True):
                    if 'relation' in data:
                        relations.add(data['relation'])
        
        return list(relations)
    
    def _query_in_memory_2hop(self, entity: str) -> List[Tuple[str, str]]:
        """Query 2-hop relation paths from in-memory graph"""
        if self.graph is None or entity not in self.graph:
            return []
        
        paths = []
        
        # Get all 2-hop paths
        for intermediate in self.graph.successors(entity):
            r1 = self.graph[entity][intermediate].get('relation', '')
            if not r1:
                continue
            
            for target in self.graph.successors(intermediate):
                r2 = self.graph[intermediate][target].get('relation', '')
                if r2:
                    paths.append((r1, r2))
        
        return paths
    
    def _query_in_memory_entities(self, entity: str, relation: str, direction: str = "head") -> List[str]:
        """Query entities connected via relation from in-memory graph"""
        if self.graph is None or entity not in self.graph:
            return []
        
        entities = []
        
        if direction == "head":
            # entity --relation--> ?
            for _, target, data in self.graph.out_edges(entity, data=True):
                if data.get('relation') == relation:
                    entities.append(target)
        else:  # tail
            # ? --relation--> entity
            for source, _, data in self.graph.in_edges(entity, data=True):
                if data.get('relation') == relation:
                    entities.append(source)
        
        return entities
    
    def get_2hop_backward_relations(self, entity_id: str) -> List[Tuple[str, str]]:
        """
        Get 2-hop BACKWARD relation paths (Phase C improvement)
        Finds paths: ? --r2--> intermediate --r1--> entity
        
        Args:
            entity_id: Freebase entity ID
            
        Returns:
            List of (relation1, relation2) tuples representing backward paths
        """
        if self.graph is None or entity_id not in self.graph:
            return []
        
        backward_paths = []
        
        # Get entities that point TO entity_id (1-hop backward)
        for intermediate, _, data in self.graph.in_edges(entity_id, data=True):
            r1 = data.get('relation', '')
            if not r1 or self._should_filter_relation(r1):
                continue
            
            # Get entities that point TO intermediate (2-hop backward)
            for source, _, data2 in self.graph.in_edges(intermediate, data=True):
                r2 = data2.get('relation', '')
                if r2 and not self._should_filter_relation(r2):
                    backward_paths.append((r2, r1))  # Order: furthest to closest
        
        return backward_paths
    
    def get_3hop_backward_relations(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """
        Get 3-hop BACKWARD relation paths (Phase C improvement)
        Finds paths: ? --r3--> e2 --r2--> e1 --r1--> entity
        
        Args:
            entity_id: Freebase entity ID
            
        Returns:
            List of (relation1, relation2, relation3) tuples representing backward paths
        """
        if self.graph is None or entity_id not in self.graph:
            return []
        
        backward_paths = []
        
        # Get entities that point TO entity_id (1-hop backward)
        for e1, _, data1 in self.graph.in_edges(entity_id, data=True):
            r1 = data1.get('relation', '')
            if not r1 or self._should_filter_relation(r1):
                continue
            
            # Get entities that point TO e1 (2-hop backward)
            for e2, _, data2 in self.graph.in_edges(e1, data=True):
                r2 = data2.get('relation', '')
                if not r2 or self._should_filter_relation(r2):
                    continue
                
                # Get entities that point TO e2 (3-hop backward)
                for source, _, data3 in self.graph.in_edges(e2, data=True):
                    r3 = data3.get('relation', '')
                    if r3 and not self._should_filter_relation(r3):
                        backward_paths.append((r3, r2, r1))  # Order: furthest to closest
        
        return backward_paths
    
    def get_all_types(self) -> List[str]:
        """
        Get all unique types from the KG
        
        For WebQSP/Freebase: Extracts types from relation patterns since explicit
        type.object.type triples are missing in subgraphs.
        
        Returns:
            List of all type IDs in the graph
        """
        if self.graph is None:
            logger.warning("Graph not loaded, cannot get all types")
            return []
        
        types = set()
        
        # Method 1: Explicit type relations (if available)
        for _, target, data in self.graph.edges(data=True):
            if data.get('relation') == 'type.object.type':
                types.add(target)
        
        # Method 2: Extract types from Freebase relation patterns
        # Freebase relations have format: domain.type.property
        # E.g., "people.person.nationality" → domain type is "people.person"
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            if relation and '.' in relation:
                parts = relation.split('.')
                if len(parts) >= 2:
                    # Extract domain type (first two parts)
                    domain_type = f"{parts[0]}.{parts[1]}"
                    types.add(domain_type)
                    
                    # Also extract range type hints from relation name
                    # E.g., "location.location.containedby" suggests range is also location
                    if len(parts) >= 3:
                        # Check if property name suggests a specific type
                        property_name = parts[-1]
                        if 'location' in property_name:
                            types.add('location.location')
                        elif 'person' in property_name or 'people' in property_name:
                            types.add('people.person')
                        elif 'organization' in property_name:
                            types.add('organization.organization')
                        elif 'country' in property_name:
                            types.add('location.country')
                        elif 'language' in property_name:
                            types.add('language.human_language')
        
        # Add common Freebase types that should always be available
        common_types = [
            'common.topic',  # Root type
            'people.person',
            'location.location',
            'location.country',
            'location.citytown',
            'organization.organization',
            'time.event',
            'language.human_language',
            'government.government_office_or_title',
            'sports.sports_team',
            'film.film',
            'music.musical_artist',
            'book.book',
            'education.educational_institution',
        ]
        types.update(common_types)
        
        logger.info(f"Found {len(types)} unique types in KG (extracted from relations + common types)")
        return list(types)
    
    def get_all_relation_types(self) -> List[Tuple[str, str, str]]:
        """
        Get all unique relation types with their domain and range types
        
        Returns:
            List of (relation, domain_type, range_type) tuples
        """
        if self.graph is None:
            logger.warning("Graph not loaded, cannot get relation types")
            return []
        
        relation_info = {}  # relation -> (domain_types, range_types)
        
        # Collect all relations and their connected entity types
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            if not relation or self._should_filter_relation(relation):
                continue
            
            if relation not in relation_info:
                relation_info[relation] = (set(), set())
            
            # Get types of source entity (domain)
            source_types = self.get_entity_types(source)
            if source_types:
                relation_info[relation][0].update(source_types)
            
            # Get types of target entity (range)
            target_types = self.get_entity_types(target)
            if target_types:
                relation_info[relation][1].update(target_types)
        
        # Convert to list of tuples
        result = []
        for relation, (domain_types, range_types) in relation_info.items():
            # For each domain-range pair, create a tuple
            if domain_types and range_types:
                # Use most common types if available
                domain = list(domain_types)[0] if domain_types else "unknown"
                range_type = list(range_types)[0] if range_types else "unknown"
                result.append((relation, domain, range_type))
            elif domain_types:
                domain = list(domain_types)[0]
                result.append((relation, domain, ""))
        
        logger.info(f"Found {len(result)} relation type mappings")
        return result
    
    def get_all_relations(self) -> List[str]:
        """
        Get all unique relations from the KG
        
        Returns:
            List of all relation IDs in the graph
        """
        if self.graph is None:
            logger.warning("Graph not loaded, cannot get all relations")
            return []
        
        relations = set()
        # Find all unique relations in edges
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            if relation and not self._should_filter_relation(relation):
                relations.add(relation)
        
        logger.info(f"Found {len(relations)} unique relations in KG")
        return list(relations)
