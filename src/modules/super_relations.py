"""
Super-relation Module for ReKnoS-style Relation Clustering
Groups semantically similar relations to reduce search space and enable bidirectional reasoning
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SuperRelationModule:
    """
    Handles super-relation construction and expansion
    Implements ReKnoS-style relation clustering by semantic similarity
    """
    
    def __init__(self, kg_interface):
        """
        Initialize Super-relation module
        
        Args:
            kg_interface: KGInterface instance for querying KG
        """
        self.kg_interface = kg_interface
        
        # Mappings
        self.super_relations = {}  # super_rel_id -> set of base relations
        self.relation_to_super = {}  # base_relation -> super_relation_id
        self.super_relation_names = {}  # super_rel_id -> human-readable name
        
        # Statistics
        self.relation_frequencies = defaultdict(int)  # relation -> count
        self.super_relation_scores = {}  # super_rel_id -> importance score
        
    def build_super_relations(self, relations: List[str], 
                              clustering_method: str = "domain",
                              semantic_model: str = "all-MiniLM-L6-v2",
                              semantic_threshold: float = 0.7) -> None:
        """
        Build super-relations from list of base relations
        
        Args:
            relations: List of base relation IDs
            clustering_method: Method to cluster ("domain", "semantic", "semantic_embeddings", "hybrid")
            semantic_model: Model name for semantic embeddings
            semantic_threshold: Similarity threshold for semantic clustering
        """
        logger.info(f"Building super-relations from {len(relations)} relations using {clustering_method} method")
        
        if clustering_method == "domain":
            self._cluster_by_domain(relations)
        elif clustering_method == "semantic":
            self._cluster_by_semantics(relations)
        elif clustering_method == "semantic_embeddings":
            # Use sentence-transformers for semantic clustering
            self.cluster_by_semantic_embeddings(relations, semantic_model, semantic_threshold)
        elif clustering_method == "hybrid":
            # First cluster by domain, then refine with semantics
            self._cluster_by_domain(relations)
            self._refine_with_semantics()
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        # Compute super-relation scores
        self._compute_super_relation_scores()
        
        logger.info(f"Created {len(self.super_relations)} super-relations")
        logger.info(f"Average cluster size: {np.mean([len(rels) for rels in self.super_relations.values()]):.2f}")
    
    def _cluster_by_domain(self, relations: List[str]) -> None:
        """
        Cluster relations by their domain (first two parts)
        e.g., "people.person.date_of_birth" -> domain: "people.person"
        
        Args:
            relations: List of base relations
        """
        domain_clusters = defaultdict(set)
        
        for relation in relations:
            domain = self.kg_interface.extract_domain(relation)
            if domain:
                domain_clusters[domain].add(relation)
            else:
                # Create singleton cluster for relations without clear domain
                domain_clusters[relation].add(relation)
        
        # Convert to super-relations
        super_rel_id = 0
        for domain, rel_set in domain_clusters.items():
            if len(rel_set) > 0:  # Only create non-empty clusters
                self.super_relations[f"SR_{super_rel_id}"] = rel_set
                self.super_relation_names[f"SR_{super_rel_id}"] = domain
                
                # Map each base relation to its super-relation
                for rel in rel_set:
                    self.relation_to_super[rel] = f"SR_{super_rel_id}"
                
                super_rel_id += 1
    
    def _cluster_by_semantics(self, relations: List[str]) -> None:
        """
        Cluster relations by semantic similarity (simplified version)
        In full implementation, would use embeddings or LLM
        
        Args:
            relations: List of base relations
        """
        # Simplified semantic clustering based on keywords
        semantic_groups = {
            'temporal': ['date', 'time', 'year', 'born', 'death', 'start', 'end', 'founded'],
            'location': ['place', 'location', 'country', 'city', 'born', 'headquarters'],
            'identity': ['name', 'title', 'label', 'alias', 'nickname'],
            'membership': ['member', 'part', 'belongs', 'participated', 'involved'],
            'creation': ['created', 'written', 'directed', 'produced', 'founded', 'established'],
            'property': ['has', 'contains', 'includes', 'features'],
            'family': ['parent', 'child', 'spouse', 'sibling', 'relative', 'family'],
            'education': ['school', 'education', 'degree', 'studied', 'university'],
            'profession': ['occupation', 'profession', 'work', 'job', 'career', 'employed'],
            'measurement': ['height', 'weight', 'size', 'length', 'width', 'area', 'population']
        }
        
        # Assign relations to semantic groups
        relation_groups = defaultdict(set)
        unassigned = set()
        
        for relation in relations:
            assigned = False
            relation_lower = relation.lower()
            
            for group_name, keywords in semantic_groups.items():
                if any(keyword in relation_lower for keyword in keywords):
                    relation_groups[group_name].add(relation)
                    assigned = True
                    break
            
            if not assigned:
                # Group by domain as fallback
                domain = self.kg_interface.extract_domain(relation)
                relation_groups[f"domain_{domain}"].add(relation)
        
        # Convert to super-relations
        super_rel_id = 0
        for group_name, rel_set in relation_groups.items():
            if len(rel_set) > 0:
                self.super_relations[f"SR_{super_rel_id}"] = rel_set
                self.super_relation_names[f"SR_{super_rel_id}"] = group_name
                
                for rel in rel_set:
                    self.relation_to_super[rel] = f"SR_{super_rel_id}"
                
                super_rel_id += 1
    
    def _refine_with_semantics(self) -> None:
        """
        Refine domain-based clusters with semantic similarity
        Splits large clusters that contain semantically diverse relations
        """
        # This would use embeddings or LLM in full implementation
        # For now, just split very large clusters
        
        max_cluster_size = 50
        new_clusters = {}
        next_id = len(self.super_relations)
        
        for super_rel_id, rel_set in list(self.super_relations.items()):
            if len(rel_set) > max_cluster_size:
                # Split large cluster
                rel_list = list(rel_set)
                chunk_size = max_cluster_size // 2
                
                for i in range(0, len(rel_list), chunk_size):
                    chunk = set(rel_list[i:i+chunk_size])
                    new_id = f"SR_{next_id}"
                    new_clusters[new_id] = chunk
                    
                    # Update mappings
                    base_name = self.super_relation_names[super_rel_id]
                    self.super_relation_names[new_id] = f"{base_name}_{i//chunk_size}"
                    
                    for rel in chunk:
                        self.relation_to_super[rel] = new_id
                    
                    next_id += 1
                
                # Remove original large cluster
                del self.super_relations[super_rel_id]
                del self.super_relation_names[super_rel_id]
            else:
                new_clusters[super_rel_id] = rel_set
        
        self.super_relations = new_clusters
    
    def _compute_super_relation_scores(self) -> None:
        """
        Compute importance scores for super-relations
        Based on frequency and diversity of base relations
        """
        for super_rel_id, rel_set in self.super_relations.items():
            # Score based on cluster size and relation frequencies
            total_freq = sum(self.relation_frequencies.get(rel, 1) for rel in rel_set)
            cluster_size = len(rel_set)
            
            # Balance between frequency and diversity
            score = np.log1p(total_freq) * np.log1p(cluster_size)
            self.super_relation_scores[super_rel_id] = score
    
    def get_super_relation(self, base_relation: str) -> Optional[str]:
        """
        Get super-relation ID for a base relation
        
        Args:
            base_relation: Base relation ID
            
        Returns:
            Super-relation ID or None if not found
        """
        return self.relation_to_super.get(base_relation)
    
    def expand_super_relation(self, super_rel_id: str, top_k: int = 10) -> List[str]:
        """
        Expand super-relation to top-k base relations
        Implements ReKnoS-style expansion from abstract to concrete
        
        Args:
            super_rel_id: Super-relation ID
            top_k: Number of base relations to return
            
        Returns:
            List of base relation IDs
        """
        if super_rel_id not in self.super_relations:
            logger.warning(f"Super-relation {super_rel_id} not found")
            return []
        
        base_relations = list(self.super_relations[super_rel_id])
        
        # Score relations by frequency
        scored_relations = [
            (rel, self.relation_frequencies.get(rel, 1))
            for rel in base_relations
        ]
        
        # Sort by frequency
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        
        return [rel for rel, score in scored_relations[:top_k]]
    
    def get_super_relations_for_domain(self, domain: str, top_k: int = 5) -> List[str]:
        """
        Get top-k super-relations related to a domain
        
        Args:
            domain: Domain string (e.g., "people.person")
            top_k: Number of super-relations to return
            
        Returns:
            List of super-relation IDs
        """
        matching_super_rels = []
        
        for super_rel_id, name in self.super_relation_names.items():
            if domain in name or name in domain:
                score = self.super_relation_scores.get(super_rel_id, 0)
                matching_super_rels.append((super_rel_id, score))
        
        # Sort by score
        matching_super_rels.sort(key=lambda x: x[1], reverse=True)
        
        return [sr_id for sr_id, score in matching_super_rels[:top_k]]
    
    def compress_relation_paths(self, relation_paths: List[Tuple], 
                                 max_super_paths: int = 100) -> List[Tuple]:
        """
        Compress relation paths using super-relations
        Reduces search space by grouping similar paths
        
        Args:
            relation_paths: List of relation path tuples (1-hop, 2-hop, or 3-hop)
            max_super_paths: Maximum number of super-paths to return
            
        Returns:
            List of super-relation path tuples
        """
        if not relation_paths:
            return []
        
        # Detect path length
        path_length = len(relation_paths[0]) if relation_paths else 0
        
        super_paths = set()
        for path in relation_paths:
            super_path = tuple(
                self.relation_to_super.get(rel, f"UNKNOWN_{rel}")
                for rel in path
            )
            super_paths.add(super_path)
        
        super_paths = list(super_paths)[:max_super_paths]
        
        logger.debug(f"Compressed {len(relation_paths)} paths to {len(super_paths)} super-paths")
        
        return super_paths
    
    def expand_super_paths(self, super_paths: List[Tuple], 
                           top_k_per_path: int = 5) -> List[Tuple]:
        """
        Expand super-relation paths back to base relation paths
        Bidirectional reasoning: compress then expand
        
        Args:
            super_paths: List of super-relation path tuples
            top_k_per_path: Number of expansions per super-path
            
        Returns:
            List of base relation path tuples
        """
        expanded_paths = []
        
        for super_path in super_paths:
            # Get top-k base relations for each super-relation in path
            expansion_options = []
            
            for super_rel in super_path:
                if super_rel.startswith("UNKNOWN_"):
                    # Keep unknown relations as-is
                    base_rel = super_rel.replace("UNKNOWN_", "")
                    expansion_options.append([base_rel])
                else:
                    base_rels = self.expand_super_relation(super_rel, top_k=top_k_per_path)
                    expansion_options.append(base_rels if base_rels else [super_rel])
            
            # Generate combinations (Cartesian product)
            if len(super_path) == 1:
                for r1 in expansion_options[0][:top_k_per_path]:
                    expanded_paths.append((r1,))
            elif len(super_path) == 2:
                for r1 in expansion_options[0]:
                    for r2 in expansion_options[1]:
                        expanded_paths.append((r1, r2))
            elif len(super_path) == 3:
                for r1 in expansion_options[0]:
                    for r2 in expansion_options[1]:
                        for r3 in expansion_options[2]:
                            expanded_paths.append((r1, r2, r3))
        
        logger.debug(f"Expanded {len(super_paths)} super-paths to {len(expanded_paths)} base paths")
        
        return expanded_paths[:100]  # Limit total expansions
    
    def update_relation_frequencies(self, relation: str, count: int = 1) -> None:
        """
        Update frequency count for a relation
        
        Args:
            relation: Base relation ID
            count: Frequency increment
        """
        self.relation_frequencies[relation] += count
    
    def cluster_by_semantic_embeddings(self, relations: List[str],
                                       model_name: str = "all-MiniLM-L6-v2",
                                       similarity_threshold: float = 0.7) -> None:
        """
        Cluster relations using semantic embeddings (Phase C improvement)
        Uses sentence-transformers to group semantically similar relations
        across different domains
        
        Args:
            relations: List of base relation IDs
            model_name: Sentence-transformers model name
            similarity_threshold: Cosine similarity threshold for clustering
        """
        logger.info(f"Clustering {len(relations)} relations using semantic embeddings")
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to domain clustering")
            self._cluster_by_domain(relations)
            return
        
        # Load model
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}, using domain clustering")
            self._cluster_by_domain(relations)
            return
        
        # Convert relation IDs to readable text
        relation_texts = []
        for rel in relations:
            # Convert "people.person.date_of_birth" to "people person date of birth"
            text = rel.replace('.', ' ').replace('_', ' ')
            relation_texts.append(text)
        
        # Compute embeddings
        logger.debug("Computing embeddings...")
        embeddings = model.encode(relation_texts, show_progress_bar=False)
        
        # Compute pairwise similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Greedy clustering by similarity
        clusters = []
        assigned = set()
        
        for i, rel in enumerate(relations):
            if i in assigned:
                continue
            
            # Start new cluster with this relation
            cluster = {rel}
            assigned.add(i)
            
            # Find similar relations
            for j in range(i + 1, len(relations)):
                if j not in assigned and similarity_matrix[i, j] >= similarity_threshold:
                    cluster.add(relations[j])
                    assigned.add(j)
            
            clusters.append(cluster)
        
        # Convert to super-relations
        self.super_relations = {}
        self.relation_to_super = {}
        self.super_relation_names = {}
        
        for idx, cluster in enumerate(clusters):
            super_rel_id = f"SR_{idx}"
            self.super_relations[super_rel_id] = cluster
            
            # Generate name from most common words in cluster
            all_words = []
            for rel in cluster:
                words = rel.replace('.', ' ').replace('_', ' ').split()
                all_words.extend(words)
            
            # Most common word as cluster name
            from collections import Counter
            if all_words:
                common_words = Counter(all_words).most_common(2)
                name = '_'.join([word for word, _ in common_words])
            else:
                name = f"cluster_{idx}"
            
            self.super_relation_names[super_rel_id] = name
            
            # Map each relation to super-relation
            for rel in cluster:
                self.relation_to_super[rel] = super_rel_id
        
        logger.info(f"Created {len(clusters)} semantic clusters")
        logger.info(f"Average cluster size: {np.mean([len(c) for c in clusters]):.2f}")
    
    def get_super_relation_summary(self) -> Dict:
        """
        Get summary statistics of super-relation module
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'num_super_relations': len(self.super_relations),
            'num_base_relations': len(self.relation_to_super),
            'avg_cluster_size': np.mean([len(rels) for rels in self.super_relations.values()]),
            'max_cluster_size': max([len(rels) for rels in self.super_relations.values()], default=0),
            'min_cluster_size': min([len(rels) for rels in self.super_relations.values()], default=0),
            'top_super_relations': sorted(
                [(sr_id, self.super_relation_names[sr_id], self.super_relation_scores[sr_id]) 
                 for sr_id in self.super_relations.keys()],
                key=lambda x: x[2],
                reverse=True
            )[:10]
        }
