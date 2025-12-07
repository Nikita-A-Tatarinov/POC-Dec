"""
SORGA: Super-relation Ontology-guided Reasoning with Graph-constrained Attention
Core reasoning engine integrating GCR, ReKnoS, and ORT methodologies
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging
import time

from .kg_interface import KGInterface
from .ontology import OntologyModule
from .ontology_ort import OntologyModuleORT, Entity
from .reknos import ReKnoSModule, SuperRelationPath
from .super_relations import SuperRelationModule
from .trie import Trie, GraphConstrainedDecoding, build_trie_from_paths

logger = logging.getLogger(__name__)


class ReasoningState:
    """Represents the current state during reasoning"""
    
    def __init__(self, question: str, topic_entities: List[str]):
        self.question = question
        self.topic_entities = topic_entities
        self.current_entities = topic_entities.copy()
        self.reasoning_paths = []
        self.abstract_paths = []
        self.super_relation_paths = []
        self.visited_entities = set(topic_entities)
        self.expected_answer_types = []
        self.llm_calls = 0
        self.start_time = time.time()
        
    def add_path(self, path: Dict):
        """Add a reasoning path to state"""
        self.reasoning_paths.append(path)
        
    def add_entity(self, entity: str):
        """Add visited entity"""
        self.visited_entities.add(entity)
        if entity not in self.current_entities:
            self.current_entities.append(entity)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed reasoning time"""
        return time.time() - self.start_time


class SORGA:
    """
    Main SORGA reasoning engine
    Integrates multiple KG reasoning methodologies:
    - Ontology-guided planning (ORT-style reverse reasoning)
    - Super-relation expansion (ReKnoS-style relation clustering)
    - Graph-constrained path generation (GCR-style trie validation)
    - Answer synthesis with inductive reasoning
    """
    
    def __init__(self,
                 kg_interface: KGInterface,
                 ontology_module: OntologyModule,
                 super_relation_module: SuperRelationModule,
                 llm_interface,
                 tokenizer=None,
                 config: Optional[Dict] = None,
                 ontology_ort_module: Optional[OntologyModuleORT] = None,
                 reknos_module: Optional[ReKnoSModule] = None):
        """
        Initialize SORGA reasoning engine
        
        Args:
            kg_interface: Knowledge graph interface
            ontology_module: Basic ontology/type hierarchy module
            super_relation_module: Super-relation clustering module
            llm_interface: LLM interface for generation
            tokenizer: Tokenizer for constrained decoding
            config: Configuration dictionary
            ontology_ort_module: ORT-style ontology module (optional)
            reknos_module: ReKnoS-style super-relation module (optional)
        """
        self.kg_interface = kg_interface
        self.ontology_module = ontology_module
        self.ontology_ort_module = ontology_ort_module
        self.reknos_module = reknos_module
        self.super_relation_module = super_relation_module
        self.llm_interface = llm_interface
        self.tokenizer = tokenizer
        
        # Configuration
        self.config = config or {}
        self.max_hops = self.config.get('max_hops', 3)
        self.beam_width = self.config.get('beam_width', 5)
        self.top_k_super_relations = self.config.get('top_k_super_relations', 10)
        self.top_k_base_relations = self.config.get('top_k_base_relations', 5)
        self.use_constrained_decoding = self.config.get('use_constrained_decoding', True)
        self.max_reasoning_time = self.config.get('max_reasoning_time', 300)  # 5 minutes
        
        # Trie for constrained decoding (built dynamically)
        self.trie = None
        self.constrained_decoder = None
        
        logger.debug("SORGA initialized with config: %s", self.config)
    
    def answer_question(self, question: str, 
                       topic_entities: List[str],
                       answer_type: Optional[str] = None) -> Dict:
        """
        Answer a question using SORGA reasoning
        
        Args:
            question: Natural language question
            topic_entities: List of topic entity IDs from question
            answer_type: Expected answer type (optional)
            
        Returns:
            Dictionary with answer, reasoning paths, and metadata
        """
        logger.debug(f"Answering question: {question}")
        logger.debug(f"Topic entities: {topic_entities}")
        
        # Initialize reasoning state
        state = ReasoningState(question, topic_entities)
        
        # Get component enablement from config (for modular evaluation)
        enabled_components = self.config.get('enabled_components', {
            'ontology': True,
            'super_relation': True,
            'kg_trie': True,
            'synthesis': True
        })
        
        try:
            # Ontology-guided planning (optional)
            abstract_paths = []
            if enabled_components.get('ontology', True):
                abstract_paths = self._ontology_planning(
                    question, topic_entities, answer_type, state
                )
            else:
                logger.debug("Ontology planning disabled")
            
            # Super-relation expansion (optional)
            candidate_relation_paths = []
            if enabled_components.get('super_relation', True):
                candidate_relation_paths = self._super_relation_expansion(
                    question, topic_entities, abstract_paths, state
                )
            else:
                logger.debug("Super-relation expansion disabled")
            
            # Graph-constrained path generation (optional)
            concrete_paths = []
            if enabled_components.get('kg_trie', True):
                concrete_paths = self._constrained_generation(
                    question, topic_entities, candidate_relation_paths, state
                )
            else:
                logger.debug("KG-Trie generation disabled")
            
            # Answer synthesis (always enabled - required for answer)
            answer = self._answer_synthesis(
                question, concrete_paths, state
            )
            
            # Prepare result
            result = {
                'question': question,
                'answer': answer,
                'reasoning_paths': state.reasoning_paths,
                'abstract_paths': abstract_paths,
                'super_relation_paths': state.super_relation_paths,
                'visited_entities': list(state.visited_entities),
                'num_llm_calls': state.llm_calls,
                'reasoning_time': state.get_elapsed_time(),
                'success': True
            }
            
            logger.debug(f"Answer: {answer} (LLM calls: {state.llm_calls}, Time: {state.get_elapsed_time():.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during reasoning: {e}", exc_info=True)
            return {
                'question': question,
                'answer': None,
                'error': str(e),
                'success': False,
                'num_llm_calls': state.llm_calls,
                'reasoning_time': state.get_elapsed_time()
            }
    
    def _ontology_planning(self, question: str,
                           topic_entities: List[str],
                           answer_type: Optional[str],
                           state: ReasoningState) -> List[List[str]]:
        """
        Ontology-guided planning: Generate abstract type-level paths from question intent.
        Uses ORT-style reverse reasoning when available.
        
        Args:
            question: Natural language question
            topic_entities: Topic entities
            answer_type: Expected answer type
            state: Reasoning state
            
        Returns:
            List of abstract type-level paths
        """
        logger.debug("Starting ontology-guided planning")
        
        # Check if ORT-style module is available
        use_ort = self.config.get('use_ort', False)
        
        if use_ort and self.ontology_ort_module:
            logger.info("Using ORT-style ontology reasoning WITH GUIDED KG TRAVERSAL")
            
            # Convert topic entities to Entity objects
            topic_entity_objects = []
            for entity_id in topic_entities:
                entity_name = self.kg_interface.get_entity_name(entity_id)
                entity_types = self.kg_interface.get_entity_types(entity_id)
                topic_entity_objects.append(Entity(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    labels=entity_types
                ))
            
            # Generate label reasoning paths using full ORT algorithm
            # NOW RETURNS: (label_reasoning_paths, condition_entities)
            max_hop = self.config.get('ontology', {}).get('max_hop', 3)
            label_reasoning_paths, condition_entities = self.ontology_ort_module.generate_label_reasoning_paths(
                question, topic_entity_objects, max_hop
            )
            
            # Store abstract paths for reference
            abstract_paths = [path.labels for path in label_reasoning_paths]
            state.abstract_paths = abstract_paths
            
            # Track LLM calls (ORT makes 2 calls: aim recognition + semantic filtering)
            state.llm_calls += 2
            
            # Store target types from aims
            if label_reasoning_paths:
                # Last label in each path is the aim (target)
                state.expected_answer_types = list(set(
                    path.labels[-1] for path in label_reasoning_paths if path.labels
                ))
            
            # CRITICAL NEW STEP: Use guided KG traversal (Section 2.3 of paper)
            # This is what was missing - we need to USE the label paths to guide KG queries!
            logger.info(f"Performing guided KG traversal with {len(label_reasoning_paths)} label paths")
            
            max_entities_per_step = self.config.get('ontology', {}).get('max_entities_per_step', 20)
            guided_entity_paths = self.ontology_ort_module.guided_kg_traversal(
                label_reasoning_paths,
                condition_entities,
                max_entities_per_step=max_entities_per_step
            )
            
            # Store guided entity paths in state (will be used in answer synthesis)
            state.guided_entity_paths = guided_entity_paths
            
            logger.info(f"ORT complete: {len(abstract_paths)} abstract paths, {len(guided_entity_paths)} concrete entity paths")
            
            # Return abstract paths for compatibility, but guided_entity_paths are stored in state
            return abstract_paths
        
        # Fallback: Basic ontology implementation
        logger.debug("Using basic ontology implementation")
        
        # Get types for topic entities
        source_types = []
        for entity in topic_entities:
            entity_types = self.kg_interface.get_entity_types(entity)
            source_types.extend(entity_types)
        
        logger.debug(f"Source types: {source_types[:5]}...")
        
        # Infer target types from question or answer_type
        target_types = self._infer_target_types(question, answer_type)
        state.expected_answer_types = target_types
        logger.info(f"Expected answer types: {target_types}")
        
        # Generate abstract paths using ontology (standard forward reasoning)
        abstract_paths = self.ontology_module.generate_abstract_path(
            source_types, target_types, max_hops=self.max_hops
        )
        
        # Use LLM to rank abstract paths by relevance to question
        if abstract_paths and self.llm_interface:
            state.llm_calls += 1
            ranked_paths = self._rank_abstract_paths_with_llm(
                question, abstract_paths
            )
            abstract_paths = ranked_paths[:self.beam_width]
        
        logger.debug(f"Generated {len(abstract_paths)} abstract paths")
        state.abstract_paths = abstract_paths
        
        return abstract_paths
    
    def _super_relation_expansion(self, question: str,
                                   topic_entities: List[str],
                                   abstract_paths: List[List[str]],
                                   state: ReasoningState) -> List[Tuple]:
        """
        Super-relation expansion: Expand abstract paths to candidate relation paths.
        Uses ReKnoS-style relation clustering and bidirectional reasoning when available.
        
        Args:
            question: Natural language question
            topic_entities: Topic entities
            abstract_paths: Abstract type-level paths
            state: Reasoning state
            
        Returns:
            List of candidate relation path tuples
        """
        logger.debug("Starting super-relation expansion")
        
        # Check if ReKnoS-style module is available
        use_reknos = self.config.get('use_reknos', False)
        
        if use_reknos and self.reknos_module:
            logger.info("Using ReKnoS-style super-relation reasoning")
            return self._reknos_reasoning(question, topic_entities, state)
        
        # Fallback: Basic super-relation implementation
        logger.debug("Using basic super-relation implementation")
        
        all_candidate_paths = []
        
        # Check if multi-hop backward is enabled (optional improvement)
        improvements = self.config.get('improvements', {})
        use_multihop_backward = improvements.get('multihop_backward', False)
        
        for entity in topic_entities:
            # Get multi-hop relations from KG with bidirectional reasoning
            # Answers may connect backward (tail->entity) in addition to forward
            forward_1hop = self.kg_interface.get_1hop_relations(entity, direction="head")
            backward_1hop = self.kg_interface.get_1hop_relations(entity, direction="tail")
            relations_1hop = forward_1hop + backward_1hop
            
            relations_2hop = self.kg_interface.get_2hop_relations(entity)
            relations_3hop = self.kg_interface.get_3hop_relations(entity) if self.max_hops >= 3 else []
            
            # Optional: Add multi-hop backward relations
            if use_multihop_backward:
                backward_2hop = self.kg_interface.get_2hop_backward_relations(entity)
                backward_3hop = self.kg_interface.get_3hop_backward_relations(entity) if self.max_hops >= 3 else []
                relations_2hop = list(set(relations_2hop + backward_2hop))
                relations_3hop = list(set(relations_3hop + backward_3hop))
                logger.debug(f"Added {len(backward_2hop)} 2-hop backward, {len(backward_3hop)} 3-hop backward relations")
            
            logger.debug(f"Entity {entity}: {len(forward_1hop)} forward, {len(backward_1hop)} backward, "
                        f"{len(relations_2hop)} 2-hop, {len(relations_3hop)} 3-hop relations")
            
            # Compress to super-relations
            super_paths_1hop = self._compress_to_super_relations([(r,) for r in relations_1hop])
            super_paths_2hop = self._compress_to_super_relations(relations_2hop)
            super_paths_3hop = self._compress_to_super_relations(relations_3hop)
            
            # Prioritize 1-hop paths (usually most relevant), then 2-hop, then 3-hop
            # This ensures simpler, more direct paths are presented first to LLM
            all_super_paths = super_paths_1hop + super_paths_2hop + super_paths_3hop
            
            # Use LLM to select relevant super-relations guided by abstract paths and expected types
            state.llm_calls += 1
            selected_super_paths = self._select_super_relations_with_llm(
                question, abstract_paths, 
                all_super_paths,
                state.expected_answer_types
            )
            
            state.super_relation_paths.extend(selected_super_paths)
            
            # Expand super-relations back to base relations
            for super_path in selected_super_paths:
                base_paths = self.super_relation_module.expand_super_paths(
                    [super_path], top_k_per_path=self.top_k_base_relations
                )
                all_candidate_paths.extend(base_paths)
        
        # Deduplicate
        all_candidate_paths = list(set(all_candidate_paths))
        
        logger.debug(f"Expanded to {len(all_candidate_paths)} candidate relation paths")
        
        return all_candidate_paths[:100]  # Limit for efficiency
    
    def _reknos_reasoning(self, question: str, topic_entities: List[str], 
                          state: ReasoningState) -> List[Tuple]:
        """
        ReKnoS-style super-relation reasoning.
        Uses LLM-based scoring, width-N search, and iterative querying.
        
        Args:
            question: Natural language question
            topic_entities: List of topic entity IDs
            state: Reasoning state
            
        Returns:
            List of candidate relation path tuples
        """
        logger.info("Starting ReKnoS-style reasoning")
        
        # Build super-relations if not already built
        if len(self.reknos_module.super_relations) == 0:
            logger.info("Building super-relations from KG")
            all_relations = self.kg_interface.get_all_relations()
            self.reknos_module.build_super_relations(all_relations)
        
        # Get entity names for reasoning
        topic_entities_with_names = []
        for entity_id in topic_entities:
            entity_name = self.kg_interface.get_entity_name(entity_id)
            if entity_name:
                topic_entities_with_names.append((entity_id, entity_name))
            else:
                topic_entities_with_names.append((entity_id, entity_id))
        
        # Run ReKnoS reasoning
        # This performs iterative LLM-guided search with super-relations
        paths = self.reknos_module.reason(question, topic_entities_with_names)
        
        # Track LLM calls
        # ReKnoS uses 2L+1 calls per question (2 per step + 1 final)
        max_length = self.reknos_module.max_length
        state.llm_calls += (2 * max_length + 1)
        
        logger.info(f"ReKnoS found {len(paths)} super-relation paths")
        
        # Extract candidate relation paths from ReKnoS paths
        candidate_paths = []
        for path in paths:
            # Each SuperRelationPath contains super_relations (list of SuperRelation objects)
            # Convert to base relation tuples
            for sr in path.super_relations:
                # Each super-relation contains multiple base relations
                # Add each base relation as a 1-tuple
                for base_rel in sr.base_relations:
                    candidate_paths.append((base_rel,))
        
        # Deduplicate
        candidate_paths = list(set(candidate_paths))
        
        logger.info(f"Expanded to {len(candidate_paths)} candidate base relation paths")
        
        return candidate_paths[:100]  # Limit for efficiency
    
    def _constrained_generation(self, question: str,
                                topic_entities: List[str],
                                candidate_paths: List[Tuple],
                                state: ReasoningState) -> List[Dict]:
        """
        Graph-constrained path generation: Generate concrete entity paths using KG constraints.
        Uses GCR-style trie validation to ensure paths exist in the knowledge graph.
        
        Args:
            question: Natural language question
            topic_entities: Topic entities
            candidate_paths: Candidate relation paths
            state: Reasoning state
            
        Returns:
            List of concrete reasoning paths with entities
        """
        logger.debug("Starting graph-constrained path generation")
        
        concrete_paths = []
        
        # Build trie from candidate paths
        if self.use_constrained_decoding and self.tokenizer:
            self._build_trie_from_relation_paths(candidate_paths)
        
        for entity in topic_entities:
            entity_name = self.kg_interface.get_entity_name(entity)
            
            for relation_path in candidate_paths[:50]:  # Limit per entity
                # Execute path on KG
                path_result = self._execute_relation_path(entity, relation_path)
                
                if path_result['success']:
                    path_dict = {
                        'topic_entity': entity,
                        'topic_entity_name': entity_name,
                        'relation_path': relation_path,
                        'intermediate_entities': path_result.get('intermediates', []),
                        'answer_entities': path_result.get('answers', []),
                        'path_string': path_result.get('path_string', ''),
                        'valid': True
                    }
                    concrete_paths.append(path_dict)
                    state.add_path(path_dict)
                    
                    # Add entities to state
                    for e in path_result.get('intermediates', []) + path_result.get('answers', []):
                        state.add_entity(e)
                
                # Check timeout
                if state.get_elapsed_time() > self.max_reasoning_time:
                    logger.warning("Max reasoning time exceeded")
                    break
        
        # Use LLM with constrained decoding for verification
        if self.use_constrained_decoding and concrete_paths and self.constrained_decoder:
            state.llm_calls += 1
            verified_paths = self._verify_paths_with_constrained_llm(
                question, concrete_paths[:20]
            )
            concrete_paths = verified_paths + concrete_paths[20:]
        
        logger.debug(f"Generated {len(concrete_paths)} concrete paths")
        
        return concrete_paths
    
    def _answer_synthesis(self, question: str,
                          concrete_paths: List[Dict],
                          state: ReasoningState) -> Optional[str]:
        """
        Answer synthesis: Synthesize final answer from concrete paths using inductive reasoning.
        Uses LLM to extract and rank answer candidates from multiple reasoning paths.
        
        CRITICAL CHANGE: Prioritize ORT-guided entity paths when available.
        These paths are type-constrained and much higher quality than generic paths.
        
        Args:
            question: Natural language question
            concrete_paths: Concrete reasoning paths (from GCR/ReKnoS)
            state: Reasoning state
            
        Returns:
            Final answer string
        """
        logger.debug("Starting answer synthesis")
        
        # CRITICAL: Check if we have ORT-guided entity paths (Section 2.3 implementation)
        paths_to_use = concrete_paths
        paths_source = "traditional"
        
        if hasattr(state, 'guided_entity_paths') and state.guided_entity_paths:
            logger.info(f"Using {len(state.guided_entity_paths)} ORT-guided entity paths (type-constrained)")
            paths_to_use = state.guided_entity_paths
            paths_source = "ort_guided"
        else:
            logger.info(f"Using {len(concrete_paths)} traditional concrete paths")
        
        if not paths_to_use:
            logger.warning("No paths available for answer synthesis")
            return None
        
        # Collect answer candidates from paths with quality scoring
        answer_scores = defaultdict(float)
        answer_evidence = defaultdict(list)
        answer_paths = defaultdict(list)
        
        if paths_source == "ort_guided":
            # ORT-guided paths: use answer_entity directly
            for path in paths_to_use:
                answer_entity = path.get('answer_entity')
                if answer_entity:
                    # ORT paths are pre-filtered by type, give them high base score
                    path_score = path.get('path_score', 1.0) * 2.0  # Boost ORT paths
                    answer_scores[answer_entity] += path_score
                    
                    # Build evidence string from entity path
                    entity_path = path.get('entity_path', [])
                    label_path = path.get('label_path', [])
                    evidence_str = f"Path: {' -> '.join(entity_path[:3])}... (types: {' -> '.join(label_path)})"
                    answer_evidence[answer_entity].append(evidence_str)
                    answer_paths[answer_entity].append(path)
        else:
            # Traditional paths: extract from answer_entities
            for path in paths_to_use:
                # Calculate path quality score
                path_score = self._calculate_path_score(path, state.expected_answer_types)
                
                for answer_entity in path.get('answer_entities', []):
                    # Weighted scoring: path quality + frequency
                    answer_scores[answer_entity] += path_score
                    answer_evidence[answer_entity].append(path['path_string'])
                    answer_paths[answer_entity].append(path)
        
        # Rank by weighted score (not just frequency)
        ranked_answers = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Top answer candidates (scored): {[(ans, f'{score:.2f}') for ans, score in ranked_answers[:5]]}")
        
        # Use LLM for final answer synthesis with inductive reasoning
        state.llm_calls += 1
        final_answer = self._synthesize_answer_with_llm(
            question, ranked_answers, answer_evidence, concrete_paths
        )
        
        return final_answer
    
    def _infer_target_types(self, question: str, answer_type: Optional[str]) -> List[str]:
        """Infer expected answer types from question using LLM"""
        if answer_type:
            return [answer_type]
        
        # Use LLM to classify question and infer answer type
        prompt = f"""Analyze this question and determine what TYPE of answer is expected.

Question: "{question}"

Common answer types:
- PERSON: Questions about people (who, which person, etc.)
- LOCATION: Questions about places (where, which country/city, etc.)
- TIME: Questions about dates/time (when, what year, what date, etc.)
- ORGANIZATION: Questions about companies, institutions
- CURRENCY: Questions about money/currency
- LANGUAGE: Questions about languages
- PROFESSION: Questions about jobs/occupations
- TITLE: Questions about positions/titles (governor, president, etc.)
- EVENT: Questions about events/occurrences
- QUANTITY: Questions about numbers (how many, how much)
- OBJECT: Questions about things/objects
- OTHER: Other types

Instructions:
1. Identify the question word (who/what/where/when/how/which)
2. Look for type-specific keywords (currency, language, born, married, etc.)
3. Consider what the question is really asking for

Examples:
Q: "who plays ken barlow in coronation street" → PERSON
Q: "what language do jamaican people speak" → LANGUAGE
Q: "where is jamarcus russell from" → LOCATION
Q: "when did andy murray start playing tennis professionally" → TIME
Q: "what currency does jamaica use" → CURRENCY
Q: "what did james k polk do before he was president" → TITLE (not PROFESSION)
Q: "who was richard nixon married to" → PERSON

Your turn - respond with ONE TYPE ONLY:
Answer type:"""
        
        try:
            response = self.llm_interface.generate(prompt, max_tokens=20, temperature=0)
            answer_type_raw = response.strip().upper()
            
            # Map to Freebase types
            type_mapping = {
                'PERSON': ['people.person'],
                'LOCATION': ['location.location', 'location.country', 'location.citytown'],
                'TIME': ['time.event', 'type.datetime'],
                'ORGANIZATION': ['organization.organization', 'business.employer'],
                'CURRENCY': ['finance.currency'],
                'LANGUAGE': ['language.human_language'],
                'PROFESSION': ['people.profession'],
                'TITLE': ['government.government_office_or_title', 'government.government_position_held'],
                'EVENT': ['time.event'],
                'QUANTITY': ['type.int', 'type.float'],
                'OBJECT': ['common.topic'],
                'OTHER': ['type.object']
            }
            
            # Find matching type
            inferred_types = []
            for key, freebase_types in type_mapping.items():
                if key in answer_type_raw:
                    inferred_types.extend(freebase_types)
                    logger.debug(f"Question type: {key} → {freebase_types}")
                    break
            
            if not inferred_types:
                # Fallback to heuristic
                inferred_types = self._heuristic_type_inference(question)
            
            return inferred_types
            
        except Exception as e:
            logger.warning(f"LLM type inference failed: {e}, using heuristic fallback")
            return self._heuristic_type_inference(question)
    
    def _heuristic_type_inference(self, question: str) -> List[str]:
        """Fallback heuristic-based type inference"""
        question_lower = question.lower()
        
        # More comprehensive type keywords with priority order
        type_rules = [
            (['currency', 'dollar', 'pound', 'euro', 'money'], ['finance.currency']),
            (['language', 'speak', 'spoken'], ['language.human_language']),
            (['when', 'date', 'year', 'born', 'died', 'start', 'begin'], ['time.event', 'type.datetime']),
            (['where', 'place', 'country', 'city', 'location', 'from'], ['location.location']),
            (['who', 'person', 'people', 'actor', 'director', 'author', 'married'], ['people.person']),
            (['what did', 'what does', 'occupation', 'job', 'profession', 'position', 'title'], 
             ['government.government_office_or_title', 'people.profession']),
            (['how many', 'how much', 'count', 'number'], ['type.int']),
            (['company', 'organization', 'institution'], ['organization.organization']),
        ]
        
        for keywords, types in type_rules:
            if any(kw in question_lower for kw in keywords):
                return types
        
        # Default
        return ['type.object']
    
    def _calculate_path_score(self, path: Dict, expected_types: List[str]) -> float:
        """
        Calculate quality score for a reasoning path.
        Simple scoring based on path length and validity.
        
        Args:
            path: Reasoning path dictionary
            expected_types: Expected answer types
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Simple scoring: shorter paths are better
        hop_count = len(path.get('relation_path', []))
        if hop_count == 1:
            score = 1.0
        elif hop_count == 2:
            score = 0.7
        elif hop_count == 3:
            score = 0.5
        else:
            score = 0.3
        
        # Boost valid paths
        if path.get('valid', True):
            score *= 1.2
        
        return min(1.0, score)
    
    def _compress_to_super_relations(self, relation_paths: List[Tuple]) -> List[Tuple]:
        """Compress relation paths to super-relation paths"""
        return self.super_relation_module.compress_relation_paths(relation_paths)
    
    def _execute_relation_path(self, entity: str, relation_path: Tuple) -> Dict:
        """
        Execute a relation path on the KG to get concrete entities
        
        Args:
            entity: Starting entity
            relation_path: Tuple of relations
            
        Returns:
            Dictionary with execution results
        """
        try:
            path_length = len(relation_path)
            
            if path_length == 1:
                answers = self.kg_interface.get_entities_via_relation(entity, relation_path[0])
                return {
                    'success': True,
                    'intermediates': [],
                    'answers': answers,
                    'path_string': f"{entity} -> {relation_path[0]} -> {answers[:3]}"
                }
            
            elif path_length == 2:
                entity_pairs = self.kg_interface.get_2hop_entities(entity, relation_path[0], relation_path[1])
                intermediates = [e1 for e1, e2 in entity_pairs]
                answers = [e2 for e1, e2 in entity_pairs]
                return {
                    'success': True,
                    'intermediates': intermediates,
                    'answers': answers,
                    'path_string': f"{entity} -> {relation_path[0]} -> {intermediates[:2]} -> {relation_path[1]} -> {answers[:3]}"
                }
            
            elif path_length == 3:
                entity_triples = self.kg_interface.get_3hop_entities(
                    entity, relation_path[0], relation_path[1], relation_path[2]
                )
                intermediates = [e1 for e1, e2, e3 in entity_triples] + [e2 for e1, e2, e3 in entity_triples]
                answers = [e3 for e1, e2, e3 in entity_triples]
                return {
                    'success': True,
                    'intermediates': intermediates,
                    'answers': answers,
                    'path_string': f"{entity} -> {relation_path[0]} -> ... -> {relation_path[2]} -> {answers[:3]}"
                }
            
            else:
                return {'success': False, 'error': 'Path too long'}
        
        except Exception as e:
            logger.error(f"Error executing path {relation_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_trie_from_relation_paths(self, relation_paths: List[Tuple]):
        """Build trie from relation paths for constrained decoding"""
        if not self.tokenizer:
            return
        
        path_strings = []
        for path in relation_paths:
            path_str = " -> ".join(path)
            path_strings.append(path_str)
        
        self.trie = build_trie_from_paths(path_strings, self.tokenizer)
        self.constrained_decoder = GraphConstrainedDecoding(
            self.tokenizer, self.trie, enable_constrained_by_default=True
        )
    
    # LLM interaction methods (to be implemented with actual LLM interface)
    
    def _rank_abstract_paths_with_llm(self, question: str, paths: List[List[str]]) -> List[List[str]]:
        """Rank abstract paths using LLM"""
        # Placeholder - implement with actual LLM
        return paths
    
    def _select_super_relations_with_llm(self, question: str, abstract_paths: List, super_paths: List, expected_types: List[str] = None) -> List:
        """
        LLM-guided super-relation scoring and selection.
        
        Args:
            question: Natural language question
            abstract_paths: Abstract type-level paths (may be empty)
            super_paths: List of super-relation path tuples
            expected_types: Expected answer types
        
        Returns:
            Top-K selected super-relation paths, sorted by LLM scores
        """
        if not super_paths:
            return []
        
        # Use scoring approach for better ranking
        use_scoring = len(super_paths) <= 30  # Only score if manageable number
        
        if use_scoring:
            return self._score_super_relations_with_llm(question, super_paths, expected_types)
        else:
            # Fall back to selection for large sets
            return self._select_super_relations_fallback(question, super_paths, expected_types)
    
    def _score_super_relations_with_llm(self, question: str, super_paths: List, expected_types: List[str] = None) -> List:
        """
        Score super-relations 1-10 using LLM.
        
        Args:
            question: Natural language question
            super_paths: List of super-relation path tuples
            expected_types: Expected answer types
            
        Returns:
            Sorted list of super-relations by score (highest first)
        """
        # Format expected types
        expected_type_str = ""
        if expected_types:
            type_names = [t.split('.')[-1].replace('_', ' ').title() for t in expected_types[:3]]
            expected_type_str = f"\nExpected answer type: {', '.join(type_names)}\n"
        
        # Format paths with readable names
        path_descriptions = []
        for i, path in enumerate(super_paths):
            if isinstance(path, tuple):
                readable_rels = []
                for rel in path:
                    key_part = rel.split('.')[-1] if '.' in rel else rel
                    readable = key_part.replace('_', ' ')
                    readable_rels.append(readable)
                path_str = " → ".join(readable_rels)
                hop_count = len(path)
            else:
                path_str = str(path)
                hop_count = 1
            path_descriptions.append(f"{i+1}. ({hop_count}-hop) {path_str}")
        
        # Create scoring prompt (ReKnoS-style)
        prompt = f"""Rate each knowledge graph path for answering this question on a scale of 1-10.
- 10: Directly answers the question
- 7-9: Very relevant, likely helpful
- 4-6: Somewhat relevant
- 1-3: Irrelevant

Question: "{question}"{expected_type_str}

Paths to score:
{chr(10).join(path_descriptions)}

Provide scores in format: "1:X, 2:Y, 3:Z" where X,Y,Z are scores 1-10.

Scores:"""
        
        try:
            response = self.llm_interface.generate(prompt, max_tokens=200, temperature=0)
            
            # Parse scores from response "1:8, 2:6, 3:9" format
            scores = {}
            for part in response.replace('\n', ',').split(','):
                if ':' in part:
                    try:
                        idx_str, score_str = part.split(':')
                        idx = int(idx_str.strip()) - 1  # Convert to 0-indexed
                        score = float(score_str.strip())
                        if 0 <= idx < len(super_paths) and 1 <= score <= 10:
                            scores[idx] = score
                    except (ValueError, IndexError):
                        continue
            
            # If no scores parsed, return top-K
            if not scores:
                logger.warning("LLM scoring parsing failed, using fallback")
                return super_paths[:self.top_k_super_relations]
            
            # Sort paths by score (highest first)
            scored_paths = [(super_paths[idx], scores.get(idx, 0)) for idx in range(len(super_paths))]
            scored_paths.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-K highest scored paths
            selected = [path for path, score in scored_paths[:self.top_k_super_relations] if score >= 4]
            logger.debug(f"LLM scored {len(scores)} paths, selected {len(selected)} with score ≥4")
            
            return selected if selected else [path for path, _ in scored_paths[:self.top_k_super_relations]]
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}, using fallback")
            return super_paths[:self.top_k_super_relations]
    
    def _select_super_relations_fallback(self, question: str, super_paths: List, expected_types: List[str] = None) -> List:
        """Fallback selection method for large super-relation sets"""
        # Format relation paths for LLM with better readability
        path_descriptions = []
        for i, path in enumerate(super_paths[:100]):  # Increased from 50 to 100 for better coverage
            # Convert tuple of relations to readable format
            if isinstance(path, tuple):
                # Make relations more readable by replacing underscores and dots
                readable_rels = []
                for rel in path:
                    # Extract the key part: "location.country.languages_spoken" -> "languages spoken"
                    if '.' in rel:
                        parts = rel.split('.')
                        key_part = parts[-1] if len(parts) > 1 else rel
                    else:
                        key_part = rel
                    # Replace underscores with spaces
                    readable = key_part.replace('_', ' ')
                    readable_rels.append(readable)
                
                path_str = " -> ".join(readable_rels)
                domain = path[0].split('.')[0] if '.' in path[0] else "general"
            else:
                path_str = str(path)
                domain = "general"
            
            # Show hop count
            hop_count = len(path) if isinstance(path, tuple) else 1
            path_descriptions.append(f"{i}. [{domain}, {hop_count}-hop] {path_str}")
        
        # Format expected types for prompt
        expected_type_str = ""
        if expected_types:
            type_names = []
            for t in expected_types:
                if 'person' in t:
                    type_names.append("PERSON")
                elif 'location' in t or 'country' in t or 'city' in t:
                    type_names.append("LOCATION")
                elif 'time' in t or 'date' in t:
                    type_names.append("TIME/DATE")
                elif 'currency' in t:
                    type_names.append("CURRENCY")
                elif 'language' in t:
                    type_names.append("LANGUAGE")
                elif 'government' in t or 'title' in t:
                    type_names.append("GOVERNMENT POSITION/TITLE")
                elif 'profession' in t:
                    type_names.append("PROFESSION")
                else:
                    type_names.append(t.split('.')[-1].upper())
            expected_type_str = f"\n**Expected answer type: {', '.join(type_names)}**\n"
        
        prompt = f"""You are analyzing knowledge graph paths to find the most relevant ones for answering a question. Select paths that DIRECTLY lead to the answer.

Question: "{question}"{expected_type_str}

Available paths:
{chr(10).join(path_descriptions)}

Instructions:
1. PRIORITIZE paths that lead to the expected answer type above.
2. Identify what type of answer the question seeks:
   - "what language/religion/currency" → look for those specific relations
   - "what did X do" or "what was X's job/position" → prefer "government positions held", "offices held" over generic "profession"
   - "where is/was X from/located/born" → look for "place of birth", "location", "headquarters"
   - "when did" → look for "date", "year", "time" relations
   - "who is/was" → look for "people", "person" relations

2. For "what did X do" questions, prioritize SPECIFIC over GENERIC:
   - PREFER: "government positions held", "offices held", "career" (specific roles)
   - AVOID: "profession", "occupation" (too generic like "politician", "writer")

3. Consider path length:
   - 1-hop is best for simple factual questions
   - 2-hop may be needed for complex questions (e.g., "positions held -> title")
   - 3-hop for very specific multi-step reasoning

4. Select the top {min(self.top_k_super_relations, len(super_paths))} paths that best match the question intent.

Output ONLY the path numbers as comma-separated integers (e.g., "3, 7, 12, 25, 41").

Selected paths:"""
        
        try:
            response = self.llm_interface.generate(prompt, max_tokens=50, temperature=0)
            
            # Parse selected indices
            selected_indices = []
            for token in response.replace(',', ' ').split():
                try:
                    idx = int(token.strip())
                    if 0 <= idx < len(super_paths):
                        selected_indices.append(idx)
                except ValueError:
                    continue
            
            # If parsing failed or no selections, fall back to top-K
            if not selected_indices:
                logger.warning("LLM selection parsing failed, using top-K fallback")
                return super_paths[:self.top_k_super_relations]
            
            # Return selected paths (limit to top_k)
            selected_paths = [super_paths[i] for i in selected_indices[:self.top_k_super_relations]]
            logger.debug(f"LLM selected {len(selected_paths)} relation paths out of {len(super_paths)}")
            
            return selected_paths
            
        except Exception as e:
            logger.error(f"LLM selection failed: {e}, using top-K fallback")
            return super_paths[:self.top_k_super_relations]
    
    def _verify_paths_with_constrained_llm(self, question: str, paths: List[Dict]) -> List[Dict]:
        """Verify paths using LLM with constrained decoding"""
        # Placeholder
        return paths
    
    def _synthesize_answer_with_llm(self, question: str, ranked_answers: List, 
                                    evidence: Dict, paths: List[Dict]) -> Optional[str]:
        """
        LLM-guided answer synthesis with inductive reasoning.
        
        Args:
            question: Natural language question
            ranked_answers: List of (entity_id, score) tuples
            evidence: Dict mapping entity_id to supporting path strings
            paths: List of concrete reasoning path dictionaries
        
        Returns:
            Final answer string
        """
        if not paths:
            logger.warning("No reasoning paths available for answer synthesis")
            if ranked_answers:
                top_entity = ranked_answers[0][0]
                return self.kg_interface.get_entity_name(top_entity)
            return None
        
        # Format top paths with their answer entities
        path_descriptions = []
        for i, path in enumerate(paths[:15]):  # Top 15 paths
            # Check if this is an ORT-guided path or traditional path
            if path.get('path_type') == 'ort_guided':
                # ORT-guided path format
                answer_entity = path.get('answer_entity')
                answer_name = path.get('answer_entity_name', self.kg_interface.get_entity_name(answer_entity))
                label_path = path.get('label_path', [])
                entity_path = path.get('entity_path', [])
                
                # Get readable names for entity path
                entity_names = []
                for eid in entity_path[:4]:  # First 4 entities
                    name = self.kg_interface.get_entity_name(eid)
                    if name != "UnknownEntity":
                        entity_names.append(name)
                
                path_descriptions.append(
                    f"{i+1}. Type path: {' -> '.join(label_path)} | Entity path: {' -> '.join(entity_names)} | Answer: {answer_name}"
                )
            else:
                # Traditional path format
                topic = path.get('topic_entity_name', path.get('topic_entity', 'Unknown'))
                relations = ' -> '.join(path.get('relation_path', []))
                answers = path.get('answer_entities', [])
                
                if not answers:
                    continue
                
                # Get human-readable names for answer entities (first 5)
                answer_names = []
                for entity_id in answers[:5]:
                    name = self.kg_interface.get_entity_name(entity_id)
                    # Use entity name if available, otherwise use entity ID (better than skipping)
                    if name != "UnknownEntity":
                        answer_names.append(name)
                    else:
                        # Use entity ID as fallback (still informative for LLM)
                        answer_names.append(entity_id)
                
                # Include path even if some names unresolved (LLM can still work with IDs)
                if answer_names:
                    path_descriptions.append(
                        f"{i+1}. {topic} -> {relations} -> [{', '.join(answer_names)}{'...' if len(answers) > 5 else ''}]"
                    )
        
        if not path_descriptions:
            # Fallback to simple frequency-based selection
            logger.warning("No valid path descriptions, using frequency fallback")
            if ranked_answers:
                top_entity = ranked_answers[0][0]
                return self.kg_interface.get_entity_name(top_entity)
            return None
        
        prompt = f"""Given a question and reasoning paths from a knowledge graph, extract the MOST SPECIFIC answer(s).

# Question:
{question}

# Reasoning paths (format: topic -> relations -> [answer entities]):
{chr(10).join(path_descriptions)}

# Instructions:
1. Identify which paths are most relevant to answering the question
2. Extract the answer entities from those paths
3. Apply these rules for specificity:
   - For "where" questions: return the MOST SPECIFIC location (e.g., "Paris" not "Paris, France, Europe")
   - For "what did X do" questions: return SPECIFIC positions/roles (e.g., "Governor of Texas") NOT generic professions (e.g., "Politician")
   - For person questions: return the person's name, not descriptions
   - For time questions: match the granularity asked (year if asked "what year", full date if asked "when exactly")
4. If multiple distinct answers exist, list them separated by commas
5. Do NOT include entity IDs (m.xxxxx), path numbers, or explanations

# Few-shot examples:
Q: "what language do jamaican people speak"
Paths: Jamaica -> languages spoken -> [Jamaican English, Jamaican Creole English Language]
A: Jamaican English, Jamaican Creole English Language

Q: "where is Samsung based"
Paths: Samsung -> headquarters location -> [Suwon, South Korea, Gyeonggi Province]
A: Suwon

Q: "what did Abraham Lincoln do before he was president"
Paths: 
1. Abraham Lincoln -> profession -> [Politician, Lawyer]
2. Abraham Lincoln -> government positions held -> basic title -> [United States Representative]
A: United States Representative

# Your turn:
Answer:"""
        
        try:
            response = self.llm_interface.generate(prompt, max_tokens=150, temperature=0)
            
            # Clean up response
            answer = response.strip()
            
            # Remove common prefixes and explanations
            prefixes_to_remove = [
                "Answer:", "The answer is", "Based on the paths,", "Based on",
                "The most relevant", "Looking at", "From the paths",
                "According to", "The final answer is", "Final answer:"
            ]
            for prefix in prefixes_to_remove:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    answer = answer.lstrip(':').strip()
            
            # Extract just the answer entities (remove any remaining explanatory text)
            # Look for the first occurrence of sentence-ending patterns and truncate there if it seems like explanation
            import re
            # If there's a pattern like ". The" or ". Based" or ". Path", truncate before it
            match = re.search(r'\.\s+(The|Based|Path|According|Looking|From|These|This)', answer)
            if match:
                answer = answer[:match.start()].strip()
            
            # Remove any trailing incomplete sentences
            if answer.endswith('.'):
                answer = answer[:-1].strip()
            
            # Filter out any entity IDs that slipped through (m.xxxxx or [m.xxxxx, ...])
            import re
            # Remove standalone entity IDs
            answer = re.sub(r'\bm\.\w+\b', '', answer)
            # Remove list-like structures with entity IDs
            answer = re.sub(r'\[m\.\w+(?:,\s*m\.\w+)*\]', '', answer)
            # Remove "A: [...]" prefix that sometimes appears
            answer = re.sub(r'^A:\s*\[.*?\]\s*', '', answer)
            # Clean up any resulting double commas or spaces
            answer = re.sub(r',\s*,', ',', answer)
            answer = re.sub(r'\s+', ' ', answer).strip().strip(',').strip()
            
            # Additional cleanup for overly verbose location answers
            # If question asks "where is X from/based/located", prefer just the city name
            if any(keyword in question.lower() for keyword in ['where is', 'where was', 'where are', 'where were', 'where did']):
                # If answer is "City, State, Country", extract just "City"
                parts = [p.strip() for p in answer.split(',')]
                if len(parts) > 1:
                    # Check if it looks like a location hierarchy
                    # Keep just the first (most specific) part
                    answer = parts[0]
            
            if answer:
                logger.info(f"LLM synthesized answer: {answer}")
                return answer
            else:
                # Fallback
                logger.warning("LLM returned empty answer, using frequency fallback")
                if ranked_answers:
                    top_entity = ranked_answers[0][0]
                    return self.kg_interface.get_entity_name(top_entity)
                return None
                
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}, using frequency fallback")
            if ranked_answers:
                top_entity = ranked_answers[0][0]
                return self.kg_interface.get_entity_name(top_entity)
            return None
