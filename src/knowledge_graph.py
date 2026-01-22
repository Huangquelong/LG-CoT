"""Build Neo4j knowledge graph from extraction results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from neo4j import GraphDatabase
except ImportError:
    logger.error("Neo4j driver not installed, please run: pip install neo4j")
    GraphDatabase = None


class KnowledgeGraphConstructor:
    """Build Neo4j KG: document, Clause, subject, object, action, condition, constrain; HAS_CLAUSE, HAS_SUBJECT, etc."""

    def __init__(self, config: Dict):
        """Init from config (Neo4j connection)."""
        if GraphDatabase is None:
            raise ImportError("Neo4j driver not installed, please run: pip install neo4j")
        
        kg_config = config.get('knowledge_graph', {})
        neo4j_config = kg_config.get('neo4j', {})
        
        self.uri = neo4j_config.get('uri', 'bolt://localhost:7687')
        self.username = neo4j_config.get('username', 'neo4j')
        self.password = neo4j_config.get('password', 'neo4j')
        self.database = neo4j_config.get('database', 'neo4j')
        self.max_connection_lifetime = kg_config.get('max_connection_lifetime', 3600)
        self.max_connection_pool_size = kg_config.get('max_connection_pool_size', 50)
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_lifetime=self.max_connection_lifetime,
            max_connection_pool_size=self.max_connection_pool_size
        )
        
        logger.info(f"Neo4j connection established: {self.uri}")
        self._create_constraints_and_indexes()

    def _create_constraints_and_indexes(self):
        """Create constraints and indexes in Neo4j."""
        constraints_and_indexes = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:subject) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:action) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:object) REQUIRE o.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (c:Clause) ON (c.type)",
            "CREATE INDEX IF NOT EXISTS FOR (cond:condition) ON (cond.variable)",
            "CREATE INDEX IF NOT EXISTS FOR (constr:constrain) ON (constr.id)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    logger.debug(f"Created constraint/index: {query}")
                except Exception as e:
                    logger.warning(f"Failed to create constraint/index (may already exist): {e}")
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def construct_from_json(self, json_data: List[Dict], document_id: Optional[str] = None) -> Dict[str, Any]:
        """Build KG from extraction JSON (e.g. results_final.json). Returns stats."""
        if not json_data:
            logger.warning("Input data is empty, skipping knowledge graph construction")
            return {"status": "skipped", "reason": "empty_data"}
        if not document_id:
            first_clause_id = json_data[0].get('id', '')
            if '_' in first_clause_id:
                document_id = first_clause_id.split('_')[0] + '_document'
            else:
                document_id = 'Document_1'
        
        stats = {
            "document_id": document_id,
            "clauses": 0,
            "subjects": 0,
            "objects": 0,
            "actions": 0,
            "conditions": 0,
            "constraints": 0,
            "relationships": 0,
            "errors": []
        }
        
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: self._create_document(tx, document_id))
            for clause_data in json_data:
                try:
                    clause_stats = session.execute_write(
                        lambda tx: self._process_clause(tx, document_id, clause_data)
                    )
                    for key in ['clauses', 'subjects', 'objects', 'actions', 'conditions', 'constraints', 'relationships']:
                        stats[key] += clause_stats.get(key, 0)
                except Exception as e:
                    error_msg = f"Error processing Clause {clause_data.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
        
        stats['status'] = 'success' if not stats['errors'] else 'partial_success'
        logger.info(f"Knowledge graph construction completed: {stats}")
        return stats

    @staticmethod
    def _create_document(tx, document_id: str):
        """Create document node."""
        query = """
        MERGE (d:document {id: $document_id})
        SET d.created_at = timestamp()
        RETURN d
        """
        tx.run(query, document_id=document_id)
    
    @staticmethod
    def _process_clause(tx, document_id: str, clause_data: Dict) -> Dict[str, int]:
        """Process single Clause and related entities; return stats."""
        stats = {
            "clauses": 0,
            "subjects": 0,
            "objects": 0,
            "actions": 0,
            "conditions": 0,
            "constraints": 0,
            "relationships": 0
        }
        
        clause_id = clause_data.get('id', '')
        rules = clause_data.get('rules', [])
        
        if not clause_id:
            logger.warning("Clause missing id, skipping")
            return stats
        context_tags = []
        logic_forms = []
        source_texts = []
        clause_types = []
        
        for rule in rules:
            if rule.get('context_tags'):
                context_tags.extend(rule['context_tags'])
            if rule.get('logic_form'):
                logic_forms.append(rule['logic_form'])
            if rule.get('source_text'):
                source_texts.append(rule['source_text'])
            if rule.get('type'):
                clause_types.append(rule['type'])
        context_tags = list(set(context_tags))
        logic_form = '; '.join(logic_forms) if logic_forms else ''
        source_text = '; '.join(source_texts) if source_texts else ''
        clause_type = clause_types[0] if clause_types else 'UNKNOWN'
        clause_query = """
        MERGE (c:Clause {id: $clause_id})
        SET c.type = $type,
            c.context_tag = $context_tag,
            c.logic_form = $logic_form,
            c.text = $text,
            c.updated_at = timestamp()
        RETURN c
        """
        tx.run(
            clause_query,
            clause_id=clause_id,
            type=clause_type,
            context_tag=','.join(context_tags) if context_tags else '',
            logic_form=logic_form,
            text=source_text
        )
        stats['clauses'] = 1
        doc_clause_query = """
        MATCH (d:document {id: $document_id}), (c:Clause {id: $clause_id})
        MERGE (d)-[:HAS_CLAUSE]->(c)
        """
        tx.run(doc_clause_query, document_id=document_id, clause_id=clause_id)
        stats['relationships'] += 1
        for rule in rules:
            rule_stats = KnowledgeGraphConstructor._process_rule(tx, clause_id, rule)
            for key in rule_stats:
                stats[key] += rule_stats[key]
        
        return stats
    
    @staticmethod
    def _process_rule(tx, clause_id: str, rule: Dict) -> Dict[str, int]:
        """Process single rule, create entities and relations; return stats."""
        stats = {
            "subjects": 0,
            "objects": 0,
            "actions": 0,
            "conditions": 0,
            "constraints": 0,
            "relationships": 0
        }
        
        core_event = rule.get('core_event', {})
        conditions = rule.get('conditions', [])
        constraints = rule.get('constraints', {})
        subject_name = core_event.get('subject', '').strip()
        action_name = core_event.get('action', '').strip()
        object_name = core_event.get('object', '').strip()
        if subject_name:
            subject_query = """
            MERGE (s:subject {name: $name})
            ON CREATE SET s.created_at = timestamp()
            ON MATCH SET s.updated_at = timestamp()
            RETURN s
            """
            tx.run(subject_query, name=subject_name)
            stats['subjects'] = 1
            clause_subject_query = """
            MATCH (c:Clause {id: $clause_id}), (s:subject {name: $subject_name})
            MERGE (c)-[:HAS_SUBJECT]->(s)
            """
            tx.run(clause_subject_query, clause_id=clause_id, subject_name=subject_name)
            stats['relationships'] += 1
        
        if action_name:
            action_query = """
            MERGE (a:action {name: $name})
            ON CREATE SET a.created_at = timestamp()
            ON MATCH SET a.updated_at = timestamp()
            RETURN a
            """
            tx.run(action_query, name=action_name)
            stats['actions'] = 1
            clause_action_query = """
            MATCH (c:Clause {id: $clause_id}), (a:action {name: $action_name})
            MERGE (c)-[:HAS_ACTION]->(a)
            """
            tx.run(clause_action_query, clause_id=clause_id, action_name=action_name)
            stats['relationships'] += 1
        
        if object_name:
            object_query = """
            MERGE (o:object {name: $name})
            ON CREATE SET o.created_at = timestamp()
            ON MATCH SET o.updated_at = timestamp()
            RETURN o
            """
            tx.run(object_query, name=object_name)
            stats['objects'] = 1
            clause_object_query = """
            MATCH (c:Clause {id: $clause_id}), (o:object {name: $object_name})
            MERGE (c)-[:HAS_OBJECT]->(o)
            """
            tx.run(clause_object_query, clause_id=clause_id, object_name=object_name)
            stats['relationships'] += 1
        for idx, condition in enumerate(conditions):
            if not condition:
                continue
            
            condition_id = f"{clause_id}_condition_{idx}"
            condition_query = """
            MERGE (cond:condition {id: $id})
            ON CREATE SET cond.variable = $variable,
                          cond.operator = $operator,
                          cond.threshold = $threshold,
                          cond.unit = $unit,
                          cond.created_at = timestamp()
            ON MATCH SET cond.variable = $variable,
                         cond.operator = $operator,
                         cond.threshold = $threshold,
                         cond.unit = $unit,
                         cond.updated_at = timestamp()
            RETURN cond
            """
            tx.run(
                condition_query,
                id=condition_id,
                variable=condition.get('variable', ''),
                operator=condition.get('operator', ''),
                threshold=condition.get('threshold', ''),
                unit=condition.get('unit', '')
            )
            stats['conditions'] += 1
            clause_condition_query = """
            MATCH (c:Clause {id: $clause_id}), (cond:condition {id: $condition_id})
            MERGE (c)-[:HAS_CONDITION]->(cond)
            """
            tx.run(clause_condition_query, clause_id=clause_id, condition_id=condition_id)
            stats['relationships'] += 1
        if constraints and any(constraints.values()):
            constraint_id = f"{clause_id}_constrain"
            constraint_query = """
            MERGE (constr:constrain {id: $id})
            ON CREATE SET constr.time_limit = $time_limit,
                          constr.frequency = $frequency,
                          constr.manner = $manner,
                          constr.created_at = timestamp()
            ON MATCH SET constr.time_limit = $time_limit,
                         constr.frequency = $frequency,
                         constr.manner = $manner,
                         constr.updated_at = timestamp()
            RETURN constr
            """
            tx.run(
                constraint_query,
                id=constraint_id,
                time_limit=constraints.get('time_limit', ''),
                frequency=constraints.get('frequency', ''),
                manner=constraints.get('manner', '')
            )
            stats['constraints'] += 1
            
            clause_constraint_query = """
            MATCH (c:Clause {id: $clause_id}), (constr:constrain {id: $constraint_id})
            MERGE (c)-[:HAS_CONSTRAINT]->(constr)
            """
            tx.run(clause_constraint_query, clause_id=clause_id, constraint_id=constraint_id)
            stats['relationships'] += 1
        
        return stats
    
    def construct_from_file(self, json_file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Build KG from JSON file. document_id default from filename."""
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file does not exist: {json_file_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not document_id:
            document_id = json_path.stem
        
        logger.info(f"Building knowledge graph from file: {json_file_path}, document ID: {document_id}")
        return self.construct_from_json(json_data, document_id)
    
    def clear_database(self, confirm: bool = False):
        """Clear Neo4j DB. Requires confirm=True."""
        if not confirm:
            logger.warning("Database clear operation requires confirmation, please set confirm=True")
            return
        
        with self.driver.session(database=self.database) as session:
            query = "MATCH (n) DETACH DELETE n"
            result = session.run(query)
            summary = result.consume()
            logger.info(f"Database cleared, deleted nodes: {summary.counters.nodes_deleted}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return KG statistics (node counts, relationship counts)."""
        with self.driver.session(database=self.database) as session:
            stats_query = """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            """
            result = session.run(stats_query)
            
            stats = {}
            for record in result:
                label = record['label']
                count = record['count']
                stats[label] = count
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            """
            rel_result = session.run(rel_query)
            
            rel_stats = {}
            for record in rel_result:
                rel_type = record['rel_type']
                count = record['count']
                rel_stats[rel_type] = count
            
            stats['relationships'] = rel_stats
            return stats

