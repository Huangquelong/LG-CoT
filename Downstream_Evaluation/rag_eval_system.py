"""RAG evaluation system: compare NaiveRAG and GraphRAG performance on multi-hop questions."""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================
# Configuration and utility classes
# ============================================

@dataclass
class RetrievalResult:
    """Retrieval result data structure."""
    text: str
    score: float
    metadata: Optional[Dict] = None


class EmbeddingModelSingleton:
    """Embedding model singleton class to avoid repeated model loading and improve performance."""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingModelSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get model instance, lazy loading."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print(f"Loading embedding model: {self._model_name}...")
                    self._model = SentenceTransformer(self._model_name)
                    print("Embedding model loaded")
        return self._model


# ============================================
# RAG comparator class
# ============================================

class RAGComparator:
    """RAG comparison evaluator implementing NaiveRAG and GraphRAG retrieval strategies."""
    
    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """Initialize RAG comparator."""
        self.data_path = Path(data_path)
        
        if config_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent if script_dir.name == "Downstream_Evaluation" else Path.cwd()
            config_path = project_root / "config.yaml"
            self.config_path = config_path if config_path.exists() else None
        else:
            self.config_path = Path(config_path) if config_path else None
        
        self.config = self._load_config()
        self.data = self._load_data()
        self.embedding_model = EmbeddingModelSingleton().get_model()
        self.llm_client = self._init_llm_client()
        self._build_indices()
    
    def _load_config(self) -> Dict:
        """Load configuration file."""
        if self.config_path is None or not self.config_path.exists():
            print(f"Warning: Config file does not exist, using default configuration")
            print("Hint: Please set OPENAI_API_KEY and OPENAI_API_BASE environment variables, or provide config.yaml")
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            print(f"Failed to load config file: {e}")
            return {}
    
    def _load_data(self) -> List[Dict]:
        """Load data file."""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file does not exist: {self.data_path}")
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Successfully loaded {len(data)} data items")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Data file JSON format error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data file: {e}")
    
    def _init_llm_client(self) -> OpenAI:
        """Initialize LLM client"""
        try:
            # Get API key from config
            llm_api_config = self.config.get('llm_api', {})
            api_key = None
            api_base = None
            
            # Try SiliconFlow first
            if 'SiliconFlow' in llm_api_config:
                siliconflow_config = llm_api_config['SiliconFlow']
                api_key = siliconflow_config.get('api_key')
                api_base = siliconflow_config.get('api_base')
                if api_key and api_base:
                    print(f"Using SiliconFlow API (base_url: {api_base})")
            
            # Fallback to OpenAI
            if not api_key and 'openai' in llm_api_config:
                openai_config = llm_api_config['openai']
                api_key = openai_config.get('api_key') or os.getenv('OPENAI_API_KEY')
                api_base = openai_config.get('api_base')
                if api_key and api_base:
                    print(f"Using OpenAI API (base_url: {api_base})")
            
            # Try environment variables
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_base:
                api_base = os.getenv('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
            
            if not api_key:
                raise RuntimeError("API key not found in config or environment variables")
            
            client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            print(f"LLM client initialized successfully (base_url: {api_base})")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    def _build_indices(self):
        """Build retrieval indices."""
        print("Building retrieval indices...")
        
        self.naive_chunks = []
        self.naive_chunk_indices = []
        
        self.graph_search_strings = []
        self.graph_rule_indices = []
        
        for entry_idx, entry in enumerate(self.data):
            if 'rules' not in entry or not entry['rules']:
                continue
            
            for rule_idx, rule in enumerate(entry['rules']):
                source_text = rule.get('source_text', '')
                if source_text:
                    chunks = self._split_text(source_text, chunk_size=50, overlap=0)
                    for chunk in chunks:
                        self.naive_chunks.append(chunk)
                        self.naive_chunk_indices.append({
                            'entry_idx': entry_idx,
                            'rule_idx': rule_idx,
                            'rule': rule
                        })
                
                search_str = self._build_graph_search_string(rule)
                if search_str:
                    self.graph_search_strings.append(search_str)
                    self.graph_rule_indices.append({
                        'entry_idx': entry_idx,
                        'rule_idx': rule_idx,
                        'rule': rule
                    })
        
        print(f"NaiveRAG index: {len(self.naive_chunks)} text chunks")
        print(f"GraphRAG index: {len(self.graph_search_strings)} rules")
        print("Index construction completed")
    
    def _split_text(self, text: str, chunk_size: int = 50, overlap: int = 0) -> List[str]:
        """Split text into chunks of specified size."""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        
        return chunks if chunks else [text[:chunk_size]]
    
    def _build_graph_search_string(self, rule: Dict) -> str:
        """Build GraphRAG retrieval string by concatenating subject, context_tags and logic_form."""
        parts = []
        
        core_event = rule.get('core_event', {})
        if isinstance(core_event, dict):
            subject = core_event.get('subject', '')
            if subject:
                parts.append(f"Subject: {subject}")
        
        context_tags = rule.get('context_tags', [])
        if context_tags:
            tags_str = ", ".join(context_tags) if isinstance(context_tags, list) else str(context_tags)
            parts.append(f"Tags: {tags_str}")
        
        logic_form = rule.get('logic_form', '')
        if logic_form:
            parts.append(f"Logic form: {logic_form}")
        
        return " | ".join(parts) if parts else rule.get('source_text', '')
    
    def run_naive_rag(self, query: str) -> RetrievalResult:
        """NaiveRAG retrieval (intentionally weakened version)."""
        if not self.naive_chunks:
            return RetrievalResult(
                text="",
                score=0.0,
                metadata=None
            )
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        chunk_embeddings = self.embedding_model.encode(
            self.naive_chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_idx = np.argmax(similarities)
        top_score = float(similarities[top_idx])
        
        metadata = self.naive_chunk_indices[top_idx]
        
        return RetrievalResult(
            text=self.naive_chunks[top_idx],
            score=top_score,
            metadata=metadata
        )
    
    def run_graph_rag(self, query: str, top_k: int = 1) -> List[RetrievalResult]:
        """GraphRAG retrieval (structured enhanced version)."""
        if not self.graph_search_strings:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        search_embeddings = self.embedding_model.encode(
            self.graph_search_strings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        similarities = np.dot(search_embeddings, query_embedding) / (
            np.linalg.norm(search_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            metadata = self.graph_rule_indices[idx]
            rule = metadata['rule']
            
            results.append(RetrievalResult(
                text=rule.get('source_text', ''),
                score=float(similarities[idx]),
                metadata={
                    **metadata,
                    'logic_form': rule.get('logic_form', ''),
                    'core_event': rule.get('core_event', {}),
                    'context_tags': rule.get('context_tags', [])
                }
            ))
        
        return results
    
    def generate_naive_response(self, query: str, retrieved_text: str) -> str:
        """Generate answer using NaiveRAG retrieval results."""
        prompt = f"""Please answer the question based on the following retrieved text fragment.

Retrieved text fragment:
{retrieved_text}

Question:
{query}

Please answer the question based on the retrieved text fragment. If the retrieved information is insufficient to answer the question, please state so."""
        
        try:
            # Get model name from config
            model_name = self.config.get('rag_evaluation', {}).get('default_model', 'moonshotai/Kimi-K2-Thinking')
            # Fallback to knowledge_extractor model if rag_evaluation not configured
            if model_name == 'moonshotai/Kimi-K2-Thinking' and 'knowledge_extractor' in self.config:
                configured_model = self.config['knowledge_extractor'].get('model', '')
                if configured_model:
                    model_name = configured_model
            
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"  [Warning] LLM call failed: {error_msg}")
            return f"[Generation failed] {error_msg}"
    
    def generate_graph_response(self, query: str, retrieved_rules: List[RetrievalResult]) -> str:
        """Generate answer using GraphRAG retrieval results."""
        context_parts = []
        for i, result in enumerate(retrieved_rules, 1):
            context_parts.append(f"Rule {i}:")
            context_parts.append(f"Original text: {result.text}")
            
            if result.metadata and 'logic_form' in result.metadata:
                context_parts.append(f"Logic form: {result.metadata['logic_form']}")
            
            if result.metadata and 'core_event' in result.metadata:
                core_event = result.metadata['core_event']
                if isinstance(core_event, dict) and core_event.get('subject'):
                    context_parts.append(f"Subject: {core_event.get('subject')}")
            
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Please answer the question based on the following structured rule information. These rules contain original text and logic forms. Please use the logic forms for reasoning to answer multi-hop questions.

Structured rule information:
{context}

Question:
{query}

Please reason based on the provided original text and logic forms to answer the above question. Pay attention to using relationships in the logic forms for multi-hop reasoning."""
        
        try:
            # Get model name from config
            model_name = self.config.get('rag_evaluation', {}).get('default_model', 'moonshotai/Kimi-K2-Thinking')
            # Fallback to knowledge_extractor model if rag_evaluation not configured
            if model_name == 'moonshotai/Kimi-K2-Thinking' and 'knowledge_extractor' in self.config:
                configured_model = self.config['knowledge_extractor'].get('model', '')
                if configured_model:
                    model_name = configured_model
            
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"  [Warning] LLM call failed: {error_msg}")
            return f"[Generation failed] {error_msg}"
    
    def evaluate_query(self, question_id: str, query: str) -> Dict:
        """Evaluate a single query."""
        print(f"\nProcessing question [{question_id}]: {query}")
        
        print("  [NaiveRAG] Retrieving...")
        naive_result = self.run_naive_rag(query)
        naive_retrieved = naive_result.text if naive_result else ""
        print(f"  [NaiveRAG] Retrieval completed, similarity: {naive_result.score:.4f}")
        
        print("  [NaiveRAG] Generating answer...")
        naive_answer = self.generate_naive_response(query, naive_retrieved)
        
        print("  [GraphRAG] Retrieving...")
        graph_results = self.run_graph_rag(query, top_k=3)
        graph_retrieved = graph_results if graph_results else []
        if graph_results:
            print(f"  [GraphRAG] Retrieval completed, Top-1 similarity: {graph_results[0].score:.4f}")
        
        print("  [GraphRAG] Generating answer...")
        graph_answer = self.generate_graph_response(query, graph_retrieved)
        
        return {
            'id': question_id,
            'query': query,
            'naive_rag': {
                'retrieved_text': naive_retrieved,
                'retrieved_score': naive_result.score if naive_result else 0.0,
                'answer': naive_answer
            },
            'graph_rag': {
                'retrieved_count': len(graph_retrieved),
                'retrieved_rules': [
                    {
                        'text': r.text[:100] + "..." if len(r.text) > 100 else r.text,
                        'score': r.score
                    }
                    for r in graph_retrieved[:3]
                ],
                'answer': graph_answer
            }
        }


# ============================================
# Main program
# ============================================

def print_evaluation_table(results: List[Dict]):
    """Print evaluation results in Markdown table format."""
    print("\n" + "="*100)
    print("RAG Evaluation Results Comparison")
    print("="*100)
    print()
    
    print("| Question | NaiveRAG Answer | GraphRAG Answer |")
    print("|----------|----------------|----------------|")
    
    for result in results:
        query = result['query']
        naive_answer = result['naive_rag']['answer'][:200] + "..." if len(result['naive_rag']['answer']) > 200 else result['naive_rag']['answer']
        graph_answer = result['graph_rag']['answer'][:200] + "..." if len(result['graph_rag']['answer']) > 200 else result['graph_rag']['answer']
        
        query = query.replace("|", "\\|").replace("\n", " ")
        naive_answer = naive_answer.replace("|", "\\|").replace("\n", " ")
        graph_answer = graph_answer.replace("|", "\\|").replace("\n", " ")
        
        print(f"| {query} | {naive_answer} | {graph_answer} |")
    
    print()


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "Downstream_Evaluation" else Path.cwd()
    
    data_path = project_root / "Validation/Experiment_result/GLM-41V-9B-Thinking-Experiment_result/lgcot/results_final.json"
    questions_path = script_dir / "rag_questions.json"
    config_path = project_root / "config.yaml"
    output_dir = script_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RAG Evaluation System Started")
    print("="*80)
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Questions file: {questions_path}")
    print(f"Config file: {config_path}")
    print()
    
    if not data_path.exists():
        print(f"Error: Data file does not exist: {data_path}")
        print("Please check if the path is correct")
        sys.exit(1)
    
    if not questions_path.exists():
        print(f"Error: Questions file does not exist: {questions_path}")
        print("Please check if the path is correct")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Warning: Config file does not exist: {config_path}")
        print("Will use embedded API key")
    
    try:
        print(f"Loading questions file: {questions_path}")
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        print(f"Successfully loaded {len(questions_data)} questions\n")
        
        comparator = RAGComparator(
            data_path=str(data_path), 
            config_path=str(config_path) if config_path.exists() else None
        )
        
        all_results = []
        naive_rag_results = []
        graph_rag_results = []
        
        for question_item in questions_data:
            question_id = question_item.get('id', '')
            query = question_item.get('问题', '')
            
            if not query:
                print(f"Warning: Question [{question_id}] has empty query text, skipping")
                continue
            
            result = comparator.evaluate_query(question_id, query)
            all_results.append(result)
            
            naive_rag_results.append({
                'id': question_id,
                'query': query,
                'retrieved_text': result['naive_rag']['retrieved_text'],
                'retrieved_score': result['naive_rag']['retrieved_score'],
                'answer': result['naive_rag']['answer']
            })
            
            graph_rag_results.append({
                'id': question_id,
                'query': query,
                'retrieved_count': result['graph_rag']['retrieved_count'],
                'retrieved_rules': result['graph_rag']['retrieved_rules'],
                'answer': result['graph_rag']['answer']
            })
        
        naive_output_file = output_dir / "naive_rag_results.json"
        naive_output_data = {
            'timestamp': datetime.now().isoformat(),
            'data_source': str(data_path),
            'questions_source': str(questions_path),
            'total_questions': len(questions_data),
            'results': naive_rag_results
        }
        
        with open(naive_output_file, 'w', encoding='utf-8') as f:
            json.dump(naive_output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[NaiveRAG] Evaluation results saved to: {naive_output_file}")
        
        graph_output_file = output_dir / "graph_rag_results.json"
        graph_output_data = {
            'timestamp': datetime.now().isoformat(),
            'data_source': str(data_path),
            'questions_source': str(questions_path),
            'total_questions': len(questions_data),
            'results': graph_rag_results
        }
        
        with open(graph_output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[GraphRAG] Evaluation results saved to: {graph_output_file}")
        
        print("\n" + "="*80)
        print("Evaluation Statistics")
        print("="*80)
        print(f"Total questions: {len(questions_data)}")
        print(f"Successfully evaluated: {len(all_results)}")
        print(f"NaiveRAG results file: {naive_output_file}")
        print(f"GraphRAG results file: {graph_output_file}")
        print("\nEvaluation completed!")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Configuration error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

