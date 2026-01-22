"""
Main pipeline (unified). Orchestrates PDF extraction -> text cleaning -> schema building -> knowledge extraction (lgcot-CoT V2).
"""

import os
import sys
import json
import random
import threading
import time
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime
import jieba
import numpy as np
try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None
    logger.warning("gensim module not installed, Word2Vec functionality unavailable")

from pdf_extractor import PDFExtractor
from text_cleaner import TextCleaner
from schema_builder import SchemaBuilder
from knowledge_extractor import KnowledgeExtractor
from validator import Validator
from knowledge_graph import KnowledgeGraphConstructor


class KnowledgeExtractionPipeline:
    """
    End-to-end pipeline: PDF (MinerU) -> text cleaning -> schema (Step0) -> Word2Vec -> lgcot-CoT extraction (Step2–Step6) -> validation -> output; optional Neo4j KG build.
    """

    def __init__(self, config_path: str = 'config.yaml', output_base_dir: Path = None):
        """
        Args:
            config_path: Path to config file.
            output_base_dir: Output root (default from config).
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

        logger.info("="*60)
        logger.info("Knowledge extraction pipeline started (merged version)")
        logger.info("="*60)

        self.llm_client = self._init_llm_client()

        if output_base_dir is None:
            output_base_dir = Path(self.config['output_dir'])
        model_name = self.config['knowledge_extractor']['model'].replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_root = Path(output_base_dir) / f"{model_name}_{timestamp}"
        self.output_root.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing modules...")
        self.pdf_extractor = PDFExtractor(self.config['pdf_extractor'])
        self.text_cleaner = TextCleaner(self.config['text_cleaner'])

        schema_config = self.config['schema_builder'].copy()
        schema_config['output_dir'] = str(self.output_root / 'schemas')
        self.schema_builder = SchemaBuilder(
            schema_config, 
            self.llm_client,
            output_root=self.output_root / 'schemas'
        )

        self.validator = Validator(
            self.config['validator'], 
            self.llm_client,
            schema_builder=self.schema_builder
        )

        self.knowledge_extractor = None
        self.word2vec_model = None
        self.text_vectors = None
        self.all_clauses = []
        self.word2vec_config = self.config.get('word2vec', {})

        self.review_repo_path = self.output_root / "review_repository.json"
        self.review_repo = self._load_repository(self.review_repo_path, default=[])

        kg_config = self.config.get('knowledge_graph', {})
        self.kg_auto_construct = kg_config.get('auto_construct', False)
        self.kg_constructor = None
        if self.kg_auto_construct:
            try:
                self.kg_constructor = KnowledgeGraphConstructor(self.config)
                logger.info("Knowledge graph construction module initialized")
            except Exception as e:
                logger.warning(f"Knowledge graph construction module initialization failed (will skip KG construction): {e}")
                self.kg_auto_construct = False

        self.pipeline_config = self.config['pipeline']
        self.checkpoint_dir = Path(self.pipeline_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.global_schema_path = Path(self.config['global_schema_path'])
        self.global_schema = self._load_global_schema()
        
        logger.info("All modules initialized")
    
    def request_stop(self):
        """Signal early stop."""
        self.stop_event.set()

    def _load_config(self, config_path: str) -> Dict:
        """Load config file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logging(self):
        """Configure logging."""
        log_config = self.config.get('logging', {})
        log_level = log_config['level']
        log_format = log_config['format']
        log_dir = Path(log_config['output_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True
        )
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=log_config['rotation'],
            retention=log_config['retention'],
            encoding='utf-8'
        )
        
        logger.info(f"Logging system configured, output: {log_file}")

    def _init_llm_client(self):
        """Initialize LLM client."""
        llm_api_config = self.config['llm_api']
        provider = self.config['schema_builder']['llm_provider']
        
        try:
            if provider == 'SiliconFlow' or provider == 'siliconflow':
                from openai import OpenAI
                api_config = llm_api_config['SiliconFlow']
                client = OpenAI(
                    api_key=api_config['api_key'],
                    base_url=api_config.get('api_base', 'https://api.siliconflow.cn/v1')
                )
                logger.info(f"SiliconFlow client initialized")
                return client
            elif provider == 'openai':
                from openai import OpenAI
                api_config = llm_api_config['openai']
                client = OpenAI(
                    api_key=api_config.get('api_key') or os.getenv('OPENAI_API_KEY'),
                    base_url=api_config.get('api_base'),
                    organization=api_config.get('organization')
                )
                logger.info(f"OpenAI client initialized")
                return client
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"LLM client initialization failed: {e}")
            raise
    
    def _load_repository(self, path: Path, default=None):
        """Load repository JSON."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Repository loading failed, using default value: {e}")
                return default if default is not None else []
        return default if default is not None else []
    
    def _save_repository(self, path: Path, data):
        """Save repository JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_clean_data(self, clean_data_dir: Optional[Path] = None) -> tuple:
        """Load cleaned results from clean_data dir. Returns (all_clauses, clause_to_doc_id)."""
        if clean_data_dir is None:
            clean_data_dir = Path(self.config['text_cleaner']['output_dir'])
        clean_data_dir = Path(clean_data_dir)
        if not clean_data_dir.exists():
            logger.warning(f"clean_data directory does not exist: {clean_data_dir}")
            return [], {}
        
        all_clauses = []
        clause_to_doc_id = {}
        json_files = list(clean_data_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No .json files in clean_data: {clean_data_dir}")
            return [], {}
        
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load clean_data {jf}: {e}")
                continue
            doc_id = data.get('doc_id') or jf.stem
            clauses = data.get('clauses', [])
            for c in clauses:
                text = (c.get('content') or '').strip()
                if not text:
                    continue
                all_clauses.append(text)
                clause_to_doc_id[text] = doc_id
        
        logger.info(f"Loaded from clean_data: {len(json_files)} files, {len(all_clauses)} clauses")
        return all_clauses, clause_to_doc_id
    
    def _build_word2vec_model(self, clauses: List[str]):
        """Build Word2Vec model for semantic retrieval."""
        logger.info("Starting Word2Vec model construction...")
        vector_size = self.word2vec_config.get('vector_size', 256)
        window = self.word2vec_config.get('window', 5)
        min_count = self.word2vec_config.get('min_count', 1)
        workers = self.word2vec_config.get('workers', 4)
        sg = self.word2vec_config.get('sg', 0)

        sentences = [list(jieba.cut(text)) for text in clauses]
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg
        )
        self.text_vectors = []
        for text in clauses:
            words = list(jieba.cut(text))
            word_vectors = [self.word2vec_model.wv[word] for word in words
                          if word in self.word2vec_model.wv]
            
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(vector_size)
            
            self.text_vectors.append(text_vector)
        
        self.text_vectors = np.array(self.text_vectors)
        self.all_clauses = clauses
        
        logger.info(f"Word2Vec model construction completed, vector dimension: {self.text_vectors.shape}")
        if self.knowledge_extractor:
            self.knowledge_extractor.update_fol_repo_vectors(self.word2vec_model)
    
    def _call_llm_unified(self, prompt: str, system_content: str = "你是一个知识抽取和逻辑形式化专家。",
                         max_retries: int = 3) -> Optional[str]:
        """Unified LLM call. Returns response text or None."""
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return None
        ke_config = self.config['knowledge_extractor']
        normal_model = ke_config['model']
        temperature = ke_config['temperature']
        max_tokens = ke_config['max_tokens']
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=normal_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM call finally failed: {e}")
                    return None
        return None
    
    def process_single_clause(self, clause_text: str, doc_id: str = None, max_retry: int = 3) -> Optional[Dict]:
        """Process a single clause (Step2–Step6)."""
        logger.info(f"Processing clause: {clause_text[:50]}...")
        retry_count = 0
        while retry_count < max_retry:
            try:
                step2_prompt = self.knowledge_extractor.step2_build_prompt(clause_text)
                step2_response = self._call_llm_unified(step2_prompt)
                if not step2_response:
                    logger.warning("Step 2 LLM call failed, retrying")
                    retry_count += 1
                    continue
                
                fol_result = self.knowledge_extractor.step2_generate_fol(clause_text, step2_response)
                if fol_result is None:
                    logger.warning("Step 2 parsing failed, retrying")
                    retry_count += 1
                    continue
                
                fol = fol_result['fol']
                new_predicates = fol_result['new_predicates']
                predicates_ok, approved_predicates = self.validator.step3_check_predicates(new_predicates, doc_id=doc_id)
                step5_prompt = self.validator.step5_build_prompt(fol, clause_text)
                step5_response = self._call_llm_unified(
                    step5_prompt,
                    system_content="You are a FOL formula quality inspector.",
                    max_retries=3
                )
                if not step5_response:
                    logger.warning("Step 5 LLM call failed, retrying")
                    retry_count += 1
                    continue
                
                similarity_ok, similarity = self.validator.step5_fol_to_text_similarity(fol, clause_text, step5_response)
                if not similarity_ok:
                    logger.warning("Step 5 self-check failed, returning to Step 2")
                    retry_count += 1
                    continue
                step6_prompt = self.knowledge_extractor._build_step6_prompt(fol, clause_text)
                step6_response = self._call_llm_unified(step6_prompt)
                if not step6_response:
                    logger.warning("Step 6 LLM call failed, retrying")
                    retry_count += 1
                    continue
                
                json_result = self.knowledge_extractor.step6_fol_to_json(fol, clause_text, step6_response)
                if json_result is None:
                    logger.warning("Step 6 parsing failed, retrying")
                    retry_count += 1
                    continue
                fol_item = {'text': clause_text, 'fol': fol}
                is_valid, error_msg = self.schema_builder._validate_fol_example(fol_item)
                if is_valid:
                    self.schema_builder.fol_repo.append(fol_item)
                    self.schema_builder._save_repository(self.schema_builder.fol_repo_path, self.schema_builder.fol_repo)
                    self.knowledge_extractor.fol_repo = self.schema_builder.fol_repo
                    self.knowledge_extractor.update_fol_repo_vectors(self.word2vec_model)
                else:
                    logger.error(f"Generated FOL does not meet format requirements: {error_msg}")
                
                logger.info(f"Clause processing succeeded: {clause_text[:50]}...")
                
                return {
                    'clause_text': clause_text,
                    'doc_id': doc_id,
                    'fol': fol,
                    'approved_predicates': approved_predicates,
                    'similarity': similarity,
                    'json_result': json_result,
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Clause processing exception: {e}")
                retry_count += 1
        logger.error(f"Clause processing failed, reached maximum retry count: {clause_text[:50]}...")
        review_item = {
            'clause': clause_text,
            'error': 'Reached maximum retry count',
            'timestamp': datetime.now().isoformat(),
            'retry_count': max_retry
        }
        existing_clauses = {item.get('clause', '') for item in self.review_repo}
        if clause_text not in existing_clauses:
            self.review_repo.append(review_item)
            self._save_repository(self.review_repo_path, self.review_repo)
        
        return {
            'clause_text': clause_text,
            'doc_id': doc_id,
            'success': False,
            'error': 'Reached maximum retry count'
        }
    
    def _format_results(self, results: List[Dict], clause_to_doc_id: Dict[str, str] = None) -> List[Dict]:
        """Format results into final output structure."""
        if clause_to_doc_id is None:
            clause_to_doc_id = {}
        
        formatted = []
        doc_results = {}
        
        for res in results:
            if not res or not res.get('success'):
                continue
            
            clause_text = res.get('clause_text', '')
            doc_id = res.get('doc_id') or clause_to_doc_id.get(clause_text)
            
            if not doc_id:
                doc_id = clause_to_doc_id.get(clause_text) or f"Clause_{len(formatted) + 1}"
            
            json_result = res.get('json_result', {})
            rules = json_result.get('rules', [])
            if not isinstance(rules, list):
                rules = [rules] if rules else []
            
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            
            for rule in rules:
                rule['logic_form'] = res.get('fol', '')
                rule['source_text'] = clause_text
                doc_results[doc_id].append(rule)
        
        for doc_id, rules in doc_results.items():
            formatted.append({
                "id": doc_id,
                "rules": rules
            })
        
        return formatted
    
    def run_step_pdf_extraction(self, input_pdf: Optional[str] = None,
                                analyze_only: bool = False,
                                use_suggestions: bool = False) -> Dict:
        """Step 1: PDF extraction."""
        logger.info("="*60)
        logger.info("[Step 1] PDF recognition and cleaning")
        logger.info("="*60)
        
        if analyze_only:
            logger.info("Mode: Analysis only, generate parameter suggestion file")
        elif use_suggestions:
            logger.info("Mode: Use previous suggestions for cleaning")
        else:
            logger.info("Mode: Analysis + cleaning (two-stage)")
        
        if input_pdf:
            pdf_result = self.pdf_extractor.extract_from_pdf(
                input_pdf, 
                analyze_only=analyze_only,
                use_suggestions=use_suggestions
            )
            
            if analyze_only:
                logger.info(f"✓ PDF analysis completed, parameter suggestion file: {pdf_result.get('suggestion_path')}")
                return {
                    'status': 'success',
                    'doc_id': Path(input_pdf).stem,
                    'suggestion_path': pdf_result.get('suggestion_path'),
                    'analysis_result': pdf_result.get('analysis_result')
                }
            
            doc_id = Path(input_pdf).stem
        else:
            pdf_dir = self.pipeline_config['input_pdf_dir']
            pdf_results = self.pdf_extractor.batch_extract(pdf_dir)
            
            if not pdf_results:
                logger.error("No successfully extracted PDF files")
                return {'status': 'error', 'message': 'No PDF extracted'}
            pdf_result = pdf_results[0]
            doc_id = Path(pdf_result['source_pdf']).stem
        
        logger.info(f"✓ PDF extraction completed: {doc_id}")
        if pdf_result.get('suggestion_path'):
            logger.info(f"  Parameter suggestion file: {pdf_result['suggestion_path']}")
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            'pdf_result': pdf_result
        }
    
    def run_step_text_cleaning(self, pdf_result: Dict) -> Dict:
        """Step 2: Text cleaning."""
        logger.info("="*60)
        logger.info("[Step 2] Text cleaning")
        logger.info("="*60)
        
        cleaned_result = self.text_cleaner.clean_text(
            pdf_result['text'],
            pdf_result.get('metadata')
        )
        doc_id = Path(pdf_result.get('source_pdf', 'unknown')).stem
        self.text_cleaner.save_cleaned_data(cleaned_result, doc_id)
        
        logger.info(f"✓ Text cleaning completed, identified {cleaned_result['statistics']['num_clauses']} clauses")
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            'cleaned_result': cleaned_result
        }
    
    def run_step_schema_building(self, cleaned_result: Dict, doc_id: str = None) -> Dict:
        """Step 3: Schema building (Step0 init)."""
        logger.info("="*60)
        logger.info("[Step 3] Schema dynamic construction")
        logger.info("="*60)
        clauses = cleaned_result.get('clauses', [])
        all_clauses = [clause.get('content', '') for clause in clauses if clause.get('content')]
        predicate_count = len([item.get('predicate') for item in self.schema_builder.predicate_repo if item.get('predicate')])
        fol_repo_count = len(self.schema_builder.fol_repo)
        
        if predicate_count == 0 or fol_repo_count == 0:
            logger.info("Domain predicate repository is empty, executing Step 0 initialization...")
            schema_config = self.config['schema_builder']
            sample_count = schema_config['sample_text_count']
            use_high_param = schema_config['use_high_param']
            max_tokens = schema_config['max_tokens']
            if len(all_clauses) > sample_count:
                sampled_texts = random.sample(all_clauses, sample_count)
            else:
                sampled_texts = all_clauses
            prompt = f"""Please analyze the following clause texts, extract domain predicates for building first-order logic (FOL), and generate complete function-call style FOL representations.

Requirements:
1. Domain predicates must be in Chinese
2. FOL format should be function-call style, e.g.: AND(Obligation(subject=Party("用人单位"), action="支付", object="劳动报酬"), Condition(desc="劳动者实际提供劳动"))
3. Except for logical operator whitelist (AND, OR, NOT, IMPLIES, IFF, XOR, FORALL, EXISTS) and domain predicate whitelist (Obligation, Permission, Prohibition, REQUIREMENT), all other content must be in Chinese
4. Output JSON format: {{"predicates": ["predicate1", "predicate2", ...], "fol_examples": [{{"text": "clause text", "fol": "FOL representation"}}]}}

Clause text list:
{json.dumps(sampled_texts, ensure_ascii=False, indent=2)}

Output only JSON, no other content."""
            llm_response = self._call_llm_unified(prompt, max_retries=3)
            if not llm_response:
                logger.error("LLM call failed, unable to initialize predicate repository")
                return {'status': 'error', 'message': 'LLM call failed'}
            self.schema_builder.step0_initialize_predicates(all_clauses, llm_response=llm_response)
        else:
            logger.info(f"Domain predicate repository exists: {predicate_count} predicates, {fol_repo_count} FOL examples")
        local_schema = self.schema_builder.build_schema(cleaned_result)
        if doc_id:
            self.schema_builder.save_schema(local_schema, doc_id)
        
        logger.info(f"✓ Schema construction completed")
        logger.info(f"  - Clause ID count: {len(local_schema.get('clause_id', []))}")
        logger.info(f"  - Predicate count: {len(local_schema.get('predicates', []))}")
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            'schema': local_schema,
            'all_clauses': all_clauses
        }
    
    def run_step_knowledge_extraction(self, sample_size: int = None) -> Dict:
        """Step 4–5: Knowledge extraction. Input from clean_data, output to extract_data and output_root."""
        logger.info("="*60)
        logger.info("[Step 4-5] Knowledge extraction & validation")
        logger.info("="*60)
        all_clauses, clause_to_doc_id = self._load_from_clean_data()
        if not all_clauses:
            logger.error("No available clauses in clean_data, please run cleaning step first and ensure JSON is saved")
            return {'status': 'error', 'message': 'No available clauses in clean_data, please run cleaning step first'}
        if sample_size and sample_size < len(all_clauses):
            sampled_clauses = random.sample(all_clauses, sample_size)
            if clause_to_doc_id:
                clause_to_doc_id = {clause: clause_to_doc_id.get(clause) for clause in sampled_clauses if clause in clause_to_doc_id}
            all_clauses = sampled_clauses
            logger.info(f"Clause count after sampling: {len(all_clauses)}")
        logger.info("Building Word2Vec model...")
        self._build_word2vec_model(all_clauses)
        ke_config = self.config['knowledge_extractor'].copy()
        ke_config['output_dir'] = str(self.output_root / 'extraction_results')
        self.knowledge_extractor = KnowledgeExtractor(
            ke_config,
            schema_builder=self.schema_builder,
            validator=self.validator,
            llm_client=self.llm_client,
            word2vec_model=self.word2vec_model,
            fol_repo=self.schema_builder.fol_repo,
            output_root=self.output_root
        )
        self.knowledge_extractor.update_fol_repo_vectors(self.word2vec_model)
        logger.info("Starting to process all clauses...")
        results = []
        lock = threading.Lock()
        progress_counter = {'processed': 0, 'total': len(all_clauses)}
        
        max_workers = self.config['knowledge_extractor']['batch_workers']
        
        def worker(clause):
            doc_id = clause_to_doc_id.get(clause) if clause_to_doc_id else None
            result = self.process_single_clause(clause, doc_id=doc_id)
            with lock:
                results.append(result)
                progress_counter['processed'] += 1
                if progress_counter['processed'] % 10 == 0:
                    logger.info(f"Progress: {progress_counter['processed']}/{progress_counter['total']}")
        
        threads = []
        
        for clause in all_clauses:
            while len([t for t in threads if t.is_alive()]) >= max_workers:
                time.sleep(0.1)
                threads = [t for t in threads if t.is_alive()]
            
            thread = threading.Thread(target=worker, args=(clause,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        formatted_results = self._format_results(results, clause_to_doc_id=clause_to_doc_id)
        output_file = self.output_root / "results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved: {output_file}")
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info("="*60)
        logger.info(f"First round processing completed: {success_count}/{len(results)} succeeded")
        logger.info(f"Review repository count: {len(self.review_repo)}")
        predicate_count = len([item.get('predicate') for item in self.schema_builder.predicate_repo if item.get('predicate')])
        logger.info(f"Domain predicate count: {predicate_count}")
        logger.info(f"FOL example count: {len(self.schema_builder.fol_repo)}")
        logger.info("="*60)
        round_num = 1
        max_rounds = 3
        while len(self.review_repo) > 0 and round_num <= max_rounds:
            round_num += 1
            logger.info("="*60)
            logger.info(f"Starting round {round_num}, {len(self.review_repo)} clauses pending in Review repository")
            logger.info("="*60)
            
            review_clauses = [item.get('clause', '') for item in self.review_repo.copy()]
            self.review_repo = []
            self._save_repository(self.review_repo_path, self.review_repo)
            
            if review_clauses:
                all_texts_for_w2v = all_clauses + review_clauses
                self._build_word2vec_model(all_texts_for_w2v)
            
            review_results = []
            review_lock = threading.Lock()
            
            def review_worker(clause):
                doc_id = clause_to_doc_id.get(clause) if clause_to_doc_id else None
                result = self.process_single_clause(clause, doc_id=doc_id)
                with review_lock:
                    review_results.append(result)
            
            review_threads = []
            for clause in review_clauses:
                if not clause:
                    continue
                while len([t for t in review_threads if t.is_alive()]) >= max_workers:
                    time.sleep(0.1)
                    review_threads = [t for t in review_threads if t.is_alive()]
                
                thread = threading.Thread(target=review_worker, args=(clause,))
                thread.start()
                review_threads.append(thread)
            
            for thread in review_threads:
                thread.join()
            results.extend(review_results)
            
            formatted_results = self._format_results(results, clause_to_doc_id=clause_to_doc_id)
            round_output_file = self.output_root / f"results_round_{round_num}.json"
            with open(round_output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, ensure_ascii=False, indent=2)
            
            if len(self.review_repo) == 0:
                logger.info("🎉 All clauses processed successfully!")
                break
            
            if round_num >= max_rounds:
                logger.warning(f"Reached maximum processing rounds {max_rounds}")
                break
        final_formatted_results = self._format_results(results, clause_to_doc_id=clause_to_doc_id)
        final_output_file = self.output_root / "results_final.json"
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_formatted_results, f, ensure_ascii=False, indent=2)
        
        extract_data_dir = Path(self.pipeline_config.get('extract_data_dir', 'output/extract_data'))
        extract_data_dir.mkdir(parents=True, exist_ok=True)
        extract_data_file = extract_data_dir / "results_final.json"
        with open(extract_data_file, 'w', encoding='utf-8') as f:
            json.dump(final_formatted_results, f, ensure_ascii=False, indent=2)
        logger.info(f"  Extraction results saved to extract_data: {extract_data_file}")
        
        final_success_count = sum(1 for r in results if r.get('success', False))
        logger.info("="*60)
        logger.info("Final statistics:")
        logger.info(f"  Total clauses: {len(results)}")
        logger.info(f"  Successfully processed: {final_success_count} ({final_success_count/len(results)*100:.1f}%)")
        logger.info(f"  Processing rounds: {round_num}")
        logger.info(f"  Domain predicate count: {predicate_count}")
        logger.info(f"  FOL example count: {len(self.schema_builder.fol_repo)}")
        logger.info(f"  Final results saved: {final_output_file}")
        logger.info("="*60)
        
        return {
            'status': 'success',
            'results': results,
            'formatted_results': final_formatted_results,
            'output_file': str(final_output_file),
            'extract_data_file': str(extract_data_file),
            'statistics': {
                'total': len(results),
                'success': final_success_count,
                'rounds': round_num
            }
        }
    
    def run(self, input_pdf: Optional[str] = None, doc_id: Optional[str] = None,
            step: Optional[str] = None, sample_size: Optional[int] = None,
            pdf_analyze_only: bool = False, pdf_use_suggestions: bool = False) -> Dict:
        """Run full pipeline or a single step (pdf, clean, schema, extract, all)."""
        logger.info("="*60)
        logger.info("Starting knowledge extraction pipeline")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            if step is None or step == 'all':
                pdf_result = self.run_step_pdf_extraction(
                    input_pdf, 
                    analyze_only=pdf_analyze_only,
                    use_suggestions=pdf_use_suggestions
                )
                if pdf_result['status'] != 'success':
                    return pdf_result
                if pdf_analyze_only:
                    return pdf_result
                
                cleaned_result_dict = self.run_step_text_cleaning(pdf_result['pdf_result'])
                if cleaned_result_dict['status'] != 'success':
                    return cleaned_result_dict
                
                schema_result = self.run_step_schema_building(
                    cleaned_result_dict['cleaned_result'],
                    doc_id=cleaned_result_dict.get('doc_id') or doc_id
                )
                if schema_result['status'] != 'success':
                    return schema_result
                current_doc_id = cleaned_result_dict.get('doc_id') or doc_id or 'unknown'
                extraction_result = self.run_step_knowledge_extraction(sample_size=sample_size)
                if extraction_result.get('status') != 'success':
                    return extraction_result
                if self.kg_auto_construct and self.kg_constructor:
                    extract_data_file = extraction_result.get('extract_data_file')
                    if extract_data_file and Path(extract_data_file).exists():
                        try:
                            kg_stats = self.kg_constructor.construct_from_file(
                                extract_data_file, document_id=current_doc_id
                            )
                            logger.info(f"Knowledge graph construction completed: {kg_stats}")
                        except Exception as e:
                            logger.warning(f"Knowledge graph construction failed: {e}")
                    else:
                        logger.warning("extract_data result file does not exist, skipping knowledge graph construction")
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = {
                    'status': 'success',
                    'doc_id': current_doc_id,
                    'output_path': extraction_result.get('output_file'),
                    'extract_data_file': extraction_result.get('extract_data_file'),
                    'statistics': extraction_result.get('statistics', {}),
                    'duration_seconds': duration
                }
                
                logger.info("\n" + "="*60)
                logger.info("Pipeline execution completed")
                logger.info("="*60)
                logger.info(f"Duration: {duration:.2f} seconds")
                logger.info("="*60)
                
                return result
                
            elif step == 'pdf':
                return self.run_step_pdf_extraction(
                    input_pdf,
                    analyze_only=pdf_analyze_only,
                    use_suggestions=pdf_use_suggestions
                )
            elif step == 'clean':
                raise ValueError("clean step requires running pdf step first, use step='all' or run step='pdf' first")
            elif step == 'schema':
                raise ValueError("schema step requires running clean step first, use step='all'")
            elif step == 'extract':
                raise ValueError("extract step requires running schema step first, use step='all'")
            else:
                raise ValueError(f"Unknown step: {step}, available steps: pdf, clean, schema, extract, all")
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.exception(e)
            
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _load_global_schema(self) -> Optional[Dict]:
        """Load global schema if present."""
        if self.global_schema_path.exists():
            try:
                with open(self.global_schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                    logger.info(f"Global Schema loaded: {self.global_schema_path}")
                    return self._ensure_schema_defaults(schema)
            except Exception as exc:
                logger.warning(f"Failed to load Global Schema: {exc}")
        else:
            logger.info("Global Schema not detected, will be created in first document")
        return None
    
    def _ensure_schema_defaults(self, schema: Optional[Dict]) -> Dict:
        """Ensure schema has required keys."""
        schema = schema or {}
        schema.setdefault('clause_id', [])
        schema.setdefault('predicates', [])
        return schema
    
    def batch_run(self):
        """Process multiple PDFs in batch."""
        pdf_dir = self.pipeline_config['input_pdf_dir']
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        results = []
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file [{i+1}/{len(pdf_files)}]: {pdf_file.name}")
            logger.info('='*60)
            
            try:
                result = self.run(input_pdf=str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Processing failed: {pdf_file.name}, error: {str(e)}")
                results.append({
                    'status': 'error',
                    'file': str(pdf_file),
                    'message': str(e)
                })
        summary = {
            'total_files': len(pdf_files),
            'success': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = Path('output') / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nBatch processing completed: {summary['success']}/{summary['total_files']} succeeded")
        logger.info(f"Summary saved: {summary_path}")
        
        return summary


def main():
    """CLI entrypoint."""
    import argparse
    parser = argparse.ArgumentParser(description='Knowledge extraction pipeline (merged version)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--input', type=str, help='Single PDF file path')
    parser.add_argument('--doc-id', type=str, help='Document ID')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--step', type=str, choices=['pdf', 'clean', 'schema', 'extract', 'all'],
                       default='all', help='Step to run (pdf=PDF extraction, clean=text cleaning, schema=Schema building, extract=knowledge extraction, all=full pipeline)')
    parser.add_argument('--sample-size', type=int, help='Sample size (only for extract step)')
    parser.add_argument('--output-base', type=str, help='Output root directory')
    parser.add_argument('--pdf-analyze-only', action='store_true', 
                       help='PDF processing mode: analysis only, generate parameter suggestion file, no cleaning')
    parser.add_argument('--pdf-use-suggestions', action='store_true',
                       help='PDF processing mode: use previous analysis suggestions for direct cleaning (skip analysis stage)')
    
    args = parser.parse_args()
    output_base = Path(args.output_base) if args.output_base else None
    pipeline = KnowledgeExtractionPipeline(config_path=args.config, output_base_dir=output_base)
    if args.batch:
        pipeline.batch_run()
    else:
        pipeline.run(
            input_pdf=args.input, 
            doc_id=args.doc_id, 
            step=args.step, 
            sample_size=args.sample_size,
            pdf_analyze_only=args.pdf_analyze_only,
            pdf_use_suggestions=args.pdf_use_suggestions
        )


if __name__ == "__main__":
    main()
