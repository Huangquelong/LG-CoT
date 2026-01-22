"""
Baseline experiment pipeline
Implements three knowledge extraction methods: standard CoT prompting, random few-shot, and semantic few-shot
"""

import os
import sys
import json
import threading
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import numpy as np
from openai import OpenAI
import jieba
from gensim.models import Word2Vec
from loguru import logger
import yaml

# Configure logging
log_dir = Path("Baseline/Validation")
log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {message}", level="INFO")
logger.add(log_dir / "baseline_experiment.log", format="{time} | {level} | {message}", level="INFO", rotation="100 MB")


class RateLimiter:
    """Rate limiter supporting RPM and TPM limits"""
    
    def __init__(self, rpm: int = 2000, tpm: int = 80000, max_sleep: float = 5.0):
        self.rpm = rpm  # Requests per minute
        self.tpm = tpm  # Tokens per minute
        self.max_sleep = max_sleep  # Maximum sleep time per wait to avoid long blocking
        self.request_times = deque()  # Request timestamp queue
        self.token_times = deque()  # (timestamp, token_count) queue
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if needed until request can be sent (segmented waiting to avoid long blocking)"""
        while True:
            with self.lock:
                now = time.time()
                one_minute_ago = now - 60
                
                # Clean expired request records
                while self.request_times and self.request_times[0] < one_minute_ago:
                    self.request_times.popleft()
                while self.token_times and self.token_times[0][0] < one_minute_ago:
                    self.token_times.popleft()
                
                # Calculate wait time needed
                rpm_wait = 0.0
                if len(self.request_times) >= self.rpm:
                    rpm_wait = max(0.0, 60 - (now - self.request_times[0]))
                
                current_tpm = sum(tokens for _, tokens in self.token_times)
                tpm_wait = 0.0
                if current_tpm + estimated_tokens > self.tpm and self.token_times:
                    tpm_wait = max(0.0, 60 - (now - self.token_times[0][0]))
                
                needed_wait = max(rpm_wait, tpm_wait)
                
                # If no wait needed, record request and return
                if needed_wait <= 0:
                    self.request_times.append(now)
                    self.token_times.append((now, estimated_tokens))
                    return
                
                # Need to wait, segmented waiting to avoid long blocking
                sleep_time = min(needed_wait, self.max_sleep)
                logger.warning(f"Rate limit: need to wait ~{needed_wait:.2f}s, waiting {sleep_time:.2f}s this time")
            
            time.sleep(sleep_time)


class BaselineExperimentPipeline:
    """Baseline experiment pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Get LLM API configuration
        llm_api_config = self.config.get('llm_api', {}).get('SiliconFlow', {})
        self.api_key = llm_api_config.get('api_key', '')
        self.base_url = llm_api_config.get('api_base', 'https://api.siliconflow.cn/v1')
        
        # Get baseline experiment configuration
        baseline_config = self.config.get('baseline_experiment', {}).get('baseline', {})
        self.model = baseline_config.get('model', 'Qwen/Qwen3-14B')
        
        # Initialize LLM client
        self.llm_client = self._init_llm_client()
        
        # Rate limiter configuration
        rate_limiter_config = baseline_config.get('rate_limiter', {})
        self.rate_limiter = RateLimiter(
            rpm=rate_limiter_config.get('rpm', 2000),
            tpm=rate_limiter_config.get('tpm', 80000),
            max_sleep=rate_limiter_config.get('max_sleep', 5.0)
        )
        
        # Output directory
        self.output_root = Path(baseline_config.get('output_dir', 'Experiment_result'))
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Store config for later use
        self.baseline_config = baseline_config
        
        # Load output template
        template_path = Path("output_template.json")
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                self.output_template = json.load(f)
        else:
            # Default template
            self.output_template = {
                "doc_id": "GB_XXXX_2024",
                "rules": [{
                    "rule_id": "R_001",
                    "type": "OBLIGATION",
                    "core_event": {
                        "subject": "Subject",
                        "action": "Action",
                        "object": "Object"
                    },
                    "conditions": [],
                    "constraints": {
                        "time_limit": "",
                        "frequency": "",
                        "manner": ""
                    },
                    "source_text": "..."
                }]
            }
        
        # Data storage
        self.all_source_texts = []
        self.source_text_to_index = {}
        self.word2vec_model = None
        self.text_vectors = None
        
        logger.info("Baseline experiment pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_llm_client(self):
        """Initialize LLM client"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info("LLM client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def _StandCoT_KE(self, clause_text: str) -> str:
        """Standard CoT prompting knowledge extraction template"""
        template_str = json.dumps(self.output_template, ensure_ascii=False, indent=2)
        prompt = f"""You are a knowledge extraction expert. Please analyze the input clause text according to the following approach and output in the given JSON structure.
1. Analyze all entities in the clause
2. Analyze all conditions and constraints in the clause
3. Fill the analyzed content into the JSON structure template
4. Only output JSON structure content, no other text

## Clause Text
{clause_text}

## JSON Structure Template
{template_str}
"""
        return prompt
    
    def _Random_few_shot_KE(self, clause_text: str, few_shot_examples: List[Dict]) -> str:
        """Random few-shot prompting knowledge extraction template"""
        examples_str = ""
        for i, example in enumerate(few_shot_examples, 1):
            example_str = json.dumps(example, ensure_ascii=False, indent=2)
            examples_str += f"\n### Example {i}\nInput clause: {example.get('source_text', '')}\nOutput:\n{example_str}\n"
        
        prompt = f"""You are a knowledge extraction expert. Please complete the same knowledge extraction task on my input clause text according to the examples I provide, and output in the given JSON structure.

## Clause Text
{clause_text}

## Examples
{examples_str}

## Note: Only output JSON structure content, no other text
"""
        return prompt
    
    def _semantic_few_shot_KE(self, clause_text: str, few_shot_examples: List[Dict]) -> str:
        """Semantic similarity few-shot prompting knowledge extraction template"""
        examples_str = ""
        for i, example in enumerate(few_shot_examples, 1):
            example_str = json.dumps(example, ensure_ascii=False, indent=2)
            examples_str += f"\n### Example {i}\nInput clause: {example.get('source_text', '')}\nOutput:\n{example_str}\n"
        
        prompt = f"""You are a knowledge extraction expert. Please complete the same knowledge extraction task on my input clause text according to the examples I provide, and output in the given JSON structure.

## Clause Text
{clause_text}

## Examples
{examples_str}

## Note: Only output JSON structure content, no other text
"""
        return prompt
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM API"""
        llm_params = self.baseline_config.get('llm_params', {})
        temperature = llm_params.get('temperature', 0.3)
        max_tokens = llm_params.get('max_tokens', 2048)
        
        for attempt in range(max_retries):
            try:
                # Estimate token count (simple estimation: 1 token ≈ 0.75 Chinese characters)
                estimated_tokens = int(len(prompt) * 1.5) + 1000
                
                # Rate limiting
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                # Call API
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"LLM call finally failed: {e}")
                    return None
        
        return None
    
    def _load_dataset(self, input_file: str) -> List[Dict]:
        """Load dataset, deduplicate identical source_text"""
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path.absolute()}")
            return []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all source_text, use dictionary for deduplication (key is source_text)
        # For identical source_text, only keep the first occurrence
        source_text_to_item = {}
        total_rules_count = 0
        
        for doc_entry in data:
            doc_id = doc_entry.get('doc_id', 'UNKNOWN')
            rules = doc_entry.get('rules', [])
            for rule in rules:
                total_rules_count += 1
                source_text = rule.get('source_text', '').strip()
                if source_text:
                    # If source_text already exists, skip (deduplication)
                    if source_text not in source_text_to_item:
                        source_text_to_item[source_text] = {
                            'doc_id': doc_id,
                            'rule_id': rule.get('rule_id', ''),
                            'source_text': source_text,
                            'original_rule': rule
                        }
        
        # Convert to list
        all_items = list(source_text_to_item.values())
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  Total original rules: {total_rules_count}")
        logger.info(f"  Deduplicated source_text count: {len(all_items)}")
        logger.info(f"  Duplicate count: {total_rules_count - len(all_items)}")
        
        return all_items
    
    def _build_word2vec_model(self, all_items: List[Dict]):
        """Build Word2Vec model and compute vectors for all texts"""
        logger.info("Building Word2Vec model...")
        
        # Prepare training data (tokenized text list)
        sentences = []
        for item in all_items:
            text = item['source_text']
            words = list(jieba.cut(text))
            sentences.append(words)
        
        # Get Word2Vec configuration
        word2vec_config = self.baseline_config.get('word2vec', {})
        vector_size = word2vec_config.get('vector_size', 100)
        window = word2vec_config.get('window', 5)
        min_count = word2vec_config.get('min_count', 1)
        workers = word2vec_config.get('workers', 4)
        sg = word2vec_config.get('sg', 0)  # 0=CBOW, 1=Skip-gram
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg
        )
        
        # Compute vectors for all texts (using average word vectors)
        self.text_vectors = []
        for item in all_items:
            text = item['source_text']
            words = list(jieba.cut(text))
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                vector_size = self.baseline_config.get('word2vec', {}).get('vector_size', 100)
                text_vector = np.zeros(vector_size)
            
            self.text_vectors.append(text_vector)
        
        self.text_vectors = np.array(self.text_vectors)
        logger.info(f"Word2Vec model built, vector dimension: {self.text_vectors.shape}")
    
    def _find_semantic_similar(self, target_text: str, target_index: int, top_k: int = 3) -> List[int]:
        """Find top_k semantically most similar texts (excluding self)"""
        if self.text_vectors is None:
            return []
        
        # Compute target text vector
        words = list(jieba.cut(target_text))
        word_vectors = []
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        if not word_vectors:
            return []
        
        target_vector = np.mean(word_vectors, axis=0)
        
        # Compute cosine similarity
        similarities = np.dot(self.text_vectors, target_vector) / (
            np.linalg.norm(self.text_vectors, axis=1) * np.linalg.norm(target_vector) + 1e-8
        )
        
        # Exclude self, get top_k
        similarities[target_index] = -1  # Exclude self
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return top_indices.tolist()
    
    def _load_checkpoint(self, checkpoint_path: Path) -> Tuple[int, List[Dict]]:
        """Load checkpoint"""
        if not checkpoint_path.exists():
            return 0, []
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            processed_count = checkpoint_data.get('processed_count', 0)
            results = checkpoint_data.get('results', [])
            
            logger.info(f"Resumed from checkpoint: processed {processed_count} items, {len(results)} results")
            return processed_count, results
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0, []
    
    def _save_checkpoint(self, checkpoint_path: Path, processed_count: int, results: List[Dict]):
        """Save checkpoint"""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            'processed_count': processed_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path} (processed {processed_count} items)")
    
    def _process_single_item(self, item: Dict, method: str, all_items: List[Dict], 
                            item_index: int, lock: threading.Lock, results: List[Dict],
                            progress_counter: Dict, processed_source_texts: set) -> Optional[Dict]:
        """Process single clause"""
        source_text = item['source_text']
        
        # Check if already processed (prevent duplicate processing)
        with lock:
            if source_text in processed_source_texts:
                logger.debug(f"Skipping already processed source_text: {source_text[:50]}...")
                return None
            processed_source_texts.add(source_text)
        
        try:
            if method == "StandCoT":
                prompt = self._StandCoT_KE(source_text)
            elif method == "Random_few_shot":
                # Randomly sample 3 examples (excluding self)
                candidate_indices = [i for i in range(len(all_items)) if i != item_index]
                if len(candidate_indices) >= 3:
                    sample_indices = random.sample(candidate_indices, 3)
                else:
                    sample_indices = candidate_indices
                
                few_shot_examples = [all_items[i]['original_rule'] for i in sample_indices]
                prompt = self._Random_few_shot_KE(source_text, few_shot_examples)
            elif method == "semantic_few_shot":
                # Semantic similarity sampling top3 (excluding self)
                similar_indices = self._find_semantic_similar(source_text, item_index, top_k=3)
                if similar_indices:
                    few_shot_examples = [all_items[i]['original_rule'] for i in similar_indices]
                else:
                    # If no similar found, randomly sample
                    candidate_indices = [i for i in range(len(all_items)) if i != item_index]
                    if candidate_indices:
                        sample_indices = random.sample(candidate_indices, min(3, len(candidate_indices)))
                        few_shot_examples = [all_items[i]['original_rule'] for i in sample_indices]
                    else:
                        few_shot_examples = []
                
                prompt = self._semantic_few_shot_KE(source_text, few_shot_examples)
            else:
                logger.error(f"Unknown method: {method}")
                return None
            
            # Call LLM
            response = self._call_llm(prompt)
            
            if response:
                # Try to parse JSON
                try:
                    # Extract JSON part (may contain markdown code blocks)
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()
                    
                    result_json = json.loads(response)
                    
                    # Add metadata
                    result = {
                        'doc_id': item['doc_id'],
                        'rule_id': item['rule_id'],
                        'source_text': source_text,
                        'method': method,
                        'extracted_result': result_json,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with lock:
                        # Check again for duplicates (prevent concurrent duplicate addition)
                        existing_source_texts = {r['source_text'] for r in results}
                        if source_text not in existing_source_texts:
                            results.append(result)
                            progress_counter['processed'] += 1
                            current_processed = progress_counter['processed']
                            if current_processed % 10 == 0:
                                logger.info(f"[{method}] Processed {current_processed}/{progress_counter['total']} items")
                        else:
                            logger.warning(f"Detected duplicate source_text, skipping save: {source_text[:50]}...")
                    
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}\nResponse content: {response[:200]}")
                    return None
            else:
                logger.warning(f"LLM returned empty: {source_text[:50]}...")
                return None
                
        except Exception as e:
            logger.error(f"Failed to process clause: {e}\nClause: {source_text[:50]}...")
            return None
    
    def _run_method(self, method: str, all_items: List[Dict], output_dir: Path):
        """Run single method"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting method: {method}")
        logger.info(f"{'='*60}")
        
        # Check checkpoint
        checkpoint_path = output_dir / f"checkpoint_{method}.json"
        start_index, existing_results = self._load_checkpoint(checkpoint_path)
        
        # Set of processed source_text
        processed_source_texts = {r['source_text'] for r in existing_results}
        
        # Filter unprocessed items
        remaining_items = []
        remaining_indices = []
        for i, item in enumerate(all_items):
            if item['source_text'] not in processed_source_texts:
                remaining_items.append(item)
                remaining_indices.append(i)
        
        logger.info(f"Total items: {len(all_items)}, Processed: {len(existing_results)}, Remaining: {len(remaining_items)}")
        
        if not remaining_items:
            logger.info(f"Method {method} already completed, no processing needed")
            return existing_results
        
        # Multi-threaded processing
        results = existing_results.copy()
        lock = threading.Lock()
        # Use set to track processed source_text (including checkpoint and in-progress)
        processed_source_texts = processed_source_texts.copy()
        progress_counter = {
            'processed': len(existing_results), 
            'total': len(all_items), 
            'last_checkpoint': len(existing_results),
            'last_saved': len(existing_results)
        }
        
        def worker(item, item_index, global_index):
            result = self._process_single_item(
                item, method, all_items, global_index, lock, results, progress_counter, processed_source_texts
            )
            return result
        
        # Create thread pool (set thread count based on RPM limit)
        max_workers_config = self.baseline_config.get('max_workers', 20)
        max_workers = min(max_workers_config, len(remaining_items))
        threads = []
        
        for idx, (item, global_idx) in enumerate(zip(remaining_items, remaining_indices)):
            # Wait until thread available
            while len([t for t in threads if t.is_alive()]) >= max_workers:
                time.sleep(0.1)
                threads = [t for t in threads if t.is_alive()]
                
                # Check if checkpoint save needed (every 10 items)
                with lock:
                    current_processed = progress_counter['processed']
                    need_save = current_processed - progress_counter.get('last_saved', 0) >= 10
                    if need_save:
                        progress_counter['last_checkpoint'] = current_processed
                        progress_counter['last_saved'] = current_processed
                        results_copy = results.copy()
                    else:
                        results_copy = None
                
                if need_save and results_copy is not None:
                    self._save_checkpoint(checkpoint_path, current_processed, results_copy)
            
            # Create new thread
            thread = threading.Thread(
                target=worker,
                args=(item, idx, global_idx)
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Final checkpoint check
        with lock:
            current_processed = progress_counter['processed']
            if current_processed > progress_counter.get('last_saved', 0):
                results_copy = results.copy()
                self._save_checkpoint(checkpoint_path, current_processed, results_copy)
        
        # Save final checkpoint
        self._save_checkpoint(checkpoint_path, len(results), results)
        
        logger.info(f"Method {method} completed, processed {len(results)} items")
        return results
    
    def run(self):
        """Run experiment"""
        logger.info("="*60)
        logger.info("Baseline experiment pipeline started")
        logger.info("="*60)
        
        # 1. Load dataset
        input_file_path = self.baseline_config.get('input_file', 'Temporary_tool/silver_standard_dataset_updated.json')
        input_file = Path(input_file_path)
        all_items = self._load_dataset(str(input_file))
        
        if not all_items:
            logger.error("Dataset is empty, exiting")
            return
        
        # 2. Build Word2Vec model (for semantic similarity calculation)
        logger.info("Building Word2Vec model...")
        self._build_word2vec_model(all_items)
        
        # 3. Run three methods
        methods = ["StandCoT", "Random_few_shot", "semantic_few_shot"]
        
        for method in methods:
            method_output_dir = self.output_root / method
            method_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run method
            results = self._run_method(method, all_items, method_output_dir)
            
            # Save results
            output_file = method_output_dir / f"results_{method}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Method {method} results saved to: {output_file}")
        
        logger.info("="*60)
        logger.info("Baseline experiment pipeline completed")
        logger.info("="*60)


def run():
    """Main function"""
    pipeline = BaselineExperimentPipeline()
    pipeline.run()


if __name__ == '__main__':
    run()
