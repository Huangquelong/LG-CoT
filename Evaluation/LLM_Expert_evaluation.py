"""LLM expert evaluation module: use DeepSeek-V3 model as judger to evaluate knowledge extraction results."""

import os
import sys
import json
import threading
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque, defaultdict
from openai import OpenAI
from loguru import logger
import yaml

# Configure logging
log_dir = Path("Evaluation")
log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {message}", level="INFO")
logger.add(log_dir / "llm_expert_evaluation.log", format="{time} | {level} | {message}", level="INFO", rotation="100 MB")


class RateLimiter:
    """Rate limiter supporting RPM and TPM limits."""
    
    def __init__(self, rpm: int = 2000, tpm: int = 80000, max_sleep: float = 5.0):
        self.rpm = rpm
        self.tpm = tpm
        self.max_sleep = max_sleep
        self.request_times = deque()
        self.token_times = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if needed until request can be sent (segmented waiting to avoid long blocking)."""
            while True:
            with self.lock:
                now = time.time()
                one_minute_ago = now - 60
                
                while self.request_times and self.request_times[0] < one_minute_ago:
                    self.request_times.popleft()
                while self.token_times and self.token_times[0][0] < one_minute_ago:
                    self.token_times.popleft()
                
                rpm_wait = 0.0
                if len(self.request_times) >= self.rpm:
                    rpm_wait = max(0.0, 60 - (now - self.request_times[0]))
                
                current_tpm = sum(tokens for _, tokens in self.token_times)
                tpm_wait = 0.0
                if current_tpm + estimated_tokens > self.tpm and self.token_times:
                    tpm_wait = max(0.0, 60 - (now - self.token_times[0][0]))
                
                needed_wait = max(rpm_wait, tpm_wait)
                
                if needed_wait <= 0:
                    self.request_times.append(now)
                    self.token_times.append((now, estimated_tokens))
                    return
                
                sleep_time = min(needed_wait, self.max_sleep)
                logger.warning(f"Rate limit, need to wait approximately {needed_wait:.2f} seconds, waiting {sleep_time:.2f} seconds this time")
            
            time.sleep(sleep_time)


class LLMExpertEvaluation:
    """LLM expert evaluation class"""
    
    def __init__(self, config_path: str = "config.yaml", silver_standard_path: str = None):
        """Initialize evaluation module
        
        Args:
            config_path: Path to configuration file
            silver_standard_path: Path to silver standard dataset (overrides config if provided)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Get LLM API configuration
        llm_api_config = self.config.get('llm_api', {}).get('SiliconFlow', {})
        self.api_key = llm_api_config.get('api_key', '')
        self.base_url = llm_api_config.get('api_base', 'https://api.siliconflow.cn/v1')
        
        # Get evaluation configuration
        eval_config = self.config.get('evaluation', {}).get('llm_expert', {})
        self.model = eval_config.get('model', 'deepseek-ai/DeepSeek-V3')
        
        # Initialize LLM client
        self.llm_client = self._init_llm_client()
        
        # Rate limiter configuration
        rate_limiter_config = eval_config.get('rate_limiter', {})
        self.rate_limiter = RateLimiter(
            rpm=rate_limiter_config.get('rpm', 2000),
            tpm=rate_limiter_config.get('tpm', 80000),
            max_sleep=rate_limiter_config.get('max_sleep', 5.0)
        )
        
        # Output directory
        output_dir = eval_config.get('output_dir', 'Evaluation/evaluation_results')
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Silver standard path
        if silver_standard_path is None:
            silver_standard_path = eval_config.get('silver_standard_path', 'Temporary_tool/silver_standard_dataset_updated.json')
        
        # Load silver standard dataset, build doc_id to ai_category_tag mapping
        self.doc_id_to_category = self._load_silver_standard(silver_standard_path)
        
        logger.info("LLM expert evaluation module initialized")
        logger.info(f"Loaded {len(self.doc_id_to_category)} doc_id category mappings")
    
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
    
    def _load_silver_standard(self, silver_standard_path: str) -> Dict[str, str]:
        """Load silver standard dataset and build doc_id to ai_category_tag mapping
        
        Args:
            silver_standard_path: Path to silver standard dataset
            
        Returns:
            Dictionary mapping doc_id to ai_category_tag
        """
        doc_id_to_category = {}
        silver_standard_file = Path(silver_standard_path)
        
        if not silver_standard_file.exists():
            logger.warning(f"Silver standard dataset file does not exist: {silver_standard_file}")
            return doc_id_to_category
        
        try:
            with open(silver_standard_file, 'r', encoding='utf-8') as f:
                silver_data = json.load(f)
            
            for item in silver_data:
                doc_id = item.get('doc_id', '')
                ai_category_tag = item.get('ai_category_tag', '')
                if doc_id and ai_category_tag:
                    doc_id_to_category[doc_id] = ai_category_tag
            
            logger.info(f"Successfully loaded {len(doc_id_to_category)} doc_id category mappings")
            category_counts = defaultdict(int)
            for category in doc_id_to_category.values():
                category_counts[category] += 1
            logger.info(f"Category statistics: {dict(category_counts)}")
            
        except Exception as e:
            logger.error(f"Failed to load silver standard dataset: {e}")
        
        return doc_id_to_category
    
    def _build_evaluation_prompt(self, candidate: Dict, source_text: str) -> str:
        """Build evaluation prompt."""
        candidate_str = json.dumps(candidate, ensure_ascii=False, indent=2)
        
        prompt = f"""
# Role
你是一位资深的自然语言处理领域专家。你的任务是评估从非结构化工程文本中提取的"Candidate"的语义质量和机器可读性。

# Core Philosophy
1) Semantic Disambiguation：高质量的抽取应能消除自然语言的模糊性，将隐含的上下文信息（如缺省的主语）显式化。
2) Logical Executability：评估结果应不仅仅是文本的分类整理，更应具备直接被规则引擎或形式化验证系统调用的潜力。
3) Information Atomicity：复合的约束条件应被拆解为最小的语义单元（变量、算符、阈值），而非保留长难句。

# Evaluation Criteria & Rubric (0-7, step=0.5)

A. IA (Information Accuracy)：
- 7.0: 精确指代消解。不仅忠实还原原文，且能根据工程语境准确补全隐式主语（如将"其"或缺省主语还原为具体的"施工人员"、"监理单位"）。
- 5.0-6.0: 核心语义正确，对主语进行了合理的具体化推断。
- 3.0-4.0: 泛化指代。关键要素无误，但使用缺乏信息量的泛化占位符（如"责任主体"、"相关人员"），导致规则执行主体不明确。
- 0.0-2.0: 出现幻觉、主客体颠倒或关键数值错误。

B. IR  (Information Recall)：
- 7.0: 全量约束捕获。完整提取所有限制性条件，包括环境前提（Preconditions）和例外条款（Exceptions），无遗漏。
- 5.0-6.0: 捕获了核心约束，仅遗漏对逻辑判定影响极小的修饰语。
- 3.0-4.0: 遗漏了次要约束，或遗漏了否定逻辑（如"不得"）。
- 0.0-2.0: 遗漏关键定量指标或限制性前提。

C. LC  (Logical Coherence)：
- 7.0: 形式化表达。逻辑关系除了依靠JSON结构，还通过显式的逻辑符号或表达式来精确定义条件与动作的关联，消除了自然语言连接词的歧义。
- 5.0-6.0: 清晰的逻辑分层。触发条件（Conditions）与执行要求（Constraints）区分清晰，逻辑链条无断裂。
- 3.0-4.0: 隐式逻辑。逻辑关系正确，但仅以自然语言列表的形式呈现，缺乏显式的逻辑算符连接。
- 0.0-2.0: 逻辑自相矛盾，或因果倒置。

D. U  (Usability):
- 7.0: 直接可计算。数值约束已实现参数化分离（变量、算符、阈值完全解耦），下游规则引擎无需二次解析即可直接使用。
- 5.0-6.0: 结构化。关键指标已从文本中剥离，但部分单位或算符仍与数值粘连，需要简单的正则处理。
- 3.0-4.0: 文本归档级 。内容被正确分类到了不同字段，但字段内部仍是自然语言描述，无法直接进行数值比较或逻辑运算。
- 0.0-2.0: 非结构化。大量使用占位符，或字段内容几乎是原句拷贝。

# Required outputs
1) scores: 四个分数 (Float, 0-7)
2) error_summary: 若有问题，列出错误类型（可多选）：[hallucination, subject_vague, relation_wrong, condition_not_atomic, logic_ambiguous, value_unit_mixed, placeholder_usage, missing_constraints]

# Output JSON only
{{
  "IA_score": <float>,
  "IR_score": <float>,
  "LC_score": <float>,
  "U_score": <float>,
  "error_summary": [ ... ],
}}

# Input
- SourceText: {source_text}
- Candidate: {candidate_str}

Please output only JSON format evaluation results, no other text.
"""
        return prompt
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM API."""
        for attempt in range(max_retries):
            try:
                estimated_tokens = int(len(prompt) * 1.5) + 2000
                
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048
                )
                
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM call finally failed: {e}")
                    return None
        
        return None
    
    def _clean_baseline_candidate(self, item: Dict) -> Dict:
        """Clean baseline experiment results, keep only extracted_result."""
        extracted_result = item.get('extracted_result', {})
        if isinstance(extracted_result, dict):
            cleaned_result = extracted_result.copy()
            cleaned_result.pop('source_text', None)
            return cleaned_result
        return extracted_result
    
    def _clean_lgcot_candidate(self, rule: Dict) -> Dict:
        """Clean LGCoT results, remove source_text field."""
        cleaned_rule = rule.copy()
        cleaned_rule.pop('source_text', None)
        return cleaned_rule
    
    def _load_baseline_results(self, baseline_dir: Path, methods: Optional[List[str]] = None) -> List[Dict]:
        """Load baseline experiment results."""
        results = []
        
        default_methods = ["StandCoT", "Random_few_shot", "semantic_few_shot"]
        methods_to_load = methods if methods is not None else default_methods
        
        for method in methods_to_load:
            method_dir = baseline_dir / method
            if not method_dir.exists():
                logger.warning(f"Method directory does not exist: {method_dir}")
                continue
            
            results_file = method_dir / f"results_{method}.json"
            if not results_file.exists():
                logger.warning(f"Results file does not exist: {results_file}")
                continue
            
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    method_results = json.load(f)
                
                model_name = baseline_dir.name.replace("-Experiment_result", "")
                
                for item in method_results:
                    source_text = item.get('source_text', '')
                    if not source_text:
                        continue
                    
                    candidate = self._clean_baseline_candidate(item)
                    
                    results.append({
                        'doc_id': item.get('doc_id', ''),
                        'model': model_name,
                        'method': method,
                        'source_text': source_text,
                        'candidate': candidate,
                        'type': 'baseline'
                    })
                
                logger.info(f"Loaded baseline results: {model_name}/{method} - {len(method_results)} items")
                
            except Exception as e:
                logger.error(f"Failed to load baseline results {results_file}: {e}")
                continue
        
        return results
    
    def _load_lgcot_results(self, model_dir: Path) -> List[Dict]:
        """Load LGCoT results from lgcot directory within model folder."""
        results = []
        
        lgcot_dir = model_dir / "lgcot"
        if not lgcot_dir.exists():
            logger.warning(f"LGCoT directory does not exist: {lgcot_dir}")
            return results
        
        results_files = list(lgcot_dir.rglob("results_final.json"))
        
        if not results_files:
            results_file = lgcot_dir / "results_final.json"
            if results_file.exists():
                results_files = [results_file]
        
        for results_file in results_files:
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    lgcot_results = json.load(f)
                
                model_name = model_dir.name.replace("-Experiment_result", "")
                
                for item in lgcot_results:
                    doc_id = item.get('id', '')
                    rules = item.get('rules', [])
                    
                    for rule in rules:
                        source_text = rule.get('source_text', '')
                        if not source_text:
                            continue
                        
                        candidate = self._clean_lgcot_candidate(rule)
                        
                        results.append({
                            'doc_id': doc_id,
                            'model': model_name,
                            'method': 'LGCoT',
                            'source_text': source_text,
                            'candidate': candidate,
                            'type': 'lgcot'
                        })
                
                logger.info(f"Loaded LGCoT results: {model_name} ({results_file}) - {len(lgcot_results)} documents")
                
            except Exception as e:
                logger.error(f"Failed to load LGCoT results {results_file}: {e}")
                continue
        
        return results
    
    def _load_all_results(self, experiment_result_dir: Path, 
                          model_filter: Optional[str] = None,
                          methods: Optional[List[str]] = None) -> List[Dict]:
        """Load all experiment results."""
        all_results = []
        
        default_methods = ["StandCoT", "Random_few_shot", "semantic_few_shot", "LGCoT"]
        methods_to_load = methods if methods is not None else default_methods
        
        if not experiment_result_dir.exists():
            logger.warning(f"Experiment result directory does not exist: {experiment_result_dir}")
            return all_results
        
        for model_dir in experiment_result_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.endswith("Experiment_result"):
                continue
            
            model_name = model_dir.name.replace("-Experiment_result", "")
            
            if model_filter is not None:
                if model_filter.lower() not in model_name.lower():
                    logger.debug(f"Skipping model: {model_name} (does not match filter: {model_filter})")
                    continue
            
            baseline_methods = [m for m in methods_to_load if m != "LGCoT"]
            if baseline_methods:
                baseline_results = self._load_baseline_results(model_dir, baseline_methods)
                all_results.extend(baseline_results)
            
            if "LGCoT" in methods_to_load:
                lgcot_results = self._load_lgcot_results(model_dir)
                all_results.extend(lgcot_results)
        
        logger.info(f"Total loaded {len(all_results)} evaluation tasks (before deduplication)")
        
        seen_keys = set()
        deduplicated_results = []
        duplicate_count = 0
        for item in all_results:
            unique_key = f"{item['doc_id']}_{item['model']}_{item['method']}"
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated_results.append(item)
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.info(f"After deduplication: {len(deduplicated_results)} items (removed {duplicate_count} duplicates)")
        else:
            logger.info(f"No duplicates, total {len(deduplicated_results)} items")
        
        return deduplicated_results
    
    def _stratified_sampling(self, items: List[Dict], simple_count: int = 25, compound_count: int = 25) -> List[Dict]:
        """Stratified sampling by ai_category_tag."""
        model_method_groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"Simple": [], "Compound": [], "Unknown": []})
        
        for item in items:
            doc_id = item.get('doc_id', '')
            model = item.get('model', '')
            method = item.get('method', '')
            category = self.doc_id_to_category.get(doc_id, 'Unknown')
            
            model_method_key = f"{model}_{method}"
            
            if category in ["Simple", "Compound"]:
                model_method_groups[model_method_key][category].append(item)
            else:
                model_method_groups[model_method_key]["Unknown"].append(item)
        
        sampled_items = []
        
        for model_method_key, category_groups in model_method_groups.items():
            simple_items = category_groups["Simple"]
            compound_items = category_groups["Compound"]
            unknown_items = category_groups["Unknown"]
            
            if len(simple_items) >= simple_count:
                sampled_simple = random.sample(simple_items, simple_count)
            else:
                sampled_simple = simple_items
                logger.warning(f"{model_method_key} - Simple category has only {len(simple_items)} items, less than required {simple_count}")
            
            if len(compound_items) >= compound_count:
                sampled_compound = random.sample(compound_items, compound_count)
            else:
                sampled_compound = compound_items
                logger.warning(f"{model_method_key} - Compound category has only {len(compound_items)} items, less than required {compound_count}")
            
            sampled_items.extend(sampled_simple)
            sampled_items.extend(sampled_compound)
            
            logger.info(f"{model_method_key} - Sampling results: Simple {len(sampled_simple)}/{len(simple_items)}, "
                       f"Compound {len(sampled_compound)}/{len(compound_items)}, "
                       f"Unknown {len(unknown_items)} (unused), total {len(sampled_simple) + len(sampled_compound)} items")
        
        logger.info(f"Stratified sampling completed: total {len(sampled_items)} items (from {len(model_method_groups)} (model, method) combinations)")
        return sampled_items
    
    def _load_checkpoint(self, checkpoint_path: Path) -> Tuple[int, List[Dict]]:
        """Load checkpoint."""
        if not checkpoint_path.exists():
            return 0, []
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            processed_count = checkpoint_data.get('processed_count', 0)
            results = checkpoint_data.get('results', [])
            
            logger.info(f"Restored from checkpoint: processed {processed_count} items, results {len(results)} items")
            return processed_count, results
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0, []
    
    def _save_checkpoint(self, checkpoint_path: Path, processed_count: int, results: List[Dict]):
        """Save checkpoint."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            'processed_count': processed_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path} (processed {processed_count} items)")
    
    def _evaluate_single_item(self, item: Dict, lock: threading.Lock, results: List[Dict],
                              progress_counter: Dict, processed_keys: set) -> Optional[Dict]:
        """Evaluate a single item."""
        unique_key = f"{item['doc_id']}_{item['model']}_{item['method']}"
        
        with lock:
            if unique_key in processed_keys:
                logger.debug(f"Skipping already processed item: {unique_key}")
                return None
            processed_keys.add(unique_key)
        
        try:
            prompt = self._build_evaluation_prompt(item['candidate'], item['source_text'])
            
            response = self._call_llm(prompt)
            
            if response:
                try:
                    cleaned = response.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

                    if "```json" in cleaned:
                        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                    elif "```" in cleaned:
                        cleaned = cleaned.split("```")[1].split("```")[0].strip()
                    
                    evaluation_result = json.loads(cleaned)
                    
                    result = {
                        'doc_id': item['doc_id'],
                        'model': item['model'],
                        'method': item['method'],
                        'candidate': item['candidate'],
                        'evaluation': evaluation_result,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with lock:
                        existing_keys = {f"{r['doc_id']}_{r['model']}_{r['method']}" for r in results}
                        if unique_key not in existing_keys:
                            results.append(result)
                            progress_counter['processed'] += 1
                            current_processed = progress_counter['processed']
                            if current_processed % 10 == 0:
                                logger.info(f"Processed {current_processed}/{progress_counter['total']} items")
                        else:
                            logger.warning(f"Detected duplicate item, skipping save: {unique_key}")
                    
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}\nResponse content: {response[:200]}")
                    return None
            else:
                logger.warning(f"LLM returned empty: {item['doc_id']}/{item['model']}/{item['method']}")
                return None
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}\nItem: {item['doc_id']}/{item['model']}/{item['method']}")
            return None
    
    def run(self, experiment_result_dir: str = "Validation/Experiment_result",
            model: Optional[str] = None,
            methods: Optional[List[str]] = None,
            max_items: Optional[int] = None):
        """Run evaluation."""
        logger.info("="*60)
        logger.info("LLM Expert Evaluation Started")
        logger.info("="*60)
        
        experiment_result_path = Path(experiment_result_dir)
        
        all_items = self._load_all_results(experiment_result_path, model_filter=model, methods=methods)
        
        if max_items is None:
            random.seed(42)
            all_items = self._stratified_sampling(all_items, simple_count=250, compound_count=250)
        else:
            grouped: Dict[str, List[Dict]] = {}
            for item in all_items:
                key = f"{item.get('model','')}_{item.get('method','')}"
                group = grouped.setdefault(key, [])
                if len(group) < max_items:
                    group.append(item)
            all_items = [it for group in grouped.values() for it in group]
        
        if not all_items:
            logger.error("No results found to evaluate, exiting")
            return
        
        all_items_keys = [f"{item['doc_id']}_{item['model']}_{item['method']}" for item in all_items]
        unique_keys_set = set(all_items_keys)
        duplicate_count = len(all_items_keys) - len(unique_keys_set)
        if duplicate_count > 0:
            logger.warning(f"⚠️ Detected {duplicate_count} duplicate items (same doc_id+model+method combination)")
            from collections import Counter
            key_counts = Counter(all_items_keys)
            duplicates = {k: v for k, v in key_counts.items() if v > 1}
            if duplicates:
                logger.warning(f"Duplicate examples (first 5): {dict(list(duplicates.items())[:5])}")
        
        checkpoint_path = self.output_root / "checkpoint_evaluation.json"
        start_index, existing_results = self._load_checkpoint(checkpoint_path)
        
        processed_keys = {f"{r['doc_id']}_{r['model']}_{r['method']}" for r in existing_results}
        
        checkpoint_keys_in_current = processed_keys & unique_keys_set
        checkpoint_keys_not_in_current = processed_keys - unique_keys_set
        if checkpoint_keys_not_in_current:
            logger.warning(f"⚠️ Checkpoint has {len(checkpoint_keys_not_in_current)} records that do not exist in current data source")
            logger.warning(f"   This may be because the data source has changed, or some result files were deleted/moved")
        
        remaining_items = []
        skipped_items = []
        for item in all_items:
            unique_key = f"{item['doc_id']}_{item['model']}_{item['method']}"
            if unique_key not in processed_keys:
                remaining_items.append(item)
            else:
                skipped_items.append(unique_key)
        
        logger.info(f"Total items: {len(all_items)}, Unique items: {len(unique_keys_set)}, Processed: {len(existing_results)}, Pending: {len(remaining_items)}")
        
        if len(remaining_items) == 0 and len(all_items) != len(existing_results):
            logger.warning("="*60)
            logger.warning(" Data mismatch analysis:")
            logger.warning(f"   - Current data source total items: {len(all_items)}")
            logger.warning(f"   - Current data source unique items: {len(unique_keys_set)}")
            logger.warning(f"   - Checkpoint processed items: {len(existing_results)}")
            logger.warning(f"   - Checkpoint unique keys: {len(processed_keys)}")
            logger.warning(f"   - Duplicate count: {duplicate_count}")
            logger.warning(f"   - Checkpoint items not in current data: {len(checkpoint_keys_not_in_current)}")
            logger.warning("="*60)
            logger.warning("Possible reasons:")
            logger.warning("  1. Data source has duplicate (doc_id, model, method) combinations")
            logger.warning("  2. Some records in checkpoint correspond to data sources that have been deleted or moved")
            logger.warning("  3. Some items failed during processing but were marked as processed")
            logger.warning("="*60)
        
        if not remaining_items:
            logger.info("Evaluation completed, no processing needed")
            results = existing_results
        else:
            results = existing_results.copy()
            lock = threading.Lock()
            processed_keys = processed_keys.copy()
            progress_counter = {
                'processed': len(existing_results),
                'total': len(all_items),
                'last_saved': len(existing_results)
            }
            
            def worker(item):
                return self._evaluate_single_item(
                    item, lock, results, progress_counter, processed_keys
                )
            
            max_workers = min(20, len(remaining_items))
            threads = []
            
            for item in remaining_items:
                while len([t for t in threads if t.is_alive()]) >= max_workers:
                    time.sleep(0.1)
                    threads = [t for t in threads if t.is_alive()]
                    
                    with lock:
                        current_processed = progress_counter['processed']
                        need_save = current_processed - progress_counter.get('last_saved', 0) >= 10
                        if need_save:
                            progress_counter['last_saved'] = current_processed
                            results_copy = results.copy()
                        else:
                            results_copy = None
                    
                    if need_save and results_copy is not None:
                        self._save_checkpoint(checkpoint_path, current_processed, results_copy)
                
                thread = threading.Thread(target=worker, args=(item,))
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()
            
            with lock:
                current_processed = progress_counter['processed']
                if current_processed > progress_counter.get('last_saved', 0):
                    results_copy = results.copy()
                    self._save_checkpoint(checkpoint_path, current_processed, results_copy)
        
        self._save_checkpoint(checkpoint_path, len(results), results)
        
        output_file = self.output_root / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to: {output_file}")
        logger.info(f"Total evaluated {len(results)} items")
        logger.info("="*60)
        logger.info("LLM Expert Evaluation Completed")
        logger.info("="*60)


def run():
    """Main function: supports specifying maximum number of samples to evaluate in terminal."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Expert Evaluation")
    parser.add_argument(
        "--experiment_result_dir",
        type=str,
        default="Validation/Experiment_result",
        help="Experiment result root directory (default: Validation/Experiment_result)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to process (default: process all models), e.g., 'glm-4-9b'",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=["StandCoT", "Random_few_shot", "semantic_few_shot", "LGCoT"],
        help="Method list to process (default: process all methods), can specify multiple, e.g., --methods StandCoT LGCoT",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per model-method combination (default None means stratified sampling: Simple and Compound 25 each), e.g., 10 means evaluate at most 10 per (model, method)",
    )
    parser.add_argument(
        "--silver_standard_path",
        type=str,
        default="Temporary_tool/silver_standard_dataset_updated.json",
        help="Silver standard dataset path (default: Temporary_tool/silver_standard_dataset_updated.json)",
    )

    args = parser.parse_args()

    evaluator = LLMExpertEvaluation(silver_standard_path=args.silver_standard_path)
    evaluator.run(
        experiment_result_dir=args.experiment_result_dir,
        model=args.model,
        methods=args.methods,
        max_items=args.max_items,
    )


if __name__ == '__main__':
    run()
