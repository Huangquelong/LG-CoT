"""Bidirectional validation: Step3 predicate check, Step5 FOL self-check (BERTScore optional)."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime


class Validator:
    """Step3: check predicates; Step5: FOL self-check vs source text."""

    def __init__(self, config: Dict, llm_client=None, schema_builder=None):
        """Init from config; llm_client, schema_builder optional."""
        self.config = config
        self.llm_client = llm_client
        self.schema_builder = schema_builder
        self.llm_provider = config['llm_provider']
        self.model = config['model']
        self.high_param_model = config['high_param_model']
        self.use_high_param = config['use_high_param']
        self.temperature = config['temperature']
        self.on_failure = config['on_failure']
        self.include_error_feedback = self.on_failure['include_error_feedback']
        self.max_retries = self.on_failure['max_retries']
        self.review_config = config['review']
        self.review_dir = Path(self.review_config['output_dir'])
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.review_file = self.review_dir / 'review_items.json'
        self.num_workers = config['num_workers']
        self.review_lock = threading.Lock()
        self._meta_tensor_detected = False
        self._custom_tokenizer = None
        self._custom_model = None
        self._custom_model_device = None
        
        import os
        os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
        
        logger.info(f"Validator V2 initialized")
        logger.info(f"Parallel threads: {self.num_workers}")
        logger.info(f"Maximum retries: {self.max_retries}")
    
    def _call_llm_general(self, prompt: str, use_high_param: bool = False,
                          temperature: float = None, max_tokens: int = 1024,
                          max_retries: int = 3) -> Optional[str]:
        """Generic LLM call (Step3). Step5 uses main pipeline LLM."""
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return None
        
        model = self.high_param_model if use_high_param else self.model
        temp = temperature if temperature is not None else self.temperature
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM call finally failed: {e}")
                    return None
        return None
    
    def step3_check_predicates(self, new_predicates: List[str], doc_id: str = None) -> Tuple[bool, List[str]]:
        """Step3: High-parameter LLM checks if newly created domain predicates are reasonable."""
        if not new_predicates:
            return True, []
        
        if not self.schema_builder:
            logger.warning("SchemaBuilder not provided, cannot execute Step3 check, returning original predicate list")
            return True, new_predicates
        
        predicates_str = json.dumps(new_predicates, ensure_ascii=False)
        existing_predicate_list = self.schema_builder.get_predicate_list()
        existing_predicates_str = json.dumps(existing_predicate_list, ensure_ascii=False)

        prompt = f"""分析以下新创建的领域谓词是否合理、是否与现有谓词重复。

新创建的谓词：
{predicates_str}

现有谓词列表：
{existing_predicates_str}

对于每个新谓词，判断：
1. 是否是合理的领域谓词（具有明确的语义）
2. 是否与现有谓词重复或相似

注意：除逻辑操作符白名单（AND, OR, NOT, IMPLIES, IFF, XOR, FORALL, EXISTS）和领域谓词白名单（Obligation, Permission, Prohibition, REQUIREMENT）外，所有其他内容必须使用中文表示。

只输出JSON格式：{{"approved_predicates": ["保留的谓词1", ...], "rejected_predicates": ["舍弃的谓词1", ...]}}"""

        response = self._call_llm_general(prompt, use_high_param=self.use_high_param, max_tokens=1024)

        if response:
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()

                result = json.loads(response)
                approved = result.get('approved_predicates', [])

                if approved and self.schema_builder:
                    self.schema_builder._add_predicates_to_repo(approved, doc_id=doc_id)
                    self.schema_builder._save_repository(
                        self.schema_builder.predicate_repo_path,
                        self.schema_builder.predicate_repo
                    )
                    approved = self.schema_builder._filter_predicates(approved)

                logger.debug(f"Step 3 predicate check: kept {len(approved)}")
                return True, approved

            except json.JSONDecodeError as e:
                logger.warning(f"Step 3 JSON parsing failed: {e}")
                return False, []
        else:
            logger.warning("Step 3 LLM call failed")
            return False, []
    
    def step5_build_prompt(self, fol: str, original_text: str) -> str:
        """Step5: Build FOL self-check prompt."""
        prompt = f"""你是FOL公式质检员。请判断下面的FOL是否正确表达了原始条款的含义。

要求：
1. 如果匹配，输出 JSON：{{"pass": true}}
2. 如果不匹配，输出 JSON：{{"pass": false}}
3. 不要添加多余文字。
4. 注意：除逻辑操作符白名单（AND, OR, NOT, IMPLIES, IFF, XOR, FORALL, EXISTS）和领域谓词白名单（Obligation, Permission, Prohibition, REQUIREMENT）外，所有其他内容必须使用中文表示。

原始条款：
{original_text}

FOL：
{fol}"""
        return prompt

    def step5_fol_to_text_similarity(self, fol: str, original_text: str,
                                     llm_response: str, threshold: float = 0.85) -> Tuple[bool, float]:
        """Step5: FOL self-check - use LLM to judge if FOL matches original text."""
        if not llm_response:
            logger.warning("Step 5 self-check failed: no response")
            return False, 0.0
        
        try:
            response = llm_response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            passed = bool(result.get("pass", False))
            reason = result.get("reason", "")
            
            if passed:
                logger.debug("Step 5 self-check passed")
                return True, 1.0
            else:
                logger.debug(f"Step 5 self-check failed: {reason}")
                return False, 0.0
                
        except Exception as e:
            logger.warning(f"Step 5 self-check parsing failed: {e}")
            return False, 0.0
    
    def _load_review_items(self) -> List[Dict]:
        """Load all items from review repository."""
        if not self.review_file.exists():
            return []
        
        try:
            with open(self.review_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load review repository: {str(e)}")
            return []
    
    def _clear_review_items(self):
        """Clear review repository."""
        with self.review_lock:
            if self.review_file.exists():
                self.review_file.unlink()
            logger.info("Review repository cleared")
    
    def batch_validate(self, validations: List[Tuple[str, Dict]], 
                      clause_ids: Optional[List[str]] = None) -> List[Dict]:
        """Batch validation (supports multi-threading)."""
        if not validations:
            return []
        
        logger.info(f"Starting batch validation: {len(validations)} items, using {self.num_workers} threads")
        
        results = []
        review_items = []
        
        def validate_single(item_data):
            idx, (original_text, fol_formula) = item_data
            clause_id = clause_ids[idx] if clause_ids and idx < len(clause_ids) else f"item_{idx}"
            
            retry_count = 0
            last_result = None
            
            # 重试循环
            while retry_count <= self.max_retries:
                try:
                    result = self.validate(original_text, fol_formula, retry_count=retry_count)
                    last_result = result
                    
                    if result['passed']:
                        return idx, result, None
                    
                    # 如果未通过但还可以重试
                    if retry_count < self.max_retries:
                        retry_count += 1
                        logger.debug(f"Clause {clause_id} validation failed, retrying {retry_count}/{self.max_retries}")
                        continue
                    else:
                        # 达到最大重试次数，需要加入review库
                        return idx, result, {
                            'idx': idx,
                            'original_text': original_text,
                            'fol_formula': fol_formula,
                            'validation_result': result,
                            'clause_id': clause_id
                        }
                        
                except Exception as e:
                    logger.error(f"Clause {clause_id} validation error: {str(e)}")
                    return idx, {
                        'passed': False,
                        'score': 0.0,
                        'generated_text': '',
                        'feedback': f'验证出错: {str(e)}',
                        'original_text': original_text,
                        'retry_count': retry_count,
                        'needs_review': True
                    }, {
                        'idx': idx,
                        'original_text': original_text,
                        'fol_formula': fol_formula,
                        'validation_result': {'error': str(e)},
                        'clause_id': clause_id
                    }
            
            return idx, last_result or {
                'passed': False,
                'score': 0.0,
                'generated_text': '',
                'feedback': '达到最大重试次数',
                'original_text': original_text,
                'retry_count': retry_count,
                'needs_review': True
            }, {
                'idx': idx,
                'original_text': original_text,
                'fol_formula': fol_formula,
                'validation_result': last_result or {},
                'clause_id': clause_id
            }
        

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(validate_single, (idx, item)): idx 
                for idx, item in enumerate(validations)
            }
            

            temp_results = [None] * len(validations)
            
            for future in as_completed(futures):
                try:
                    idx, result, review_item = future.result()
                    temp_results[idx] = result
                    if review_item:
                        review_items.append(review_item)
                except Exception as e:
                    original_idx = futures[future]
                    logger.error(f"验证任务失败 (索引 {original_idx}): {str(e)}")
                    temp_results[original_idx] = {
                        'passed': False,
                        'score': 0.0,
                        'generated_text': '',
                        'feedback': f'任务执行失败: {str(e)}',
                        'original_text': validations[original_idx][0] if original_idx < len(validations) else '',
                        'needs_review': True
                    }
        
        results = [r for r in temp_results if r is not None]

        if review_items:
            for item in review_items:
                clause_id = clause_ids[item.get('idx', 0)] if clause_ids and 'idx' in item else item.get('clause_id')
                self._save_to_review(
                    item['original_text'],
                    item['fol_formula'],
                    item['validation_result'],
                    clause_id=clause_id
                )
        

        passed_count = sum(1 for r in results if r.get('passed', False))
        review_count = sum(1 for r in results if r.get('needs_review', False))
        avg_score = sum(r.get('score', 0.0) for r in results) / len(results) if results else 0
        
        logger.info(f"Batch validation completed: {passed_count}/{len(results)} passed, {review_count} items added to review repository, average score: {avg_score:.4f}")
        
        return results
    
    def process_review_items(self, clear_context: bool = True) -> List[Dict]:
        """Process all items in review repository."""
        review_items = self._load_review_items()
        
        if not review_items:
            logger.info("Review repository is empty, no processing needed")
            return []
        
        logger.info(f"Starting to process review repository: {len(review_items)} items")
        
        if clear_context and self.llm_client:
            logger.info("Clearing LLM context, creating new client")
        
        validations = [
            (item['original_text'], item['fol_formula'])
            for item in review_items
        ]
        
        clause_ids = [
            item.get('clause_id', f"review_{i}")
            for i, item in enumerate(review_items)
        ]
        
        results = []
        for idx, (original_text, fol_formula) in enumerate(validations):
            try:
                result = self.validate(original_text, fol_formula, retry_count=0)
                result['review_item_id'] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Review item {idx} validation failed: {str(e)}")
                results.append({
                    'passed': False,
                    'score': 0.0,
                    'generated_text': '',
                    'feedback': f'验证出错: {str(e)}',
                    'original_text': original_text,
                    'review_item_id': idx,
                    'needs_review': True
                })
        

        self._clear_review_items()
        
        passed_count = sum(1 for r in results if r.get('passed', False))
        logger.info(f"Review库处理完成: {passed_count}/{len(results)} 通过")
        
        return results

