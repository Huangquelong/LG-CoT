"""lgcot-CoT (V2): Step2 FOL generation, Step6 FOL→JSON. Logic-guided chain-of-thought extraction."""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from loguru import logger
import yaml
import numpy as np
import jieba
from gensim.models import Word2Vec


class KnowledgeExtractor:
    """Step2: FOL from predicates + 3 similar examples; Step6: FOL → JSON."""

    def __init__(self, config: Dict, schema_builder, validator=None, llm_client=None,
                 word2vec_model=None, fol_repo=None, output_root: Path = None):
        """Init from config; schema_builder required; validator, llm_client, word2vec, fol_repo optional."""
        self.config = config
        self.schema_builder = schema_builder
        self.validator = validator
        self.llm_client = llm_client
        self.llm_provider = config['llm_provider']
        self.normal_model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.max_retries = config['max_retries']
        
        self.batch_workers = config['batch_workers']
        self.batch_chunk_size = config['batch_chunk_size']
        
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[1]
        self.repository_dir = project_root / "Repository"
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.fol_repo_path = self.repository_dir / "fol_repo.json"
        self.predicate_repo_path = self.repository_dir / "predicate_repo.json"
        self.badcase_repo_path = self.repository_dir / "badcase_repo.json"
        self.fol_repo = self._load_repository(self.fol_repo_path, default=[])
        self.predicate_repo = self._load_repository(self.predicate_repo_path, default=[])
        self.badcase_repo = self._load_repository(self.badcase_repo_path, default=[])
        if fol_repo:
            existing_texts = {item.get('text', '') for item in self.fol_repo if item.get('text')}
            for item in fol_repo:
                if item.get('text') and item.get('text') not in existing_texts:
                    self.fol_repo.append(item)
                    existing_texts.add(item.get('text'))
            self._save_repository(self.fol_repo_path, self.fol_repo)
        self.word2vec_model = word2vec_model
        self.fol_repo_vectors = None
        template_path = config['template_path']
        self.template = self._load_template(template_path)
        
        logger.info(f"KnowledgeExtractor V2 initialized")
        logger.info(f"Model: {self.normal_model}")
        logger.info(f"Repository directory: {self.repository_dir}")
        logger.info(f"FOL repository size: {len(self.fol_repo)}")
        logger.info(f"Predicate repository size: {len(self.predicate_repo)}")
        logger.info(f"Badcase repository size: {len(self.badcase_repo)}")

    def _load_template(self, template_path: str) -> Dict:
        """Load JSON output template."""
        resolved_path = None
        project_root = Path(__file__).resolve().parents[1]
        path_str = template_path or ''
        try:
            if path_str.startswith('@'):
                resolved_path = project_root / path_str[1:]
            else:
                path_candidate = Path(path_str)
                resolved_path = path_candidate if path_candidate.is_absolute() else project_root / path_candidate
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            try:
                template = json.loads(raw_content)
            except json.JSONDecodeError:
                stripped = self._strip_json_comments(raw_content)
                template = json.loads(stripped)
            
            logger.info(f"Template loaded: {resolved_path}")
            return template
        except Exception as e:
            logger.warning(f"Failed to load template: {str(e)}, using default template")
            return {}

    def _load_repository(self, path: Path, default=None):
        """Load repository file."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load repository {path}: {e}, using default value")
                self._save_repository(path, default if default is not None else [])
                return default if default is not None else []
        else:
            self._save_repository(path, default if default is not None else [])
            return default if default is not None else []
    
    def _save_repository(self, path: Path, data):
        """Save repository file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _strip_json_comments(json_text: str) -> str:
        """Remove // and /* */ comments from JSON."""
        comment_re = re.compile(
            r"""
            ("(\\.|[^"\\])*") |
            (//[^\n\r]*|/\*.*?\*/)
            """,
            re.VERBOSE | re.DOTALL
        )

        def _replacer(match):
            text = match.group(0)
            if text.startswith(('"', "'")):
                return text
            return ''

        return re.sub(comment_re, _replacer, json_text)

    def _clean_llm_response(self, response: str) -> str:
        """Remove code block wrappers and extra text from LLM response."""
        cleaned = response.strip()
        if not cleaned:
            return cleaned

        code_block = re.search(r"```(?:json|JSON)?\s*(.*?)```", cleaned, re.DOTALL)
        if code_block:
            cleaned = code_block.group(1).strip()

        start_candidates = [idx for idx in (cleaned.find('{'), cleaned.find('[')) if idx != -1]
        if start_candidates:
            cleaned = cleaned[min(start_candidates):]
        end_candidates = [idx for idx in (cleaned.rfind('}'), cleaned.rfind(']')) if idx != -1]
        if end_candidates:
            cleaned = cleaned[:max(end_candidates) + 1]

        return cleaned.strip()

    def _parse_llm_json(self, response: str):
        """Robustly parse LLM response."""
        cleaned = self._clean_llm_response(response)
        if not cleaned:
            raise ValueError("LLM response is empty")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.debug(f"JSON parsing failed, attempting to fix: {exc}")
            fixed = self._try_fix_json(cleaned)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            try:
                parsed = yaml.safe_load(cleaned)
                if parsed is None:
                    raise ValueError("YAML parsing result is empty")
                return parsed
            except Exception as yaml_exc:
                snippet = cleaned[:200]
                logger.error(f"Failed to parse LLM response, content snippet: {snippet}")
                raise ValueError(f"Unable to parse LLM response: {yaml_exc}") from yaml_exc

    def _try_fix_json(self, text: str) -> Optional[str]:
        """Try to fix common non-strict JSON formats."""
        if not text:
            return None

        fixed = text
        fixed = re.sub(r"'", '"', fixed)
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
        fixed = re.sub(r"\bTrue\b", "true", fixed)
        fixed = re.sub(r"\bFalse\b", "false", fixed)
        fixed = re.sub(r"\bNone\b", "null", fixed)

        if fixed != text:
            logger.debug("Detected non-strict JSON format, attempted automatic fix and re-parsing.")
            return fixed
        return None


    def _find_semantic_similar(self, target_text: str, top_k: int = 3) -> List[int]:
        """Find top_k semantically similar texts based on fol_repo text field."""
        if self.fol_repo_vectors is not None and len(self.fol_repo_vectors) > 0:
            vectors = self.fol_repo_vectors
            repo_size = len(self.fol_repo)
        else:
            return []
        
        if not self.word2vec_model:
            return []
        
        words = list(jieba.cut(target_text))
        word_vectors = [self.word2vec_model.wv[word] for word in words 
                       if word in self.word2vec_model.wv]
        
        if not word_vectors:
            return []
        
        target_vector = np.mean(word_vectors, axis=0)
        
        similarities = np.dot(vectors, target_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        top_indices = [idx for idx in top_indices if idx < repo_size]
        
        return top_indices

    def update_fol_repo_vectors(self, word2vec_model):
        """Update vectors based on fol_repo text field for semantic retrieval."""
        self.word2vec_model = word2vec_model
        if not self.fol_repo or not self.word2vec_model:
            self.fol_repo_vectors = None
            return
        
        fol_texts = [item.get('text', '').strip() for item in self.fol_repo if item.get('text', '').strip()]
        
        if not fol_texts:
            self.fol_repo_vectors = None
            return
        fol_vectors = []
        for text in fol_texts:
            words = list(jieba.cut(text))
            word_vectors = [self.word2vec_model.wv[word] for word in words 
                          if word in self.word2vec_model.wv]
            
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(256)
            
            fol_vectors.append(text_vector)
        
        self.fol_repo_vectors = np.array(fol_vectors) if fol_vectors else None

        def step2_build_prompt(self, clause_text: str) -> str:
        similar_indices = self._find_semantic_similar(clause_text, top_k=3)
        similar_examples = [self.fol_repo[i] for i in similar_indices
                          if i < len(self.fol_repo)]

        examples_str = ""
        for i, example in enumerate(similar_examples, 1):
            examples_str += f"\n### 样例 {i}\n条款：{example.get('text', '')}\nFOL：{example.get('fol', '')}\n"
        predicate_list = self.schema_builder.get_predicate_list()
        predicates_str = json.dumps(predicate_list, ensure_ascii=False)

        prompt = f"""根据条款文本、谓词列表和参考样例，生成函数调用式FOL表示。

条款文本：
{clause_text}

谓词列表（可使用这些谓词，也可以创建新的领域谓词）：
{predicates_str}

参考样例：
{examples_str}

要求：
1. FOL格式为函数调用式，如：AND(Obligation(subject=Party("用人单位"), action="支付", object="劳动报酬"), Condition(desc="劳动者实际提供劳动"))
2. 除逻辑操作符白名单（AND, OR, NOT, IMPLIES, IFF, XOR, FORALL, EXISTS）和领域谓词白名单（Obligation, Permission, Prohibition, REQUIREMENT）外，所有其他内容必须使用中文表示
3. 谓词必须使用中文，不能为纯英文
4. 将识别的实体填入FOL中

只输出JSON格式：{{"fol": "FOL表示", "new_predicates": ["新创建的谓词1", ...]}}"""

        return prompt

    def step2_generate_fol(self, clause_text: str, llm_response: str) -> Optional[Dict]:
        """Step2: Generate FOL framework based on LLM response."""
        if not llm_response:
            logger.warning("Step 2 LLM response is empty")
            return None

        try:
            result = self._parse_llm_json(llm_response)
            fol = result.get('fol', '')
            new_predicates = result.get('new_predicates', [])

            logger.debug(f"Step 2 generated FOL: {fol[:100]}...")
            return {'fol': fol, 'new_predicates': new_predicates}

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Step 2 JSON parsing failed: {e}")
            return None

    def step6_fol_to_json(self, fol: str, original_text: str, llm_response: str) -> Optional[Dict]:
        """Step6: Map FOL to JSON structure."""
        if not llm_response:
            logger.warning("Step 6 LLM response is empty")
            return None

        try:
            result = self._parse_llm_json(llm_response)
            logger.debug(f"Step 6 mapped to JSON completed")
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Step 6 JSON parsing failed: {e}")
            return None

        def _build_step6_prompt(self, fol: str, original_text: str) -> str:
        return f"""将以下FOL表示映射到标准JSON结构，字段要求：
1) type: 规则类型，取值 OBLIGATION / PERMISSION / PROHIBITION（无法判断用 OBLIGATION）
2) core_event: subject / action / object
3) conditions: 列表，元素为对象，包含 variable / operator / threshold / unit
4) constraints: time_limit / frequency / manner
5) context_tags: 语境标签列表

注意：除逻辑操作符白名单（AND, OR, NOT, IMPLIES, IFF, XOR, FORALL, EXISTS）和领域谓词白名单（Obligation, Permission, Prohibition, REQUIREMENT）外，所有其他内容必须使用中文表示。

FOL：
{fol}

原始条款：
{original_text}

输出JSON格式：
{{
  "rules": [
    {{
      "type": "",
      "core_event": {{
        "subject": "",
        "action": "",
        "object": ""
      }},
      "conditions": [
        {{
          "variable": "",
          "operator": "",
          "threshold": "",
          "unit": ""
        }}
      ],
      "constraints": {{
        "time_limit": "",
        "frequency": "",
        "manner": ""
      }},
      "context_tags": []
    }}
  ]
}}

只输出JSON，不要其他内容。"""

    def extract_knowledge(self, clause_text: str, clause_id: str = None, doc_id: str = None) -> Dict:
        """Extract knowledge from a single clause (compatible with old interface)."""
        logger.info(f"Starting knowledge extraction: {clause_id or 'Unknown'}")
        
        logger.warning("extract_knowledge method requires external LLM call, please use step2_build_prompt and step2_generate_fol")
        return {
            "doc_id": doc_id or "UNKNOWN",
            "rules": [],
            "success": False
        }
        if json_result is None:
            logger.warning("Step 6 failed")
            return {
                "doc_id": doc_id or "UNKNOWN",
                "rules": [],
                "success": False
            }
        
        rules = json_result.get('rules', [])
        if not isinstance(rules, list):
            rules = [rules] if rules else []
        for rule in rules:
            rule['logic_form'] = fol
            rule['source_text'] = clause_text
            if clause_id:
                rule['rule_id'] = clause_id
        
        return {
            "doc_id": doc_id or "UNKNOWN",
            "rules": rules,
            "fol": fol,
            "new_predicates": new_predicates,
            "success": True
        }

    def batch_extract(
        self,
        clauses: List[Dict],
        doc_id: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        checkpoint_callback: Optional[Callable[[List[Dict]], None]] = None
    ) -> List[Dict]:
        """Batch extract knowledge."""
        if not clauses:
            return []

        all_results: List[Dict] = []
        chunk_size = max(1, int(self.batch_chunk_size or 20))
        total = len(clauses)

        try:
            for chunk_start in range(0, total, chunk_size):
                if stop_event and stop_event.is_set():
                    logger.info("Early stop request detected, stopping remaining clauses.")
                    break

                chunk = clauses[chunk_start:chunk_start + chunk_size]
                logger.info(
                    f"Processing clauses {chunk_start + 1}-{chunk_start + len(chunk)} in parallel "
                    f"(threads: {min(len(chunk), self.batch_workers)})"
                )

                chunk_results = self._process_clause_chunk(
                    chunk=chunk,
                    chunk_offset=chunk_start,
                    doc_id=doc_id,
                    stop_event=stop_event
                )

                all_results.extend(chunk_results)

                if checkpoint_callback and all_results:
                    checkpoint_callback(list(all_results))

                if stop_event and stop_event.is_set():
                    logger.info("Early stop signal activated, ending batch processing.")
                    break

        except KeyboardInterrupt:
            logger.warning("Batch extraction interrupted, returning partial results.")
            if stop_event:
                stop_event.set()

        logger.info(f"Batch extraction completed: {len(all_results)}/{len(clauses)}")
        return all_results

    def _process_clause_chunk(
        self,
        chunk: List[Dict],
        chunk_offset: int,
        doc_id: Optional[str],
        stop_event: Optional[threading.Event]
    ) -> List[Dict]:
        """Process a single clause chunk in parallel."""
        if not chunk:
            return []

        local_results: List[Optional[Dict]] = [None] * len(chunk)
        workers = min(len(chunk), max(1, int(self.batch_workers or 20)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}

            for local_idx, clause in enumerate(chunk):
                global_idx = chunk_offset + local_idx

                if stop_event and stop_event.is_set():
                    logger.info("Stop signal triggered, canceling new task scheduling.")
                    break

                future = executor.submit(
                    self.extract_knowledge,
                    clause_text=clause.get('content', ''),
                    clause_id=clause.get('clause_id', f'clause_{global_idx}'),
                    doc_id=doc_id or clause.get('doc_id')
                )
                future_to_idx[future] = local_idx

            try:
                for future in as_completed(future_to_idx):
                    local_idx = future_to_idx[future]
                    global_idx = chunk_offset + local_idx
                    clause_id = chunk[local_idx].get('clause_id', f'clause_{global_idx}')

                    if stop_event and stop_event.is_set():
                        logger.info("Stop signal triggered, skipping remaining result collection.")
                        break

                    try:
                        local_results[local_idx] = future.result()
                        logger.info(f"Clause {clause_id} (#{global_idx + 1}) processing completed")
                    except Exception as exc:
                        logger.error(f"Clause {clause_id} processing failed: {exc}")
            except KeyboardInterrupt:
                logger.warning("Clause chunk processing interrupted.")
                if stop_event:
                    stop_event.set()

        return [res for res in local_results if res]

    def add_to_badcase_repo(self, clause_text: str, error: str, details: Optional[Dict] = None):
        """Add error case to badcase_repo."""
        badcase_item = {
            'clause_text': clause_text,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        existing_texts = {item.get('clause_text', '') for item in self.badcase_repo}
        if clause_text not in existing_texts:
            self.badcase_repo.append(badcase_item)
            self._save_repository(self.badcase_repo_path, self.badcase_repo)
            logger.debug(f"Added error case to badcase_repo: {error[:50]}")
    
    def update_fol_repo(self, fol_item: Dict):
        """Update FOL repository."""
        existing_texts = {item.get('text', '') for item in self.fol_repo if item.get('text')}
        if fol_item.get('text') and fol_item.get('text') not in existing_texts:
            self.fol_repo.append(fol_item)
            self._save_repository(self.fol_repo_path, self.fol_repo)
            logger.debug(f"Updated FOL repository: {fol_item.get('text', '')[:50]}")
    
    def update_predicate_repo(self, predicate_item: Dict):
        """Update predicate repository."""
        existing_predicates = {item.get('predicate', '') for item in self.predicate_repo if item.get('predicate')}
        if predicate_item.get('predicate') and predicate_item.get('predicate') not in existing_predicates:
            self.predicate_repo.append(predicate_item)
            self._save_repository(self.predicate_repo_path, self.predicate_repo)
            logger.debug(f"Updated predicate repository: {predicate_item.get('predicate', '')}")
    
    def save_result(self, result: Dict, filename: str) -> str:
        """Save extraction result."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Extraction result saved: {output_path}")

        return str(output_path)

