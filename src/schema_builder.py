"""Schema builder (V2): Step0 init predicate/FOL repo via LLM."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime


class SchemaBuilder:
    """Step0: sample texts, LLM → predicates + FOL examples; validate & store."""

    def __init__(self, config: Dict, llm_client=None, output_root: Path = None):
        """Init from schema_builder config; llm_client optional; output_root for repos."""
        self.config = config
        self.output_root = Path(output_root) if output_root else Path(config['output_dir'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.llm_provider = config['llm_provider']
        self.high_param_model = config['high_param_model']
        self.normal_model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.use_high_param = config['use_high_param']
        self.predefined_logical_connectives = config.get('logical_connectives', [])
        self.predefined_modal_operators = config.get('modal_operators', [])
        self.llm_client = llm_client
        project_root = Path(__file__).resolve().parents[1]
        repository_dir = project_root / "Repository"
        repository_dir.mkdir(parents=True, exist_ok=True)
        self.predicate_repo_path = repository_dir / "predicate_repo.json"
        self.fol_repo_path = repository_dir / "fol_repo.json"
        self.predicate_repo = self._load_repository(self.predicate_repo_path, default=[])
        self.fol_repo = self._load_repository(self.fol_repo_path, default=[])
        self.predicate_repo = self._normalize_predicate_repo(self.predicate_repo)
        if self.predicate_repo:
            self._save_repository(self.predicate_repo_path, self.predicate_repo)
        if self.fol_repo:
            original_count = len(self.fol_repo)
            valid_fol_repo = []
            for example in self.fol_repo:
                is_valid, _ = self._validate_fol_example(example)
                if is_valid:
                    valid_fol_repo.append({
                        'text': example.get('text', '').strip(),
                        'fol': example.get('fol', '').strip()
                    })
            self.fol_repo = valid_fol_repo
            if len(self.fol_repo) < original_count:
                logger.info(f"Cleaned FOL repository: {original_count} -> {len(self.fol_repo)}")
                self._save_repository(self.fol_repo_path, self.fol_repo)
        
        logger.info(f"SchemaBuilder V2 initialized")
        logger.info(f"Predicate repository: {len(self.predicate_repo)} predicates")
        logger.info(f"FOL repository: {len(self.fol_repo)} examples")
    
    def _load_repository(self, path: Path, default=None):
        """Load repository JSON."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Repository loaded successfully: {path}")
                return data
            except Exception as e:
                logger.warning(f"Repository loading failed, using default value: {e}")
                self._save_repository(path, default if default is not None else [])
                return default if default is not None else []
        else:
            self._save_repository(path, default if default is not None else [])
            return default if default is not None else []
    
    def _save_repository(self, path: Path, data):
        """Save repository JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Repository saved successfully: {path}")
    
    def _normalize_predicate_repo(self, repo_data) -> List[Dict]:
        """Normalize predicate repo (support legacy list-of-str or list-of-dict)."""
        if not repo_data:
            return []
        normalized = []
        predicate_dict = {}
        if isinstance(repo_data, list) and len(repo_data) > 0:
            if isinstance(repo_data[0], str):
                for pred in repo_data:
                    pred = pred.strip()
                    if pred and self._validate_predicate(pred):
                        if pred not in predicate_dict:
                            predicate_dict[pred] = set()
            elif isinstance(repo_data[0], dict):
                for item in repo_data:
                    pred = item.get('predicate', '').strip()
                    clause_ids = item.get('clause_ids', item.get('doc_ids', []))
                    if pred and self._validate_predicate(pred):
                        if pred not in predicate_dict:
                            predicate_dict[pred] = set()
                        if isinstance(clause_ids, list):
                            predicate_dict[pred].update(clause_ids)
        
        for pred, clause_ids_set in predicate_dict.items():
            normalized.append({
                "predicate": pred,
                "clause_ids": sorted(list(clause_ids_set))
            })
        
        return normalized
    
    def _validate_predicate(self, predicate: str) -> bool:
        """Validate if predicate meets requirements."""
        if not isinstance(predicate, str):
            return False
        
        predicate = predicate.strip()
        
        if not predicate:
            return False
        
        if len(predicate) > 4:
            return False
        
        if any(char.isascii() and char.isalpha() for char in predicate):
            return False
        
        if '=' in predicate or '\\' in predicate:
            return False
        
        return True
    
    def _filter_predicates(self, predicates: List[str]) -> List[str]:
        """Filter predicate list, remove predicates that do not meet requirements."""
        filtered = []
        for pred in predicates:
            if self._validate_predicate(pred):
                filtered.append(pred)
        return filtered
    
    def _validate_fol_example(self, fol_example: Dict) -> Tuple[bool, str]:
        """Validate if FOL example meets requirements."""
        if not isinstance(fol_example, dict):
            return False, "FOL example must be a dictionary type"
        
        if 'text' not in fol_example or 'fol' not in fol_example:
            return False, "FOL example missing required fields (text or fol)"
        
        text = fol_example.get('text', '').strip()
        fol = fol_example.get('fol', '').strip()
        
        if not text or not fol:
            return False, "FOL example text or fol field is empty"
        
        if '(' not in fol or ')' not in fol:
            return False, "FOL format incorrect: missing parentheses (should be function-call style)"
        
        return True, ""
    
    
    def step0_initialize_predicates(self, sample_texts: List[str], llm_response: str = None) -> bool:
        """Step0: Randomly sample texts, let high-parameter LLM analyze domain predicates and FOL."""
        logger.info("="*60)
        logger.info("Step 0: 初始化领域谓词库")
        logger.info("="*60)
        
        # 从配置读取采样数量
        sample_count = self.config['sample_text_count']
        
        # 随机采样
        if len(sample_texts) > sample_count:
            sampled_texts = random.sample(sample_texts, sample_count)
        else:
            sampled_texts = sample_texts
        
        logger.info(f"Sampled text count: {len(sampled_texts)}")
        
        if not llm_response:
            logger.error("Step 0 requires LLM response but not provided. Please call LLM externally and pass the response.")
            return False
        
        response = llm_response
        if response:
            try:
                # 解析JSON
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response)
                
                new_predicates = result.get('predicates', [])
                original_count = len([item.get('predicate') for item in self.predicate_repo if item.get('predicate')])
                self._add_predicates_to_repo(new_predicates, clause_id="Step0_Init")
                current_count = len([item.get('predicate') for item in self.predicate_repo if item.get('predicate')])
                if current_count < original_count + len(new_predicates):
                    logger.warning(f"Step 0: Some predicates were filtered (do not meet requirements or already exist)")
                self._save_repository(self.predicate_repo_path, self.predicate_repo)
                
                fol_examples = result.get('fol_examples', [])
                valid_fol_examples = []
                for example in fol_examples:
                    is_valid, error_msg = self._validate_fol_example(example)
                    if is_valid:
                        valid_fol_examples.append({
                            'text': example.get('text', '').strip(),
                            'fol': example.get('fol', '').strip()
                        })
                    else:
                        logger.warning(f"Step 0: Discarded invalid FOL example: {error_msg}, example: {example}")
                if len(valid_fol_examples) < len(fol_examples):
                    logger.warning(f"Step 0: Filtered {len(fol_examples) - len(valid_fol_examples)} FOL examples that do not meet requirements")
                self.fol_repo.extend(valid_fol_examples)
                self._save_repository(self.fol_repo_path, self.fol_repo)
                
                logger.info(f"Step 0 completed: Extracted {len(new_predicates)} domain predicates, {len(fol_examples)} FOL examples")
                return True
                
            except json.JSONDecodeError as e:
                logger.error(f"Step 0 JSON parsing failed: {e}\nResponse: {response[:500]}")
                return False
        else:
            logger.error("Step 0 LLM response is empty")
            return False
    
    def _add_predicates_to_repo(self, predicates: List[str], clause_id: str = None):
        """Add predicates to repository and update clause_id list."""
        filtered_predicates = self._filter_predicates(predicates)
        
        predicate_to_clause_ids = {}
        for item in self.predicate_repo:
            pred = item.get('predicate', '')
            if pred:
                predicate_to_clause_ids[pred] = set(item.get('clause_ids', []))
        
        for pred in filtered_predicates:
            if pred not in predicate_to_clause_ids:
                predicate_to_clause_ids[pred] = set()
            if clause_id:
                predicate_to_clause_ids[pred].add(clause_id)
        
        self.predicate_repo = [
            {
                "predicate": pred,
                "clause_ids": sorted(list(clause_ids_set))
            }
            for pred, clause_ids_set in predicate_to_clause_ids.items()
        ]
        
        self.predicate_repo.sort(key=lambda x: x.get('predicate', ''))
    
    def get_predicate_list(self) -> List[str]:
        """Extract predicate string list from predicate repository (for LLM prompt)."""
        return [item.get('predicate', '') for item in self.predicate_repo if item.get('predicate')]
    
    def get_fol_repo(self) -> List[Dict]:
        """Get FOL repository."""
        return self.fol_repo
    
    def get_predicate_repo(self) -> List[Dict]:
        """Get predicate repository."""
        return self.predicate_repo
    
    def build_schema(self, cleaned_data: Dict, llm_response: str = None) -> Dict:
        """Build domain Schema."""
        logger.info("Starting schema construction")
        
        clauses = cleaned_data.get('clauses', [])
        sample_texts = [clause.get('content', '') for clause in clauses if clause.get('content')]
        clause_ids = [clause.get('clause_id', '') for clause in clauses if clause.get('clause_id')]
        
        predicate_count = len([item.get('predicate') for item in self.predicate_repo if item.get('predicate')])
        if predicate_count == 0 or len(self.fol_repo) == 0:
            logger.info("Domain predicate repository is empty, executing Step 0 initialization...")
            if llm_response:
                self.step0_initialize_predicates(sample_texts, llm_response=llm_response)
            else:
                logger.warning("Repository is empty but LLM response not provided, cannot initialize. Please call LLM externally and pass the response.")
        else:
            logger.info(f"Domain predicate repository exists: {predicate_count} predicates, {len(self.fol_repo)} FOL examples")
        
        predicates = [item.get('predicate', '') for item in self.predicate_repo if item.get('predicate')]
        return {
            'clause_id': clause_ids,
            'predicates': predicates
        }
    
    def save_schema(self, schema: Dict, doc_name: str) -> str:
        """Save Schema to file (compatible with old interface)."""
        output_path = self.output_root / f"{doc_name}_schema.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Schema saved: {output_path}")
        
        return str(output_path)


if __name__ == "__main__":
    # 测试代码
    import yaml
    from openai import OpenAI
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化LLM客户端
    llm_api_config = config.get('llm_api', {})
    api_config = llm_api_config.get('SiliconFlow', {})
    llm_client = OpenAI(
        api_key=api_config.get('api_key'),
        base_url=api_config.get('api_base', 'https://api.siliconflow.cn/v1')
    )
    
    # 初始化构建器
    builder = SchemaBuilder(config['schema_builder'], llm_client)
    
    # 模拟清洗数据
    mock_cleaned_data = {
        'extracted_metadata': {
            '目录': '1. 第一章\n2. 第二章',
            '前言': '本规范适用于危险化学品存储管理...'
        },
        'clauses': [
            {'clause_id': '1.1', 'content': '1.1 本规范规定了基本要求。'},
            {'clause_id': '2.1', 'content': '2.1 当温度超过30℃时，应启动通风系统。'}
        ]
    }
    
    # 构建schema
    schema = builder.build_schema(mock_cleaned_data)
    print(json.dumps(schema, ensure_ascii=False, indent=2))
