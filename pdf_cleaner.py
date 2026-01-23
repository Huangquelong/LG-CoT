"""Markdown text cleaning module: filter content based on clause numbers, clean empty lines and spaces."""

import re
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class PDFCleaner:
    """Markdown text cleaner: filter based on clause numbers, clean empty lines and spaces."""
    
    def __init__(self, config: Dict):
        """Initialize cleaner."""
        self.config = config

        self.clause_patterns = {
            'chapter': re.compile(r'^第[一二三四五六七八九十百千]+章'),  # 章
            'section': re.compile(r'^第[一二三四五六七八九十百千]+节'),  # 节
            'article': re.compile(r'^第[一二三四五六七八九十百千]+条'),  # 条
            'standard': re.compile(r'^([0-9]+)(\.[0-9]+){0,5}\b'),  # 标准条款：最多6层
            'letter': re.compile(r'^[A-Z]\.(\d+)(\.\d+){0,2}\b'),  # 字母编号：A.1.2
            'chinese_paren': re.compile(r'^[（(][一二三四五六七八九十百千]+[)）]'),  # 中文括号：（一）
            'digit_paren': re.compile(r'^[（(]\d+[)）]'),  # 数字括号：（1）
            'digit_dot': re.compile(r'^\d+\.'),  # 数字点：1. 2.
            'circled_digit': re.compile(r'^[①②③④⑤⑥⑦⑧⑨⑩]')  # 带圈数字
        }
        
        # 保留旧模式以兼容
        self.clause_xxx_pattern = re.compile(r'^\d+\.\d+\.\d+')
        self.clause_xx_pattern = re.compile(r'^\d+\.\d+')
        self.clause_x_pattern = re.compile(r'^#\s+\d+\s+')
        
    def clean(self, text: str, analysis_result: Optional[Dict] = None) -> str:
        """Execute cleaning."""
        logger.info("Starting Markdown text cleaning...")
        
        filter_mode = None
        detected_types = []
        clause_stats = {}
        
        if analysis_result:
            if 'suggestions' in analysis_result:
                filter_mode = analysis_result['suggestions'].get('filter_mode')
            if 'statistics' in analysis_result and 'clause_stats' in analysis_result['statistics']:
                clause_stats = analysis_result['statistics']['clause_stats']
                detected_types = clause_stats.get('detected_types', [])
        
        if filter_mode and filter_mode != 'none':
            text = self._filter_by_clause_numbers(text, filter_mode, detected_types, clause_stats)
            logger.info(f"Filter mode applied: {filter_mode}")
        
        text = self._filter_appendix_and_references(text)
        
        text = self._clean_empty_lines_and_spaces(text)
        
        logger.info("Markdown text cleaning completed")
        return text.strip()
    
    def _filter_appendix_and_references(self, text: str) -> str:
        """Discard content after appendix, references, etc."""
        discard_keywords = [
            '附录',
            '引用法律法规和标准规范目录',
            '参考文献',
            '参考书目',
            '致谢',
            '后记'
        ]
        
        lines = text.split('\n')
        filtered_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            is_title = line_stripped.startswith('#')
            
            should_discard = False
            for keyword in discard_keywords:
                if is_title and keyword in line_stripped:
                    should_discard = True
                    logger.info(f"Detected keyword '{keyword}', discarding subsequent content (starting from line {i+1})")
                    break
                elif line_stripped.startswith(keyword):
                    should_discard = True
                    logger.info(f"Detected keyword '{keyword}', discarding subsequent content (starting from line {i+1})")
                    break
            
            if should_discard:
                break
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _filter_by_clause_numbers(self, text: str, filter_mode: str, 
                                   detected_types: List[str] = None, 
                                   clause_stats: Dict = None) -> str:
        """Filter content based on clause numbers with multiple patterns."""
        if detected_types is None:
            detected_types = []
        if clause_stats is None:
            clause_stats = {}
        
        has_standard = 'standard' in detected_types
        has_other = any(t in detected_types for t in ['chapter', 'section', 'article', 'letter', 
                                                       'chinese_paren', 'digit_paren', 'digit_dot', 'circled_digit'])
        
        lines = text.split('\n')
        filtered_lines = []
        i = 0
        
        def match_clause_pattern(line: str) -> tuple:
            """Match line against all clause patterns, return (matched_type, level, match_obj)."""
            line_stripped = line.strip()
            if not line_stripped:
                return (None, 0, None)
            
            # 按优先级匹配
            # 1. 标准条款
            match = self.clause_patterns['standard'].match(line_stripped)
            if match:
                level = match.group(0).count('.') + 1
                return ('standard', level, match)
            
            # 2. 字母编号
            match = self.clause_patterns['letter'].match(line_stripped)
            if match:
                level = match.group(0).count('.')
                return ('letter', level, match)
            
            # 3. 章
            match = self.clause_patterns['chapter'].match(line_stripped)
            if match:
                return ('chapter', 1, match)
            
            # 4. 节
            match = self.clause_patterns['section'].match(line_stripped)
            if match:
                return ('section', 2, match)
            
            # 5. 条
            match = self.clause_patterns['article'].match(line_stripped)
            if match:
                return ('article', 3, match)
            
            # 6. 中文括号
            match = self.clause_patterns['chinese_paren'].match(line_stripped)
            if match:
                return ('chinese_paren', 4, match)
            
            # 7. 数字括号
            match = self.clause_patterns['digit_paren'].match(line_stripped)
            if match:
                return ('digit_paren', 5, match)
            
            # 8. 数字点
            match = self.clause_patterns['digit_dot'].match(line_stripped)
            if match:
                return ('digit_dot', 6, match)
            
            # 9. 带圈数字
            match = self.clause_patterns['circled_digit'].match(line_stripped)
            if match:
                return ('circled_digit', 7, match)
            
            return (None, 0, None)
        
        def should_keep_line(clause_type: str, level: int) -> bool:
            """Determine if line should be kept based on filter mode."""
            if clause_type is None:
                return False
            

            if has_standard:
                if clause_type == 'standard':
                    if filter_mode == 'A':
                        return level >= 2  # 二级及以下
                    elif filter_mode == 'B':
                        return level >= 3  # 三级及以下
                    elif filter_mode == 'C':
                        return level >= 4  # 四级及以下
                    elif filter_mode == 'D':
                        return level >= 5  # 五级及以下
                    elif filter_mode == 'E':
                        return level >= 6  # 六级及以下
                    elif filter_mode == 'S':
                        return True  # 特殊模式：保留所有
                    else:
                        return False
                elif clause_type == 'letter':
                    # 字母编号也按层级过滤
                    if filter_mode == 'A':
                        return level >= 2  # A.1 及以下
                    elif filter_mode == 'B':
                        return level >= 3  # A.1.1 及以下
                    elif filter_mode in ['C', 'D', 'E', 'S']:
                        return True
                    else:
                        return False
                else:
                    # 其他类型在标准条款模式下
                    if filter_mode == 'S':
                        return True  # 特殊模式：保留所有
                    # 其他模式下，如果有标准条款，也保留主要类型（章/节/条）
                    # 这样可以保持文档结构的完整性
                    return clause_type in ['chapter', 'section', 'article']
            

            else:
                if filter_mode == 'A':
                    # 保留所有检测到的类型
                    return clause_type in detected_types
                elif filter_mode == 'B':
                    # 只保留主要类型（章/节/条）
                    return clause_type in ['chapter', 'section', 'article']
                elif filter_mode == 'C':
                    # 只保留特殊类型（括号/点/带圈数字）
                    return clause_type in ['chinese_paren', 'digit_paren', 'digit_dot', 'circled_digit']
                elif filter_mode == 'S':
                    # 特殊模式：保留所有类型
                    return True
            
            return False
        
        def is_clause_line(line: str) -> bool:
            """Check if line is a clause number line."""
            clause_type, _, _ = match_clause_pattern(line)
            return clause_type is not None
        
        while i < len(lines):
            line = lines[i]
            clause_type, level, _ = match_clause_pattern(line)
            
            if should_keep_line(clause_type, level):
                # 保留这一行
                filtered_lines.append(line)
                i += 1
                
                # 继续保留后续内容，直到遇到下一个条款编号
                while i < len(lines):
                    next_line = lines[i]
                    if is_clause_line(next_line):
                        break
                    filtered_lines.append(next_line)
                    i += 1
                continue
            
            i += 1
        
        return '\n'.join(filtered_lines)
    
    def _clean_empty_lines_and_spaces(self, text: str) -> str:
        """Clean empty lines and spaces."""
        lines = text.split('\n')
        cleaned_lines = []
        
        consecutive_empty = 0
        
        for line in lines:
            line = line.rstrip()
            
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty <= 2:
                    cleaned_lines.append('')
            else:
                consecutive_empty = 0
                line = re.sub(r' +', ' ', line)
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result
    
    def save_cleaned_data(self, cleaned_text: str, doc_id: str, metadata: Optional[Dict] = None):
        """Save cleaned data (only save md file)."""
        output_dir = Path('output/clean_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_path = output_dir / f"{doc_id}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"Cleaned text saved to: {md_path}")
