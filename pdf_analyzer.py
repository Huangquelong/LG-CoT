"""Markdown text analysis module: analyze structure, identify clause numbers (X.X.X format), generate filtering suggestions."""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class PDFAnalyzer:
    """Markdown text analyzer: analyze structure, identify clause numbers, generate filtering suggestions."""
    
    def __init__(self, config: Dict):
        """Initialize analyzer."""
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

        self.clause_xxx_pattern = re.compile(r'^\d+\.\d+\.\d+')
        self.clause_xx_pattern = re.compile(r'^\d+\.\d+')
        self.clause_x_pattern = re.compile(r'^#\s+\d+\s+')
        
    def analyze_pdf(self, extracted_content: Dict) -> Dict:
        """Analyze Markdown content, identify clause numbers, generate filtering suggestions."""
        text = extracted_content.get('text', '')
        metadata = extracted_content.get('metadata', {})
        
        logger.info("Starting Markdown structure analysis...")
        
        clause_stats = self._identify_clause_numbers(text)
        
        filter_mode = self._interactive_confirm_filter_mode(clause_stats)
        
        suggestions = {
            'filter_mode': filter_mode,
            'clause_statistics': clause_stats,
            'comment': self._get_filter_mode_description(filter_mode)
        }
        
        return {
            'statistics': {
                'total_lines': len(text.split('\n')),
                'clause_stats': clause_stats
            },
            'suggestions': suggestions,
            'metadata': metadata
        }
    
    def _identify_clause_numbers(self, text: str) -> Dict:
        """Identify clause numbers with multiple patterns."""
        lines = text.split('\n')

        clause_types = {
            'chapter': [],      # 章
            'section': [],      # 节
            'article': [],      # 条
            'standard': [],     # 标准条款（数字.数字.数字）
            'letter': [],       # 字母编号
            'chinese_paren': [], # 中文括号
            'digit_paren': [],  # 数字括号
            'digit_dot': [],    # 数字点
            'circled_digit': [] # 带圈数字
        }
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            

            matched = False

            match = self.clause_patterns['standard'].match(line_stripped)
            if match:
                clause_num = match.group(0)

                level = clause_num.count('.') + 1
                clause_types['standard'].append({
                    'line': line_num,
                    'number': clause_num,
                    'level': level,
                    'text': line_stripped[:100]
                })
                matched = True
            

            if not matched:
                match = self.clause_patterns['letter'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    level = clause_num.count('.')
                    clause_types['letter'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': level,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['chapter'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['chapter'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 1,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['section'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['section'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 2,
                        'text': line_stripped[:100]
                    })
                    matched = True

            if not matched:
                match = self.clause_patterns['article'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['article'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 3,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['chinese_paren'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['chinese_paren'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 4,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['digit_paren'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['digit_paren'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 5,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['digit_dot'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['digit_dot'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 6,
                        'text': line_stripped[:100]
                    })
                    matched = True
            

            if not matched:
                match = self.clause_patterns['circled_digit'].match(line_stripped)
                if match:
                    clause_num = match.group(0)
                    clause_types['circled_digit'].append({
                        'line': line_num,
                        'number': clause_num,
                        'level': 7,
                        'text': line_stripped[:100]
                    })
                    matched = True
        

        result = {}
        for clause_type, clauses in clause_types.items():
            result[f'{clause_type}_count'] = len(clauses)
            result[f'{clause_type}_samples'] = clauses[:10]
            if clauses:
                # 计算层级分布（仅对标准条款有意义）
                if clause_type == 'standard':
                    level_dist = {}
                    for clause in clauses:
                        level = clause.get('level', 0)
                        level_dist[level] = level_dist.get(level, 0) + 1
                    result[f'{clause_type}_level_dist'] = level_dist
        

        result['xxx_count'] = sum(1 for c in clause_types['standard'] if c.get('level', 0) >= 3)
        result['xx_count'] = sum(1 for c in clause_types['standard'] if c.get('level', 0) == 2)
        result['x_count'] = sum(1 for c in clause_types['standard'] if c.get('level', 0) == 1)
        result['xxx_samples'] = [c for c in clause_types['standard'] if c.get('level', 0) >= 3][:10]
        result['xx_samples'] = [c for c in clause_types['standard'] if c.get('level', 0) == 2][:10]
        result['x_samples'] = [c for c in clause_types['standard'] if c.get('level', 0) == 1][:10]
        
        result['detected_types'] = [k for k, v in clause_types.items() if len(v) > 0]
        result['all_clauses'] = clause_types
        
        return result
    
    def _interactive_confirm_filter_mode(self, clause_stats: Dict) -> str:
        """Interactively confirm filter mode based on detected clause types."""
        detected_types = clause_stats.get('detected_types', [])
        
        logger.info("=" * 80)
        logger.info("Clause number identification results:")
        logger.info("=" * 80)
        

        type_names = {
            'chapter': '章',
            'section': '节',
            'article': '条',
            'standard': '标准条款（数字.数字...）',
            'letter': '字母编号（A.1.2）',
            'chinese_paren': '中文括号（（一））',
            'digit_paren': '数字括号（（1））',
            'digit_dot': '数字点（1.）',
            'circled_digit': '带圈数字（①②③）'
        }
        
        has_standard = 'standard' in detected_types
        has_other = any(t in detected_types for t in ['chapter', 'section', 'article', 'letter', 
                                                       'chinese_paren', 'digit_paren', 'digit_dot', 'circled_digit'])
        
        for clause_type in detected_types:
            count = clause_stats.get(f'{clause_type}_count', 0)
            type_name = type_names.get(clause_type, clause_type)
            logger.info(f"  {type_name}: {count}")
            

            samples = clause_stats.get(f'{clause_type}_samples', [])
            if samples:
                logger.info(f"    Samples (first 3):")
                for i, sample in enumerate(samples[:3], 1):
                    logger.info(f"      {i}. Line {sample['line']}: {sample['text']}")
        

        if has_standard and 'standard_level_dist' in clause_stats:
            level_dist = clause_stats['standard_level_dist']
            logger.info(f"\n  标准条款层级分布:")
            for level in sorted(level_dist.keys()):
                logger.info(f"    {level}层: {level_dist[level]}个")
        
        logger.info("=" * 80)
        
        if not detected_types:
            logger.warning("No clause numbers detected, will keep all content")
            return 'none'
        

        logger.info("\n" + "=" * 80)
        logger.info("Please select filter mode:")
        
        if has_standard:

            max_level = max(clause_stats.get('standard_level_dist', {}).keys(), default=1)
            logger.info("  A: Keep level 2 and below (e.g., 1.1, 1.1.1, 1.1.1.1...)")
            logger.info("  B: Keep level 3 and below (e.g., 1.1.1, 1.1.1.1...)")
            if max_level >= 4:
                logger.info("  C: Keep level 4 and below")
            if max_level >= 5:
                logger.info("  D: Keep level 5 and below")
            if max_level >= 6:
                logger.info("  E: Keep level 6 and below")
        else:

            logger.info("  A: Keep all detected clause types")
            logger.info("  B: Keep only primary types (章/节/条)")
            logger.info("  C: Keep only special types (括号/点/带圈数字)")
        
        if has_standard and has_other:
            logger.info("  S: Special mode - configure each type separately")
        
        logger.info("=" * 80)
        
        while True:
            try:
                if has_standard:
                    max_level = max(clause_stats.get('standard_level_dist', {}).keys(), default=1)
                    valid_options = ['A', 'B']
                    if max_level >= 4:
                        valid_options.append('C')
                    if max_level >= 5:
                        valid_options.append('D')
                    if max_level >= 6:
                        valid_options.append('E')
                    if has_other:
                        valid_options.append('S')
                    
                    choice = input(f"Please enter option ({'/'.join(valid_options)}): ").strip().upper()
                    if choice in valid_options:
                        logger.info(f"Selected filter mode: {choice}")
                        return choice
                    else:
                        logger.warning(f"Invalid option, please enter one of {', '.join(valid_options)}")
                else:
                    valid_options = ['A', 'B', 'C']
                    if has_other:
                        valid_options.append('S')
                    choice = input(f"Please enter option ({'/'.join(valid_options)}): ").strip().upper()
                    if choice in valid_options:
                        logger.info(f"Selected filter mode: {choice}")
                        return choice
                    else:
                        logger.warning(f"Invalid option, please enter one of {', '.join(valid_options)}")
            except (EOFError, KeyboardInterrupt):
                logger.warning("User interrupted, using default mode: A")
                return 'A'
    
    def _get_filter_mode_description(self, filter_mode: str) -> str:
        """Get filter mode description."""
        descriptions = {
            'A': 'Keep level 2 and below (e.g., 1.1, 1.1.1, 1.1.1.1...)',
            'B': 'Keep level 3 and below (e.g., 1.1.1, 1.1.1.1...)',
            'C': 'Keep level 4 and below',
            'D': 'Keep level 5 and below',
            'E': 'Keep level 6 and below',
            'S': 'Special mode - configure each type separately',
            'none': 'No clause numbers detected, keep all content'
        }
        return descriptions.get(filter_mode, 'Unknown mode')
    
    def save_suggestions(self, analysis_result: Dict, output_path: Path):
        """Save analysis results and suggestions to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis results saved to: {output_path}")
        
        report_path = output_path.with_suffix('.txt')
        self._generate_readable_report(analysis_result, report_path)
    
    def _generate_readable_report(self, analysis_result: Dict, report_path: Path):
        """Generate human-readable analysis report."""
        stats = analysis_result['statistics']
        suggestions = analysis_result['suggestions']
        clause_stats = stats.get('clause_stats', {})
        detected_types = clause_stats.get('detected_types', [])
        
        type_names = {
            'chapter': '章',
            'section': '节',
            'article': '条',
            'standard': '标准条款（数字.数字...）',
            'letter': '字母编号（A.1.2）',
            'chinese_paren': '中文括号（（一））',
            'digit_paren': '数字括号（（1））',
            'digit_dot': '数字点（1.）',
            'circled_digit': '带圈数字（①②③）'
        }
        
        report_lines = [
            "=" * 80,
            "Markdown Text Analysis Report",
            "=" * 80,
            "",
            f"Total lines: {stats.get('total_lines', 0)}",
            "",
            "【Clause Number Statistics】",
            "-" * 80,
        ]
        

        if detected_types:
            for clause_type in detected_types:
                count = clause_stats.get(f'{clause_type}_count', 0)
                type_name = type_names.get(clause_type, clause_type)
                report_lines.append(f"{type_name}: {count}")
                

                if clause_type == 'standard' and 'standard_level_dist' in clause_stats:
                    level_dist = clause_stats['standard_level_dist']
                    for level in sorted(level_dist.keys()):
                        report_lines.append(f"  {level}层: {level_dist[level]}个")
        else:
            report_lines.append("No clause numbers detected")
        

        report_lines.extend([
            "",
            "【Legacy Format Statistics (for compatibility)】",
            "-" * 80,
            f"X.X.X format (specific clauses): {clause_stats.get('xxx_count', 0)}",
            f"X.X format (section titles): {clause_stats.get('xx_count', 0)}",
            f"X format (level 1 titles): {clause_stats.get('x_count', 0)}",
        ])
        
        report_lines.extend([
            "",
            "【Filter Mode】",
            "-" * 80,
            f"Selected mode: {suggestions.get('filter_mode', 'unknown')}",
            f"Description: {suggestions.get('comment', '')}",
            "",
        ])
        
        # 显示样本
        for clause_type in detected_types:
            samples = clause_stats.get(f'{clause_type}_samples', [])
            if samples:
                type_name = type_names.get(clause_type, clause_type)
                report_lines.extend([
                    f"【{type_name} Samples (first 10)】",
                    "-" * 80,
                ])
                for i, sample in enumerate(samples[:10], 1):
                    report_lines.append(f"{i}. Line {sample['line']}: {sample['text']}")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "Note: Please check JSON config file for detailed parameters",
            "=" * 80
        ])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Readable report saved to: {report_path}")
