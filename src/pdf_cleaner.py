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
        
        self.clause_xxx_pattern = re.compile(r'^\d+\.\d+\.\d+')
        self.clause_xx_pattern = re.compile(r'^\d+\.\d+')
        self.clause_x_pattern = re.compile(r'^#\s+\d+\s+')
        
    def clean(self, text: str, analysis_result: Optional[Dict] = None) -> str:
        """Execute cleaning."""
        logger.info("Starting Markdown text cleaning...")
        
        filter_mode = None
        if analysis_result and 'suggestions' in analysis_result:
            filter_mode = analysis_result['suggestions'].get('filter_mode')
        
        if filter_mode and filter_mode != 'none':
            text = self._filter_by_clause_numbers(text, filter_mode)
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
    
    def _filter_by_clause_numbers(self, text: str, filter_mode: str) -> str:
        """Filter content based on clause numbers."""
        lines = text.split('\n')
        filtered_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            is_xxx = bool(self.clause_xxx_pattern.match(line_stripped))
            is_xx = bool(self.clause_xx_pattern.match(line_stripped))
            is_x = bool(self.clause_x_pattern.match(line_stripped))
            
            should_keep = False
            
            if filter_mode == 'A':
                if is_xxx:
                    should_keep = True
                    filtered_lines.append(line)
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        next_stripped = next_line.strip()
                        if (self.clause_xxx_pattern.match(next_stripped) or
                            self.clause_xx_pattern.match(next_stripped) or
                            self.clause_x_pattern.match(next_stripped)):
                            break
                        filtered_lines.append(next_line)
                        i += 1
                    continue
            
            elif filter_mode == 'C':
                if is_xxx or is_xx:
                    should_keep = True
                    filtered_lines.append(line)
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        next_stripped = next_line.strip()
                        if (self.clause_xxx_pattern.match(next_stripped) or
                            self.clause_xx_pattern.match(next_stripped) or
                            self.clause_x_pattern.match(next_stripped)):
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
