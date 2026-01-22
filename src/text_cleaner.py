"""
Rule-based text cleaning: extract metadata (TOC, preface, refs), drop headers, keep clause text.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger


class TextCleaner:
    """Extract metadata, strip headers, retain clause text with clause-id patterns."""

    def __init__(self, config: Dict):
        """Init from text_cleaner config."""
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extract_sections = config['extract_sections']
        self.remove_headers = config['remove_headers']
        self.clause_patterns = [re.compile(p) for p in config['clause_patterns']]
        
        logger.info(f"TextCleaner initialized")
        logger.info(f"Extract sections: {self.extract_sections}")
        logger.info(f"Clause pattern count: {len(self.clause_patterns)}")
    
    def clean_text(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Clean text: extract metadata, parse clauses, strip headers. Return clean_text, clauses, statistics."""
        logger.info("Starting text cleaning")
        extracted_metadata, remaining_text = self._extract_metadata_sections(text)
        clauses = self._parse_clauses(remaining_text)
        logger.info(f"Identified {len(clauses)} clauses")
        clean_text = self._remove_headers_and_build_text(clauses)
        statistics = {
            'original_length': len(text),
            'clean_length': len(clean_text),
            'num_clauses': len(clauses),
            'metadata_sections': list(extracted_metadata.keys())
        }
        
        logger.info(f"Markdown cleaning completed, original length: {statistics['original_length']}, "
                   f"cleaned: {statistics['clean_length']}")
        
        return {
            'clean_text': clean_text,
            'extracted_metadata': extracted_metadata,
            'clauses': clauses,
            'statistics': statistics
        }
    
    def _extract_metadata_sections(self, text: str) -> Tuple[Dict, str]:
        """Extract TOC, preface, etc. into metadata; return (metadata, remaining text)."""
        extracted = {}
        remaining_text = text
        for section_name in self.extract_sections:
            pattern = rf'(?:#+\s*)?{section_name}(?:\s|：|:)?\n(.*?)(?=\n#+\s+|\Z)'
            matches = re.finditer(pattern, remaining_text, re.DOTALL | re.IGNORECASE)
            section_content = []
            for match in matches:
                content = match.group(0).strip()
                section_content.append(content)
                remaining_text = remaining_text.replace(match.group(0), '\n\n')
            
            if section_content:
                extracted[section_name] = '\n\n'.join(section_content)
                logger.debug(f"Extracted {section_name}: {len(section_content)} fragments")
        
        return extracted, remaining_text
    
    def _parse_clauses(self, text: str) -> List[Dict]:
        """Parse clauses from text; return list of {clause_id, content, header, header_level}."""
        lines = text.split('\n')
        clauses = []
        current_clause = None
        current_header = None
        current_header_level = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if header_match:
                current_header_level = len(header_match.group(1))
                current_header = header_match.group(2)
                continue
            is_clause_start = False
            clause_id = None
            
            for pattern in self.clause_patterns:
                match = pattern.match(line)
                if match:
                    is_clause_start = True
                    clause_id = match.group(0)
                    break
            
            if is_clause_start:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = {
                    'clause_id': clause_id,
                    'content': line,
                    'header': current_header,
                    'header_level': current_header_level
                }
            elif current_clause:
                current_clause['content'] += '\n' + line
        if current_clause:
            clauses.append(current_clause)
        
        return clauses
    
    def _remove_headers_and_build_text(self, clauses: List[Dict]) -> str:
        """Strip configured header levels and build clean text from clause content."""
        remove_levels = set()
        for header_config in self.remove_headers:
            level = header_config.get('level')
            if level:
                remove_levels.add(level)
        clean_lines = []
        for clause in clauses:
            header_level = clause.get('header_level', 0)
            if header_level not in remove_levels and clause.get('header'):
                pass
            clean_lines.append(clause['content'])
        
        return '\n\n'.join(clean_lines)
    
    def save_cleaned_data(self, result: Dict, source_file: str) -> str:
        """Save cleaned Markdown and full JSON to clean_data for extraction."""
        base_name = Path(source_file).stem
        md_output_path = self.output_dir / f"{base_name}.md"
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(result['clean_text'])
        logger.info(f"Saved cleaned Markdown file: {md_output_path}")
        json_result = {**result, 'doc_id': base_name}
        json_output_path = self.output_dir / f"{base_name}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved cleaning result JSON: {json_output_path}")
        
        return str(md_output_path)
    
    def process_file(self, input_file: str) -> Dict:
        """Process single markdown file; return cleaned result."""
        logger.info(f"Processing file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        result = self.clean_text(text)
        output_path = self.save_cleaned_data(result, input_file)
        result['output_path'] = output_path
        result['source_file'] = input_file
        
        return result
    
    def batch_process(self, input_dir: str, pattern: str = "*_processed.md") -> List[Dict]:
        """Batch process files in directory; return list of results."""
        input_dir = Path(input_dir)
        files = list(input_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} files to process")
        
        results = []
        for file in files:
            try:
                result = self.process_file(str(file))
                results.append(result)
            except Exception as e:
                logger.error(f"Processing failed: {file.name}, error: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(files)}")
        return results



