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
        """Identify clause numbers."""
        lines = text.split('\n')
        
        xxx_clauses = []
        xx_clauses = []
        x_clauses = []
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if self.clause_xxx_pattern.match(line_stripped):
                match = self.clause_xxx_pattern.match(line_stripped)
                clause_num = match.group(0)
                xxx_clauses.append({
                    'line': line_num,
                    'number': clause_num,
                    'text': line_stripped[:100]
                })
            
            elif self.clause_xx_pattern.match(line_stripped):
                match = self.clause_xx_pattern.match(line_stripped)
                clause_num = match.group(0)
                xx_clauses.append({
                    'line': line_num,
                    'number': clause_num,
                    'text': line_stripped[:100]
                })
            
            elif self.clause_x_pattern.match(line_stripped):
                match = re.search(r'\d+', line_stripped)
                if match:
                    clause_num = match.group(0)
                    x_clauses.append({
                        'line': line_num,
                        'number': clause_num,
                        'text': line_stripped[:100]
                    })
        
        return {
            'xxx_count': len(xxx_clauses),
            'xx_count': len(xx_clauses),
            'x_count': len(x_clauses),
            'xxx_samples': xxx_clauses[:10],
            'xx_samples': xx_clauses[:10],
            'x_samples': x_clauses[:10]
        }
    
    def _interactive_confirm_filter_mode(self, clause_stats: Dict) -> str:
        """Interactively confirm filter mode."""
        xxx_count = clause_stats.get('xxx_count', 0)
        xx_count = clause_stats.get('xx_count', 0)
        x_count = clause_stats.get('x_count', 0)
        
        logger.info("=" * 80)
        logger.info("Clause number identification results:")
        logger.info(f"  X.X.X format (specific clauses): {xxx_count}")
        logger.info(f"  X.X format (section titles): {xx_count}")
        logger.info(f"  X format (level 1 titles): {x_count}")
        logger.info("=" * 80)
        
        if xxx_count == 0:
            logger.warning("No X.X.X format clause numbers detected, will keep all content")
            return 'none'
        
        if clause_stats.get('xxx_samples'):
            logger.info("\nX.X.X format samples (first 5):")
            for i, sample in enumerate(clause_stats['xxx_samples'][:5], 1):
                logger.info(f"  {i}. {sample['text']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Please select filter mode:")
        logger.info("  A: Keep only X.X.X level and smaller (e.g., 1.1.1, 1.1.2), discard all other content")
        logger.info("  C: Keep X.X and X.X.X levels, discard other content")
        logger.info("=" * 80)
        
        while True:
            try:
                choice = input("Please enter option (A/C): ").strip().upper()
                if choice in ['A', 'C']:
                    logger.info(f"Selected filter mode: {choice}")
                    return choice
                else:
                    logger.warning("Invalid option, please enter A or C")
            except (EOFError, KeyboardInterrupt):
                logger.warning("User interrupted, using default mode: A")
                return 'A'
    
    def _get_filter_mode_description(self, filter_mode: str) -> str:
        """Get filter mode description."""
        descriptions = {
            'A': 'Keep only X.X.X level and smaller (e.g., 1.1.1, 1.1.2), discard all other content',
            'C': 'Keep X.X and X.X.X levels, discard other content',
            'none': 'No X.X.X format detected, keep all content'
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
        
        report_lines = [
            "=" * 80,
            "Markdown Text Analysis Report",
            "=" * 80,
            "",
            f"Total lines: {stats.get('total_lines', 0)}",
            "",
            "【Clause Number Statistics】",
            "-" * 80,
            f"X.X.X format (specific clauses): {clause_stats.get('xxx_count', 0)}",
            f"X.X format (section titles): {clause_stats.get('xx_count', 0)}",
            f"X format (level 1 titles): {clause_stats.get('x_count', 0)}",
            "",
            "【Filter Mode】",
            "-" * 80,
            f"Selected mode: {suggestions.get('filter_mode', 'unknown')}",
            f"Description: {suggestions.get('comment', '')}",
            "",
        ]
        
        if clause_stats.get('xxx_samples'):
            report_lines.extend([
                "【X.X.X Format Samples (first 10)】",
                "-" * 80,
            ])
            for i, sample in enumerate(clause_stats['xxx_samples'][:10], 1):
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
