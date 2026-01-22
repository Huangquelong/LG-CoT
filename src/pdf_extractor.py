"""
PDF extraction via MinerU CLI: extract -> analyze -> suggest -> clean.
"""

import os
import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from .pdf_analyzer import PDFAnalyzer
    from .pdf_cleaner import PDFCleaner
except ImportError:
    from pdf_analyzer import PDFAnalyzer
    from pdf_cleaner import PDFCleaner

HAS_MINERU_CLI = False
MINERU_CLI_PATH = None

try:
    result = subprocess.run(['mineru', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        HAS_MINERU_CLI = True
        MINERU_CLI_PATH = 'mineru'
        logger.info("Detected MinerU CLI tool (mineru CLI)")
except (FileNotFoundError, subprocess.TimeoutExpired):
    try:
        result = subprocess.run(['python', '-m', 'mineru', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            HAS_MINERU_CLI = True
            MINERU_CLI_PATH = ['python', '-m', 'mineru']
            logger.info("Detected MinerU CLI tool (python -m mineru)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

if not HAS_MINERU_CLI:
    logger.warning("MinerU CLI tool not detected, please ensure mineru is installed")


class PDFExtractor:
    """PDF-to-markdown extractor using MinerU CLI."""

    def __init__(self, config: Dict):
        """Init from pdf_extractor config."""
        if not HAS_MINERU_CLI:
            raise ImportError(
                "MinerU CLI tool not installed, please install first:\n"
                "  pip install mineru[all]\n"
                "  or ensure mineru command is in PATH"
            )
        self.config = config
        self.input_pdf_dir = Path(config.get('input_pdf_dir', 'data'))
        self.output_dir = Path(config.get('output_dir', 'output/markdown'))
        self.suggestion_dir = Path(config.get('suggestion_dir', 'output/pdf_analysis'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suggestion_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_mode = config.get('mineru_mode', 'txt')
        self.mineru_timeout = config.get('mineru_timeout', 3600)
        self.analyzer = PDFAnalyzer(config)
        self.cleaner = PDFCleaner(config)
        
        logger.info(f"PDFExtractor initialized (using MinerU CLI tool)")
        logger.info(f"  Input directory: {self.input_pdf_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Suggestion directory: {self.suggestion_dir}")

    def _call_mineru_cli(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run MinerU CLI on PDF; return extraction result dict."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")
        pdf_name = pdf_path.stem if pdf_path.is_file() else pdf_path.name
        logger.info(f"Starting MinerU parsing: {pdf_path} (mode: {self.mineru_mode})")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if MINERU_CLI_PATH is None:
                cmd = ['mineru']
            elif isinstance(MINERU_CLI_PATH, list):
                cmd = MINERU_CLI_PATH.copy()
            else:
                cmd = [MINERU_CLI_PATH]
            cmd.extend([
                '-p', str(pdf_path),
                '-o', str(output_dir),
                '-m', self.mineru_mode,
            ])
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            logger.info("Starting processing, this may take several minutes (large files may take longer)...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            output_lines = []
            last_log_time = time.time()
            try:
                while True:
                    if process.poll() is not None:
                        remaining_output = process.stdout.read()
                        if remaining_output:
                            output_lines.append(remaining_output)
                        break
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line)
                        current_time = time.time()
                        if current_time - last_log_time > 30:
                            elapsed_minutes = int((current_time - last_log_time) / 60)
                            logger.info(f"MinerU processing... (running for {elapsed_minutes} minutes)")
                            last_log_time = current_time
                    else:
                        time.sleep(0.1)
                return_code = process.wait(timeout=1)
                output_text = ''.join(output_lines)
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                timeout_minutes = self.mineru_timeout / 60
                logger.error(f"MinerU CLI execution timeout (exceeded {timeout_minutes:.0f} minutes)")
                logger.error(f"Last output: {''.join(output_lines[-20:])}")
                raise RuntimeError(
                    f"PDF parsing timeout (running for more than {timeout_minutes:.0f} minutes).\n"
                    f"Possible causes and solutions:\n"
                    f"1. PDF file too large - try processing smaller PDFs or process in segments\n"
                    f"2. GPU not properly used - run 'nvidia-smi' to check GPU usage\n"
                    f"3. Model not downloaded - first run may need to download models, please wait\n"
                    f"4. Insufficient VRAM - check if other programs are using it\n"
                    f"5. Wrong mode used - if it's a scanned PDF, may need to use '-m auto' instead of '-m txt'\n"
                )
            
            if return_code != 0:
                logger.error(f"MinerU CLI execution failed, return code: {return_code}")
                logger.error(f"Full output:\n{output_text}")
                raise RuntimeError(f"MinerU CLI execution failed (return code: {return_code}):\n{output_text[-1000:]}")
            logger.info("MinerU CLI command execution completed, searching for output files...")
            md_path = None
            if pdf_path.is_file():
                pdf_name = pdf_path.stem
                possible_paths = [
                    output_dir / f"{pdf_name}.md",
                    output_dir / pdf_name / f"{pdf_name}.md",
                    output_dir / pdf_name / "markdown" / f"{pdf_name}.md",
                ]
                
                for path in possible_paths:
                    if path.exists():
                        md_path = path
                        break
                if md_path is None:
                    md_files = list(output_dir.rglob("*.md"))
                    if md_files:
                        for md_file in md_files:
                            if pdf_name in md_file.stem or md_file.stem == pdf_name:
                                md_path = md_file
                                break
                        if md_path is None:
                            md_path = md_files[0]
                        logger.info(f"Found Markdown file: {md_path}")
            elif pdf_path.is_dir():
                md_files = list(output_dir.rglob("*.md"))
                if md_files:
                    md_path = md_files[0]
                    logger.info(f"找到 Markdown 文件: {md_path}")
            if md_path is None or not md_path.exists():
                logger.warning(f"Output directory contents: {list(output_dir.iterdir())}")
                if output_dir.exists():
                    for subdir in output_dir.iterdir():
                        if subdir.is_dir():
                            logger.warning(f"Subdirectory {subdir.name} contents: {list(subdir.iterdir())}")
                raise FileNotFoundError(
                    f"Markdown file not generated, output directory: {output_dir}\n"
                    f"Input path: {pdf_path}\n"
                    f"MinerU output (last 500 chars): {output_text[-500:]}"
                )
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            image_dir = None
            possible_image_dirs = [
                md_path.parent / "images",
                output_dir / "images",
                output_dir / pdf_name / "images",
            ]
            for img_dir in possible_image_dirs:
                if img_dir.exists() and img_dir.is_dir():
                    image_dir = img_dir
                    break
            
            logger.info(f"MinerU CLI parsing completed, generated text length: {len(markdown_text)} characters")
            return {
                'text': markdown_text,
                'metadata': {
                    'pdf_path': str(pdf_path),
                    'pdf_name': pdf_name,
                    'text_length': len(markdown_text),
                    'markdown_path': str(md_path),
                    'image_dir': str(image_dir) if image_dir else None,
                    'method': 'CLI'
                }
            }
            
        except subprocess.TimeoutExpired:
            logger.error("MinerU CLI execution timeout")
            raise RuntimeError("PDF parsing timeout")
        except FileNotFoundError as e:
            raise
        except Exception as e:
            logger.error(f"MinerU CLI parsing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def extract_from_pdf(self, pdf_path: str, analyze_only: bool = False,
                         use_suggestions: bool = False) -> Dict[str, Any]:
        """Extract text from PDF. Optionally analyze-only or use suggestions."""
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        logger.info(f"Starting PDF processing: {pdf_name}")
        try:
            pdf_output_dir = self.output_dir / pdf_name
            extracted_content = self._call_mineru_cli(pdf_path, pdf_output_dir)
            markdown_text = extracted_content['text']
            logger.info("Executing PDF structure analysis...")
            analysis_result = self.analyzer.analyze_pdf(extracted_content)
            suggestion_path = self.suggestion_dir / f"{pdf_name}_suggestions.json"
            self.analyzer.save_suggestions(analysis_result, suggestion_path)
            if analyze_only:
                return {
                    'text': markdown_text,
                    'metadata': extracted_content['metadata'],
                    'analysis_result': analysis_result,
                    'cleaned_text': None,
                    'suggestion_path': str(suggestion_path),
                    'source_pdf': str(pdf_path)
                }
            if use_suggestions:
                logger.info("Automatically applying analysis suggestions...")
            logger.info("Executing PDF cleaning...")
            cleaned_text = self.cleaner.clean(markdown_text, analysis_result)
            self.cleaner.save_cleaned_data(
                cleaned_text, 
                pdf_name,
                metadata={
                    **extracted_content['metadata'],
                    'suggestion_path': str(suggestion_path),
                    'analysis_result': analysis_result
                }
            )
            clean_data_path = Path('output/clean_data') / f"{pdf_name}.md"
            logger.info(f"PDF processing completed, cleaned text saved to: {clean_data_path}")
            
            return {
                'text': cleaned_text,
                'original_text': markdown_text,
                'metadata': {
                    **extracted_content['metadata'],
                    'output_path': str(clean_data_path),
                    'suggestion_path': str(suggestion_path)
                },
                'analysis_result': analysis_result,
                'cleaned_text': cleaned_text,
                'source_pdf': str(pdf_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def batch_extract(self, pdf_dir: str) -> List[Dict]:
        """Batch-extract PDFs from directory. Returns list of result dicts with source_pdf."""
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF目录不存在: {pdf_dir}")
        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in directory: {pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files, starting batch processing...")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing progress: {i}/{len(pdf_files)} - {pdf_path.name}")
            try:
                result = self.extract_from_pdf(str(pdf_path))
                if 'source_pdf' not in result:
                    result['source_pdf'] = str(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                results.append({
                    'source_pdf': str(pdf_path),
                    'status': 'error',
                    'error': str(e),
                    'text': ''
                })
        
        success_count = sum(1 for r in results if r.get('status') != 'error' and 'text' in r)
        logger.info(f"Batch processing completed, succeeded: {success_count}, "
                   f"failed: {len(results) - success_count}")
        
        return results


