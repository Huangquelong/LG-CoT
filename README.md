# Knowledge Graph Construction (KGC) System —— LG-CoT Pipeline

A comprehensive knowledge extraction system that converts PDF documents into structured knowledge graphs using Large Language Models (LLMs) and First-Order Logic (FOL) representation.

## Overview

This system implements a complete pipeline for extracting structured knowledge from PDF documents, particularly suited for regulatory documents, standards, and technical specifications. It uses a Logic-Guided Chain-of-Thought (LG-CoT) approach to extract knowledge in First-Order Logic format, which is then mapped to JSON structures and optionally imported into Neo4j knowledge graphs.


## ✨ Highlights

**End-to-end Pipeline**: PDF Parsing → Cleaning → Schema Construction → FOL Extraction → Validation → JSON Output → (Optional) Neo4j Ingestion
**LG-CoT Extraction**: For legal regulations, standards, technical specifications, etc., output traceable logical expressions
**Multi-baseline Comparison**: StandCoT / Random Few-shot / Semantic Few-shot
**Evaluation and Downstream**: Supports LLM Expert Evaluation and RAG (NaiveRAG vs GraphRAG) comparison
**Configurable**: All parameters are unified in `config.yaml`

## Installation

### Prerequisites

- Python 3.8+
- Neo4j (optional, for knowledge graph features)
- CUDA-capable GPU (recommended for MinerU)
- (建议) Linux / WSL 环境运行 MinerU 与 GPU 推理更稳定



### Setup

1. Clone the repository:
```bash
git clone https://github.com/Huangquelong/LG-CoT.git
cd KGC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `config.yaml`:
```yaml
llm_api:
  SiliconFlow:
    api_key: "your_api_key_here"
    api_base: "https://api.siliconflow.cn/v1"

Important: Replace all placeholder API keys in config.yaml with your actual keys before running.
```

## Configuration

All configuration is managed through `config.yaml`. Key sections include:

### LLM API Configuration
```yaml
llm_api:
  SiliconFlow:
    api_key: "your_api_key_here"
    api_base: "https://api.siliconflow.cn/v1"
  openai:
    api_key: "your-openai-api-key"
    api_base: "https://api.openai.com/v1"
```

### ⚠️MinerU Model Download and Placement
Your project structure includes the "model/MinerU/" directory (for high-quality PDF parsing), but due to the large size of the model, the MinerU model files are not uploaded in this repository. You need to download and install the MinerU model from the official MinerU project to your local machine and place the weights/resources under the "model/MinerU/" directory.

Clone the MinerU repository (please refer to the official MinerU repository page; if the link changes, please search for "MinerU" on GitHub):
```bash
git clone <MinerU-repo-url> MinerU
mkdir -p model/MinerU
# Copy the weights/resources downloaded from MinerU to the "model/MinerU" directory.
cp -r <path_to_mineru_models_or_assets>/* model/MinerU/
```


### PDF Extractor
- MinerU model configuration
- Layout parameters (headers, footers, page numbers)
- Filtering rules (tables, images, appendices)

### Knowledge Extractor
- Model selection
- Temperature and max_tokens
- Batch processing parameters
- Parallel worker configuration

### Baseline Experiment
- Model and API configuration
- Rate limiter settings (RPM, TPM)
- Word2Vec parameters
- Input/output paths

### Evaluation
- LLM expert evaluation settings
- RAG evaluation configuration

**Important**: Replace all placeholder API keys in `config.yaml` with your actual keys before running.

## Usage

### Main Pipeline

Run the complete knowledge extraction pipeline:

```bash
python src/main.py --input data/example.pdf
```

Options:
- `--input`: Single PDF file path
- `--batch`: Batch processing mode
- `--step`: Run specific step (pdf, clean, schema, extract, all)
- `--config`: Custom config file path
- `--sample-size`: Limit number of clauses processed (for testing)

### Baseline Experiments

Run baseline comparison experiments:

```python
from Validation.baseline_experiment_pipeline import BaselineExperimentPipeline

pipeline = BaselineExperimentPipeline()
pipeline.run()
```

This will run three methods:
1. **StandCoT**: Standard Chain-of-Thought prompting
2. **Random_few_shot**: Random few-shot examples
3. **Semantic_few_shot**: Semantically similar few-shot examples

### RAG Evaluation

Evaluate RAG systems:

```bash
cd Downstream_Evaluation
python rag_eval_system.py
```

This compares:
- **NaiveRAG**: Simple text chunking with vector similarity
- **GraphRAG**: Structured retrieval using extracted knowledge

### LLM Expert Evaluation

Run expert evaluation on extraction results:

```python
from Evaluation.LLM_Expert_evaluation import LLMExpertEvaluation

evaluator = LLMExpertEvaluation()
# Run evaluation...
```

## 🧱Project Structure

```
.
├── src/                          # Main pipeline modules
│   ├── main.py                   # Main pipeline orchestrator
│   ├── pdf_extractor.py          # PDF to markdown conversion (MinerU)
│   ├── text_cleaner.py           # Text cleaning and preprocessing
│   ├── schema_builder.py         # Dynamic schema generation
│   ├── knowledge_extractor.py    # LG-CoT extraction
│   ├── validator.py              # Bidirectional verification
│   └── knowledge_graph.py        # Neo4j integration
├── Validation/                   # Baseline experiments
│   └── baseline_experiment_pipeline.py
├── Evaluation/                   # Evaluation modules
│   └── LLM_Expert_evaluation.py
├── Downstream_Evaluation/         # RAG evaluation
│   ├── rag_eval_system.py
│   └── rag_questions.json
├── model/
│   └── MinerU/                   # (NOT included) MinerU PDF extraction model assets
├── Repository/                   # Predicate and FOL repositories
├── output/                       # Output directories (logs/results/json)
├── config.yaml                   # Main configuration file
├── output_template.json          # JSON output template
├── global_schema.json            # Global schema definitions
└── requirements.txt              # Python dependencies

```

## 🔄Workflow

1. **PDF Extraction**: Convert PDF to structured markdown using MinerU
2. **Text Cleaning**: Remove headers, footers, extract metadata
3. **Schema Building**: 
   - Step 0: Sample texts and generate domain predicates
   - Build FOL example repository
4. **Knowledge Extraction**:
   - Step 2: Generate FOL representation
   - Step 3: Validate predicates
   - Step 5: Self-verify FOL correctness
   - Step 6: Map FOL to JSON structure
5. **Validation**: Bidirectional checks ensure quality
6. **Output**: Structured JSON ready for knowledge graph import

## Key Components

### PDF Extractor
- Uses MinerU for high-quality PDF parsing
- Configurable layout detection (headers, footers, page numbers)
- Filtering rules for tables, images, appendices

### Schema Builder
- Dynamically generates domain-specific predicates
- Builds FOL example repository for few-shot learning
- Supports cross-document predicate sharing

### Knowledge Extractor
- Implements lgcot-CoT (Logic-Guided Chain-of-Thought)
- Uses Word2Vec for semantic similarity search
- Multi-threaded batch processing

### Validator
- Step 3: Validates generated predicates against domain
- Step 5: Self-checks FOL formula correctness
- Review repository for failed cases

## 📦Output Format

The system outputs structured JSON following `output_template.json`:

```json
{
  "id": "doc_id",
  "rules": [
    {
      "type": "OBLIGATION",
      "core_event": {
        "subject": "Subject entity",
        "action": "Action",
        "object": "Object entity"
      },
      "conditions": [...],
      "constraints": {
        "time_limit": "...",
        "frequency": "...",
        "manner": "..."
      },
      "logic_form": "FOL representation",
      "source_text": "Original clause text"
    }
  ]
}
```

## Configuration Guide

### Setting Up API Keys

1. Open `config.yaml`
2. Find the `llm_api` section
3. Replace placeholder values:
   - `your_siliconflow_api_key_here` → Your SiliconFlow API key
   - `your-openai-api-key` → Your OpenAI API key (if using)

### Adjusting Rate Limits

In `config.yaml`, configure rate limiters:

```yaml
baseline_experiment:
  baseline:
    rate_limiter:
      rpm: 2000  # Requests per minute
      tpm: 80000  # Tokens per minute
```

### Model Selection

Choose your LLM model:

```yaml
knowledge_extractor:
  model: "Qwen/Qwen3-14B"  # Change to your preferred model
```

### Common Issues

1. **API Key Errors**: Ensure API keys are correctly set in `config.yaml`
2. **Rate Limiting**: Adjust RPM/TPM limits in config if hitting API limits
3. **Memory Issues**: Reduce `batch_workers` in knowledge_extractor config
4. **PDF Parsing Errors**: Check MinerU installation and model availability

### Logs

Logs are stored in:
- Main pipeline: `output/logs/`
- Baseline experiments: `Baseline/Validation/baseline_experiment.log`
- Evaluation: `Evaluation/llm_expert_evaluation.log`

## MinerU Citation
@misc{niu2025mineru25decoupledvisionlanguagemodel,
      title={MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing}, 
      author={Junbo Niu and Zheng Liu and Zhuangcheng Gu and Bin Wang and Linke Ouyang and Zhiyuan Zhao and Tao Chu and Tianyao He and Fan Wu and Qintong Zhang and Zhenjiang Jin and Guang Liang and Rui Zhang and Wenzheng Zhang and Yuan Qu and Zhifei Ren and Yuefeng Sun and Yuanhong Zheng and Dongsheng Ma and Zirui Tang and Boyu Niu and Ziyang Miao and Hejun Dong and Siyi Qian and Junyuan Zhang and Jingzhou Chen and Fangdong Wang and Xiaomeng Zhao and Liqun Wei and Wei Li and Shasha Wang and Ruiliang Xu and Yuanyuan Cao and Lu Chen and Qianqian Wu and Huaiyu Gu and Lindong Lu and Keming Wang and Dechen Lin and Guanlin Shen and Xuanhe Zhou and Linfeng Zhang and Yuhang Zang and Xiaoyi Dong and Jiaqi Wang and Bo Zhang and Lei Bai and Pei Chu and Weijia Li and Jiang Wu and Lijun Wu and Zhenxiang Li and Guangyu Wang and Zhongying Tu and Chao Xu and Kai Chen and Yu Qiao and Bowen Zhou and Dahua Lin and Wentao Zhang and Conghui He},
      year={2025},
      eprint={2509.22186},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.22186}, 
}

@misc{wang2024mineruopensourcesolutionprecise,
      title={MinerU: An Open-Source Solution for Precise Document Content Extraction}, 
      author={Bin Wang and Chao Xu and Xiaomeng Zhao and Linke Ouyang and Fan Wu and Zhiyuan Zhao and Rui Xu and Kaiwen Liu and Yuan Qu and Fukai Shang and Bo Zhang and Liqun Wei and Zhihao Sui and Wei Li and Botian Shi and Yu Qiao and Dahua Lin and Conghui He},
      year={2024},
      eprint={2409.18839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.18839}, 
}

@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}

## Acknowledgments

- MinerU for high-quality PDF extraction capabilities (required by model/MinerU)
- OpenAI/SiliconFlow for LLM APIs
- Neo4j for graph database support

---

## 开源声明 / Open Source Disclaimer
本项目仅供科研与学习使用（For research and learning purposes only）。

如有商业用途需求，请提前联系作者获取授权。未经授权不得将本项目用于任何商业目的。

For commercial use, please contact the author for permission. Unauthorized commercial use is prohibited.






