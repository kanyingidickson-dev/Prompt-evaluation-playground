# Enterprise Prompt Evaluation Playground

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

A professional, extensible framework for systematically evaluating, scoring, and comparing Large Language Model (LLM) prompts. Designed for ML Engineers and Researchers who need to move beyond ad-hoc testing to rigorous, metric-driven prompt engineering.

## 🚀 Project Overview

In production environments, prompt engineering cannot be based on "vibes" or single examples. This playground provides a deterministic test harness to:

1.  **Quantify Quality**: Assign numerical scores to subjective metrics like Relevance, Clarity, and Safety.
2.  **Compare Variations**: A/B test prompt templates on consistent datasets to statistically determine the winner.
3.  **Detect Regressions**: Catch failures (hallucinations, refusals, safety breaches) before deployment.
4.  **Abstract Backends**: Seamlessly switch between OpenAI, Anthropic, or Local models for evaluation.

## 🏗 Architecture

The system is designed with a modular "Plug-and-Play" architecture:

```ascii
[Datasets]      [Prompts]       [Models]
    |               |               |
    v               v               v
    +-------------------------------+
    |    Prompt Execution Engine    |
    +---------------+---------------+
                    |
                    v
            [Raw Responses]
                    |
    +---------------+---------------+
    |     Evaluation Framework      |
    | (Relevance, Safety, Accuracy) |
    +---------------+---------------+
                    |
                    v
    [Results Store] -> [Analysis & Reporting]
```

### Core Components

*   **`models/`**: Abstracted LLM clients. Includes a `LocalModelClient` for cost-free testing and `OpenAIClient` for production usage.
*   **`evaluators/`**: Scoring logic. Supports both heuristic-based rules (regex, keywords) and LLM-as-a-Judge patterns.
*   **`scripts/`**: CLI tools for running experiments (`run_experiment.py`) and analyzing results (`compare_prompts.py`).
*   **`config/`**: YAML-based declarative configuration for reproducible experiments.

## 📊 Scoring Methodology

We employ a multi-dimensional scoring rubric. Each dimension scores from **0.0 to 10.0**.

| Dimension | Description | Method |
| :--- | :--- | :--- |
| **Relevance** | Does the response directly address the user query? | NLP Keyword Overlap / Vector Similarity |
| **Accuracy** | Is the information factually correct relative to ground truth? | Sequence Matching / Fact Checking Evaluator |
| **Clarity** | Is the response well-structured (bullet points, length)? | Heuristic Rules |
| **Safety** | Does the response avoid harmful content and policy violations? | Keyword Blacklist / Classifier |

*Note: For production use cases, we recommend extending `evaluators/base.py` to use a strong LLM (e.g., GPT-4) as a Judge for nuanced scoring.*

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/kanyingidickson-dev/Prompt-evaluation-playground.git
cd Prompt-evaluation-playground
pip install -r requirements.txt
```

### 2. Configuration

Set your API keys (if using real models):
```bash
export OPENAI_API_KEY="sk-..."
```

Review the experiment config in `config/evaluation.yaml` to select your models and datasets.

### 3. Run an Experiment

Execute the main test harness:
```bash
python scripts/run_experiment.py --config config/evaluation.yaml
```

**Output:**
```text
🚀 Starting Experiment: financial-advisor-v1-benchmark
📋 Loaded Evaluators: ['relevance', 'safety', 'accuracy', 'clarity']
...
✅ Experiment Complete. Results saved to results/
```

### 4. Analyze Results

Compare prompts and rank performance:
```bash
python scripts/compare_prompts.py --results results/results.csv
```

Detect constraints failures:
```bash
python scripts/analyze_failures.py --threshold 5.0
```

## 🛠 Extending the System

### Adding a New Evaluator
Create a new file in `evaluators/` inheriting from `BaseEvaluator`:

```python
from .base import BaseEvaluator, EvaluationResult

class ToneEvaluator(BaseEvaluator):
    def evaluate(self, query, response, reference=None):
        # Your custom logic here
        return EvaluationResult(score=8.5, reasoning="Professional tone detected.")
```

### Adding a New Model
Implement the `BaseModelClient` interface in `models/`:

```python
class AnthropicClient(BaseModelClient):
    def generate(self, prompt, **kwargs):
        # Call Claude API
        pass
```

## 📄 License

Proprietary / Enterprise intent.

---
**Maintained by the AI Platform Team.**
