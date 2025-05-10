# NLP‑Project‑CS‑5624 — LLMThinkBench Experiments

<!-- [![Paper](https://img.shields.io/badge/Paper-PDF-blue)](LLMThinkBench___Project_Report.pdf) -->
[![Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-Streamlit-orange)](https://llmthinkbench-leaderboard.streamlit.app/)
[![LLMThinkBench Package](https://img.shields.io/pypi/v/llmthinkbench)](https://pypi.org/project/llmthinkbench/)
[![GitHub stars](https://img.shields.io/github/stars/ctrl-gaurav/NLP-Project-CS-5624?style=social)](https://github.com/ctrl-gaurav/NLP-Project-CS-5624/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A course project for **CS 5624 (Natural Language Processing)**, Virginia Tech.

We measure how much large language models *over‑think* on **basic** symbolic‑reasoning tasks and explore three ways to keep answers short without losing accuracy:

1. **Prompt engineering** – ask for the boxed answer only.  
2. **Token‑budget caps** – force every model to stay within a task-specific limit.  
3. **Distillation + SFT** – fine‑tune smaller Qwen‑2.5 models on concise Gemini‑2.0‑Flash traces.

All code builds on the open‑sourced benchmark **[LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench)** and feeds scores to a public leaderboard.

---

## 🚀 Quick start

```bash
# clone the repo
git clone https://github.com/ctrl-gaurav/NLP-Project-CS-5624.git
cd NLP-Project-CS-5624

# install core dependencies (CUDA 12.x recommended)
conda create -n thinkbench python=3.10 -y
conda activate thinkbench
pip install -r requirements.txt          # torch, transformers, vllm, bitsandbytes …
pip install llmthinkbench                # CLI + tasks as a PyPI package

# one‑line evaluation (example: Llama‑3‑Instruct‑8B on the SORT task)
llmthinkbench \
  --model meta-llama/Llama-3-8B-Instruct \
  --tasks sorting \
  --datapoints 1000 --folds 3 \
  --output-dir results/llama3_8b_sort
```

Fine‑tuning scripts expect 4× A100 80 GB GPUs but can be run on smaller hardware by lowering `--batch_size` and enabling gradient accumulation.

## 📊 Results Snapshot

| Model | Params | Avg. Accuracy ↑ | Avg. Tokens ↓ |
|-------|--------|-----------------|---------------|
| Qwen‑2.5‑3B (baseline) | 3 B | 74 % | 145 |
| Qwen‑2.5‑3B + token cap | 3 B | 73 % | 71 |
| Qwen‑2.5‑3B + SFT | 3 B | 91 % | 68 |

Full, always‑up‑to‑date numbers are available on the [leaderboard](https://llmthinkbench-leaderboard.streamlit.app/).

## 📁 Repository Structure

| Path | Description |
|------|-------------|
| [`Prompt Engineering/`](./Prompt%20Engineering) | Prompt templates and helper scripts for direct‑answer vs. standard CoT prompting. Contains experimental prompt variations, ablation studies, and parameter sweeps. |
| [`RegularHyperParams/`](./RegularHyperParams) | Baseline runs with vanilla sampling settings (temperature=0.7, top_p=0.9). Includes model outputs and evaluation metrics across all benchmark tasks. |
| [`TruncatingBasedOnTradeoff/`](./TruncatingBasedOnTradeoff) | Experiments that search for the best accuracy/token‑count trade‑off and then re‑run models under those token caps. Contains analysis scripts and visualizations of optimal trade-off points. |
| [`benchmarking-website-results/`](./benchmarking-website-results) | JSON output files automatically ingested by the Streamlit leaderboard. These files contain structured results for all evaluated models. |
| [`dataset/`](./dataset) | Synthetic data used in the paper (global benchmark set and Gemini distillation set). Includes both training and evaluation splits across all reasoning tasks. |
| [`distillation + sft/`](./distillation%20%2B%20sft) | Training configs, LoRA adapters, and logs for supervised fine‑tuning with Gemini‑2.0‑Flash supervision. Contains hyperparameters, loss curves, and adapter weights. |
| [`important-scripts/`](./important-scripts) | Core CLI (`cli.py`), model loader, evaluation utilities—entry points for most experiments. These are the backbone scripts used throughout the project. |
| [`llmthinkbench-tasks/`](./llmthinkbench-tasks) | All 14 benchmark tasks (sorting, counting, min/max, etc.) and their robust regex parsers. Each task has its own directory with problem generators and evaluation code. |
| [`plots.ipynb`](./plots.ipynb) | Jupyter notebook that reproduces all figures in the report (accuracy vs. tokens, trade‑off curves, etc.). |
| [`temp_and_top_p.zip`](./temp_and_top_p.zip) | Compressed results grid from the sampling‑parameter sweep (temperature × top‑p). Contains detailed outputs from parameter studies. |
| `LICENSE` | MIT License — free to use and modify. |
| `README.md` | This file - project overview and documentation. |
| Misc. | `.gitignore`, `.DS_Store`, original README.md placeholder. |

## 📚 Benchmark Tasks

LLMThinkBench includes 14 reasoning tasks across multiple categories:

### 🔢 Arithmetic
- Basic addition, subtraction, multiplication, division
- Multi-step arithmetic operations
- Sign recognition

### 🧮 Statistical
- Finding mean/average values
- Min/max identification
- Count operations
- Sum of series

### 🔀 Algorithmic
- Sorting numbers
- Even/odd number identification
- Comparison operations
- Pattern completion

Each task evaluates both answer accuracy and verbosity (token count).

## 🔬 Methodology

Our approach explores different methods to reduce LLM verbosity while maintaining or improving accuracy:

1. **Prompt Engineering**:
   - Direct answer prompting ("Give only the boxed final answer")
   - Chain-of-thought ablations
   - Instruction variations

2. **Token Budget Enforcement**:
   - Task-specific token caps based on empirical accuracy trade-offs
   - Model-specific optimal truncation points

3. **Knowledge Distillation**:
   - Teacher-student training with Gemini-2.0-Flash as teacher
   - LoRA fine-tuning of Qwen-2.5 models
   - Direct supervision on concise yet accurate reasoning paths

## 🔗 Key Links

<!-- - [Paper PDF (project report)](LLMThinkBench___Project_Report.pdf) -->
- [Benchmark Code Repository](https://github.com/ctrl-gaurav/LLMThinkBench)
- [Live Leaderboard](https://llmthinkbench-leaderboard.streamlit.app/)
- [LLMThinkBench PyPI Package](https://pypi.org/project/llmthinkbench/)

## 📈 Live Benchmarking

You can contribute to our leaderboard by running:

```bash
# Run a full benchmark suite and upload results
llmthinkbench --model YOUR_MODEL_NAME --tasks all --upload-results

# Run specific tasks only
llmthinkbench --model YOUR_MODEL_NAME --tasks sorting,counting,mean --upload-results
```

Results will be automatically submitted to the leaderboard if the `--upload-results` flag is used.

## 🧠 Key Findings

- Large models spend significant computational effort "thinking" on tasks that could be solved more efficiently
- Token budget caps can reduce verbosity by 40-60% with minimal accuracy loss
- Distillation from concise teacher models provides the best accuracy/token trade-off
- Prompt engineering alone yields inconsistent results across model families
- Certain tasks (e.g., complex sorting) benefit more from explicit reasoning than others

<!-- See our [paper](LLMThinkBench___Project_Report.pdf) for detailed analysis and findings. -->

## 📝 Citing

If you use this code or the benchmark in your research, please cite:

```bibtex
@misc{srivastava2025llmthinkbench,
  title   = {Are Language Models Overthinking on Simple Math Reasoning?},
  author  = {Srivastava, Gaurav and Hussain, Aafiya and Srinivasan, Sriram and Chauhan, Aninditaa},
  year    = {2025},
  url     = {https://github.com/ctrl-gaurav/LLMThinkBench}
}
```

## 👥 Authors

- **Gaurav Srivastava** 
- **Aafiya Hussain**
- **Sriram Srinivasan**
- **Aninditaa Chauhan**

For questions, please open an issue or contact us via the emails listed in the report.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.