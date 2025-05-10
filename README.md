# NLP-Project-CS-5624 — LLMThinkBench Experiments

[![Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-Streamlit-orange)](https://llmthinkbench-leaderboard.streamlit.app/)
[![LLMThinkBench Package](https://img.shields.io/pypi/v/llmthinkbench)](https://pypi.org/project/llmthinkbench/)
<!-- [![GitHub](https://img.shields.io/github/stars/ctrl-gaurav/LLMThinkBench?style=social)](https://github.com/ctrl-gaurav/LLMThinkBench) -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A course project for CS 5624 (Natural Language Processing), Virginia Tech.

We measure how much large language models over-think on basic symbolic-reasoning tasks and explore three ways to keep answers short without losing accuracy:

1. **Prompt engineering** – Ask for the boxed answer only without showing work
2. **Token-budget caps** – Force every model to stay within a task-specific limit
3. **Distillation + SFT** – Fine-tune smaller Qwen-2.5 models on concise Gemini-2.0-Flash traces

All code builds on the open-sourced benchmark [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench) and feeds scores to a public leaderboard.

<div align="center">
  <img src="https://raw.githubusercontent.com/ctrl-gaurav/LLMThinkBench/main/assets/llmthinkbench_banner.png" alt="LLMThinkBench" width="600"/>
</div>

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ctrl-gaurav/NLP-Project-CS-5624.git
cd NLP-Project-CS-5624

# Install core dependencies (CUDA 12.x recommended)
conda create -n thinkbench python=3.10 -y
conda activate thinkbench
pip install -r requirements.txt          # torch, transformers, vllm, bitsandbytes ...
pip install llmthinkbench                # CLI + tasks as a PyPI package

# One-line evaluation (example: Llama-3-Instruct-8B on the SORT task)
llmthinkbench \
  --model_id meta-llama/Llama-3-8B-Instruct \
  --tasks sorting \
  --datapoints 1000 --folds 3 \
  --output_dir results/llama3_8b_sort
```

Fine-tuning scripts expect 4× A100 80 GB GPUs but can be run on smaller hardware by lowering `--batch_size` and enabling gradient accumulation.

## Repository Structure

| Path | What it Contains |
|------|------------------|
| **Prompt Engineering/** | Prompt templates and helper scripts for direct-answer vs. standard CoT prompting |
| **RegularHyperParams/** | Baseline runs with vanilla sampling settings (temperature=0.7, top_p=0.9) |
| **TruncatingBasedOnTradeoff/** | Experiments that search for the best accuracy/token-count trade-off and then re-run models under those token caps |
| **benchmarking-website-results/** | JSON output files automatically ingested by the Streamlit leaderboard |
| **dataset/** | Synthetic data used in the project (global benchmark set and Gemini distillation set) |
| **distillation + sft/** | Training configs, LoRA adapters, and logs for supervised fine-tuning with Gemini-2.0-Flash supervision |
| **important-scripts/** | Core CLI (cli.py), model loader, evaluation utilities—entry points for most experiments |
| **llmthinkbench-tasks/** | All 14 benchmark tasks (sorting, comparison, min/max, etc.) and their robust regex parsers |
| **plots.ipynb** | Jupyter notebook that reproduces all figures in the report (accuracy vs. tokens, trade-off curves, etc.) |
| **temp_and_top_p.zip** | Compressed results grid from the sampling-parameter sweep (temperature × top_p) |
| **LICENSE** | MIT License — free to use and modify |
| **Misc.** | .gitignore, .DS_Store, original README.md placeholder |

## Benchmark Tasks

LLMThinkBench evaluates models on 14 different reasoning tasks:

| Task Type | Tasks |
|-----------|-------|
| **Basic Operations** | Sorting, Comparison, Sum, Subtraction, Multiplication, Division |
| **List Processing** | Find Maximum, Find Minimum, Odd Count, Even Count |
| **Statistical** | Mean, Median, Mode |
| **Advanced** | Absolute Difference |

## Results Snapshot

| Model | Params | Avg. Accuracy ↑ | Avg. Tokens ↓ |
|-------|--------|-----------------|---------------|
| Qwen-2.5-3B (baseline) | 3B | 74% | 145 |
| Qwen-2.5-3B + token cap | 3B | 73% | 71 |
| Qwen-2.5-3B + SFT | 3B | 91% | 68 |

Full, always-up-to-date numbers are available on the [leaderboard](https://llmthinkbench-leaderboard.streamlit.app/).

## Key Experiments

### Prompt Engineering

In the `Prompt Engineering/` directory, we explore different prompt templates:
- Standard Chain-of-Thought prompting
- Direct answer prompting (no thinking shown)
- Instruction variations that affect verbosity

### Token Budget Analysis

The `TruncatingBasedOnTradeoff/` directory contains:
- Scripts to find optimal token limits per task
- Re-evaluation with enforced token limits
- Pareto curve analysis scripts

### Supervised Fine-Tuning

The `distillation + sft/` directory includes:
- SFT scripts to train Qwen-2.5 models
- Gemini-2.0-Flash generated training data
- LoRA adapters and configs

## Links

- **Benchmark code**: [github.com/ctrl-gaurav/LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench)
- **Live leaderboard**: [llmthinkbench-leaderboard.streamlit.app](https://llmthinkbench-leaderboard.streamlit.app/)
- **PyPI package**: [pypi.org/project/llmthinkbench](https://pypi.org/project/llmthinkbench/)

## Citing

If you use this code or the benchmark, please cite:

```bibtex
@misc{srivastava2025llmthinkbench,
  title   = {Are Language Models Overthinking on Simple Math Reasoning?},
  author  = {Srivastava, Gaurav and Hussain, Aafiya and Srinivasan, Sriram and Chauhan, Aninditaa},
  year    = {2025},
  url     = {https://github.com/ctrl-gaurav/LLMThinkBench}
}
```

## Authors

- **Gaurav Srivastava**
- **Aafiya Hussain** 
- **Sriram Srinivasan** 
- **Aninditaa Chauhan** 

For questions, please open an issue or contact us via the emails listed in our GitHub profiles.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.