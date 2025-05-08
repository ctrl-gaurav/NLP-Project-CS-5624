# 🧠 LLMThinkBench

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.4-blue)](https://pypi.org/project/llmthinkbench/0.1.4/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-orange)](https://github.com/vllm-project/vllm)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)

## A Framework for Evaluating Reasoning Capabilities and Overthinking of Large Language Models

LLMThinkBench is a comprehensive framework designed to rigorously evaluate the reasoning capabilities of Large Language Models (LLMs). Through standardized and reproducible benchmarks, it offers valuable insights into how well models perform on various reasoning tasks, from basic arithmetic to complex logical operations.

<div align="center">
  <img src="https://raw.githubusercontent.com/ctrl-gaurav/LLMThinkBench/main/assets/llmthinkbench_banner.png" alt="LLMThinkBench" width="600"/>
</div>

## 📰 News & Releases

- **v0.1.4** (Current) - Major improvements to parsing robustness for fair evaluations. Enhanced result validation mechanisms and edge case handling.
- **v0.1.3** - Added mean, median, mode tasks and implemented GPU customization options, allowing users to specify GPU memory allocation.
- **v0.1.2** - Expanded task library with find_maximum, find_minimum, absolute_difference, and division tasks. Improved documentation.
- **v0.1.1** - Fixed several inference issues and optimized performance using vLLM for high-throughput evaluation.
- **v0.1.0** - Initial release with core functionality including sorting and comparison tasks.

## 🌟 Key Features

- **Comprehensive Evaluation**: Test LLMs on a range of mathematical and logical reasoning tasks
- **Modular Architecture**: Easily extend with custom evaluation tasks
- **Fair Comparison**: Standardized methodology for comparing models
- **Efficient Inference**: Built on vLLM for high-throughput batched evaluation
- **Detailed Metrics**: Comprehensive reports on accuracy, instruction following, and output characteristics
- **Multi-GPU Support**: Scale evaluations across multiple GPUs
- **Reproducible Results**: Consistent methodology across model comparisons
- **Output Analysis**: Identify when and how models make reasoning errors

## 📊 Supported Tasks

| Task Type | Task | Description |
|-----------|------|-------------|
| **Basic Operations** | Sorting | Evaluates ability to correctly sort lists of numbers |
| | Comparison | Tests number comparison abilities (greater than, less than, equal to) |
| | Sum | Assesses ability to calculate the sum of multiple numbers |
| | Subtraction | Measures accuracy in subtracting two numbers |
| | Multiplication | Tests multiplication of numbers |
| | Division | Evaluates division operations |
| **List Processing** | Find Maximum | Finds the largest value in a list |
| | Find Minimum | Identifies the smallest value in a list |
| | Odd Count | Counts odd numbers in a list |
| | Even Count | Counts even numbers in a list |
| **Statistical** | Mean | Calculates the arithmetic mean of a list |
| | Median | Finds the median value of a list |
| | Mode | Identifies the most frequent value(s) in a list |
| **Advanced** | Absolute Difference | Calculates the absolute difference between numbers |

## 🚀 Installation

```bash
# Install from PyPI
pip install llmthinkbench

# Install from source
git clone https://github.com/ctrl-gaurav/llmthinkbench.git
cd llmthinkbench
pip install -e .
```

## 📈 Quick Start

### Command Line Interface

```bash
# Basic usage
llmthinkbench --model_id "Qwen/Qwen2.5-1.5B-Instruct" --tasks sorting comparison

# Comprehensive evaluation
llmthinkbench --model_id "meta-llama/Llama-2-7b-chat-hf" \
  --tensor_parallel_size 2 \
  --tasks sorting comparison sum multiplication \
  --datapoints 1000 \
  --list_sizes 8 16 32 \
  --folds 3 \
  --range -1000 1000 \
  --store_details \
  --output_dir "./llama2_evaluation_results"
```

### Python API

```python
from llmthinkbench import evaluate

# Simple evaluation
results = evaluate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tasks=["sorting", "comparison", "sum"]
)

# Advanced configuration
results = evaluate(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tasks=["sorting", "comparison", "sum", "multiplication"],
    datapoints=500,
    list_sizes=[8, 16, 32],
    folds=3,
    range=[-1000, 1000],
    store_details=True,
    output_dir="./custom_results",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)
```

### Detailed API Usage

```python
from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask
from llmthinkbench.utils.reporting import generate_final_report

# Initialize model
model_handler = ModelHandler(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)

# Configure output directory
output_dir = "llama2_eval_results"

# Run sorting task
sorting = SortingTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-100,
    max_val=100,
    num_folds=3,
    num_samples=500,
    store_details=True,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Evaluate multiple list sizes
list_sizes = [8, 16, 32]
sorting_metrics = sorting.run_evaluation(list_sizes)

# Run comparison task
comparison = ComparisonTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-100,
    max_val=100,
    num_folds=3,
    num_samples=500,
    store_details=True,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

comparison_metrics = comparison.run_evaluation()

# Generate comprehensive report
all_metrics = sorting_metrics + comparison_metrics
report = generate_final_report(all_metrics, list_sizes, output_dir)
```

## 📊 Example Results

Below is an example report generated by LLMThinkBench:

```
+------------------+----------------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| Task             | Accuracy                   | Instruction Followed              | Tokens                  | Chars                        | Words                        |
+------------------+----------------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| sorting_8        | 95.20% ± 3.60              | 98.80% ± 1.20                     | 186.23 ± 32.61          | 612.57 ± 98.35               | 93.45 ± 15.62                |
| sorting_16       | 87.40% ± 4.80              | 96.70% ± 2.30                     | 312.45 ± 48.72          | 982.32 ± 156.47              | 167.85 ± 26.93               |
| sorting_32       | 68.60% ± 7.20              | 92.40% ± 3.50                     | 645.65 ± 92.18          | 1872.15 ± 283.62             | 348.76 ± 52.84               |
| comparison       | 99.20% ± 1.20              | 99.60% ± 0.50                     | 93.75 ± 16.24           | 324.83 ± 52.16               | 48.27 ± 8.14                 |
| sum_8            | 97.80% ± 2.10              | 99.30% ± 0.70                     | 134.62 ± 23.85          | 452.16 ± 78.32               | 68.92 ± 11.67                |
| multiplication   | 94.60% ± 3.50              | 98.40% ± 1.60                     | 114.27 ± 19.43          | 386.71 ± 64.28               | 58.35 ± 9.74                 |
+------------------+----------------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
```

## 📈 Visualization

You can visualize LLMThinkBench results using any plotting library:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open("final_report.json") as f:
    results = json.load(f)

# Create dataframe for plotting
data = []
for task, metrics in results.items():
    data.append({
        "Task": task,
        "Accuracy": metrics["accuracy"]["mean"] * 100,
        "Instruction Following": metrics["instruction_followed"]["mean"] * 100
    })

df = pd.DataFrame(data)

# Plot results
plt.figure(figsize=(12, 6))
df.plot(x="Task", y=["Accuracy", "Instruction Following"], kind="bar")
plt.title("LLMThinkBench Results")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results_comparison.png")
```

## ⚙️ Advanced Configuration

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_id` | Hugging Face model ID | *Required* |
| `--tasks` | Tasks to evaluate | `["sorting"]` |
| `--datapoints` | Number of samples per test case | `1000` |
| `--folds` | Number of evaluation folds | `1` |
| `--range` | Number range for evaluation | `[-100, 100]` |
| `--list_sizes` | List sizes for list-based tasks | `[8]` |
| `--store_details` | Store detailed per-example results | `False` |
| `--output_dir` | Directory to save results | Auto-generated |
| `--tensor_parallel_size` | Number of GPUs to use | `1` |
| `--gpu_memory_utilization` | GPU memory utilization threshold | `0.9` |
| `--temperature` | Sampling temperature | `0.7` |
| `--top_p` | Sampling top_p value | `0.9` |
| `--max_tokens` | Maximum tokens for sampling | `512` |

## 🧩 Extending with Custom Tasks

LLMThinkBench is designed to be easily extensible. Here's how to create a custom evaluation task:

1. Create a new task module:

```python
# llmthinkbench/tasks/custom_task.py
import random
from ..tasks.base_task import BaseTask

class CustomTask(BaseTask):
    """Implementation of a custom task"""
    
    @property
    def task_name(self):
        return "custom_task"
    
    def generate_data(self):
        """Generate random data for the task"""
        data = []
        for _ in range(self.num_samples):
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            result = a * b + a  # Example operation
            data.append({"a": a, "b": b, "expected": result})
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for the task"""
        return (f"Calculate a * b + a where a = {data_point['a']} and b = {data_point['b']}.\n\n"
                f"Provide your answer in the format \\boxed{{result}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for the task"""
        from ..utils.custom_parsing import parse_boxed_answer
        
        boxed_answer = parse_boxed_answer(response)
        instruction_followed = boxed_answer is not None
        accuracy = 0
        
        if instruction_followed and boxed_answer:
            try:
                parsed_answer = int(boxed_answer[0])
                accuracy = 1 if parsed_answer == data_point['expected'] else 0
            except:
                pass
        
        return {
            "a": data_point['a'],
            "b": data_point['b'],
            "expected": data_point['expected'],
            "parsed_answer": boxed_answer[0] if boxed_answer and len(boxed_answer) > 0 else None,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self):
        """Run evaluation for custom task"""
        all_metrics = []
        
        # Generate evaluation data
        data = self.generate_data()
        
        # Run each fold
        for fold in range(1, self.num_folds + 1):
            metrics = self.run_fold(data, "custom_task", fold)
            all_metrics.append(metrics)
        
        return all_metrics
```

2. Create a parsing utility for your task:

```python
# llmthinkbench/utils/custom_parsing.py
import re

def parse_boxed_answer(text):
    """Parse boxed answers from text.
    
    Args:
        text (str): Model response text
        
    Returns:
        list: List of answers found in \boxed{} notation
    """
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    
    if not matches:
        # Alternative formats
        matches = re.findall(r'\[([^]]*)\]', text)
        
    return matches if matches else None
```

3. Add your task to the available tasks in `__init__.py`

4. Use your custom task:

```bash
llmthinkbench --model_id "meta-llama/Llama-2-7b-chat-hf" --tasks custom_task
```

## 🔍 Contributing

Contributions to LLMThinkBench are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

LLMThinkBench is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use LLMThinkBench in your research, please cite:

```
@software{llmthinkbench2025,
  author = {Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan, Aninditaa Chauhan},
  title = {LLMThinkBench: Advanced Reasoning and Overthinking Evaluation Framework for LLMs},
  year = {2025},
  url = {https://github.com/ctrl-gaurav/LLMThinkBench/},
  version = {0.1.4}
}
```

## 📧 Contact

- **Issues**: For questions, issues, or feedback, please [open an issue](https://github.com/ctrl-gaurav/LLMThinkBench/issues) on GitHub.
- **PyPI**: [pypi.org/project/llmthinkbench](https://pypi.org/project/llmthinkbench/)
