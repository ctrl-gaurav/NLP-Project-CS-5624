from abc import ABC, abstractmethod
import logging
import os
import json
import numpy as np
from tqdm import tqdm
from vllm import SamplingParams

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using LoRA
from datasets import load_dataset
import re
import json
from tqdm import tqdm

class BaseTask(ABC):
    """Abstract base class for all evaluation tasks"""
    
    def __init__(self, model_handler, output_dir, min_val, max_val, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        """
        Initialize base task with common parameters
        
        Args:
            model_handler: Model handler for inference
            output_dir: Directory to save results
            min_val: Minimum value for number generation
            max_val: Maximum value for number generation
            num_folds: Number of evaluation folds
            num_samples: Number of samples to generate per test case
            store_details: Whether to store detailed results
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.model_handler = model_handler
        self.output_dir = output_dir
        self.min_val = min_val
        self.max_val = max_val
        self.num_folds = num_folds
        self.num_samples = num_samples
        self.store_details = store_details
        self.seed = seed
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Create task-specific directory
        self.task_dir = os.path.join(output_dir, self.task_name)
        os.makedirs(self.task_dir, exist_ok=True)
    
    @property
    @abstractmethod
    def task_name(self):
        """Return task name, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def generate_data(self, **kwargs):
        """Generate evaluation data, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def create_prompt(self, data_point):
        """Create prompt for the task, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def evaluate_response(self, response, data_point):
        """Evaluate model response, to be implemented by subclasses"""
        pass
    
    def save_detailed_results(self, results, test_case_id, fold):
        """Save detailed results for each test case and fold"""
        if not self.store_details:
            return
            
        case_dir = os.path.join(self.task_dir, f"test_case_{test_case_id}")
        os.makedirs(case_dir, exist_ok=True)
        
        filename = os.path.join(case_dir, f"detailed_results_fold_{fold}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved detailed results for test case {test_case_id} fold {fold} to {filename}")
    
    def process_fold_metrics(self, fold_results):
        """Calculate metrics for a single fold"""
        metrics = {
            'total': len(fold_results),
            'correct': sum(item['accuracy'] for item in fold_results),
            'instruction_followed': sum(item['instruction_followed'] for item in fold_results),
            'response_lengths': [item['string_len'] for item in fold_results],
            'word_counts': [item['words'] for item in fold_results],
            'output_tokens': [item['tokens'] for item in fold_results]
        }
        
        metrics['accuracy'] = round(metrics['correct'] / metrics['total'], 4)
        metrics['instruction_followed_pct'] = round(metrics['instruction_followed'] / metrics['total'], 4)
        metrics['avg_response_length'] = round(np.mean(metrics['response_lengths']), 2)
        metrics['avg_word_count'] = round(np.mean(metrics['word_counts']), 2)
        metrics['avg_output_tokens'] = round(np.mean(metrics['output_tokens']), 2)
        
        return metrics
        
    def run_fold(self, data, test_case_id, fold):
        """Run a single evaluation fold"""
        fold_results = []
        
        for data_point in tqdm(data, desc=f"Test case {test_case_id} - Fold {fold}"):
            
            # Create prompt
            prompt = self.create_prompt(data_point)

            response = self.model_handler.generate_answer(prompt)

            tokens = len(self.model_handler.tokenizer.encode(response, add_special_tokens=False))
        
            # Generate response
            # messages = [{"role": "user", "content": prompt}]
            # formatted_prompt = self.model_handler.tokenizer.apply_chat_template(
            #     messages, add_generation_prompt=True, tokenize=False
            # )
            
            # try:
            #     outputs = self.model_handler.model.generate(
            #         formatted_prompt, self.sampling_params, use_tqdm=False
            #     )
            #     response = outputs[0].outputs[0].text
            #     tokens = len(outputs[0].outputs[0].token_ids)
            # except Exception as e:
            #     logging.error(f"Generation error: {e}")
            #     response = ""
            #     tokens = 0
            
            # Evaluate response
            eval_result = self.evaluate_response(response, data_point)
            
            # Add metrics
            eval_result.update({
                "prompt": prompt,
                "model_response": response,
                "string_len": len(response),
                "words": len(response.split()),
                "tokens": tokens,
            })
            
            fold_results.append(eval_result)
        
        # Save detailed results if requested
        self.save_detailed_results(fold_results, test_case_id, fold)
        
        # Calculate fold metrics
        metrics = self.process_fold_metrics(fold_results)
        metrics['test_case_id'] = test_case_id
        metrics['fold'] = fold
        metrics['task'] = self.task_name
        
        return metrics