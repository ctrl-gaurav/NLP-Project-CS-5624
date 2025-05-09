import logging
from transformers import AutoTokenizer
from vllm import LLM

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using LoRA
from datasets import load_dataset
import re
import json
from tqdm import tqdm

class ModelHandler:
    """Handler for model loading and inference"""
    
    def __init__(self, model_path, device="cuda", tensor_parallel_size=1, gpu_memory_utilization=0.9):
        """
        Initialize model handler
        
        Args:
            model_id: Hugging Face model ID
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: GPU memory utilization threshold
        """
        # self.model_id = model_id
        
        # try:
        #     # Load tokenizer
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
        #     # Load model with vLLM
        #     self.model = LLM(
        #         model=model_id,
        #         tensor_parallel_size=tensor_parallel_size,
        #         max_model_len=8192,
        #         gpu_memory_utilization=gpu_memory_utilization
        #     )
            
        #     logging.info(f"Loaded model {model_id} with tensor_parallel_size={tensor_parallel_size} "
        #                 f"and gpu_memory_utilization={gpu_memory_utilization}")
        # except Exception as e:
        #     logging.error(f"Error loading model: {e}")
        #     raise e

        """Load the fine-tuned model"""
        try:
            # First try loading as a PEFT/LoRA model
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            print("Loaded as PEFT/LoRA model")
        except Exception as e:
            print(f"Could not load as PEFT model, trying regular model: {e}")
            # If that fails, try loading as a regular model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, prompt, max_new_tokens=1024, temperature=0.7):
        """Generate an answer for the given question"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's reply (remove the prompt)
        response = response.replace(prompt, "").strip()
        
        return response