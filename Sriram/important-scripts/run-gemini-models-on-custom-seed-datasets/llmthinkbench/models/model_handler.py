import logging
from transformers import AutoTokenizer
from vllm import LLM

class ModelHandler:
    """Handler for model loading and inference"""
    
    def __init__(self, model_id, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        """
        Initialize model handler
        
        Args:
            model_id: Hugging Face model ID
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: GPU memory utilization threshold
        """
        self.model_id = model_id
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load model with vLLM
            self.model = LLM(
                model=model_id,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=8192,
                gpu_memory_utilization=gpu_memory_utilization
            )
            
            logging.info(f"Loaded model {model_id} with tensor_parallel_size={tensor_parallel_size} "
                        f"and gpu_memory_utilization={gpu_memory_utilization}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e