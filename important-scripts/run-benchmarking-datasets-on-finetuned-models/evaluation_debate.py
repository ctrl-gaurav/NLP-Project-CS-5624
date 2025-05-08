import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using LoRA
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# ================= CONFIGURATION =================
MODEL_PATH = "./finetuned-qwen-debate/final_model"  # Path to your fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
# =================================================

def load_model(model_path, device="cuda"):
    """Load the fine-tuned model"""
    try:
        # First try loading as a PEFT/LoRA model
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else "cpu",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        print("Loaded as PEFT/LoRA model")
    except Exception as e:
        print(f"Could not load as PEFT model, trying regular model: {e}")
        # If that fails, try loading as a regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else "cpu",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_prompt(question, tokenizer):
    """Format the prompt for your fine-tuned model"""

    prompt_template = """Can you solve this math problem?
Your final answer must be in the format \\boxed{{answer}} at the end.

Problem: {question}"""

    messages = [
        {"role": "user", "content": prompt_template.format(question=question)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_answer(model, tokenizer, question, max_new_tokens=1024, temperature=0.7):
    """Generate an answer for the given question"""
    prompt = format_prompt(question, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's reply (remove the prompt)
    response = response.replace(prompt, "").strip()
    
    return response

def extract_final_answer(response):
    """Extract the final numeric answer from the model's response"""
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"\*\*Final Answer\*\*:.*?([$\£€]?[\d,]+(?:\.\d+)?(?:[ million| thousand|%])?)",
        r"Therefore,?\s+(?:the|our|we get|we have)?(?:\s+(?:answer|result|solution|value))?\s+(?:is|equals|=)?\s*([$\£€]?[\d,]+(?:\.\d+)?(?:[ million| thousand|%])?)",
        r"(?:The|Our|We get) (?:answer|result|solution|value) (?:is|equals|=) ?([$\£€]?[\d,]+(?:\.\d+)?(?:[ million| thousand|%])?)",
        r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)",
        r"(\d+\.?\d*)"
    ]
    
    for pattern in patterns:
        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            try:
                raw_answer = match[-1].strip()
                # Clean and standardize the answer
                cleaned = raw_answer.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
                
                # Handle LaTeX formatting
                cleaned = re.sub(r'\\text\{[^}]*\}', '', cleaned).strip()
                
                if "%" in raw_answer:
                    cleaned = cleaned.replace("%", "")
                
                # Extract numeric value before "million" or "thousand"
                if "million" in raw_answer.lower():
                    num_match = re.search(r'(\d+\.?\d*)', cleaned)
                    if num_match:
                        cleaned = num_match.group(1)
                        cleaned = str(int(float(cleaned) * 1_000_000))
                elif "thousand" in raw_answer.lower():
                    num_match = re.search(r'(\d+\.?\d*)', cleaned)
                    if num_match:
                        cleaned = num_match.group(1)
                        cleaned = str(int(float(cleaned) * 1_000))
                
                return cleaned.split(".")[0]  # Return integer part
            except Exception:
                # If any error occurs during processing, continue to next pattern
                continue
                
    return "Unable to Extract"

def extract_ground_truth(answer):
    """Extract answer using #### split"""
    return answer.split("####")[-1].strip()

def clean_numeric_string(s):
    """Extract and normalize numeric values from strings"""
    if not isinstance(s, str):
        return None
    
    matches = re.findall(r"[-+]?\d*\.?\d+|\d+", s)
    if not matches:
        return None
    
    number_str = matches[-1].replace(",", "").strip()
    
    try:
        if "." in number_str:
            return round(float(number_str), 2)
        return int(number_str)
    except:
        return None

def evaluate_model_on_gsm8k(model, tokenizer, split="test", sample_limit=None):
    """Evaluate the model on GSM8K dataset"""
    # Load dataset
    gsm8k = load_dataset("gsm8k", "main", split=split)
    
    if sample_limit:
        gsm8k = gsm8k.select(range(min(sample_limit, len(gsm8k))))
    
    results = []
    correct = 0
    total = 0
    
    for idx, sample in tqdm(enumerate(gsm8k), total=len(gsm8k)):
        question = sample["question"]
        ground_truth = sample["answer"]
        extracted_gt = extract_ground_truth(ground_truth)
        
        # Generate response using fine-tuned model
        response = generate_answer(model, tokenizer, question, 
                                  max_new_tokens=MAX_NEW_TOKENS, 
                                  temperature=TEMPERATURE)
        
        # Extract the final answer
        model_answer = extract_final_answer(response)
        
        # Clean answers for comparison
        clean_gt = clean_numeric_string(extracted_gt)
        clean_answer = clean_numeric_string(model_answer)
        
        # Calculate match
        match = False
        if clean_gt is not None and clean_answer is not None:
            if abs(clean_gt - clean_answer) < 1e-9:
                match = True
        
        if match:
            correct += 1
        total += 1
        
        # Store results
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "extracted_gt": extracted_gt,
            "model_response": response,
            "model_answer": model_answer,
            "match": match
        })
        
        # Print progress
        if (idx+1) % 10 == 0:
            print(f"Processed {idx+1} samples | Current Accuracy: {correct/total:.2%}")
    
    # Save results
    output_file = f"finetuned_model_results_gsm8k_{split}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Accuracy: {correct/total:.2%} ({correct}/{total})")
    return correct/total, results

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model(MODEL_PATH, device=DEVICE)
    
    # You can choose to evaluate on test or train split
    # Optionally limit the number of samples for quicker testing
    split = "test"  # or "train"
    sample_limit = None  # Set to a number like 100 for quick testing
    
    print(f"Evaluating on GSM8K {split} set...")
    accuracy, results = evaluate_model_on_gsm8k(model, tokenizer, split=split, sample_limit=sample_limit)
    
    print(f"Evaluation complete! Results saved to finetuned_model_results_gsm8k_{split}.json")
    print(f"Final accuracy: {accuracy:.2%}")