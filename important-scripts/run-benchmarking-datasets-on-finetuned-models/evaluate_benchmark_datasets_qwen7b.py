import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using LoRA
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# ================= CONFIGURATION =================
MODEL_PATH = "/home/sriramsrinivasan/SFT/finetuned-Qwen2.5-7B-Instruct/final_model"  # Path to your fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
DATASETS_TO_EVALUATE = ["gsm_8k", "arc_challenge", "commonsense_qa"]
# DATASETS_TO_EVALUATE = ["commonsense_qa"]

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

def format_prompt_gsm8k(question, tokenizer):
    """Format the prompt for GSM8K"""
    prompt_template = """Can you solve this math problem?
Your final answer must be in the format \\boxed{{answer}} at the end.

Problem: {question}"""
    messages = [
        {"role": "user", "content": prompt_template.format(question=question)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_prompt_arc(question, choices, tokenizer):
    """Format the prompt for ARC challenge"""
    prompt_template = """Answer the following multiple choice question:
{question}
{choices}
Your answer should be one of the choices provided."""
    choice_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    messages = [
        {"role": "user", "content": prompt_template.format(question=question, choices=choice_str)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_prompt_csqa(question, choices, tokenizer):
    """Format the prompt for CommonSenseQA"""
    prompt_template = """Answer the following question by choosing the best option:
{question}
{choices}
Your answer should be a single letter corresponding to the correct choice."""
    choice_str = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    messages = [
        {"role": "user", "content": prompt_template.format(question=question, choices=choice_str)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_answer(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7):
    """Generate an answer for the given prompt"""
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

def extract_final_answer_gsm8k(response):
    """Extract the final numeric answer from the model's response for GSM8K"""
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

def extract_final_answer_arc(response):
    """Extract the final answer (single letter) for ARC challenge"""
    match = re.search(r"Answer:?\s*([A-E])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return response.strip()[:1].upper() if response.strip() else "Unable to Extract"

def extract_final_answer_csqa(response):
    """Extract the final answer (single letter) for CommonSenseQA"""
    match = re.search(r"Answer:?\s*([A-E])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return response.strip()[:1].upper() if response.strip() else "Unable to Extract"

def extract_ground_truth_gsm8k(answer):
    """Extract answer using #### split for GSM8K"""
    return answer.split("####")[-1].strip()

def extract_ground_truth_arc(answer):
    """Extract the correct answer for ARC challenge"""
    return answer

def extract_ground_truth_csqa(answer):
    """Extract the correct answer for CommonSenseQA"""
    return chr(ord('A') + answer)

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

def evaluate_model(model, tokenizer, dataset_name, split="test", sample_limit=None):
    """Evaluate the model on a given dataset"""
    print(f"Evaluating on {dataset_name} {split} set...")
    results = []
    correct = 0
    total = 0

    if dataset_name == "gsm_8k":
        dataset = load_dataset("gsm8k", "main", split=split)
        format_prompt_fn = format_prompt_gsm8k
        extract_final_answer_fn = extract_final_answer_gsm8k
        extract_ground_truth_fn = extract_ground_truth_gsm8k
        evaluate_numeric = True
    elif dataset_name == "arc_challenge":
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        format_prompt_fn = format_prompt_arc
        extract_final_answer_fn = extract_final_answer_arc
        extract_ground_truth_fn = extract_ground_truth_arc
        evaluate_numeric = False
    elif dataset_name == "commonsense_qa":
        dataset = load_dataset("commonsense_qa", split=split)
        format_prompt_fn = format_prompt_csqa
        extract_final_answer_fn = extract_final_answer_csqa
        extract_ground_truth_fn = extract_ground_truth_csqa
        evaluate_numeric = False
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    if sample_limit:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        if dataset_name == "gsm_8k":
            question = sample["question"]
            ground_truth_raw = sample["answer"]
            prompt = format_prompt_fn(question, tokenizer)
        elif dataset_name == "arc_challenge":
            question = sample["question"]
            choices = sample["choices"]["text"]
            ground_truth_raw = sample["answerKey"]
            prompt = format_prompt_fn(question, choices, tokenizer)
        elif dataset_name == "commonsense_qa":
            question = sample["question"]
            choices = sample["choices"]["text"]
            ground_truth_raw = sample["answerKey"]
            prompt = format_prompt_fn(question, choices, tokenizer)

            if ground_truth_raw and ground_truth_raw.isdigit():
                extracted_gt = extract_ground_truth_fn(int(ground_truth_raw))
            else:
                extracted_gt = ""
        else:
            ground_truth_raw = ""
            extracted_gt = ""

        # Generate response using fine-tuned model
        response = generate_answer(model, tokenizer, prompt,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=TEMPERATURE)

        # Extract the final answer
        model_answer = extract_final_answer_fn(response)

        match = False
        if evaluate_numeric and dataset_name == "gsm_8k":
            clean_gt = clean_numeric_string(ground_truth_raw)
            clean_answer = clean_numeric_string(model_answer)
            if clean_gt is not None and clean_answer is not None:
                if abs(clean_gt - clean_answer) < 1e-9:
                    match = True
        elif dataset_name == "commonsense_qa":
            match = model_answer.strip().upper() == extracted_gt.strip().upper()
        else:
            match = model_answer.strip().upper() == extract_ground_truth_fn(ground_truth_raw).strip().upper()

        if match:
            correct += 1
        total += 1

        # Store results
        result_entry = {
            "ground_truth": ground_truth_raw,
            "model_response": response,
            "model_answer": model_answer,
            "match": match
        }
        if dataset_name == "gsm_8k":
            result_entry["question"] = question
            result_entry["extracted_gt"] = ground_truth_raw
        elif dataset_name == "arc_challenge":
            result_entry["question"] = question
            result_entry["choices"] = choices
        elif dataset_name == "commonsense_qa":
            result_entry["question"] = question
            result_entry["choices"] = choices
            result_entry["extracted_gt"] = extracted_gt

        results.append(result_entry)

        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} samples | Current Accuracy: {correct / total:.2%}")

    # Save results
    output_file = f"finetuned_model_results_{dataset_name}_{split}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy on {dataset_name} {split}: {accuracy:.2%} ({correct}/{total})")

    output_file1 = f"finetuned_model_accuracy_{dataset_name}_{split}.json"
    with open(output_file1, "w") as f:
        json.dump(accuracy, f, indent=2)

    return accuracy, results

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model(MODEL_PATH, device=DEVICE)

    # You can choose to evaluate on test or train split
    split = "test"  # or "train" for gsm8k, "dev" or "test" for others
    sample_limit = None  # Set to a number like 100 for quick testing

    for dataset_name in DATASETS_TO_EVALUATE:
        accuracy, results = evaluate_model(model, tokenizer, dataset_name, split=split, sample_limit=sample_limit)
        print(f"Evaluation for {dataset_name} complete! Results saved to finetuned_model_results_{dataset_name}_{split}.json")
        print(f"Final accuracy on {dataset_name}: {accuracy:.2%}")