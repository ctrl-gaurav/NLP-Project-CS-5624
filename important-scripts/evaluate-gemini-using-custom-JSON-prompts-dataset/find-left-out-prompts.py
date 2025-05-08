import json
import sys
import re

def load_json_file(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if the file contains a JSON array or multiple JSON objects
            content = f.read()
            try:
                # Try parsing as a JSON array
                return json.loads(content)
            except json.JSONDecodeError:
                # If it fails, try parsing as multiple JSON objects (one per line)
                result = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        result.append(json.loads(line))
                return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def normalize_text(text):
    """Normalize text for comparison by removing extra whitespace and standardizing line breaks."""
    if text is None:
        return ""
    # Replace all whitespace sequences (including newlines) with a single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized

def extract_prompts(json_data):
    """Extract prompts from JSON data with their original form and normalized form."""
    normalized_prompts = set()
    normalized_to_original = {}
    normalized_to_full_entry = {}
    
    for item in json_data:
        if isinstance(item, dict) and 'prompt' in item:
            original = item['prompt']
            normalized = normalize_text(original)
            normalized_prompts.add(normalized)
            normalized_to_original[normalized] = original
            normalized_to_full_entry[normalized] = item
    
    return normalized_prompts, normalized_to_original, normalized_to_full_entry

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1.json> <file2.json>")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    print(f"Loading {file1_path}...")
    file1_data = load_json_file(file1_path)
    print(f"Loading {file2_path}...")
    file2_data = load_json_file(file2_path)
    
    # Extract normalized prompts and mapping back to original
    file1_prompts_set, file1_map, file1_full_entries = extract_prompts(file1_data)
    file2_prompts_set, _, _ = extract_prompts(file2_data)
    
    # Find prompts in file1 that are not in file2
    unique_prompts_normalized = file1_prompts_set - file2_prompts_set
    
    # Get full entries for unique prompts
    unique_entries = [file1_full_entries[normalized] for normalized in unique_prompts_normalized]
    
    print(f"\nFound {len(unique_entries)} prompts in {file1_path} that are not in {file2_path}")
    
    # Save only the prompts to a JSON file
    output_file = "unique_prompts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json_output = [{"prompt": entry["prompt"]} for entry in unique_entries]
        json.dump(json_output, f, indent=2)
    
    print(f"Unique prompts saved to {output_file}")
    print(f"Total entries in file 1: {len(file1_data)}")
    print(f"Total entries in file 2: {len(file2_data)}")
    
    # Print first few examples of prompts that weren't matched (if any)
    if unique_entries:
        print("\nFirst 3 examples of prompts only in file 1:")
        for i, entry in enumerate(unique_entries[:min(3, len(unique_entries))]):
            prompt = entry["prompt"]
            print(f"Example {i+1} (first 100 chars): {prompt[:100]}...")
            if "task" in entry:
                print(f"  Task: {entry['task']}")
            if "list_size" in entry:
                print(f"  List size: {entry['list_size']}")

if __name__ == "__main__":
    main()