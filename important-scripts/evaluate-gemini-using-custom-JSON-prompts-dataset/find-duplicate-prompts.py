import json
import os
from collections import defaultdict

def find_duplicate_prompts(json_file_path, output_file=None):
    """
    Extract duplicate prompts from a JSON file and optionally export them to a new JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_file (str, optional): Path to save the duplicate prompts
        
    Returns:
        dict: Dictionary with prompts as keys and their counts as values (only for duplicates)
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Count occurrences of each prompt
        prompt_counter = defaultdict(int)
        for item in data:
            if 'prompt' in item:
                prompt_counter[item['prompt']] += 1
        
        # Filter only duplicates (count > 1)
        duplicates = {prompt: count for prompt, count in prompt_counter.items() if count > 1}
        
        # Print results
        if duplicates:
            print(f"Found {len(duplicates)} duplicate prompts.")
            
            # Export duplicates to a JSON file if output_file is specified
            if output_file:
                # Create a dictionary with prompts as keys and counts as values
                export_data = {}
                for prompt in duplicates:
                    export_data[prompt] = duplicates[prompt]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
                print(f"Duplicate prompts exported to {output_file}")
        else:
            print("No duplicate prompts found.")
            
        return duplicates
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Generate default output filename based on input filename
        input_base = os.path.basename(file_path)
        input_name = os.path.splitext(input_base)[0]
        default_output = f"{input_name}_duplicates.json"
        
        # Use second argument as output file if provided, otherwise use default
        output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
        
        find_duplicate_prompts(file_path, output_file)
    else:
        print("Usage: python duplicate_prompts.py <path_to_json_file> [output_file.json]")