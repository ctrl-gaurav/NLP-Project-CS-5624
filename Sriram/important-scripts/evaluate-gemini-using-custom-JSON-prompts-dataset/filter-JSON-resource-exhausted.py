import json
import os

def filter_resource_exhausted_prompts(input_file, output_file):
    """
    Reads a JSON file, filters prompts with 'Error: 429 RESOURCE_EXHAUSTED' responses,
    and saves them to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the filtered prompts
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter prompts with resource exhausted errors
        resource_exhausted_prompts = []
        for item in data:
            if "response" in item and "Error: 429 RESOURCE_EXHAUSTED" in item["response"]:
                resource_exhausted_prompts.append(item["prompt"])
        
        # Save filtered prompts to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resource_exhausted_prompts, f, indent=2)
        
        print(f"Processing complete!")
        print(f"Found {len(resource_exhausted_prompts)} prompts with resource exhausted errors")
        print(f"Results saved to '{output_file}'")
        
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not valid JSON")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # Ask for input file path if not provided
    input_file = input("Enter the path to your JSON file: ")
    output_file = "gemini-1.5-flash-resource-exhausted-prompts.json"
    filter_resource_exhausted_prompts(input_file, output_file)