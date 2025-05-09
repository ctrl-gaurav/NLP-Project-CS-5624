import json
import os
from google import genai
import time

# Set up the Gemini API with your API key
# Replace this with your actual API key when running the script

API_KEY1 = "AIzaSyD5j3BKVI51gbh5C4EmmlC9vSz8FbupDpo"
API_KEY2 = "AIzaSyBOvrZRqg9HRZRniqntLqFr7WMih5DBK_A"

# Function to get response from Gemini model
def get_gemini_response(prompt, api_key):

    client = genai.Client(api_key=api_key)

    try:
        # Generate response using gemini-2.0-flash model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Return the text response
        return response.text
    except Exception as e:
        print(f"Error getting response for prompt: {e}")
        return f"Error: {str(e)}"

# Function to process a single JSON file
def process_file(file_path):
    with open(file_path, 'r') as f:
        prompts_data = json.load(f)
    
    results = []
    times = 1
    for item in prompts_data:
        prompt = item.get('prompt', '')
        
        # Get response from Gemini
        if times < 1350:
            response = get_gemini_response(prompt, API_KEY1)
        else:
            response = get_gemini_response(prompt, API_KEY2)
        
        response = response.strip()
        
        time.sleep(5)

        print(f"Processing prompt: {prompt[:50]}... and response: {response[:50]}...")  # Print first 50 chars for logging
        
        # Add to results
        results.append({
            "prompt": prompt,
            "response": response
        })
        
        times += 1

        if (times % 14 == 0):
            print(f"Processed {times} prompts so far.")
            # Add a small delay to avoid rate limiting
            time.sleep(60)
    
    return results

# Main function to process all files and save results
def main():
    # Paths to the input files
    input_files = [
        '/home/sriramsrinivasan/gemini_data/unique_prompts/unique_prompts_gemini_2.0_flash.json'
    ]
    
    # Output file
    output_file = '/home/sriramsrinivasan/gemini_data/unique_prompts/unique_responses_gemini_2.0_flash.json'
    
    # Process all files and collect results
    all_results = []
    for file_path in input_files:
        print(f"Processing file: {file_path}")
        results = process_file(file_path)
        all_results.extend(results)

        # Add a delay between file processing to avoid rate limiting
        print(f"Finished processing {file_path}. Sleeping for 6 seconds to avoid rate limiting.")
        # Sleep for 1 hour to avoid rate limiting
        time.sleep(6)
    
    # Save all results to a single JSON file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()