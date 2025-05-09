import time
from google import genai

# Function to get response from Gemini model
def get_gemini_response(prompt):

    client = genai.Client(api_key="AIzaSyD5j3BKVI51gbh5C4EmmlC9vSz8FbupDpo")

    try:
        # Generate response using gemini-2.0-flash model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Return the response
        return response
    except Exception as e:
        print(f"Error getting response for prompt: {e}")
        return f"Error: {str(e)}"

def main():
    # Generate response
    prompt = "What is the capital of France?"
    return_value = get_gemini_response(prompt)

    response = return_value.text
    tokens = int(return_value.usage_metadata.candidates_token_count)
    time.sleep(5)

    print("Response: ", response)
    print("Tokens: ", tokens)

main()