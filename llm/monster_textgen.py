import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from the environment variables
MONSTER_API_TOKEN = os.getenv('MONSTER_API_KEY')

# Base URL for the API
API_ENDPOINT = "https://api.monsterapi.ai/v1/generate/llama2-7b-chat"

# Function to initiate a text generation request
def initiate_text_request(user_prompt, context_prompt=""):
    payload = {
        "system_prompt": context_prompt,
        "prompt": user_prompt,
        "beam_size": 1,
        "max_length": 1024,
        "repetition_penalty": 1.2,
        "temp": 0.98,
        "top_k": 40,
        "top_p": 0.9
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {MONSTER_API_TOKEN}"
    }

    # Make the POST request to initiate the process
    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    return response.json().get('process_id')

# Function to check the status of a request
def fetch_process_status(process_identifier):
    status_url = f"https://api.monsterapi.ai/v1/status/{process_identifier}"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {MONSTER_API_TOKEN}"
    }

    # Make the GET request to check the process status
    response = requests.get(status_url, headers=headers)
    return response.json()

# Function to generate text and wait for completion
def text_generator(user_prompt, context_prompt=""):
    process_identifier = initiate_text_request(user_prompt, context_prompt)
    status_response = fetch_process_status(process_identifier)

    # Poll the API until the process status indicates completion
    while status_response.get('status') != "COMPLETED":
        status_response = fetch_process_status(process_identifier)
        time.sleep(0.1)

    # Extract and return the generated text from the response
    generated_text = status_response.get('result', {}).get('text')
    return generated_text
