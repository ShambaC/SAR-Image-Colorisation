# Load environment variables
from dotenv import load_dotenv
env_res = load_dotenv()

if not env_res :
    raise Exception("No .env file provided")

import os
import random
import requests
import json

def get_country(lat: float, lon: float) -> str :
    api_url = f'https://api.api-ninjas.com/v1/reversegeocoding?lat={lat}&lon={lon}'

    API_KEY_LIST = ["API_NINJA_TOKEN_1", "API_NINJA_TOKEN_2"]

    response = requests.get(api_url, headers={'X-Api-Key': os.environ[random.choice(API_KEY_LIST)]})
    if response.status_code == requests.codes.ok:
        try:
            if response.text == "[]" :
                return "error"
            
            x = json.loads(response.text[1:-1])
            return x["country"]
        except :
            print(f"Response: {response.text}")
            raise Exception("API Ninja error occured")

        
    else:
        print("Error:", response.status_code, response.text)
        return "error"

if __name__ == "__main__" :
    get_country(76.16237104394712, -109.54512274485435)