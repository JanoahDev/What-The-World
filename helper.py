import requests
import cv2
import numpy as np
import base64
import os

from dotenv import load_dotenv

load_dotenv()

## --- Diffusion Functionality --- ##

#Create the api url based on the Runpod ID
api_url = "https://"+os.getenv("RUNPOD_ID")+"-5000.proxy.runpod.net/generate"

# Function that calls the api with a prompt and returns the image
def get_image_from_api(prompt: str):
    response = requests.post(api_url, json={'prompt': prompt})
    data = response.json()
    img_str = data['image_url'].split(",")[1]
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
