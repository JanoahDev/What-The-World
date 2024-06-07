import requests
import cv2
import numpy as np
import base64

api_url = 'https://qbwvbikcp6d37t-5000.proxy.runpod.net/generate'

def get_image_from_api(prompt):
    response = requests.post(api_url, json={'prompt': prompt})
    data = response.json()
    img_str = data['image_url'].split(",")[1]
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

prompt = "A dog wearing a robe in a kings chair"
image = get_image_from_api(prompt)

# Display the generated image
cv2.imshow("Generated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()