from flask import Flask, request, jsonify
from PIL import Image
import base64
from io import BytesIO
import torch

from diffusers import DiffusionPipeline

app = Flask(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the diffusion pipeline on the selected device
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to(device)

def run_diffusion(prompt):
    # Run diffusion model
    with torch.no_grad():
        results = pipe(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
        )
    # Get the generated image
    img = results.images[0]
    return img

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Missing or empty prompt'}), 400

    # Run diffusion model to generate image
    generated_image = run_diffusion(prompt)

    # Convert image to base64
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return image URL
    response = {
        'image_url': f"data:image/png;base64,{img_str}"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)