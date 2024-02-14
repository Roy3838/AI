

"""
Code designed to call OpenAI API with GPT-4, using an image prompt and a text prompt to generate a response.

"""

# import key from non-tracked file
from key import key
from openai import OpenAI
import os
import wandb
import matplotlib.image as img
import requests
import base64
from PIL import Image
import io

key = key()

# Set OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = key

# Define a function to encode an image to a base64 string
def encode_image(image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        
        # Resize the image to 256x256
        image = image.resize((256, 256))
        
        # Save the resized image to a bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")  # You can change the format if needed
        
        # Get the byte data from the buffer
        byte_data = buffer.getvalue()
        
        # Encode the byte data to a base64 string
        string = base64.b64encode(byte_data).decode('utf-8')
        
        return string


# Path to your image
image_path = "EconomCirc/agave.jpg"

# Text prompt
text_prompt = "Utiliza la imagen y descripción adjuntas para informar a los patrocinadores sobre el estado mensual de su planta de agave en peligro de extinción. Describe el crecimiento y cambios notables de la planta. Finaliza con un agradecimiento sincero por su continuo apoyo en este boletín mensual.Genera un mensaje amable para actualizar a los patrocinadores sobre el estado de su planta de agave en peligro de extinción. Incluye agradecimiento por su apoyo."

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": text_prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
