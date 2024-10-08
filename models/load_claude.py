import anthropic 

from PIL import Image 
import os
import json 
import tqdm
import base64


client = anthropic.Anthropic(
    api_key="YOUR_API_KEY"
)

def gen_with_model(image1_media_type, image1_data, prompt):
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": image1_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
    except:
        return "None"
    
    return message.content[0].text


type_map = {"png": "image/png", "jpg": "image/jpeg"}

def call_model(image_file,prompt):
    
    img = base64.b64encode(open(image_file, "rb").read()).decode("utf-8")
    return gen_with_model(type_map[image_file.split('.')[-1]], img, prompt)
    
    
