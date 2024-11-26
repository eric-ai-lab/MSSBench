import anthropic 

from PIL import Image 
import os
import io
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

def resize_image(image_path, max_size=5242880):
    """Resize the image to be within the max_size if it exceeds the limit."""
    # Open the image file.
    with Image.open(image_path) as img:
        # If image size is more than max_size
        if os.path.getsize(image_path) > max_size:
            # Calculate the reduction factor
            reduction_factor = (max_size / os.path.getsize(image_path)) ** 0.5
            # Calculate the new size
            new_size = (int(img.width * reduction_factor), int(img.height * reduction_factor))
            # Resize the image
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        # Save the image into a BytesIO object
        output = io.BytesIO()
        img = img.convert("RGB")
        img.save(output, format='JPEG')
    output.seek(0)
    return output


def call_model(image_file,prompt):
    img = resize_image(image_file)
    img = base64.b64encode(img.read()).decode("utf-8")
    return gen_with_model("image/jpeg", img, prompt)
    
    
