import argparse
import json
import os
import time
import base64
import re
from pathlib import Path
from mimetypes import guess_type
import logging
from tqdm import tqdm
from openai import AzureOpenAI
import random
api_base =  "your_api_link"
api_key = "your_api_key"
deployment_name = "your_deployment_name"
api_version = 'your_api_version'

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

USER_PROMPT = '{information}'

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_model(image_path, prompt):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": local_image_to_data_url(image_path)
                        }
                    }
                ]}
            ],
            max_tokens=2000
        )

        response = json.loads(response.json())
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "None"
