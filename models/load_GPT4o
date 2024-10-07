import json
import time
import base64
from mimetypes import guess_type
from openai import AzureOpenAI

# Configure logging (optional but recommended)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
API_BASE = "your_api_link"
API_KEY = "your_api_key"
DEPLOYMENT_NAME = "your_deployment_name"
API_VERSION = 'your_api_version'

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    base_url=f"{API_BASE}/openai/deployments/{DEPLOYMENT_NAME}"
)

USER_PROMPT_TEMPLATE = '{information}'

def local_image_to_data_url(image_path):
    """
    Converts a local image file to a data URL.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Data URL of the image.
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    try:
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        data_url = f"data:{mime_type};base64,{base64_encoded_data}"
        logger.debug(f"Converted image {image_path} to data URL.")
        return data_url
    except Exception as e:
        logger.error(f"Failed to convert image to data URL: {e}")
        raise

def call_model(image_path, prompt):
    """
    Calls the Azure OpenAI model with the provided image and prompt.

    Args:
        image_path (str): Path to the image file.
        prompt (str): The prompt to send to the model.

    Returns:
        str: The model's response.
    """
    try:
        # Convert the local image to a data URL
        image_data_url = local_image_to_data_url(image_path)

        # Prepare the messages payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]

        # Send the request to the Azure OpenAI API
        logger.info("Sending request to Azure OpenAI API...")
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=2000
        )

        # Parse the JSON response
        response_json = response.json()
        logger.debug(f"API response: {response_json}")

        # Extract and return the content
        content = response_json['choices'][0]['message']['content']
        logger.info("Received response from model.")
        return content

    except Exception as e:
        logger.error(f"Error calling the model: {e}")
        return None
