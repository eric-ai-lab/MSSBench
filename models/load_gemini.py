# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
from PIL import Image 

GOOGLE_API_KEY="YOUR_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-pro")


def gen_with_model(img, prompt):
    try:
        response = model.generate_content([img, prompt])
        
        return response.text
    except:
        return "None"

def call_model(img, prompt):
    
    return gen_with_model(Image.open(img), prompt) 

