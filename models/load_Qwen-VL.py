from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import tqdm
import random
from PIL import Image
torch.manual_seed(1234)



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Function to generate caption with grounding
def generate_caption_with_grounding(image_path, text_prompt):

    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': text_prompt},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response