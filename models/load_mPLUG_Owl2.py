import torch
from PIL import Image
from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
import json
import os
import tqdm
import random



# Function to load model
def load_model(model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device="cuda"
    )
    return tokenizer, model, image_processor


# Function to generate output with mPLUG-Owl
def call_model(tokenizer, model, image_processor, image_path, query):
    image = Image.open(image_path).convert('RGB')
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    image_tensor = process_images([image], image_processor).to(model.device, dtype=torch.float16)

    conv = conv_templates["mplug_owl2"].copy()
    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs



# ------------------------------------------------------- Second choice -------------------------------------------------
import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


model_path = 'MAGAer13/mplug-owl2-llama2-7b'

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")


def call_model(image_file, query):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles
    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    return tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
