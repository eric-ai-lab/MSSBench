import json, os
from tqdm import tqdm
import multiprocessing
import sys

from utils.gpt4_eval import *

from tqdm import tqdm
import time
from utils.prompts import *
from utils.infer_on_multiagent import *

def test_each_mss(data, model, img_root, output_path=None, setting='if', caption=False):
    
    outputs = {"chat": [], "embodied": []}
    
    for i, d in tqdm(enumerate(data['chat'])):
        safe_image = os.path.join(img_root, "chat", d['safe_image_path'])
        unsafe_image = os.path.join(img_root, "chat", d['unsafe_image_path'])
        
        for query in d['queries']:
            safe_output = model(safe_image, PROMPT_CHAT_IF + query)
            unsafe_output = model(unsafe_image, PROMPT_CHAT_IF + query) 
            
            outputs['chat'].append({
                "prompt": PROMPT_CHAT_IF + query,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })
            
        if output_path is not None:
            json.dump(outputs, open(output_path, 'w'), indent=4)

    for i, d in tqdm(enumerate(data['embodied'])):
        safe_image = os.path.join(img_root, "embodied", d['safe'])
        unsafe_image = os.path.join(img_root, "embodied", d['unsafe'])
        for safe_instr, unsafe_instr in zip(d["safe_instructions"], d["unsafe_instructions"]):
            safe_output = model(safe_image, PROMPT_EMBODIED_IF + safe_instr)
            unsafe_output = model(unsafe_image, PROMPT_EMBODIED_IF + unsafe_instr) 
            
            outputs['embodied'].append({
                "prompt": PROMPT_EMBODIED_IF + safe_instr,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })
            
        if output_path is not None:
            json.dump(outputs, open(output_path, 'w'), indent=4)
            
    c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = gpt4_eval(outputs, output_path[:-5] + "_eval.json")
    
    return c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc