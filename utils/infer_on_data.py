import json, os
from tqdm import tqdm
import multiprocessing
import sys

from utils.gpt4_eval import *

from tqdm import tqdm
import time
from utils.prompts import *
from utils.infer_on_multiagent import *

def test_each_mss(data, model, img_root, output_path=None, setting='if'):
    
    outputs = {"chat": [], "embodied": []}
    
    if setting == 'if':
        chat_prompt = PROMPT_CHAT_IF
        embodied_prompt = PROMPT_EMBODIED_IF
    elif setting == 'qc':
        chat_prompt = PROMPT_CHAT_QC
        embodied_prompt = PROMPT_EMBODIED_QC
    elif setting == 'ic':
        chat_prompt = PROMPT_CHAT_IC
        embodied_prompt = PROMPT_EMBODIED_IC
    elif setting == 'ic_cap':
        chat_prompt = PROMPT_CHAT_IC_CAP
        embodied_prompt = PROMPT_EMBODIED_IC_CAP
        
    for i, d in tqdm(enumerate(data['chat'])):
        safe_image = os.path.join(img_root, "chat", d['safe_image_path'])
        unsafe_image = os.path.join(img_root, "chat", d['unsafe_image_path'])
        
        if 'ic' in setting:
            intent = d['intent']
            safe_output = model(safe_image, chat_prompt + intent)
            if 'cap' in setting:
                unsafe_output = model(unsafe_image, chat_prompt + intent + "\nThe caption is: " + d["unsafe_image"])
            else:
                unsafe_output = model(unsafe_image, chat_prompt + intent) 
            outputs['chat'].append({
                "prompt": chat_prompt + intent,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })
        else:
            for query in d['queries']:
                safe_output = model(safe_image, chat_prompt + query)
                unsafe_output = model(unsafe_image, chat_prompt + query) 
                
                outputs['chat'].append({
                    "prompt": chat_prompt + query,
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
        
        if 'ic' in setting:
            safe_instr = d['safe_instruction']
            unsafe_instr = d['unsafe_instruction']
            if 'cap' in setting:
                safe_output = model(safe_image, embodied_prompt + safe_instr + "\nThe caption is: " + d["observation_safe"])
                unsafe_output = model(unsafe_image, embodied_prompt + unsafe_instr + "\nThe caption is: " + d["observation_unsafe"])
            else:
                # For IC tasks, we use the same instruction for both safe and unsafe images
                safe_output = model(safe_image, embodied_prompt + safe_instr)
                unsafe_output = model(unsafe_image, embodied_prompt + unsafe_instr)
            outputs['embodied'].append({
                "prompt": embodied_prompt + unsafe_instr,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })
        else:
            for safe_instr, unsafe_instr in zip(d["safe_instructions"], d["unsafe_instructions"]):
                safe_output = model(safe_image, embodied_prompt + safe_instr)
                unsafe_output = model(unsafe_image, embodied_prompt + unsafe_instr) 
                
                outputs['embodied'].append({
                    "prompt": embodied_prompt + safe_instr,
                    "safe_img": safe_image,
                    "unsafe_img": unsafe_image,
                    "safe_output": safe_output,
                    "unsafe_output": unsafe_output
                })
            
        if output_path is not None:
            json.dump(outputs, open(output_path, 'w'), indent=4)
            
    c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = gpt4_eval(outputs, output_path[:-5] + "_eval.json")
    
    return c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc