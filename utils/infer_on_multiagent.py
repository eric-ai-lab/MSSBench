import json, os
from tqdm import tqdm
import multiprocessing
import sys
from utils.gpt4_eval import *
from tqdm import tqdm
import time
from prompts import *


def chat_response(img, safety_output, query):
    if 'Answer: Yes' in safety_output:
        prompt = PROMPT_QA_AGENT_YES
     elif 'Answer: No' in safety_output or safety_output == "None":
        prompt = PROMPT_QA_AGENT_NO
    else:
        prompt = PROMPT_QA_AGENT_ELSE
    return  model(img, query)


def embodied_response(safe_imge, unsafe_imge, safe_instr, unsafe_instr):
    safe_vf = model(safe_image, PROMPT_IMPORTANT_POINT_AGENT + safe_instr).split("Important environment state:")[-1].strip()
    prompt_safe = PROMPT_RESPONSE_AGENT  + safe_instr + " A point to consider is: " + safe_vf
    safe_output = model(safe_imge, prompt_safe)

    unsafe_vf = model(safe_imge, PROMPT_IMPORTANT_POINT_AGENT  + unsafe_instr).split("Important environment state:")[-1].strip()
    prompt_unsafe = PROMPT_RESPONSE_AGENT  + unsafe_instr + " An point to consider is: " + unsafe_vf
    unsafe_output = model(unsafe_imge, prompt_unsafe)

    return safe_output, unsafe_output


def test_multiagent(data, model, img_root, output_path=None, set_name=None, parallel=False):
    outputs = {"chat": [], "embodied": []}
    for i, d in tqdm(enumerate(data['chat'])):
        safe_image = os.path.join(img_root, "chat", d['safe_image_path'])
        unsafe_image = os.path.join(img_root, "chat", d['unsafe_image_path'])

        for query in d['queries']:
            safe_intent = model(safe_image, PROMPT_CHAT_INTENT_AGENT + query)
            unsafe_intent = model(unsafe_image, PROMPT_CHAT_INTENT_AGENT + query)

            safe_judge = model(safe_image, PROMPT_CHAT_JUDGE_AGENT + query)
            unsafe_judge = model(unsafe_image, PROMPT_CHAT_JUDGE_AGENT + query)

            safe_output = chat_response(safe_image, safe_judge, query)
            unsafe_output = chat_response(unsafe_image, unsafe_judge, query)

            outputs['chat'].append({
                "prompt": PROMPT_CHAT_IF + query,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            json.dump(outputs, open(output_path, 'w'), indent=4)

    for i, d in tqdm(enumerate(data['embodied'])):

        for safe_instr, unsafe_instr in zip(d["safe_instructions"], d["unsafe_instructions"]):

            safe_output, unsafe_output = embodied_response(safe_imge, unsafe_imge, safe_instr, unsafe_instr)

            outputs['embodied'].append({
                "prompt": PROMPT_EMBODIED_IF + safe_instr,
                "safe_img": safe_image,
                "unsafe_img": unsafe_image,
                "safe_output": safe_output,
                "unsafe_output": unsafe_output
            })

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            json.dump(outputs, open(output_path, 'w'), indent=4)

    c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = gpt4_eval(outputs, output_path[
                                                                                                      :-5] + "_eval.json")

    return c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc