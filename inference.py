import os, sys

# load data
import json
import argparse
import importlib
from utils.infer_on_data import *
from models.load_LLaVA import *

mllm_to_module = {
    "gpt4": "load_GPT4o",
    "llava": "load_LLaVA",
    "minigpt4": "load_MiniGPT4",
    "deepseek": "load_deepseek",
    "mplug": "load_mPLUG_Owl2",
    'qwenvl': 'load_Qwen_VL',
    "gemini": "load_gemini",
    "claude": "load_claude",
}

# args
parser = argparse.ArgumentParser()
parser.add_argument("--mllm", type=str, default="gpt4", choices=mllm_to_module.keys())
parser.add_argument("--data_root", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

# Dynamic import based on mllm argument
module_name = f"models.{mllm_to_module[args.mllm]}"
model_module = importlib.import_module(module_name)
globals().update(vars(model_module))

val_data = json.load(open(os.path.join(args.data_root, "combine.json"), 'r'))

c_safe_acc, c_unsafe_acc, c_total_acc, e_safe_acc, e_unsafe_acc, e_total_acc = \
    test_each_multipanel(val_data, call_model, args.data_root, output_dir=os.path.join(args.output_dir, f"{args.mllm}_instruction_following.json"))
    
print(f"Chat Safe Acc: {c_safe_acc}, Chat Unsafe Acc: {c_unsafe_acc}, Chat Total Acc: {c_total_acc}")
print(f"Embodied Safe Acc: {e_safe_acc}, Embodied Unsafe Acc: {e_unsafe_acc}, Embodied Total Acc: {e_total_acc}")
