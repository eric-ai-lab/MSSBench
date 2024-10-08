# Multimodal Situational Safety 
[Kaiwen Zhou](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">*,1</sup>, [Chengzhi Liu](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">*,1</sup>, [Xuandong Zhao](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">,2</sup>,  [Anderson Compalas](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">,1</sup>, [Dawn Song](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">,2</sup>,  [Xin Eric Wang†](https://sheehan1230.github.io/)<sup style="color: #FFB6C1;">,1</sup>

<sup style="color: #FFB6C1;">1</sup>University of California, Santa Cruz, <sup style="color: #FFB6C1;">2</sup>University of California, Berkley
<sup style="color: #FFB6C1;">*</sup>Equal contribution

<a href='https://arxiv.org/abs/2406.08407'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://mmworld-bench.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/Xuehai/MMWorld'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>

![Teaser figure](figures/fig1.png)
Multimodal Large Language Models (MLLMs) are rapidly evolving, demonstrating impressive capabilities as multimodal assistants that interact with both humans and their environments. However, this increased sophistication introduces significant safety concerns. In this paper, we present the first evaluation and analysis of a novel safety challenge termed Multimodal Situational Safety, which explores how safety considerations vary based on the specific situation in which the user or agent is engaged. We argue that for an MLLM to respond safely—whether through language or action—it often needs to assess the safety implications of a language query within its corresponding visual context. To evaluate this capability, we develop the Multimodal Situational Safety benchmark (MSSBench) to assess the situational safety performance of current MLLMs. The dataset comprises 1,820 language query-image pairs, half of which the image context is safe, and the other half is unsafe. We also develop an evaluation framework that analyzes key safety aspects, including explicit safety reasoning, visual understanding, and, crucially, situational safety reasoning. Our findings reveal that current MLLMs struggle with this nuanced safety problem in the instruction-following setting and struggle to tackle these situational safety challenges all at once, highlighting a key area for future research. Furthermore, we develop multi-agent pipelines to coordinately solve safety challenges, which shows consistent improvement in safety over the original MLLM response.

## Dataset Structure
The [Dataset](https://huggingface.co/datasets/kzhou35/mssbench/tree/main) can be downloaded from Hugging Face.

Each entry in the Chat Task dataset contains the following fields:
- `safe_image_path`: the file path to the safe image. 
- `intent`: The user's intent in the context of images.
- `unsafe_image`: The description of  unsafe image.
- `unsafe_image_path`: the file path to the unsafe image. 
- `Type`: The multimodal situational safety category of the entry.
- `queries`: The user's question in  Chat Task.

Each entry in the Embodied Task dataset contains the following fields:
- `task`: the specific embodied task. 
- `category`: The multimodal situational safety category of the entry.
- `safe_instruction/safe_instructions`: The user's safe instructions and related variations.
- `unsafe_instruction/unsafe_instructions`: The user's unsafe instructions and related variations.
- `safe`: the file path to the safe image.
- `unsafe`: the file path to the unsafe image.

## Evaluation

