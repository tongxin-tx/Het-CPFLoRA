import os

import fire
import gradio as gr
import torch
import transformers
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import time

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
import torch
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.prompter import Prompter
from sklearn.manifold import TSNE  # 导入 t-SNE
import numpy as np
import torch.nn.functional as F

import json
import random

import subprocess
import time
import sys
from datetime import datetime
import os
def writeFile(s, path):
    with open(path,'a+',encoding='utf-8') as f1:
        f1.write(s+'\n')


def calculate_perplexity(text, model,tokenizer,  device='cuda'):
    """
    计算单个句子的困惑度
    """
    # 编码文本
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    # 计算困惑度: exp(loss)
    perplexity = torch.exp(loss)
    # print(perplexity.item())
    num_tokens = len(tokenizer.encode(text))

    # 归一化困惑度
    perplexity = perplexity / num_tokens

    return perplexity.item()

def format_prompt(instruction, input_text=""):
    """
    使用模板格式化提示

    Args:
        instruction: 指令文本
        input_text: 输入文本（可选）

    Returns:
        formatted_prompt: 格式化后的提示
    """
    template = {
        "description": "A shorter template to experiment with.",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }
    input_text = " "
    return template["prompt_input"].format(
        instruction=instruction,
        input=input_text
    )

def weight_lora_matrices(dict,r,w_s,w_p):
    """
    将per_dict中的LoRA矩阵拼接到share_dict对应矩阵后面

    Args:
        share_dict: 共享字典，包含B矩阵和A矩阵
        per_dict: 个人字典，包含要拼接的B矩阵和A矩阵

    Returns:
        拼接后的字典
    """

    # 遍历所有键
    for key , value in dict.items():
        if key in dict:
            # print(key)
            # 检查是B矩阵还是A矩阵
            if "lora_B" in key:
                value[:, :r]=value[:, :r].clone()#泛化
                value[:, r:] = value[:, r:].clone()#个性化
                # print(value[:, :r].shape)

            elif "lora_A" in key:
                # A矩阵拼接：在第一个维度（行）拼接
                # share_dict A矩阵形状: [r1, in_features]
                # per_dict A矩阵形状: [r2, in_features]
                # 拼接后形状: [r1 + r2, in_features]
                value[:r, :] = value[:r, :].clone()*w_s#泛化
                value[r:, :] = value[r:, :].clone()*w_p#个性化
    return dict

def load_data_from_json(json_path):
    """
    从JSON文件加载数据并随机抽取样本

    Args:
        json_path: JSON文件路径
        num_samples: 要抽取的样本数量

    Returns:
        samples: 使用模板格式化后的样本列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机抽取样本
    # selected_samples = random.sample(data, min(num_samples, len(data)))

    # 使用模板格式化每个样本的instruction
    samples = []
    for item in data:
        # 如果样本中有input字段，使用它；否则使用空字符串
        input_text = item.get('input', '')
        formatted_prompt = format_prompt(item['instruction'], input_text)
        samples.append(formatted_prompt)

    print(f"从 {json_path} 成功加载 {len(samples)} 个样本")
    return samples


def load_eval_data_from_json(json_path, num_samples=5):
    """
    从JSON文件加载数据并随机抽取样本

    Args:
        json_path: JSON文件路径
        num_samples: 要抽取的样本数量

    Returns:
        samples: 使用模板格式化后的样本列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机抽取样本
    # selected_samples = random.sample(data, min(num_samples, len(data)))
    #
    # # 使用模板格式化每个样本的instruction
    samples = []
    samples_data=[]
    for item in data:
        # 如果样本中有input字段，使用它；否则使用空字符串
        input_text = item.get('input', '')
        formatted_prompt = format_prompt(item['instruction'], input_text)
        samples.append(formatted_prompt)
        samples_data.append(item)

    print(f"从 {json_path} 成功加载 {len(samples)} 个样本")
    return samples,samples_data
def euclidean_similarity(vec1, vec2):
    """系数"""
    similarity = 1 - (abs(vec1 - vec2) / max(vec1, vec2))
    return similarity

def weight_comp(eval_sample,mean_ref_kunhuo_list,model2, tokenizer):
    sim_list = []
    sim1_list = []
    sim2_list = []
    # base_model_name = "meta-llama/Llama-3.2-1B"  # 替换为实际的LLaMA模型名称tokenizer,
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    kunhuo_sample=calculate_perplexity(eval_sample, model2, tokenizer, device)
    # sim = euclidean_similarity(mean_ref_list, mean_sample)
    for i in range(len(mean_ref_kunhuo_list)):
        sim1 = euclidean_similarity(mean_ref_kunhuo_list[i], kunhuo_sample)
        sim1_list.append(sim1)
   
    sim1_list = torch.stack(sim1_list)
    sim_mean=torch.mean(sim1_list)

    # sim_mean=0.1
    w_p=sim_mean*1
    w_s=(1-sim_mean)*1
    # print(sim_list.shape)

    # print(sim_mean,w_p,w_s)
    return w_p,w_s


def evaluate(model,
    instruction,
    tokenizer,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=80,
    stream_output=True,
    input_ids=None,
):
    prompter = Prompter("alpaca_short")
    # base_model_name = "meta-llama/Llama-3.2-1B"  # 替换为实际的LLaMA模型名称
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if input_ids is not None:
        input_ids = input_ids.to(device)
        #print(input_ids)
    else:
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    if len(generation_output.sequences) ==1:
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        ans = prompter.get_response(output)
    else:
        s = generation_output.sequences.cpu()
        output = tokenizer.batch_decode(s)
        ans = [prompter.get_response(t).split('</s>')[0] for t in output]
    return ans


def writeFile(s, path):
    with open(path, 'a+', encoding='utf-8') as f1:
        f1.write(s + '\n')


def main(
        r: int = 4,
        allr: int = 8,
        file: str = "lora-1b-8-het-data2-sw",
        local: str = "4",
        data: str='data2'
):
    # prompt_no_input = template["prompt_no_input"].format(instruction=instruction)
    # 获取加上LoRA模块后的最后一层激活值
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # file = '44'
    # r = 4
    # local = '0'
    # 加载LLaMA模型和tokenizer（需要替换为实际模型名称）
    base_model_name = "meta-llama/Llama-3.2-3B"  # 替换为实际的LLaMA模型名称
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    lora_checkpoint_path = f"/data/tongxin/FedDPA-main/{file}/8/9/new_local_output_{local}"  # LoRA模型的路径/data/tongxin/FedDPA-main/lora-1b-71-8-data2/8/9/weight_new_local_output_7
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig.from_pretrained(lora_checkpoint_path)
    # print(config)
    lora_weights_p = torch.load(lora_checkpoint_path + '/pytorch_model.bin')
    model = PeftModel(model, config)
    set_peft_model_state_dict(model, lora_weights_p)

    lora_checkpoint_path = f"/data/tongxin/FedDPA-main/{file}/8/9/local_output_{local}"  # LoRA模型的路径/data/tongxin/FedDPA-main/lora-1b-71-8-data2/8/9/weight_new_local_output_7
    model2 = AutoModelForCausalLM.from_pretrained(base_model_name)
    model2.to(device)
    model2.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model2 = prepare_model_for_kbit_training(model2)
    config2 = LoraConfig.from_pretrained(lora_checkpoint_path)
    lora_weights_p_weight = torch.load(lora_checkpoint_path + '/pytorch_model.bin')
    model2 = PeftModel(model2, config2)
    set_peft_model_state_dict(model2, lora_weights_p_weight)
    # print(config)
    # lora_checkpoint_path = "/data/tongxin/FedDPA-main/lora-1b-44-8-data2/8/9"  # LoRA模型的路径/data/tongxin/FedDPA-main/lora-1b-71-8-data2/8/9/local_output_7
    # model = prepare_model_for_kbit_training(model)
    # config = LoraConfig.from_pretrained("/data/tongxin/FedDPA-main/lora-1b-44-8-data2/8")
    # lora_weights_p = torch.load(lora_checkpoint_path+'/adapter_model.bin')

    # 计算本地训练样本的激活：
    samples = load_data_from_json(f'/data/tongxin/FedDPA-main/data/dataset2/8/local_training_{local}.json')
    # samples = random.sample(samples, 5)
    mean_ref_kun_list = []
    for i in range(len(samples)):
        # calculate_perplexity
        mean_kunhuo = calculate_perplexity(samples[i], model2,tokenizer, device)
        # mean_ = mean_.squeeze(0)
        # mean_=torch.tensor(mean_)
        # print(mean_)
        mean_ref_kun_list.append(mean_kunhuo)

    mean_ref_kun_list = torch.tensor(mean_ref_kun_list)
   
    # mean_ref_list = torch.stack(mean_ref_list)  # 形状: [5, 2048]
    print( mean_ref_kun_list.shape)
   
    eval_data, samples_data = load_eval_data_from_json(
        '/data/tongxin/FedDPA-main/data/dataset2/flan_test_200_selected_nstrict_1_change.json')
    
    save = f'Qwen/result_{file}_{local}_all_newtest-1.jsonl'
    
    count = 0
    lora_weights_o = torch.load(
        f'/data/tongxin/FedDPA-main/{file}/8/9/new_local_output_{local}/pytorch_model.bin')
    # mean_ref_=torch.mean(mean_ref_list)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    w_p_lora=allr-r
    w_s_lora=r

    for i in range(len(eval_data)):
        lora_weights_o_2 = copy.deepcopy(lora_weights_o)
        # set_peft_model_state_dict(model, lora_weights_p)

        # samples_list=copy.deepcopy(samples)
        # samples_list.append(eval_data[i])
        # w_p=activate_get_weight(samples_list,tokenizer, model2, num=5, w=w_p_lora/(w_p_lora+w_s_lora), emb_type='last')
        # w_s=1-w_p
        start_time = time.time()
        w_p, w_s = weight_comp(eval_data[i], mean_ref_kun_list,model2, tokenizer)
        w_p = w_p * (max(w_s_lora,w_p_lora) / (w_p_lora + w_s_lora))
        w_s = 1 - w_p

        print(w_p, w_s )
        # print(r)
        lora_weights_c = weight_lora_matrices(lora_weights_o_2, r, w_s, w_p)
        end_time = time.time()
        time_delay = end_time - start_time
        print(f"Time delay for one forward and backward pass: {time_delay:.4f} seconds")
        set_peft_model_state_dict(model, lora_weights_c)
        res = evaluate(model, samples_data[i]['instruction'],tokenizer)
        tmp = {}

        tmp['text'] = samples_data[i]['instruction']
        tmp['answer'] = res
        tmp['category'] = samples_data[i]['category']
        writeFile(json.dumps(tmp, ensure_ascii=False), save)


if __name__ == "__main__":
    fire.Fire(main)
