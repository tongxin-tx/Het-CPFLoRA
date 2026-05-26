import copy
import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import time
# from code.peft1 import prepare_model_for_kbit_training
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import numpy as np
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient, FedAvg_all_rank, FedAvg_all_rank2
import datasets
from utils.prompter import Prompter
# import wandb
import sys

datasets.utils.logging.set_verbosity_error()


def get_lora_BA_weights_per_layer(lora_state_dict):
    lora_BA_weights = {}

    # 遍历 lora_state_dict 获取每一层的 lora_A 和 lora_B
    for name, param in lora_state_dict.items():
        if "lora_A" in name:
            layer_name = name.split('lora_A')[0]  # 提取层名
            if layer_name not in lora_BA_weights:
                lora_BA_weights[layer_name] = {}
            lora_BA_weights[layer_name]["A"] = param
        elif "lora_B" in name:
            layer_name = name.split('lora_B')[0]  # 提取层名
            if layer_name not in lora_BA_weights:
                lora_BA_weights[layer_name] = {}
            lora_BA_weights[layer_name]["B"] = param

    # 创建 lora_BA (A 和 B 的乘积)
    for layer_name, weights in lora_BA_weights.items():
        lora_A = weights.get("A")
        lora_B = weights.get("B")

        if lora_A is not None and lora_B is not None:
            # 计算 A 和 B 的乘积
            lora_BA_weights[layer_name]["BA"] = torch.matmul(lora_A, lora_B)

    return lora_BA_weights


def creat_W():
    weight_dict = {}

    # 层数和权重类型
    num_layers = 16  # 层数16,28
    weight_types = ["q_proj", "v_proj"]

    # 构建字典
    for layer in range(num_layers):
        for weight_type in weight_types:
            key = f"base_model.model.model.layers.{layer}.self_attn.{weight_type}"
            weight_dict[key] = 0  # 或者 torch.randn(shape) 初始化为权重矩阵

    return weight_dict


def low_rank_approximation(A, rank):
    u, s, v = torch.svd(A)
    u = u[:, :rank]
    s = s[:rank]
    v = v.T[:rank, :]

    s = torch.diag(s)

    B = u @ s  # *4 #@ torch.diag(s) #/ merge_rate
    A0 = v  # @v

    # B = u   # *4 #@ torch.diag(s) #/ merge_rate
    # A0 =s@v  # @v
    # B=U@S_matrix
    # A0=Vt.T
    # print(torch.diag(s))
    reconstruction = B @ A0
    error = torch.norm(A - reconstruction, p="fro").item()
    print(f"Reconstruction Error: {error:.10f}")
    return B, A0


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = '../data',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        subsets: int = 0,
        # Local training hyperparams
        local_batch_size: int = 24,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "../data/dataset1/48/local_training_0.json",
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        local_model: bool = False,
        glocal: bool = False,
        local_weight: float = 0.5,
        # LoRA hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        # #wandb
        # wandb_project: str = "",
        # wandb_run_name: str = "",
        # wandb_watch: str = "",  # options: false | gradients | all
        # wandb_log_model: str = "",  # options: false | true
):
    print(client_selection_strategy)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"
    eval_loss_list = []
    # time.sleep(30*60)
    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,

    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     quantization_config=quantization_config,
    #     device_map=device_map,
    #     trust_remote_code=script_args.trust_remote_code,
    #     torch_dtype=torch_dtype,
    # )
    tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=False)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    # model.config.pad_token_id=tokenizer.pad_token_id

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if 'input' in data_point.keys() else None,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        # print(tokenized_full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    def per_share(r, r_p, dict):
        new_rank = r - r_p
        per_dict = {}
        share_dict = {}

        for key, value in dict.items():
            # 判断是A矩阵还是B矩阵
            if "lora_A" in key:
                r = value.shape[0]
                if r >= new_rank:
                    share_dict[key] = value[:new_rank, :].clone()
                    per_dict[key] = value[new_rank:, :].clone()
                else:
                    print("错误，个性化秩超过原始秩")
            elif "lora_B" in key:
                r = value.shape[1]
                if r >= new_rank:
                    share_dict[key] = value[:, :new_rank].clone()
                    per_dict[key] = value[:, new_rank:].clone()
                else:
                    print("错误，个性化秩超过原始秩")

        return share_dict, per_dict

    # model = prepare_model_for_int8_training(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(config)
    model = get_peft_model(model, config)

    global_dict_list = []

    r_p_list = [1, 4, 4, 8, 10, 8, 4, 12, 8, 6, 12, 8, 12, 8, 8, 12, 4, 4, 12, 12, 10, 12, 6, 2, 12, 10, 16, 10, 12, 12, 8, 18, 8, 4, 4, 16, 8, 8, 8, 2, 3, 10, 8, 8, 16, 12, 8, 4]
    r_s_list = [7, 36, 12, 32, 22, 8, 8, 32,
                12,10,36,20,16,28,24,24,4,6,
                20,16,18,24,10,6,32,30,16,18,
                16,32,20,26,8,4,8,24,16,8,
                28,6,5,30,12,24,20,12,24,8
                ]
    rank=[8, 40, 16, 40, 32, 16, 12,  44,
          20, 16, 48, 28, 28, 36, 32, 36, 8, 10,
          32, 28, 28, 36, 16, 8, 44, 40, 32, 28,
          28, 44, 28, 44, 16, 8, 12, 40, 24, 16,
          36, 8, 8, 40, 20, 32, 36, 24, 32, 12]
    for i in range(len(rank)):
        model0 = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,

        )
        tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=False)
        tokenizer.pad_token_id = (
            0
        )
        tokenizer.padding_side = "left"
        model0.config.pad_token_id = tokenizer.pad_token_id
        model0 = prepare_model_for_kbit_training(model0, use_gradient_checkpointing=True)
        model0.enable_input_require_grads()
        print('***********',rank[i])
        config = LoraConfig(
            r=rank[i],
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(config)
        model0 = get_peft_model(model0, config)
        model0.print_trainable_parameters()
        # models.append(model0)
        global_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model0)))
    del model0
    torch.cuda.empty_cache()

    # global_dict = copy.deepcopy(get_peft_model_state_dict(model, adapter_name="default"))
    local_global_dict_list = copy.deepcopy(global_dict_list)
    # lora_state_dict=[copy.deepcopy(global_dict) for i in range(num_clients)]

    share_lora_state_dict = []
    per_lora_state_dict = []
    for i in range(num_clients):
        share_global_dict_, per_global_dict_ = per_share(rank[i], r_p_list[i], local_global_dict_list[i])
        share_lora_state_dict.append(share_global_dict_)
        per_lora_state_dict.append(per_global_dict_)
    # share_lora_state_dict=[copy.deepcopy(share_global_dict) for i in range(num_clients)]
    # per_lora_state_dict=[copy.deepcopy(per_global_dict) for i in range(num_clients)]

    # del global_dict
    # del per_global_dict

    linshi_w = creat_W()
    global_dict_allw = copy.deepcopy(linshi_w)
    local_dict_list_allw = [copy.deepcopy(linshi_w) for i in range(num_clients)]
    del linshi_w
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()

    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    prv = {}
    # torch.set_printoptions(precision=10)

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        print('client_selection_strategy', client_selection_strategy)
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch, subsets=subsets)

        for client_id in selected_clients_set:

            model0 = AutoModelForCausalLM.from_pretrained(
                global_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,

            )
            tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=False)
            tokenizer.pad_token_id = (
                0
            )
            tokenizer.padding_side = "left"
            model0.config.pad_token_id = tokenizer.pad_token_id
            model0 = prepare_model_for_kbit_training(model0, use_gradient_checkpointing=True)
            model0.enable_input_require_grads()

            config = LoraConfig(
                r=rank[client_id],
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(config)
            model0 = get_peft_model(model0, config)
            model0.print_trainable_parameters()

            client = GeneralClient(client_id, model0, data_path, output_dir)

            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            print('ssssss', model0.print_trainable_parameters())
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            if epoch == 0:
                client.initiate_local_training()
            else:
                # print(share_global_dict['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'].shape)
                # print(per_lora_state_dict[client_id]['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'].shape)
                # print(share_global_dict['base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight'].shape)
                # print(per_lora_state_dict[client_id][
                #           'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight'].shape)
                client.initiate_local_training2(per_lora_state_dict[client_id], share_lora_state_dict[client_id])

            print("Local training starts ... ")
            client.train()

            # 获取本地LoRA参数
            # lora_state_dict[client_id] = get_peft_model_state_dict(client.model, adapter_name="default")

            print("\nTerminating the local training of Client_{}".format(client_id))

            model0, local_dataset_len_dict, previously_selected_clients_set, last_client_id, local_global_dict_list[
                client_id] = client.terminate_local_training_svd(
                epoch, local_dataset_len_dict, previously_selected_clients_set, config, glocal=glocal)

            share_lora_state_dict[client_id], per_lora_state_dict[client_id] = per_share(rank[client_id],
                                                                                         r_p_list[client_id],
                                                                                         local_global_dict_list[
                                                                                             client_id])

            print("全秩计算")

            for key in share_lora_state_dict[client_id].keys():
                if "lora_A.weight" in key:
                    # 找到对应的 lora_B
                    layer_name = key.replace(".lora_A.weight", "")  # 提取层的名称
                    lora_A = share_lora_state_dict[client_id][key]
                    lora_B_key = key.replace(".lora_A.weight", ".lora_B.weight")
                    # print(lora_B_key)
                    if lora_B_key in share_lora_state_dict[client_id]:
                        lora_B = share_lora_state_dict[client_id][lora_B_key]
                        # 矩阵相乘 (A @ B)
                        # print(layer_name)
                        # print(key,lora_B_key)
                        # if key in "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" and lora_B_key in "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                        #     print('12122',lora_B,lora_A)
                        #     A=torch.matmul(lora_B,lora_A)
                        #     print(A)
                        local_dict_list_allw[client_id][layer_name] = torch.matmul(lora_B, lora_A)
                        # print(local_dict_list_w[client][layer_name])
                        # print(local_dict_list_w[client][layer_name])
            print('全秩计算finish')
            # print(local_dict_list_allw[client_id]["base_model.model.model.layers.0.self_attn.q_proj"])
            del model0
            del client

        print("Collecting the weights of clients and performing aggregation")
        # model = FedAvg(model,
        #                selected_clients_set,
        #                output_dir,
        #                local_dataset_len_dict,
        #                epoch,
        #                )
        # config = model.peft_config["default"]    r_s_list=[7,36,12,32,22,40,10,8]
        global_dict_allw = FedAvg_all_rank2(
                                            selected_clients_set,
                                            output_dir,
                                            local_dict_list_allw,
                                            r_s_list,
                                            epoch)

        for client_id in selected_clients_set:
            for i in range(len(global_dict_allw)):
                layer_base = list(global_dict_allw.keys())[i]
                B, A = low_rank_approximation(global_dict_allw[layer_base], rank=r_s_list[client_id])

                share_lora_state_dict[client_id][f"{layer_base}.lora_A.weight"] = A
                share_lora_state_dict[client_id][f"{layer_base}.lora_B.weight"] = B

            model_ = AutoModelForCausalLM.from_pretrained(
                global_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,

            )
            tokenizer_ = AutoTokenizer.from_pretrained(global_model, use_fast=False)
            tokenizer_.pad_token_id = (
                0
            )
            tokenizer_.padding_side = "left"
            model_ = prepare_model_for_kbit_training(model_, use_gradient_checkpointing=True)
            model_.enable_input_require_grads()

            config_ = LoraConfig(
                r=r_s_list[client_id],
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(config_)
            model_ = get_peft_model(model_, config_)
            set_peft_model_state_dict(model_, share_lora_state_dict[client_id], "default")
            new_adapter_weight = get_peft_model_state_dict(
                model_, adapter_name='default'
            )
            torch.save(new_adapter_weight,
                       os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "share_adapter_model.bin"))
            # torch.save(share_global_dict, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
            # val_data_path=f"../data/dataset2/8/local_training_{client_id}.json"
            # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
            eval_loss = global_evaluation(model_, val_data_path, generate_and_tokenize_prompt, 1, 'cuda')
            eval_loss_list.append(eval_loss)
            print('communication round: ', epoch, ' the eval loss: ', eval_loss)
            del model_
        if client_selection_strategy == 'subset':
            for tmp_id in selected_clients_set:
                if tmp_id not in prv.keys():
                    prv[tmp_id] = 0
                prv[tmp_id] = epoch
            print(prv)
    filepath = os.path.join(output_dir, 'eval_loss_list.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(item) for item in eval_loss_list))

        # torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))

        print('communication round: ', epoch, ' the eval loss: ', eval_loss)
        # wandb.log({"eval_loss": eval_loss})
        print('jjjjj')


if __name__ == "__main__":
    fire.Fire(fl_finetune)
