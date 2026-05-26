import copy

from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
def low_rank_approximation(A, rank):

    u, s, v = torch.svd(A)
    u = u[:, :rank]
    s = s[:rank]
    v = v.T[:rank, :]

    s=torch.diag(s)

    B = u@s#*4 #@ torch.diag(s) #/ merge_rate
    A0 =v#@v

    # B = u   # *4 #@ torch.diag(s) #/ merge_rate
    # A0 =s@v  # @v
    # B=U@S_matrix
    # A0=Vt.T
    # print(torch.diag(s))
    reconstruction = B @ A0
    error = torch.norm(A - reconstruction, p="fro").item()
    print(f"Reconstruction Error: {error:.10f}")
    return B,A0

def creat_W():
    weight_dict = {}

    # 层数和权重类型
    num_layers =28   # 层数16,28
    weight_types = ["q_proj", "v_proj"]

    # 构建字典
    for layer in range(num_layers):
        for weight_type in weight_types:
            key = f"base_model.model.model.layers.{layer}.self_attn.{weight_type}.lora_B.weight"
            weight_dict[key] = 0  # 或者 torch.randn(shape) 初始化为权重矩阵
            key = f"base_model.model.model.layers.{layer}.self_attn.{weight_type}.lora_A.weight"
            weight_dict[key] = 0  # 或者 torch.randn(shape) 初始化为权重矩阵

    return weight_dict
def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    # weights_array = normalize(
    #     torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    weights_array = normalize(
        torch.tensor([1 for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model

def FedAvg_all_rank(model, selected_clients_set, output_dir, local_dict_list_allw, epoch,rank):
    # weights_array = normalize(
    #     torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    weights_array = normalize(
        torch.tensor([1 for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    global_dict=copy.deepcopy(creat_W())

    for k, client_id in enumerate(selected_clients_set):
        # single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
        #                                  "pytorch_model.bin")
        # print(lora_state_dict[client_id])
        single_weights = local_dict_list_allw[client_id]

        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}
    # print('11111',weighted_single_weights["base_model.model.model.layers.0.self_attn.q_proj"])
    for i in range(len(weighted_single_weights)):
        layer_base = list(weighted_single_weights.keys())[i]
        B, A = low_rank_approximation(weighted_single_weights[layer_base], rank=rank)

        global_dict[f"{layer_base}.lora_A.weight"] = A
        global_dict[f"{layer_base}.lora_B.weight"] = B
        # # 获取该层的 lora_A 和 lora_B 参数
    # for k, client_id in enumerate(selected_clients_set):

    # print('2222',weighted_single_weights["base_model.model.model.layers.0.self_attn.q_proj"])
    # lora_A = global_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    # lora_B = global_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    #
    # print(lora_A)
    # print(lora_B)
    #分解后的全局低秩矩阵
    # set_peft_model_state_dict(model, global_dict, "default")


    return global_dict

def FedAvg_all_rank2(selected_clients_set, output_dir, local_dict_list_allw, r_s_list,epoch):
    # weights_array = normalize(
    #     torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    
    # weights_array = normalize(
    #     torch.tensor([1 for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    # 计算总和
    # total_sum = sum(r_s_list)

    # 对每个数字进行归一化处理，即除以总和
    weights_array = torch.tensor([x / sum(r_s_list) for x in r_s_list],dtype=torch.float32)

    global_dict = copy.deepcopy(creat_W())

    for k, client_id in enumerate(selected_clients_set):
        # single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
        #                                  "pytorch_model.bin")
        # print(lora_state_dict[client_id])
        single_weights = local_dict_list_allw[client_id]
        # device = single_weights[key].device .to(single_weights[key].device)

        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key]+ single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    return weighted_single_weights

def FedAvg_share(model, selected_clients_set, output_dir, share_lora_state_dict, epoch,rank):
    # weights_array = normalize(
    #     torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    weights_array = normalize(
        torch.tensor([1 for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    # global_dict=copy.deepcopy(creat_W())

    for k, client_id in enumerate(selected_clients_set):
        # single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
        #                                  "pytorch_model.bin")
        # print(lora_state_dict[client_id])
        single_weights = share_lora_state_dict[client_id]

        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}
    # print('11111',weighted_single_weights["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"])
    # print('11111', weighted_single_weights["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])
    return weighted_single_weights
