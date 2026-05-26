from typing import Dict, List
from t5.evaluation import metrics
import tqdm
import json
import os
import random
import wordninja
import numpy as np
# 设置 Python 随机种子
random.seed(42)

def read_list(file,k):
    dic={}
    lines=open(file).readlines()
    for line in lines:
        line = line.strip()
        data = json.loads(line)
        if data['category'] not in dic.keys():
            dic[data['category']] = []
        tmpd=data[k]
        if data[k].endswith('</s>'):
            tmpd = data[k].split('</s>')[0]
        #if data['category'] in ['paraphrase','question_classification']:
           # tmpd = tmpd.split(' ')[0].strip(',')
        dic[data['category']].append(tmpd)
    return dic

# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source: https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
  """Computes ROUGE by taking the max ROUGE-N per reference and N."""
  # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
  # Identify best reference per response and ROUGE type.
  rouge_types = ["rouge1", "rouge2", "rougeLsum"]
  max_references = {rouge_type: [] for rouge_type in rouge_types}
  for targ_for_resp, resp in tqdm.tqdm(zip(targets, predictions), total=len(targets)):
    # Compute individual scores per example/ref pair.
    resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
    # Find best scoring references for generated output and ROUGE type.
    for rouge_type in rouge_types:
      best_score_index = max(range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type])
      best_ref = targ_for_resp[best_score_index]
      # Add the reference to the new reference list.
      max_references[rouge_type].append(best_ref)
  # Compute metric for each of the reference lists for a ref type.
  results = {}
  for rouge_type in rouge_types:
    results[rouge_type] = metrics.rouge(max_references[rouge_type], predictions)[rouge_type]
  return results

def rouge(targets, predictions):
    results = metrics.rouge(targets, predictions)
    return results
def restore_spaces(text):
    # 使用wordninja进行拆分并恢复空格
    return " ".join(wordninja.split(text))
# def process_text_list(text_list):
#     """
#     对文本列表中的每个项进行空格恢复。
#     :param text_list: 包含没有空格的文本列表
#     :return: 恢复空格后的文本列表
#     """
#     return [restore_spaces(text) for text in text_list]
def get_result(targets, predictions, save):
    results = {}
    total_target=[]
    total_pre=[]
    total_result=[]
    for k in targets.keys():
        predictions[k] = [text.replace('<|end_of_text|>', ' ') for text in predictions[k]]
        predictions[k] = [text.replace('</s>', ' ') for text in predictions[k]]
        # print(targets[k][0])
        # print('******************************')</s>
        # predictions[k] = [text.split('\n', 1)[0] for text in predictions[k]]
        # predictions[k] = process_text_list(predictions[k])
        # print(predictions[k][0])

        result = rouge(targets[k], predictions[k])
        total_result.append(result['rouge1'])
        # print(result['rouge1'])
        results[k] = result
        total_target.extend(targets[k])
        total_pre.extend(predictions[k])
    # print(total_target)
    # total_pre = [text.replace('<|end_of_text|>', '') for text in total_pre]
    # total_pre = [text.split('\n', 1)[0] for text in total_pre]
    #
    # print(total_pre)
    # results['total'] = rouge(total_target, total_pre)
    results['total'] ={'rouge1':np.sum(total_result)/len(total_result)}

    rouge1_list = [(task, scores["rouge1"]) for task, scores in results.items()]

    print(rouge1_list)
    with open(save, 'w') as f:
        f.write(json.dumps(results))

#result_old_8_0
ps='62'
# path_list=['our/result_old_glocal','our/result_71_svd_glocal','our/result_62_svd_glocal','weight/result_old_0_per','weight/result_old_1_per','weight/result_old_2_per','weight/result_old_3_per','weight/result_old_4_per','weight/result_old_5_per','weight/result_old_6_per','weight/result_old_7_per','weight/result_44_8_0_all','weight/result_44_8_0_all_per','weight/result_44_8_1_all','weight/result_44_8_1_all_per','weight/result_44_8_2_all','weight/result_44_8_2_all_per','weight/result_44_8_3_all','weight/result_44_8_3_all_per','weight/result_44_8_4_all','weight/result_44_8_4_all_per','weight/result_44_8_5_all','weight/result_44_8_5_all_per','weight/result_44_8_6_all','weight/result_44_8_6_all_per','weight/result_44_8_7_all','weight/result_44_8_7_all_per','weight/result_44_8_all_share']
# path='our/result_71_svd_glocal'

path_='Qwen/'
# path_list=['result_old_0_per','result_old_1_per','result_old_2_per','result_old_3_per','result_old_4_per','result_old_5_per','result_old_6_per','result_old_7_per','result_old_share','result_44_8_0_all_per','result_44_8_1_all_per','result_44_8_2_all_per','result_44_8_3_all_per','result_44_8_4_all_per','result_44_8_5_all_per','result_44_8_6_all_per','result_44_8_7_all_per','result_44_8_0_all','result_44_8_1_all','result_44_8_2_all','result_44_8_3_all','result_44_8_4_all','result_44_8_5_all','result_44_8_6_all','result_44_8_7_all','result_44_8_all_share']
path_list=['result_1b_8_44_data2_0_all','result_1b_8_44_data2_1_all','result_1b_8_44_data2_2_all','result_1b_8_44_data2_3_all','result_1b_8_44_data2_4_all','result_1b_8_44_data2_5_all','result_1b_8_44_data2_6_all','result_1b_8_44_data2_7_all']

path_list=['lora-1b-71-8-data2_0_all','lora-1b-71-8-data2_1_all','lora-1b-71-8-data2_2_all','lora-1b-71-8-data2_3_all','lora-1b-71-8-data2_4_all','lora-1b-71-8-data2_5_all','lora-1b-71-8-data2_6_all','lora-1b-71-8-data2_7_all']
path_list=['result_lora-1b-8-our-data2_0_all_avg','result_lora-1b-8-our-data2_1_all_avg','result_lora-1b-8-our-data2_2_all_avg','result_lora-1b-8-our-data2_3_all_avg','result_lora-1b-8-our-data2_4_all_avg','result_lora-1b-8-our-data2_5_all_avg','result_lora-1b-8-our-data2_6_all_avg','result_lora-1b-8-our-data2_7_all_avg']
path_list=['lora-1b-71-8-data1_0_all','lora-1b-71-8-data1_1_all','lora-1b-71-8-data1_2_all','lora-1b-71-8-data1_3_all','lora-1b-71-8-data1_4_all','lora-1b-71-8-data1_5_all','lora-1b-71-8-data1_6_all','lora-1b-71-8-data1_7_all']

           # 'result_71_8_6_all_per','result_71_8_6_all','result_62_8_6_all_per','result_62_8_6_all','result_44_8_6_all_per','result_44_8_6_all','result_old_6_per','result_old_r=4_6_per']
#             'result_71_8_5_all_per','result_71_8_5_all','result_62_8_5_all_per','result_62_8_5_all','result_44_8_5_all_per','result_44_8_5_all','result_old_5_per','result_old_r=4_5_per',
#            'result_71_8_4_all_per','result_71_8_4_all','result_62_8_4_all_per','result_62_8_4_all','result_44_8_4_all_per','result_44_8_4_all','result_old_4_per','result_old_r=4_4_per',
# #
# #             'result_71_8_3_all_per','result_71_8_3_all','result_62_8_3_all_per','result_62_8_3_all','result_44_8_3_all_per','result_44_8_3_all','result_old_3_per','result_old_r=4_3_per',
# #            'result_71_8_2_all_per','result_71_8_2_all','result_62_8_2_all_per','result_62_8_2_all','result_44_8_2_all_per','result_44_8_2_all','result_old_2_per','result_old_r=4_2_per',
# #             'result_71_8_1_all_per','result_71_8_1_all','result_62_8_1_all_per','result_62_8_1_all','result_44_8_1_all_per','result_44_8_1_all','result_old_1_per','result_old_r=4_1_per',
# #            'result_71_8_0_all_per','result_71_8_0_all','result_62_8_0_all_per','result_62_8_0_all','result_44_8_0_all_per','result_44_8_0_all','result_old_0_per','result_old_r=4_0_per'
# #             ,'result_71_8_all_share','result_62_8_all_share' ,'result_44_8_all_share','result_old_share','result_old_r=4_share']
# # path_list=['result_align_44_0_all','result_align_44_1_all','result_align_44_2_all','result_align_44_3_all','result_align_44_4_all','result_align_44_5_all','result_align_44_6_all','result_align_44_7_all']
# path_list=['result_62_8_0_all','result_62_8_1_all','result_62_8_2_all','result_62_8_3_all','result_62_8_4_all','result_62_8_5_all','result_62_8_6_all','result_62_8_7_all','result_62_8_all_share']
# path_list=['result_old_16_2_avg__6_all','result_old_16_2_avg__7_all','result_old_16_2_avg__4_all','result_old_16_2_avg__5_all','result_old_16_2_avg__3_all','result_old_16_2_avg__1_all','result_old_16_2_avg__0_all','result_old_16_avg_7_all']
# path_list=['result_old_16_avg_6_all','result_old_16_avg_7_all','result_old_16_avg__4_all','result_old_16_avg__5_all','result_old_16_avg__3_all','result_old_16_avg__1_all','result_old_16_avg__0_all','result_old_16_avg__2_all']

# path_list=['result_16_2_het_avg__0_all','result_16_2_het_avg__1_all','result_16_2_het_avg__2_all','result_16_2_het_avg__3_all','result_16_2_het_avg__4_all','result_16_2_het_avg__5_all','result_16_2_het_avg__6_all','result_16_2_het_avg__7_all']
# path_list=['result_our_16_het_avg__0_all','result_our_16_het_avg__1_all','result_our_16_het_avg__2_all','result_our_16_het_avg__3_all','result_our_16_het_avg__4_all','result_our_16_het_avg__5_all','result_our_16_het_avg_6_all','result_our_16_het_avg_7_all']
# result_old_1_all_avg
# path_list=['result_8_old_avg__0_all','result_8_old_avg__1_all','result_8_old_avg__2_all','result_8_old_avg__3_all','result_8_old_avg__4_all','result_8_old_avg__5_all','result_8_old_avg__6_all','result_8_old_avg__7_all']#
# path_list=['result_8_het_avg__0_all','result_8_het_avg__1_all','result_8_het_avg__2_all','result_8_het_avg__3_all','result_8_het_avg__4_all','result_8_het_avg__5_all','result_8_het_avg__6_all','result_8_het_avg__7_all']
# path_list=['result_lora-1b-8-our-data2_0_all_autotest','result_lora-1b-8-our-data2_1_all_autotest','result_lora-1b-8-our-data2_2_all_autotest','result_lora-1b-8-our-data2_3_all_autotest','result_lora-1b-8-our-data2_4_all_autotest','result_lora-1b-8-our-data2_5_all_autotest']
# path_list=['result_lora-1b-8-our-data1_0_all_autotest','result_lora-1b-8-our-data1_1_all_autotest','result_lora-1b-8-our-data1_2_all_autotest_','result_lora-1b-8-our-data1_3_all_autotest_','result_lora-1b-8-our-data1_4_all_autotest_','result_lora-1b-8-our-data1_5_all_autotest_','result_lora-1b-8-our-data1_6_all_autotest','result_lora-1b-8-our-data1_7_all_autotest_']

# path_list=['result_8_swdobhet_avg__0_all','result_8_swdobhet_avg__1_all','result_8_swdobhet_avg__2_all','result_8_swdobhet_avg__3_all','result_8_swdobhet_avg__4_all','result_8_swdobhet_avg__5_all','result_8_swdobhet_avg__6_all','result_8_swdobhet_avg__7_all']


# path_list=['result_8_douhet_avg__0_all','result_8_douhet_avg__1_all','result_8_douhet_avg__2_all','result_8_douhet_avg__3_all','result_8_douhet_avg__4_all','result_8_douhet_avg__5_all','result_8_douhet_avg__6_all','result_8_douhet_avg__7_all']
# path_list=['result_lora-1b-8-het-data2-sw_2_all_newtest-33']

#'result_lora-1b-8-het-data1-sw_2_all_newtest-1','result_lora-1b-8-het-data1-sw_3_all_newtest-1','result_lora-1b-8-het-data1-sw_4_all_newtest-1','result_lora-1b-8-het-data1-sw_5_all_newtest-1','result_lora-1b-8-het-data1-sw_6_all_newtest-1',

# result_44_8_3_all 'result_44_8_4_all_local',result_lora-1b-8-our-data1_6_all_autotest——
# 'result_44_7_all_autoweight',
# # path_list=['result_44_0.5_8_0_all_rate','result_44_8_0_all_rate','result_44_8_0_all','result_44_0.5_8_1_all_rate','result_44_8_1_all_rate','result_44_8_1_all','result_44_0.5_8_2_all_rate','result_44_8_2_all_rate','result_44_8_2_all','result_44_0.5_8_3_all_rate','result_44_8_3_all_rate','result_44_8_3_all','result_44_0.5_8_4_all_rate','result_44_8_4_all_rate','result_44_8_4_all','result_44_0.5_8_5_all_rate','result_44_8_5_all_rate','result_44_8_5_all','result_44_0.5_8_6_all_rate','result_44_8_6_all_rate','result_44_8_6_all','result_44_0.5_8_7_all_rate','result_44_8_7_all_rate','result_44_8_7_all','result_old_share','old0','result_old_0_all_avg','result_old_0_per']
# # path_list=['result_44_0:2_8_0_all_rate','result_44_8_0_all_local','result_44_8_1_all_local','result_44_8_2_all_local','result_44_8_3_all_local','result_44_8_4_all_local','result_44_8_5_all_local','result_44_8_6_all_local','result_44_8_7_all_local']
# path_list=['result_old_7_all_autoweight','result_62_7_all_autotest1','result_62_7_all_autotest2','result_62_7_all_autotest3','result_71_7_all_autotest1','result_71_7_all_autotest2','result_71_7_all_autotest3','result_71_7_all_autotest4','result_71_7_all_autotest5','result_71_7_all_autotest6','result_44_7_all_autotest2','result_44_7_all_autotest3','result_44_7_all_autotest4','result_44_7_all_autotest5','result_44_7_all_autotest6','result_44_7_all_autotest7','result_44_7_all_autotest8','result_44_7_all_autotest9','result_44_7_all_autotest10']
# path_list=['result_old_0_all_autoweight','result_old_1_all_autoweight','result_old_2_all_autoweight','result_old_3_all_autoweight','result_old_4_all_autoweight','result_old_5_all_autoweight','result_old_6_all_autoweight','result_old_7_all_autoweight']
# client='6'
# path_list=[f'result_old_7_all_autoweight',f'result_old_6_all_autoweight',f'result_old_5_all_autoweight',f'result_old_4_all_autoweight',f'result_old_3_all_autoweight',f'result_old_2_all_autoweight',f'result_old_1_all_autoweight',f'result_old_0_all_autoweight']

# path_list=['result_lora-1b-44-8-data1_0_all_newtest-2','result_lora-1b-44-8-data1_1_all_newtest-2','result_lora-1b-44-8-data1_2_all_newtest-2','result_lora-1b-44-8-data1_3_all_newtest-2','result_lora-1b-44-8-data1_4_all_newtest-2','result_lora-1b-44-8-data1_5_all_newtest-2','result_lora-1b-44-8-data1_6_all_newtest-2','result_lora-1b-44-8-data1_7_all_newtest-2']
# path_list=['lora-3b-8-het-data2-sw_0_all','lora-3b-8-het-data2-sw_1_all','lora-3b-8-het-data2-sw_2_all','lora-3b-8-het-data2-sw_3_all','lora-3b-8-het-data2-sw_4_all','lora-3b-8-het-data2-sw_5_all','lora-3b-8-het-data2-sw_6_all','lora-3b-8-het-data2-sw_7_all']
# path_list=['lora-3b-8-old-data2_0_all','lora-3b-8-old-data2_1_all','lora-3b-8-old-data2_2_all','lora-3b-8-old-data2_3_all','lora-3b-8-old-data2_4_all','lora-3b-8-old-data2_5_all','lora-3b-8-old-data2_6_all','lora-3b-8-old-data2_7_all']
# path_list=['lora-1b-71-8-data1_0_all','lora-1b-71-8-data1_1_all','lora-1b-71-8-data1_2_all','lora-1b-71-8-data1_3_all','lora-1b-71-8-data1_4_all','lora-1b-71-8-data1_5_all','lora-1b-71-8-data1_6_all','lora-1b-71-8-data1_7_all']
#
# for path in path_list:
#     print(path)
#     targets = read_list('../data/dataset1/flan_test_200_selected_nstrict_1.jsonl', 'output')
#     # print(targets)
#     predictions = read_list(f'{path_}{path}.jsonl', 'answer')
#     # print(predictions)
#     get_result(targets, predictions, f'{path_}{path}.json')
#
# path_list=['lora-1b-62-8-data1_0_all','lora-1b-62-8-data1_1_all','lora-1b-62-8-data1_2_all','lora-1b-62-8-data1_3_all','lora-1b-62-8-data1_4_all','lora-1b-62-8-data1_5_all','lora-1b-62-8-data1_6_all','lora-1b-62-8-data1_7_all']
#
# for path in path_list:
#     print(path)
#     targets = read_list('../data/dataset1/flan_test_200_selected_nstrict_1.jsonl', 'output')
#     # print(targets)
#     predictions = read_list(f'{path_}{path}.jsonl', 'answer')
#     # print(predictions)
#     get_result(targets, predictions, f'{path_}{path}.json')
#
# path_list=['lora-1b-53-8-data1_0_all','lora-1b-53-8-data1_1_all','lora-1b-53-8-data1_2_all','lora-1b-53-8-data1_3_all','lora-1b-53-8-data1_4_all','lora-1b-53-8-data1_5_all','lora-1b-53-8-data1_6_all','lora-1b-53-8-data1_7_all']
#
# for path in path_list:
#     print(path)
#     targets = read_list('../data/dataset1/flan_test_200_selected_nstrict_1.jsonl', 'output')
#     # print(targets)
#     predictions = read_list(f'{path_}{path}.jsonl', 'answer')
#     # print(predictions)
#     get_result(targets, predictions, f'{path_}{path}.json')

# path_list=['lora-1b-44-8-data2_0_all','lora-1b-44-8-data2_1_all','lora-1b-44-8-data2_2_all','lora-1b-44-8-data2_3_all','lora-1b-44-8-data2_4_all','lora-1b-44-8-data2_5_all','lora-1b-44-8-data2_6_all','lora-1b-44-8-data2_7_all']
# path_list=['result_1b_8_44_data1_0_all','result_1b_8_44_data1_1_all','result_1b_8_44_data1_2_all','result_1b_8_44_data1_3_all','result_1b_8_44_data1_4_all','result_1b_8_44_data1_5_all','result_1b_8_44_data1_6_all','result_1b_8_44_data1_7_all']


# path_list=['lora-1b-44-8-data1_0_all','lora-1b-44-8-data1_1_all','lora-1b-44-8-data1_2_all','lora-1b-44-8-data1_3_all','lora-1b-44-8-data1_4_all','lora-1b-44-8-data1_5_all','lora-1b-44-8-data1_6_all','lora-1b-44-8-data1_7_all']

# path_list=['lora-3b-8-het-data1-sw_{}_all']lora-3b-8-old-data2_3_all.jsonl  lora-7b-8-het-data1-sw_7_all_new result_lora-7b-8-het-data1-sw_0_all_auto*2_new_auto*1_new

for i in range(8):
    path=f'lora-0.6b-het-data2-sw__{i}_all'
    print(path)
    targets = read_list('../data/dataset2/flan_test_200_selected_nstrict_1.jsonl', 'output')
    # print(targets)
    predictions = read_list(f'{path_}{path}.jsonl', 'answer')
    # print(predictions)
    get_result(targets, predictions, f'{path_}{path}.json')