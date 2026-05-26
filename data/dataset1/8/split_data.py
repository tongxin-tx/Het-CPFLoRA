import json
import math
from pathlib import Path


def split_json_file(input_file, num_parts=5,ind=0):
    """
    将 JSON 文件均分为多个文件

    参数:
        input_file: 输入 JSON 文件路径
        num_parts: 要分割成的文件数量
    """
    # 1. 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始文件: {input_file}")
    print(f"总数据项数: {len(data)}")
    print(f"分割份数: {num_parts}")

    # 2. 计算每份的大小
    total_items = len(data)
    items_per_file = math.ceil(total_items / num_parts)
    print(f"每个文件大约 {items_per_file} 项")

    # 3. 创建输出文件名模板
    input_path = Path(input_file)
    stem = input_path.stem
    suffix = input_path.suffix

    # 4. 分割并保存
    for i in range(num_parts):
        # 计算当前文件的起始和结束索引
        start_idx = i * items_per_file
        end_idx = min((i + 1) * items_per_file, total_items)

        # 提取子数据
        sub_data = data[start_idx:end_idx]

        # 生成输出文件名
        output_file = input_path.parent / f"local_training_{i+ind}{suffix}"

        print(output_file)
        path=f'/data/tongxin/FedDPA-main/data/dataset1/50/{output_file}'
        # 保存为新的 JSON 文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sub_data, f, ensure_ascii=False, indent=2)

        print(f"已保存: {output_file.name} - {len(sub_data)} 项")

    print("\n分割完成！")


# 使用示例
if __name__ == "__main__":
    # split_json_file("local_training_2.json", num_parts=6,ind=12)
    # split_json_file("local_training_3.json", num_parts=6,ind=18)
    # split_json_file("local_training_4.json", num_parts=6,ind=24)
    # split_json_file("local_training_5.json", num_parts=6,ind=30)
    # split_json_file("local_training_6.json", num_parts=6,ind=36)
    # split_json_file("local_training_7.json", num_parts=6,ind=42)
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
    difference = [r - s for r, s in zip(rank, r_s_list)]
    print(difference)






















