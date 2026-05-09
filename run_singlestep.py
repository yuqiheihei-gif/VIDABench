# -*- coding: utf-8 -*-

# 导入所需的基础库
import argparse
import pandas as pd
import requests
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from PIL import Image

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/yuanxi_xui_agent_1/xui_mid_train/models/experiment/260224/checkpoint/30BA3B-v5-20260228-225812-HF"

# 图片保存路径
DEFAULT_IMAGE_DIR = "/workspaces/"

# 系统 Prompt
MOBILE_USE_QWEN = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* click: Click the point on the screen with coordinate (x, y).\n* swipe: Swipe from the starting point with coordinate (x, y) to the end point with coordinate2 (x2, y2).\n* type: Click the point on the screen with coordinate (x, y) to activate the input box and input the specified text into the activated input box.", "enum": ["click", "swipe", "type"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by action=click, action=type, and action=swipe.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by action=swipe.", "type": "array"}, "text": {"description": "Required only by action=type.", "type": "string"}, "call_user": {"description": "Whether to prompt the user for input or confirmation at this step. If True, the system will pause and wait for user interaction; if False, the action proceeds automatically without user intervention.", "type": "boolean", "enum": ["True", "False"]}}, "required": ["action", "call_user"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.
"""




# ================= 模型加载 (全局加载一次) =================
print("正在加载模型和处理器，请稍候...")
try:
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)


# ================= 推理核心函数 =================
def process_csv_inference(input_csv_path: str, output_csv_path: str, image_save_dir: str):
    """
    处理单个CSV文件进行推理
    """
    print(f"\n{'='*20} 开始处理任务 {'='*20}")
    print(f"输入文件: {input_csv_path}")
    print(f"输出文件: {output_csv_path}")

    try:
        # 读取CSV，注意原文使用了gbk编码
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        print(f"成功读取CSV文件，共 {len(df)} 行数据。")
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
        print(f"已创建图片保存目录: {image_save_dir}")

    results_list = []
    print("开始逐行进行推理...")

    for index, row in df.iterrows():
        try:
            query = row['query']
            image_url = row['service_image_url']

            print(f"正在处理第 {index + 1}/{len(df)} 行 | 正在下载: {image_url[:70]}...")

            history_actions = ""  # 如果有历史记录列，可以在此添加
            
            
            user_query = f'''The user query:  {query} Task progress (You have done the following operation on the current device): {history_actions}'''

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": MOBILE_USE_QWEN},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": user_query},
                    ],
                }
            ]

            # 准备推理输入
            device = model.device
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)

            # 生成输出
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text_list = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            if output_text_list:
                result = output_text_list[0]
                print(result)
                results_list.append(result)
                print(f"  └── 推理成功。")
            else:
                error_message = "推理无返回结果"
                results_list.append(error_message)
                print(f"  └── 推理失败: {error_message}")

        except (requests.exceptions.RequestException, ValueError) as e:
            error_message = f"下载或验证图片时失败: {e}"
            print(f"  └── {error_message}")
            results_list.append(error_message)
            continue

        except Exception as e:
            error_message = f"推理阶段发生异常: {e}"
            print(f"  └── {error_message}")
            results_list.append(error_message)
            continue
        
    # 保存结果
    if len(results_list) == len(df):
        df['juanji-0302-30b'] = results_list
        print(f"\n正在保存结果到: {output_csv_path}")
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"保存成功！")
    else:
        print("\n错误：处理得到的结果数量与CSV的行数不匹配，为防止数据错乱，已终止保存。")


# ================= 主程序入口 =================
if __name__ == '__main__':
    # 定义需要处理的任务列表 (对应你提供的三个代码中的路径)
    tasks = [
        {
            "name": "任务1: 点击",
            "input": './单步导航评测集/业务单步导航点击_val.csv',
            "output": './评测结果/业务单步导航-点击2-juanji-0302-30b.csv'
        },
        {
            "name": "任务2: 输入",
            "input": './单步导航评测集/业务单步导航输入_val.csv',
            "output": './评测结果/业务单步导航-输入2-juanji-0302-30b.csv'
        },
        {
            "name": "任务3: 滑动",
            "input": './单步导航评测集/业务单步导航滑动_val.csv',
            "output": './评测结果/业务单步导航-滑动2-juanji-0302-30b.csv'
        }
    ]

    print(f"总共有 {len(tasks)} 个任务待处理。")

    # 循环处理每个任务
    for i, task in enumerate(tasks):
        print(f"\n>>> 正在执行第 {i+1} 个任务: {task['name']}")
        
        # 检查输入文件是否存在，避免报错中断后续任务
        if not os.path.exists(task['input']):
            print(f"错误: 找不到输入文件 {task['input']}，跳过此任务。")
            continue
            
        process_csv_inference(
            input_csv_path=task['input'],
            output_csv_path=task['output'],
            image_save_dir=DEFAULT_IMAGE_DIR
        )

    print("\n所有任务执行完毕！")
