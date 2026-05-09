# -*- coding: utf-8 -*-

# 导入所需的基础库
import argparse
import pandas as pd
import requests
import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from PIL import Image

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/yuanxi_xui_agent_1/xui_mid_train/models/experiment/260224/checkpoint/30BA3B-v5-20260228-225812-HF"


MOBILE_USE_QWEN = """
\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinate2 (x2, y2).\\n* `type`: Click the point on the screen with coordinate (x, y) to activate the input box and input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `wait`: Wait for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=type`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to press. Required only by `action=long_press`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface. Required only by `action=system_button`\", \"enum\": [\"Back\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"call_user\": {\"description\": \"Whether to prompt the user for input or confirmation at this step. If True, the system will pause and wait for user interaction; if False, the action proceeds automatically without user intervention.\", \"type\": \"boolean\", \"enum\": [\"True\", \"False\"]}}, \"required\": [\"action\", \"call_user\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Action: a short imperative describing what to do in the UI.\n2) A single <tool_call>...</tool_call> block containing only the JSON: {\"name\": <function-name>, \"arguments\": <args-json-object>}.
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
def process_trajectory_inference(input_root_dir: str, output_csv_path: str):
    """
    遍历指定文件夹中的子文件夹，处理轨迹JSON数据并进行推理
    """
    print(f"\n{'='*20} 开始处理轨迹任务 {'='*20}")
    print(f"输入根目录: {input_root_dir}")
    print(f"输出文件: {output_csv_path}")

    if not os.path.exists(input_root_dir):
        print(f"错误: 输入目录 {input_root_dir} 不存在！")
        return

    results_list = []
    num = 0
    # 遍历根目录下的所有内容
    for root, dirs, files in os.walk(input_root_dir):
        num += 1
        print(f"正在执行第{num}条轨迹")
        # 查找当前目录下的所有 json 文件
        json_files = [f for f in files if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            print(f"\n找到轨迹文件: {json_path}")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)
            except Exception as e:
                print(f"读取JSON文件 {json_path} 时发生错误: {e}")
                continue
                
            # 确保按 step_id 排序，以防 JSON 数据乱序
            trajectory_data = sorted(trajectory_data, key=lambda x: x.get('step_id', 0))
            
            # 开始处理该轨迹的每一步
            for i, step_data in enumerate(trajectory_data):
                try:
                    episode_id = step_data.get('episode_id', '')
                    step_id = step_data.get('step_id', i)
                    query = step_data.get('instruction', '')
                    
                    # === 1. 构建前 i 步的历史动作 ===
                    history_actions_list = []
                    for j in range(i):
                        action_text = trajectory_data[j].get('result_action_text', '未知动作')
                        history_actions_list.append(f"第{j}步：{action_text}")
                    
                    history_actions = "，".join(history_actions_list) if history_actions_list else "无"
                    
                    # === 2. 拼接本地图片路径 ===
                    # json 里的 image_path 类似 "test/xxx/xxx_0.jpeg"，我们取文件名并与当前 json 所在目录拼接
                    image_filename = os.path.basename(step_data.get('image_path', ''))
                    local_img_path = os.path.abspath(os.path.join(root, image_filename))
                    
                    # Qwen-VL 读取本地图片推荐使用 file:// 协议
                    image_uri = f"{local_img_path}"
                    
                    print(f"  正在处理 Episode: {episode_id[:20]}... | Step: {step_id}")
                    
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
                                {"type": "image", "image": image_uri},
                                {"type": "text", "text": user_query},
                            ],
                        }
                    ]

                    # 检查本地图片是否存在
                    if not os.path.exists(local_img_path):
                        print(f"    └── 警告: 找不到本地截图 {local_img_path}")
                        output_text = "图片不存在，推理失败"
                    else:
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
                            output_text = output_text_list[0]
                            print(f"    └── 推理成功: {output_text}...")
                        else:
                            output_text = "推理无返回结果"
                            print(f"    └── 推理失败: {output_text}")

                except Exception as e:
                    output_text = f"推理阶段发生异常: {e}"
                    print(f"    └── {output_text}")
                
                # === 3. 组装当前步的保存结果 ===
                result_row = {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "messages": json.dumps(messages, ensure_ascii=False), # 将模型输入转为文本保存
                    "拼接历史轨迹": history_actions,
                    "当前截图path": local_img_path,
                    "instruction": query,
                    "image_width": step_data.get("image_width", ""),
                    "image_height": step_data.get("image_height", ""),
                    "ui_positions": step_data.get("ui_positions", ""),
                    "result_action_type": step_data.get("result_action_type", ""),
                    "result_action_text": step_data.get("result_action_text", ""),
                    "result_touch_yx": step_data.get("result_touch_yx", ""),
                    "result_lift_yx": step_data.get("result_lift_yx", ""),
                    "duration": step_data.get("duration", ""),
                    "模型完整输出文本": output_text,
                    "完整该step对应的json": json.dumps(step_data, ensure_ascii=False)
                }
                results_list.append(result_row)

    # ================= 保存结果 =================
    if results_list:
        df = pd.DataFrame(results_list)
        print(f"\n正在保存结果，共 {len(df)} 条数据。")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"成功保存至: {output_csv_path}")
    else:
        print("\n未找到任何可处理的数据，无需保存。")


# ================= 主程序入口 =================
if __name__ == '__main__':
    # 定义需要处理的任务列表（支持新的文件夹输入格式）
    tasks = [
        {
            "name": "任务1: 轨迹数据推理",
            # 这里替换为包含所有子文件夹（轨迹数据）的总文件夹路径
            "input_dir": './多步导航评测集/多步导航评测集数据', 
            "output_csv": './评测结果/test_多步轨迹推理结果-juanji-0302-30b.csv'
        },
        # 根据需要可以添加更多任务
    ]

    print(f"总共有 {len(tasks)} 个任务待处理。")

    # 循环处理每个任务
    for i, task in enumerate(tasks):
        print(f"\n>>> 正在执行第 {i+1} 个任务: {task['name']}")
        
        process_trajectory_inference(
            input_root_dir=task['input_dir'],
            output_csv_path=task['output_csv']
        )

    print("\n所有任务执行完毕！")