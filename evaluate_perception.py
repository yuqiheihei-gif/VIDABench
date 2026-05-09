# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
import requests
import torch
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# Transformers
# Qwen3-VL 建议使用 AutoModelForConditionalGeneration 加载
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

# ==============================================================================
# 1. 全局配置与 API 设置
# ==============================================================================

API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 6ne2trZuLA5CX6lT5fn3CnPOK24nVRBx",
}

# -------------------- 评测 Prompt 模板库 (保持不变) --------------------

PROMPT_TEMPLATE_非业务_组件理解 = """作为一个公正且具备极高视觉理解能力的评测专家，请根据提供的图片、问题和参考答案，对“考生回答”进行打分。

    问题：{question}
    参考答案：{gt_response}
    考生回答：{model_output}

    【核心准则】：
    1. **参考答案是最高判定标准**：
    
    2. **裁判逻辑**：
       - **方位描述的相对性（重点）**：
         - 当考生描述“左侧是A，右侧是B”时，通常指**A与B的相对位置关系**（即B在A的右边），而**未必是指B处于容器的最右边缘**。
         - **示例**：图片顺序为 [图标] [文字] [箭头]。若考生回答“左侧是图标，右侧是文字”，应视为**正确**（因为文字确实在图标右侧，且是主要内容）。裁判不应因为忽略了最右侧的箭头而判错，除非问题专门询问“最右侧的小图标是什么”。
       - **次要元素的豁免**：
         - 如果考生准确描述了组件的主体形状、背景、核心图标和核心文字位置，仅遗漏了**非核心的装饰性元素**（如列表末尾的小箭头、分割线等），应视为**回答正确**。
       - **颜色问题（广义匹配原则）**：
         - 核心原则：参考答案是主要基准。
         - **【相近色豁免】**：如果考生回答的颜色与参考答案属于**同一色系、深浅不同或视觉上极易混淆的邻近色**，**均视为正确**。
            - 示例：黑vs灰、黄vs橙、蓝vs浅蓝等，均视为一致。

    【评分标准】：
    0分：回答是根本错误的（如位置完全颠倒），或者在颜色题中与参考答案**完全不沾边**（如红说成绿），或者产生严重幻觉，或者完全答非所问。
    1分：回答符合参考答案和图片中的视觉事实（包括合理的相对位置描述），或在颜色题中符合广义匹配原则。
        - 只要核心语义正确（即使遗漏末尾箭头等次要细节），能够回答问题，即可得1分。
        - **方位描述符合人类自然语言习惯（相对位置正确）即可得1分。**
        - （颜色题）若出现**同色系**或**相近色**的对应，必须给1分。

    请按照以下步骤输出：
    1. 【思考过程】：
    Step 1: 判断问题类型。关注是否涉及方位布局或颜色。
    Step 2: 检查参考答案。
        - 若是方位问题：检查考生的描述是否符合图片的**相对视觉流向**（如从左到右），不要因为遗漏边缘小图标而死板判错。
        - 若是颜色问题：检查是否落入“相近色/同色系”宽容范围。
    Step 3: 对比考生回答与判定标准，给出评判理由。**注意：针对布局描述，只要文字在图标右侧，即便忽略了更右侧的箭头，也应判定为描述符合事实。**
    2. 【打分】：在思考结束后，在最后一行严格按照格式 "Score: X" 输出分数，其中 X 为 0 或 1。

    示例输出：
    思考过程：问题询问组件形状和颜色。考生回答“左侧有图标，右侧是文字”，虽然图片最右侧还有一个箭头，但考生描述了文字在图标右侧这一核心事实，属于合理的相对位置描述；颜色“黑色”与参考答案“灰色”属于邻近色豁免。因此判定为正确。
    Score: 1
    """

PROMPT_TEMPLATE_非业务_元素指代 = """作为一个专业的UI交互体验评估专家，请根据“问题”和“参考答案”，对“考生回答”进行打分。

    【输入信息】
    问题：{question}
    参考答案：{gt_response}
    考生回答：{model_output}

    【核心评判准则】：
    请重点判断考生是否理解了组件的**核心功能方向**，而非扣字眼。

    1. **功能方向必须一致（Pass/Fail的关键）**：
       - **正确**：只要操作的**最终结果**一致，即视为正确。
         - *Case A*：参考答案说“点击收藏”，考生说“点击后变成已收藏状态” -> **判定为正确**（动作 vs 结果，意图一致）。
         - *Case B*：参考答案说“点击展开菜单”，考生说“点击显示更多选项” -> **判定为正确**。
       - **错误**：只有当**功能方向完全相反**或**严重歪曲**时，才判错误。
         - *Case C*：参考答案说“点击添加”，考生说“点击取消/删除” -> **判定为错误**（方向反了）。
         - *Case D*：参考答案说“提交表单”，考生说“清空表单” -> **判定为错误**。

    2. **容错机制**：
       - **忽略幻觉细节**：如果考生编造了具体的弹窗文案（如“提示操作成功”），但核心逻辑（添加成功）是对的，**不扣分**。
       - **忽略状态描述的角度**：无论是描述“动作”（去添加）还是描述“动作后的视觉反馈”（变亮/打钩），只要逻辑通顺，**均视为正确**。

    【评分标准】：
    - **Score 0 (不合格)**：
        - **功能严重错误**：功能描述出现了严重的错误或者功能说反了，例如把“选中”说成“取消”，把“进入”说成“退出”。
        - **对象错误**：完全认错了组件（如把“搜索栏”说成“输入框”以外的东西，如“按钮”）。
        - **答非所问**：没有描述交互功能，只描述了外观（形状颜色）。
    - **Score 1 (合格)**：
        - 核心功能的**作用方向**正确（如：Add, Select, Expand, Confirm）。
        - 即使包含多余的视觉细节描述或文案幻觉，只要不改变功能本质，都给1分。
    

    请按照以下步骤输出：
    1. 【思考过程】：
       Step 1: 提取参考答案的核心意图（Key Intent），例如：是“从无到有（添加）”还是“从有到无（取消）”？
       Step 2: 提取考生回答的核心意图。
       Step 3: **意图比对**。询问自己：考生的描述是否会导致用户达到相同的最终状态？
          - 如果参考是“添加”，考生说是“结果显示已添加”，这算**通过**。
          - 只有方向相反（如“删除”）才算失败。
    2. 【打分】：在思考结束后，在最后一行严格按照格式 "Score: X" 输出分数，其中 X 为 0 或 1。
    """

PROMPT_TEMPLATE_业务_组件理解 = """作为一个具备极高视觉理解能力和UI交互专业知识的评测专家，请根据提供的【图片】、【问题】和【参考答案】，对【考生回答】进行打分。

【输入信息】
问题：{question}
参考答案：{gt_response}
考生回答：{model_output}

【核心准则】：
# 1. **参考答案是最高判定标准**。

【第一步：判断题目类型并选择评分逻辑】
请先分析问题属于以下哪一类，并严格应用对应的评分准则：

---

### 场景一：组件位置/布局理解 (Location & Layout)
*适用问题：询问组件在哪里、相对位置、方位描述等。*

**【判定核心：空间逻辑必须精确符合视觉事实】**

**1. 严禁“轴向错误”（致命伤 -> Score 0）**：
   - **横排 vs 竖排**：如果组件 A 和 B 在图片中是**同一行横向排列**，考生绝对不能说 A 在 B 的**“下方”**或**“上方”**。
   - **包含关系错误**：如果 A 包含 B，不能说 A 在 B 旁边。

**2. 严禁“隔空邻接”（致命伤 -> Score 0）**：
   - **紧邻/挨着/旁边**：如果考生使用了**“紧邻”、“挨着”、“紧靠”**等表示直接相邻的词汇，而实际上两个组件中间**隔着其他明显的独立组件**，必须判错。

**3. 参照物多样性豁免（仅限逻辑正确时）：**
   - 允许更换参照物，但前提是描述必须精准。

**4. “之间”的维度兼容性（关键修正）：**
   - **语义定义**：考生使用“在...之间” (between) 时，**既可以指横向排列，也可以指纵向垂直排列**。

---

### 场景二：组件功能/交互 (Functionality)
*适用问题：询问组件的作用、点击后的效果、功能跳转等。*

**【核心评分哲学】**：
重点判断是否理解了**核心功能**。

1. **功能方向必须一致（Pass/Fail的关键）**：
       - **正确**：只要操作的**最终结果**一致，即视为正确。
       - **错误**：只有当**功能方向完全相反**或**严重歪曲**时，才判错误。

2. **容错机制**：
       - **忽略幻觉细节**：如果考生编造了具体的弹窗文案，但核心逻辑是对的，**不扣分**。

---

### 场景三：组件颜色/形状 (Color & Shape)
*适用问题：询问组件是什么颜色、什么形状、图标样式等。*

**【核心评分哲学】**：
广义匹配原则。

1. **颜色豁免**：同色系或相近色均视为正确。
2. **次要元素豁免**：如果准确描述了核心图标/文字，仅遗漏末尾的小箭头、装饰线，视为正确。

---

### 场景四：组件属性/状态 (Attribute & Status)
*适用问题：询问是否选中、xx是否可点击、开关状态等。*

**【核心评分哲学】**：
严格事实核对。

1. **判分**：状态描述必须准确。“已选中”不能说成“未选中”。

---

### 【通用评分标准】
**Score 0 (错误)**：
- 方位完全相反/参照物错误（场景一）。
- 功能方向严重错误/对象认错（场景二）。
- 颜色严重不符/产生严重幻觉（场景三/四）。

**Score 1 (正确)**：
- **位置**：核心定位准确。
- **功能**：核心意图正确。
- **颜色/形状**：符合广义匹配。
- **状态**：符合事实。

---

请按照以下步骤输出：
1. 【思考过程】：
   Step 1: 判定题目所属场景。
   Step 2: 根据该场景的特定规则，对比考生回答与图片事实/参考答案。
   Step 3: 得出结论。
2. 【打分】：在思考结束后，在最后一行严格按照格式 "Score: X" 输出分数，其中 X 为 0 或 1。
"""

PROMPT_TEMPLATE_业务_元素指代 = """作为一个专业的UI交互体验评估专家，请根据“问题”和“参考答案”，对“考生回答”进行打分。

    【输入信息】
    问题：{question}
    参考答案：{gt_response}
    考生回答：{model_output}

    【核心评判准则】：
    请重点判断考生是否理解了组件的**核心功能方向**，而非扣字眼。

    1. **功能方向必须一致（Pass/Fail的关键）**：
       - **正确**：只要操作的**最终结果**一致，即视为正确。
         - *Case A*：参考答案说“点击收藏”，考生说“点击后变成已收藏状态” -> **判定为正确**（动作 vs 结果，意图一致）。
         - *Case B*：参考答案说“点击展开菜单”，考生说“点击显示更多选项” -> **判定为正确**。
       - **错误**：只有当**功能方向完全相反**或**严重歪曲**时，才判错误。
         - *Case C*：参考答案说“点击添加”，考生说“点击取消/删除” -> **判定为错误**（方向反了）。
         - *Case D*：参考答案说“提交表单”，考生说“清空表单” -> **判定为错误**。

    2. **容错机制**：
       - **忽略幻觉细节**：如果考生编造了具体的弹窗文案（如“提示操作成功”），但核心逻辑（添加成功）是对的，**不扣分**。
       - **忽略状态描述的角度**：无论是描述“动作”（去添加）还是描述“动作后的视觉反馈”（变亮/打钩），只要逻辑通顺，**均视为正确**。

    【评分标准】：
    - **Score 0 (不合格)**：
        - **功能严重错误**：功能描述出现了严重的错误或者功能说反了，例如把“选中”说成“取消”，把“进入”说成“退出”。
        - **对象错误**：完全认错了组件（如把“搜索栏”说成“输入框”以外的东西，如“按钮”）。
        - **答非所问**：没有描述交互功能，只描述了外观（形状颜色）。
    - **Score 1 (合格)**：
        - 核心功能的**作用方向**正确（如：Add, Select, Expand, Confirm）。
        - 即使包含多余的视觉细节描述或文案幻觉，只要不改变功能本质，都给1分。
    

    请按照以下步骤输出：
    1. 【思考过程】：
       Step 1: 提取参考答案的核心意图（Key Intent），例如：是“从无到有（添加）”还是“从有到无（取消）”？
       Step 2: 提取考生回答的核心意图。
       Step 3: **意图比对**。询问自己：考生的描述是否会导致用户达到相同的最终状态？
          - 如果参考是“添加”，考生说是“结果显示已添加”，这算**通过**。
          - 只有方向相反（如“删除”）才算失败。
    2. 【打分】：在思考结束后，在最后一行严格按照格式 "Score: X" 输出分数，其中 X 为 0 或 1。
    """

# ==============================================================================
# 2. 辅助工具类与函数
# ==============================================================================

def ask_for_qwen3VL(image_afts_url, prompt):
    """
    调用 Qwen3-VL API (Judger) 进行打分
    注意：这里的 Qwen3-VL 是指作为裁判的 API 模型，不是你正在评测的本地模型
    """
    url = "https://antchat.alipay.com/v1/chat/completions"
    body = {
        "model": "Qwen3-VL-235B-A22B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_afts_url},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "stream": False
    }
    
    max_retries = 5
    wait_seconds = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=API_HEADERS, json=body, timeout=600)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error (Attempt {attempt + 1}): Status {response.status_code}")
        except Exception as e:
            print(f"API Exception (Attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(wait_seconds)
    return None

def process_instruction_coords(text, img_w, img_h):
    """
    专门处理 '元素指代' 任务中的相对坐标转绝对坐标
    """
    match = re.search(r'\[(.*?)\]', text)
    if match:
        coords_str = match.group(1)
        try:
            coords_rel = [float(x) for x in coords_str.split(',')]
            if len(coords_rel) == 4:
                abs_x1 = int(coords_rel[0] / 999.0 * img_w)
                abs_y1 = int(coords_rel[1] / 999.0 * img_h)
                abs_x2 = int(coords_rel[2] / 999.0 * img_w)
                abs_y2 = int(coords_rel[3] / 999.0 * img_h)
                coords_abs = [abs_x1, abs_y1, abs_x2, abs_y2]
            else:
                coords_abs = [int(x) for x in coords_rel]
        except ValueError:
            return text
        
        new_text = f"请描述绝对像素坐标区域{coords_abs}的组件的交互功能。回答格式要求：说明用户的操作（如点击、输入、滑动等）以及触发的后果（如跳转、弹窗）。严禁只描述该区域显示的文字或图片内容。"
        return new_text
    return text

def calculate_judge_score(image_url, question, gt_response, model_output, template_type="general"):
    """
    根据任务类型选择 Prompt 模板进行打分
    """
    if template_type == "非业务-组件理解":
        judge_prompt = PROMPT_TEMPLATE_非业务_组件理解.format(question=question, gt_response=gt_response, model_output=model_output)
    elif template_type == "非业务-元素指代":
        judge_prompt = PROMPT_TEMPLATE_非业务_元素指代.format(question=question, gt_response=gt_response, model_output=model_output)
    elif template_type == "业务-组件理解":
        judge_prompt = PROMPT_TEMPLATE_业务_组件理解.format(question=question, gt_response=gt_response, model_output=model_output)
    elif template_type == "业务-元素指代":
        judge_prompt = PROMPT_TEMPLATE_业务_元素指代.format(question=question, gt_response=gt_response, model_output=model_output)
    try:
        res = ask_for_qwen3VL(image_url, judge_prompt)
        if res and 'choices' in res:
            content = res['choices'][0]['message']['content'].strip()
            # 提取 Score: 0 或 Score: 1
            match = re.search(r'Score[:：]\s*([01])', content, re.IGNORECASE)
            if not match:
                # 兜底策略：看最后一行是否有数字
                match = re.search(r'([01])', content.split('\n')[-1])
            score = int(match.group(1)) if match else 0
            return score, content
    except Exception as e:
        print(f"Judge Error: {e}")
    return 0, ""

def save_summary(result_dict, num_samples, output_path, is_final=False):
    summary = {}
    # 动态计算存在的指标
    for key, val_list in result_dict.items():
        if key == 'Accuracy':
            if val_list[1] > 0:
                summary['Accuracy'] = val_list[0] / val_list[1]
        elif val_list: # 如果列表不为空
            summary[f'mean_{key}'] = sum(val_list) / len(val_list)
            
    if is_final:
        print(f"\n[{output_path}] 最终结果: {json.dumps(summary, indent=4, ensure_ascii=False)}")
    
    base, ext = os.path.splitext(output_path)
    save_file = output_path if is_final else f"{base}_summary_checkpoint{ext}"
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

# ==============================================================================
# 3. 核心评测逻辑
# ==============================================================================

def run_evaluation_for_task(task_config, models, args):
    """
    运行单个任务的评测循环
    models: 包含推理模型 (inference_model, processor)
    """
    task_name = task_config['name']
    file_path = task_config['file_path']
    # 配置选项
    use_coord_transform = task_config.get('use_coord_transform', False)
    judge_template = task_config.get('judge_template', 'general') # interaction 或 general
    
    output_summary_path = f"{args.output_path}/output_{task_name}_{args.model_name}.txt"
    output_detail_path = f"{args.output_path}/detail_{task_name}_{args.model_name}.jsonl"
    
    print(f"\n{'='*20} 开始任务: {task_name} {'='*20}")
    print(f"数据源: {file_path}")
    print(f"输出到: {output_detail_path}")

    # 1. 加载数据
    test_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_list.append(json.loads(line))
    else:
        print(f"错误: 文件不存在 {file_path}")
        return

    # 2. 断点恢复
    # 仅保留 Qwen Judge Score 和 基础 Accuracy
    result_dict = {'qwen_judge_score': [], 'Accuracy': [0, 0]}
        
    processed_count = 0
    right_num = 0
    
    if os.path.exists(output_detail_path) and not args.re_eval:
        print("检测到历史进度，正在恢复...")
        with open(output_detail_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_count += 1
                    scores = data.get('scores', {})
                    
                    # 恢复 judge 分数
                    if 'qwen_judge_score' in scores:
                        result_dict['qwen_judge_score'].append(scores['qwen_judge_score'])
                    
                    # 恢复 Accuracy (简单字符串包含匹配)
                    gt = data.get('gt_response', '')
                    pred = data.get('model_output', '')
                    if pred and (pred == gt or gt in pred):
                        right_num += 1
                except: continue
        result_dict['Accuracy'] = [right_num, processed_count]
        print(f"已恢复 {processed_count} 条。")
    elif args.re_eval and os.path.exists(output_detail_path):
        os.remove(output_detail_path)

    # 3. 采样
    target_list = test_list[::args.inter]
    if processed_count >= len(target_list):
        print("该任务已完成。")
        save_summary(result_dict, processed_count, output_summary_path, is_final=True)
        return

    # 4. 循环推理
    inference_model, processor = models['inference']
    
    for i, test_data in enumerate(tqdm(target_list)):
        if i < processed_count: continue
        
        generated_text = ""
        current_scores = {}
        
        try:
            image_path = test_data['images'][0]
            raw_instruction = test_data['messages'][0]['content']
            gt_response = test_data['messages'][1]['content']
            
            # --- 图片处理 (为了获取宽高) ---
            img_w, img_h = 999, 999
            try:
                if image_path.startswith(('http', 'https')):
                    resp = requests.get(image_path, timeout=10)
                    with Image.open(BytesIO(resp.content)) as img:
                        img_w, img_h = img.size
                elif os.path.exists(image_path):
                    with Image.open(image_path) as img:
                        img_w, img_h = img.size
            except: pass

            # --- 指令处理 ---
            if use_coord_transform:
                instruction = process_instruction_coords(raw_instruction, img_w, img_h)
            else:
                instruction = raw_instruction + args.add

            if i % 10 == 0:
                print(f"\n[Q]: {instruction[:100]}...")

            # --- 模型推理 (Qwen3-VL 适配) ---
            # Qwen3-VL 通常支持标准的 chat template。
            # 注意：新版 Transformers/Qwen3VL 可能需要特定的 processor 调用方式
            # 这里的 messages 格式符合 Qwen-VL 系列的标准要求
            messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": instruction}]}]
            
            # 使用 apply_chat_template 处理文本和图像
            # trust_remote_code=True 的 processor 会处理 image 路径的读取
            inputs = processor.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(inference_model.device)
            
            # 生成
            generated_ids = inference_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            

            # 截取新生成的 token
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if '</think>' in generated_text:
                generated_text = generated_text.split('</think>')[-1].strip()

            if i % 10 == 0:
                print(f"[A]: {generated_text[:100]}...")

            # --- 指标计算 (仅 API Judge) ---
            if generated_text:
                # 1. Qwen Judge
                score, reasoning = calculate_judge_score(image_path, instruction, gt_response, generated_text, template_type=judge_template)
                current_scores['qwen_judge_score'] = score
                current_scores['qwen_judge_reasoning'] = reasoning
                result_dict['qwen_judge_score'].append(score)
                
                # 2. 简单 Accuracy (用于YNQA或严格匹配)
                is_correct = (generated_text == gt_response or gt_response in generated_text)
                if is_correct: right_num += 1
                result_dict['Accuracy'][0] = right_num
                result_dict['Accuracy'][1] = i + 1

            # --- 保存结果 ---
            save_item = test_data.copy()
            save_item['model_output'] = generated_text
            save_item['scores'] = current_scores
            
            with open(output_detail_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_item, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"Error at index {i}: {e}")
            import traceback
            traceback.print_exc()
            # 填充0分防止长度不一致
            result_dict['qwen_judge_score'].append(0.0)
            continue
        
        if (i + 1) % 500 == 0:
            save_summary(result_dict, i + 1, output_summary_path)

    save_summary(result_dict, len(target_list), output_summary_path, is_final=True)


# ==============================================================================
# 4. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Midtrain-30BA3B-Instruct-260224', help='模型名称标识，用于输出文件命名')
    # 请在这里填入 Qwen3-VL 的实际绝对路径
    parser.add_argument('--model_path', type=str, default='/yuanxi_xui_agent_1/xui_mid_train/models/experiment/260224/checkpoint/30BA3B-v5-20260228-225812-HF', help='模型权重路径')
    parser.add_argument('--output_path', type=str, default='./评测结果', help='结果输出根目录')
    parser.add_argument('--inter', type=int, default=1, help='采样间隔')
    parser.add_argument('--re_eval', action='store_true', help='强制重跑')
    parser.add_argument('--add', type=str, default='', help='额外prompt')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # --------------------------------------------------------------------------
    # 配置区域：在这里添加你想跑的任务
    # --------------------------------------------------------------------------
    TASKS_TO_RUN = [
        {
            "name": "非业务-组件理解",
            "file_path": "./感知评测集/业务组件理解val.jsonl",
            "use_coord_transform": False,     
            "judge_template": "业务-组件理解"  # 使用交互功能的打分 Prompt
        },
        {
            "name": "非业务-元素指代",
            "file_path": "./感知评测集/业务-元素指代-val.jsonl",
            "use_coord_transform": False,    
            "judge_template": "业务-元素指代"      # 使用通用UI理解的打分 Prompt
        }
    ]
    # --------------------------------------------------------------------------

    # 1. 加载推理模型 (Qwen3-VL 适配版)
    print(f'\n>>> 正在加载模型: {args.model_name} 从路径: {args.model_path} ...')
    
    try:
        # 使用 AutoModelForConditionalGeneration 并开启 trust_remote_code=True
        # 这是加载 Qwen3-VL 等新架构模型的标准方法
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        )
        
        # 加载 Processor
        processor = AutoProcessor.from_pretrained(
            args.model_path, 
            trust_remote_code=True
        )
        
        print(">>> 推理模型加载成功。")
    except Exception as e:
        print(f"Model load failed: {e}")
        print("建议检查: 1. model_path 是否正确; 2. transformers 版本是否支持 Qwen3; 3. 网络是否允许加载远程代码(trust_remote_code)")
        exit(1)
    
    # 打包模型对象
    models_pack = {
        'inference': (model, processor)
    }

    # 2. 遍历任务执行
    for task_conf in TASKS_TO_RUN:
        run_evaluation_for_task(task_conf, models_pack, args)
        
    print("\n所有任务执行完毕！")
