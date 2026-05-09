import pandas as pd
import json
import re
import ast
import difflib
import os
import base64
import requests

# ================= API 配置区域 =================

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 6ne2trZuLA5CX6lT5fn3CnPOK24nVRBx",
}

def image_to_base64_data_uri(file_path):
    """将本地图片文件转换为 base64 data URI，供外部 API 读取"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # 假设截图主要是jpeg或png
    return f"data:image/jpeg;base64,{encoded_string}"

def ask_for_qwen3VL(image_b64_url, prompt):
    """调用 Qwen3-VL 接口进行等价性判断"""
    url = "https://antchat.alipay.com/v1/chat/completions"
    body = {
        "model": "Qwen3-VL-235B-A22B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": image_b64_url # 使用base64传入
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "stream": False
    }
    try:
        response = requests.post(url, headers=HEADERS, json=body, timeout=60)
        return response.json()
    except Exception as e:
        print(f"API 调用异常: {e}")
        return None

# ================= 辅助函数 =================

def calculate_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, str(text1), str(text2)).ratio()

def extract_type_text(action_text):
    match = re.search(r'输入(.*?)(?:，|$)', str(action_text))
    if match:
        return match.group(1).strip()
    return str(action_text)

def parse_model_output(output_str):
    output_str = str(output_str)
    action_text = ""
    action_match = re.search(r'Action:\s*(.*?)(?=\n<tool_call>|$)', output_str, re.DOTALL)
    if action_match:
        action_text = action_match.group(1).strip()

    tool_call = {}
    json_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', output_str, re.DOTALL)
    if json_match:
        try:
            tool_call = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    return action_text, tool_call

# ================= 主评估逻辑 =================

def evaluate_metrics(csv_path, print_limit=20):
    df = pd.read_csv(csv_path)

    total_valid_steps = 0

    # 修正后的指标统计（含 Qwen）
    tm_correct_count = 0
    em_correct_count = 0
    tdfr_fault_count = 0
    tdfr_denominator = 0
    episode_results = {}

    # ===== 新增：原始指标统计（不含 Qwen） =====
    raw_tm_correct_count = 0
    raw_em_correct_count = 0
    raw_tdfr_fault_count = 0
    raw_tdfr_denominator = 0
    raw_episode_results = {}
    # ==========================================

    # === 按 GT 动作类型分类统计 ===
    gt_action_stats = {}

    error_cases = []
    error_print_count = 0

    # 评测集中标准的动作列表（包含模型可能输出的所有动作）
    # 注意：system_button 和 wait 需要 API 兜底判断，long_press 直接映射判断位置
    standard_gt_actions = ["click", "swipe", "type", "system_button", "terminate", "long_press", "wait"]

    # 模型预测动作 -> GT 动作的映射（用于 TM 判断）
    # 直接映射：action 类型可以直接判断等价
    # API兜底：只用 TM 判断，EM（参数）需要 API 判断
    ACTION_MAPPING = {
        # 直接映射（action 类型可直接比较）
        "click": "click",
        "long_press": "click",   # TM: → click; EM: 位置在 GT 目标框内
        "swipe": "swipe",
        "type": "type",
        "terminate": "terminate",
        # API 兜底（TM 直接判断 action 类型，EM 参数需要 API 判断）
        "system_button": "system_button",  # TM: 直接判断; EM: button 参数 API 判断
        "wait": None,             # 无等价 GT，API 判断是否可接受
    }

    print("="*40)
    print("开始评测并打印错误案例（包含 Qwen 动态评审）...")
    print("="*40)
    print(f"数据总行数: {len(df)}")

    for index, row in df.iterrows():
        ep_id = row['episode_id']
        step_id = row['step_id']

        # 过滤多路径页
        if '完整该step对应的json' in row and pd.notna(row['完整该step对应的json']):
            try:
                step_json_data = json.loads(str(row['完整该step对应的json']))
                if step_json_data.get('page_cls_eval') == '多路径页':
                    continue
            except Exception:
                pass

        # 1. 解析 Ground Truth
        gt_type_code = row['result_action_type']
        gt_action_text = row['result_action_text']
        gt_action_type = ""

        try:
            touch_yx = ast.literal_eval(str(row['result_touch_yx']))
            lift_yx = ast.literal_eval(str(row['result_lift_yx']))
        except:
            touch_yx, lift_yx = [0,0], [0,0]

        if gt_type_code == 4:
            gt_action_type = "click" if touch_yx == lift_yx else "swipe"
        elif gt_type_code == 3:
            gt_action_type = "type"
        elif gt_type_code in [6, 7]:
            gt_action_type = "system_button"
        elif gt_type_code == 10:
            gt_action_type = "terminate"

        if not gt_action_type:
            continue

        total_valid_steps += 1
        # 初始化该 GT 动作类型的统计
        if gt_action_type not in gt_action_stats:
            gt_action_stats[gt_action_type] = {
                "total": 0,
                "tm_correct": 0,
                "em_correct": 0,
                "tdfr_fault": 0,
                "tdfr_denom": 0
            }
        gt_action_stats[gt_action_type]["total"] += 1

        if ep_id not in episode_results:
            episode_results[ep_id] = []

        # 2. 解析模型输出
        model_action_desc, model_tool_call = parse_model_output(row['模型完整输出文本'])
        model_args = model_tool_call.get('arguments', {})
        model_action_type = model_args.get('action', '')

        # 提取GT核心参数用于判断和展示
        gt_params = ""
        if gt_action_type == "click":
            try:
                ui_pos = ast.literal_eval(str(row['ui_positions']))[0]
                gt_params = f"目标框[y, x, h, w]: {ui_pos}"
            except:
                gt_params = "解析失败"
        elif gt_action_type == "swipe":
            gt_params = f"touch_yx: {touch_yx}, lift_yx: {lift_yx}"
        elif gt_action_type == "type":
            gt_params = f"文本: {extract_type_text(gt_action_text)}"

        # 3. 计算准确率 (基础规则判断)
        # 使用映射判断动作类型是否等价
        mapped_action_type = ACTION_MAPPING.get(model_action_type, model_action_type)
        tm_correct = (gt_action_type == mapped_action_type)
        em_correct = False

        # 判断是否需要 API 兜底
        # wait: 无等价 GT，TM 和 EM 都需要 API 判断
        # system_button: TM 直接判断，EM 参数需要 API 判断
        tm_need_api = (model_action_type in ACTION_MAPPING and ACTION_MAPPING[model_action_type] is None)
        em_need_api = (model_action_type == "system_button" or model_action_type == "wait")
        need_api_judge = tm_need_api or em_need_api

        if tm_correct:
            if mapped_action_type == "click":
                try:
                    ui_positions = ast.literal_eval(str(row['ui_positions']))[0]
                    gt_y, gt_x, gt_h, gt_w = ui_positions
                    model_coord = model_args.get('coordinate', [0, 0])
                    model_x, model_y = model_coord[0], model_coord[1]
                    norm_model_x = model_x / 1000.0
                    norm_model_y = model_y / 1000.0
                    if (gt_y <= norm_model_y <= gt_y + gt_h) and (gt_x <= norm_model_x <= gt_x + gt_w):
                        em_correct = True
                except:
                    pass
            elif mapped_action_type == "swipe":
                try:
                    model_c1 = model_args.get('coordinate', [0, 0])
                    model_c2 = model_args.get('coordinate2', [0, 0])
                    mdx = model_c2[0] - model_c1[0]
                    mdy = model_c2[1] - model_c1[1]
                    gdx = lift_yx[1] - touch_yx[1]
                    gdy = lift_yx[0] - touch_yx[0]
                    if (mdx * gdx) + (mdy * gdy) > 0:
                        em_correct = True
                except:
                    pass
            elif mapped_action_type == "type":
                gt_text = extract_type_text(gt_action_text)
                model_text = model_args.get('text', '')
                if calculate_text_similarity(gt_text, model_text) > 0.5:
                    em_correct = True
            # system_button 改为 API 兜底判断（参数不确定，不能直接比较）
            # elif mapped_action_type == "system_button":
            #     if model_args.get('button', '') == 'Back':
            #         em_correct = True
            elif mapped_action_type == "terminate":
                if model_args.get('status', '') == 'success':
                    em_correct = True
        else:
            # ======== 新增：处理 GT为click，但模型预测为type 的合并/跨步情况 ========
            # Prompt定义中 type 包含 "Click the point... to activate the input box"
            if gt_action_type == "click" and model_action_type == "type":
                try:
                    ui_positions = ast.literal_eval(str(row['ui_positions']))[0]
                    gt_y, gt_x, gt_h, gt_w = ui_positions
                    model_coord = model_args.get('coordinate', [0, 0])
                    model_x, model_y = model_coord[0], model_coord[1]
                    norm_model_x = model_x / 1000.0
                    norm_model_y = model_y / 1000.0
                    
                    # 只要模型 type 的坐标点在了 GT 要求点击的输入框内，即认为动作完全正确
                    if (gt_y <= norm_model_y <= gt_y + gt_h) and (gt_x <= norm_model_x <= gt_x + gt_w):
                        tm_correct = True
                        em_correct = True
                        print(f"💡 [智能等价] GT为click，模型提前预测为type，坐标命中目标框，判定为正确 (Ep: {ep_id}, Step: {step_id})")
                except Exception:
                    pass
            # ==============================================================

        # ===== 新增：记录不使用 Qwen 修正的原始结果 =====
        raw_tm_correct = tm_correct
        raw_em_correct = em_correct

        if raw_tm_correct:
            raw_tm_correct_count += 1

        if raw_em_correct:
            raw_em_correct_count += 1
        else:
            raw_tdfr_denominator += 1
            if calculate_text_similarity(model_action_desc, gt_action_text) < 0.5:
                raw_tdfr_fault_count += 1

        if ep_id not in raw_episode_results:
            raw_episode_results[ep_id] = []
        raw_episode_results[ep_id].append({
            'step_id': step_id,
            'em_correct': raw_em_correct
        })
        # ============================================

        # ================= 4. Qwen API 兜底评测 (当模型动作需要 API 判断等价性) =================
        api_judgement_detail = ""
        # 需要 API 兜底的情况：
        # 1. 模型输出 system_button（参数不确定，需 API 判断）
        # 2. 模型输出 wait（无等价 GT，API 判断是否可接受）
        if need_api_judge and model_action_type:
            print(f"🔄 检测到非标准动作 [{model_action_type}]，正在调用 Qwen API 评审...")

            img_path = str(row.get('当前截图path', ''))
            instruction = str(row.get('instruction', ''))
            history = str(row.get('拼接历史轨迹', ''))

            # 构建让 Qwen 判断等价性的 Prompt
            judge_prompt = f"""你是一个专业的手机 UI 自动化操作评估专家。

用户当前任务：{instruction}
历史操作步骤：{history}

【真实标签（标准答案）】
真实动作类型：{gt_action_type}
动作详细参数：{gt_params}
人工动作描述：{gt_action_text}

【模型预测】
预测动作类型：{model_action_type}
预测详细参数：{model_args}
预测思考过程：{model_action_desc}

请结合截图内容判断：模型预测的动作是否与真实标签等价？或者在当前上下文中是否也是合法且能推进任务的？（例如：真实答案是click，但屏幕可能在加载，模型输出 wait 也是可以接受的；或者 long_press 达到了同样的效果等）。
请在回答的第一行严格输出 "【等价】" 或 "【不等价】"，然后在第二行简要说明理由。"""

            # 转换本地图片为 base64
            b64_img = image_to_base64_data_uri(img_path)

            if b64_img:
                api_res = ask_for_qwen3VL(b64_img, judge_prompt)
                if api_res and 'choices' in api_res:
                    content = api_res['choices'][0]['message']['content']
                    api_judgement_detail = content

                    if "【等价】" in content:
                        print(f"✅ Qwen API 判定为【等价】!")
                        # wait: TM 和 EM 都由 API 判断
                        # system_button: 只有 EM 由 API 判断，TM 保持原判断
                        if tm_need_api:
                            tm_correct = True
                        em_correct = True
                    else:
                        print(f"❌ Qwen API 判定为【不等价】!")
                        # system_button 且 API 判断不等价时，TM 才设为 False
                        if model_action_type == "system_button":
                            tm_correct = False
                else:
                    print("⚠️ API 返回结果为空或解析失败，按错误处理。")
            else:
                print(f"⚠️ 找不到截图 {img_path}，无法调用 API。")

        # 统计最终修正后的结果
        if tm_correct:
            tm_correct_count += 1

        if em_correct:
            em_correct_count += 1
        else:
            # 记录并打印错误信息
            error_type = "动作类型错误 (TM Error)" if not tm_correct else "动作参数/坐标错误 (EM Error)"
            if api_judgement_detail:
                error_type += " [API 已驳回]"

            error_info = {
                "episode_id": ep_id,
                "step_id": step_id,
                "错误分类": error_type,
                "GT_Action": gt_action_type,
                "GT_Params": gt_params,
                "GT_Description": gt_action_text,
                "Model_Action": model_action_type,
                "Model_Params": str(model_args),
                "Model_Description": model_action_desc,
                "API_Judge_Detail": api_judgement_detail  # 新增 API 意见字段
            }
            error_cases.append(error_info)

            if error_print_count < print_limit:
                print(f"❌ [错误] Ep: {ep_id} | Step: {step_id} | {error_type}")
                print(f"   [真实 GT] 动作: {gt_action_type:<12} | 参数: {gt_params}")
                print(f"             描述: {gt_action_text}")
                print(f"   [模型输出] 动作: {model_action_type:<12} | 参数: {model_args}")
                if api_judgement_detail:
                    print(f"   [API意见] {api_judgement_detail.replace(chr(10), ' ')}")
                print("-" * 60)
                error_print_count += 1
            elif error_print_count == print_limit:
                print(f"⚠️ 错误太多，已折叠控制台输出（共打印前 {print_limit} 条）。完整错误请查看导出的 CSV 文件。\n")
                error_print_count += 1

        # 思考偏差占比计算 (修正后)
        if not em_correct:
            tdfr_denominator += 1
            if calculate_text_similarity(model_action_desc, gt_action_text) < 0.5:
                tdfr_fault_count += 1

        # 记录修正后轨迹结果
        episode_results[ep_id].append({
            'step_id': step_id,
            'em_correct': em_correct
        })

        # === 按 GT 动作类型统计 ===
        if tm_correct:
            gt_action_stats[gt_action_type]["tm_correct"] += 1
        if em_correct:
            gt_action_stats[gt_action_type]["em_correct"] += 1
        else:
            gt_action_stats[gt_action_type]["tdfr_denom"] += 1
            if calculate_text_similarity(model_action_desc, gt_action_text) < 0.5:
                gt_action_stats[gt_action_type]["tdfr_fault"] += 1

    # === 保存错误分析表格 ===
    if error_cases:
        error_df = pd.DataFrame(error_cases)
        error_csv_path = csv_path.replace(".csv", "_Error_Analysis.csv")
        error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📁 发现 {len(error_cases)} 个错误预测。详细错误数据已保存至: {error_csv_path}\n")

    # === 计算整体指标 ===
    total_episodes = len(episode_results)

    # 1. 包含 Qwen 修正的指标
    tm_rate = tm_correct_count / total_valid_steps if total_valid_steps > 0 else 0
    em_rate = em_correct_count / total_valid_steps if total_valid_steps > 0 else 0
    tdfr_rate = tdfr_fault_count / tdfr_denominator if tdfr_denominator > 0 else 0

    tsr_success_count = 0
    total_leml = 0

    for ep_id, steps in episode_results.items():
        steps = sorted(steps, key=lambda x: x['step_id'])
        leml = 0
        for step in steps:
            if step['em_correct']:
                leml += 1
            else:
                break
        total_leml += leml
        if leml == len(steps) and len(steps) > 0:
            tsr_success_count += 1

    avg_leml = total_leml / total_episodes if total_episodes > 0 else 0
    tsr_rate = tsr_success_count / total_episodes if total_episodes > 0 else 0

    # 2. 不包含 Qwen 修正的指标 (Raw)
    raw_tm_rate = raw_tm_correct_count / total_valid_steps if total_valid_steps > 0 else 0
    raw_em_rate = raw_em_correct_count / total_valid_steps if total_valid_steps > 0 else 0
    raw_tdfr_rate = raw_tdfr_fault_count / raw_tdfr_denominator if raw_tdfr_denominator > 0 else 0

    raw_tsr_success_count = 0
    raw_total_leml = 0

    for ep_id, steps in raw_episode_results.items():
        steps = sorted(steps, key=lambda x: x['step_id'])
        leml = 0
        for step in steps:
            if step['em_correct']:
                leml += 1
            else:
                break
        raw_total_leml += leml
        if leml == len(steps) and len(steps) > 0:
            raw_tsr_success_count += 1

    raw_avg_leml = raw_total_leml / total_episodes if total_episodes > 0 else 0
    raw_tsr_rate = raw_tsr_success_count / total_episodes if total_episodes > 0 else 0

    # === 打印输出 ===
    print("="*40)
    print("📊 大模型多模态评测指标计算结果")
    print("="*40)
    print(f"有效 Step 总数: {total_valid_steps}")
    print(f"有效轨迹 (Episode) 总数: {total_episodes}\n")

    print("【1. 不包含 Qwen 修正的原始指标】")
    print(f"1. 动作类型准确率 (TM): {raw_tm_rate:.2%} ({raw_tm_correct_count}/{total_valid_steps})")
    print(f"2. 动作内容准确率 (EM): {raw_em_rate:.2%} ({raw_em_correct_count}/{total_valid_steps})")
    print(f"3. 思考偏差占比 (TDFR): {raw_tdfr_rate:.2%} ({raw_tdfr_fault_count}/{raw_tdfr_denominator})")
    print(f"4. 平均连续动作准确步长 (LEML): {raw_avg_leml:.2f} 步")
    print(f"5. 静态执行成功率 (TSR): {raw_tsr_rate:.2%} ({raw_tsr_success_count}/{total_episodes})\n")

    print("【2. 包含 Qwen 动态评审修正后的最终指标】")
    print(f"1. 动作类型准确率 (TM): {tm_rate:.2%} ({tm_correct_count}/{total_valid_steps})")
    print(f"2. 动作内容准确率 (EM): {em_rate:.2%} ({em_correct_count}/{total_valid_steps})")
    print(f"3. 思考偏差占比 (TDFR): {tdfr_rate:.2%} ({tdfr_fault_count}/{tdfr_denominator})")
    print(f"4. 平均连续动作准确步长 (LEML): {avg_leml:.2f} 步")
    print(f"5. 静态执行成功率 (TSR): {tsr_rate:.2%} ({tsr_success_count}/{total_episodes})")

    # === 按 GT 动作类型分类统计 ===
    print("\n【3. 按 GT 动作类型分类指标】")
    action_type_names = {
        "click": "点击",
        "swipe": "滑动",
        "type": "输入",
        "system_button": "系统按钮",
        "terminate": "结束"
    }

    for gt_type in ["click", "swipe", "type", "system_button", "terminate"]:
        if gt_type not in gt_action_stats:
            continue
        stats = gt_action_stats[gt_type]
        total = stats["total"]
        if total == 0:
            continue

        tm_rate_gt = stats["tm_correct"] / total
        em_rate_gt = stats["em_correct"] / total
        tdfr_rate_gt = stats["tdfr_fault"] / stats["tdfr_denom"] if stats["tdfr_denom"] > 0 else 0

        action_name = action_type_names.get(gt_type, gt_type)
        print(f"  【{action_name}】总数: {total}")
        print(f"    TM: {tm_rate_gt:.2%} ({stats['tm_correct']}/{total})")
        print(f"    EM: {em_rate_gt:.2%} ({stats['em_correct']}/{total})")
        print(f"    TDFR: {tdfr_rate_gt:.2%} ({stats['tdfr_fault']}/{stats['tdfr_denom']})")

    print("="*40)

    # 顺便把原始指标也合并返回给外部调用者
    return {
        "TM": tm_rate,
        "EM": em_rate,
        "TDFR": tdfr_rate,
        "LEML": avg_leml,
        "TSR": tsr_rate,
        "Raw_TM": raw_tm_rate,
        "Raw_EM": raw_em_rate,
        "Raw_TDFR": raw_tdfr_rate,
        "Raw_LEML": raw_avg_leml,
        "Raw_TSR": raw_tsr_rate
    }

if __name__ == "__main__":
    csv_path = "./评测结果/test_多步轨迹推理结果-juanji-0302-30b.csv"
    evaluate_metrics(csv_path, print_limit=20)


# =============================================================================
# =================== 模型预测 → GT 动作空间 映射关系 ===================
# =============================================================================
# 本脚本用于评测 Midtrain 30B 模型，推理脚本 system prompt 规定的预测空间与 GT 的映射关系如下：
# 注意：Midtrain 30B 的 system prompt 与 8B 版本完全一致！
#
# 分类标准：
# - 直接映射：可以从模型输出和GT参数直接判断等价，无需API
# - API兜底：参数无法直接比较，需要调用 API 判断语义等价性
#
# ┌─────────────────┬─────────────────┬─────────────────────────────────────────┐
# │ 模型输出 (action) │ TM 映射         │ EM 映射                                 │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ click          │ 直接: click     │ 坐标 ∈ GT 目标框                        │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ long_press     │ 直接: → click   │ 坐标 ∈ GT 目标框                        │
# │                │ (位置正确→TM✓)  │ (位置错误→EM✗)                          │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ swipe          │ 直接: swipe     │ 方向一致                                │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ type           │ 直接: type      │ 文本相似度 > 0.5                        │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ system_button  │ 直接: action    │ API 判断 button 参数是否等价            │
# │  (button=Back) │ == "system_btn" │ (TM 先判断类型，API 纠正 TM/EM)         │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ wait           │ API 兜底判断    │ API 判断 wait 是否可接受                │
# │                │ (无等价GT)       │ (如：页面加载中，wait 合理)             │
# ├─────────────────┼─────────────────┼─────────────────────────────────────────┤
# │ terminate      │ 直接: terminate │ status == 'success'                    │
# └─────────────────┴─────────────────┴─────────────────────────────────────────┘
#
# 参数级别映射规则（用于 EM 判断）：
# ┌────────────────┬───────────────────────────────────────────────────────────┐
# │ 动作类型       │ 参数映射规则                                              │
# ├────────────────┼───────────────────────────────────────────────────────────┤
# │ click /       │ model coordinate ∈ GT ui_positions 目标框                 │
# │ long_press    │ 坐标归一化：model_x / 1000, model_y / 1000                │
# ├────────────────┼───────────────────────────────────────────────────────────┤
# │ swipe         │ 方向一致：(mdx * gdx) + (mdy * gdy) > 0                    │
# │                │ mdx, mdy = model_c2 - model_c1                           │
# │                │ gdx, gdy = lift_yx - touch_yx                            │
# ├────────────────┼───────────────────────────────────────────────────────────┤
# │ type          │ 文本相似度 > 0.5 (difflib.SequenceMatcher)                │
# ├────────────────┼───────────────────────────────────────────────────────────┤
# │ system_button │ API 兜底判断                                               │
# ├────────────────┼───────────────────────────────────────────────────────────┤
# │ terminate     │ model_args['status'] == 'success'                        │
# └────────────────┴───────────────────────────────────────────────────────────┘
#
# API 兜底策略说明：
# 需要 API 兜底的情况：
# 1. 模型输出 wait：无等价 GT，TM 和 EM 都需 API 判断
# 2. 模型输出 system_button：TM 直接判断 action 类型，EM 参数需 API 判断
#
# 兜底流程：
# 1. 检测到需要 API 兜底的动作类型
# 2. 调用 Qwen3-VL-235B-A22B API
# 3. 传入截图、任务指令、历史操作、GT 标签、模型预测
# 4. API 返回【等价】或【不等价】
# 5. wait: 等价则 TM=✓, EM=✓; 不等价则 TM=✗, EM=✗
# 6. system_button: 等价则 EM=✓（TM 保持原判断）; 不等价则 TM=✗, EM=✗
# =============================================================================