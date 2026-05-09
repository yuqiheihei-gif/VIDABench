# -*- coding: utf-8 -*-

import os
import re
import json
import requests
import difflib
import pandas as pd
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageDraw

# ==============================================================================
# 🎯 统一配置区 (所有的路径、列名、参数都在这里修改)
# ==============================================================================

# ----------------- [全局通用配置] -----------------
MODEL_PRED_COL = 'juanji-0302-30b'
# 新增：指标结果保存路径
SUMMARY_TXT_PATH = './评测结果/单步导航评测指标汇总.txt' 


# ----------------- [1. 点击 (Click) 评测配置] -----------------
RUN_CLICK = True  
CLICK_CSV_PATH = './评测结果/业务单步导航-点击2-juanji-0302-30b.csv'
CLICK_GT_COL = '框选-PICTURE-undefined'  
CLICK_IMG_COL = 'service_image_url'      


# ----------------- [2. 输入 (Type) 评测配置] -----------------
RUN_TYPE = True   
TYPE_CSV_PATH = './单步导航评测集/业务单步导航输入_val.csv'
TYPE_GT_COL = 'input_text'               
TYPE_ERROR_SUFFIX = '_errors.csv'        
TYPE_FILTER_QUERY = '电子客票号码为9876543210987654'  


# ----------------- [3. 滑动 (Swipe) 评测配置] -----------------
RUN_SWIPE = True  
SWIPE_CSV_PATH = '/ossfs/node_61280994/workspace/lanxun/业务单步导航-滑动2-juanji-0302-30b.csv'
SWIPE_OUTPUT_FOLDER = './评测结果/业务单步导航-滑动2-juanji-0302-30b' 
SWIPE_GT_COL = '标注环节结果'            
SWIPE_IMG_COL = 'service_image_url'      


# ==============================================================================
# 核心逻辑代码
# ==============================================================================

def evaluate_click(file_path, col_name_pred, gt_col_name, img_col_name):
    print(f"\n>>> [1] 开始进行【点击 Click】任务评测: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return f"点击任务读取失败: {e}\n"

    action_accuracies, detail_accuracies, overall_accuracies, error_rows = [], [], [], []
    dir_name, base_name = os.path.split(file_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    error_img_dir = os.path.join(dir_name, f"{file_name_no_ext}_ErrorImages")
    if not os.path.exists(error_img_dir): os.makedirs(error_img_dir)

    total_samples = 0
    for index, row in df.iterrows():
        current_action_acc, current_detail_acc = 0, 0
        error_reason, img_width, img_height, scaled_x, scaled_y, img_obj = "", None, None, None, None, None
        try:
            if col_name_pred not in row: raise ValueError(f"列缺失: {col_name_pred}")
            img_path = str(row[img_col_name])
            if img_path.startswith(('http://', 'https://')):
                response = requests.get(img_path, timeout=10)
                img_obj = Image.open(BytesIO(response.content))
            else:
                img_obj = Image.open(img_path)
            img_width, img_height = img_obj.size
            
            model_content = row[col_name_pred]
            tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', str(model_content), re.DOTALL)
            total_samples += 1
            
            if tool_call_match:
                tool_call_str = tool_call_match.group(1).strip()
                if tool_call_str.startswith('```'): tool_call_str = tool_call_str.replace('```json', '').replace('```', '').strip()
                tool_call_json = json.loads(tool_call_str)
                
                pred_coords = tool_call_json.get('arguments', {}).get('coordinate')
                if pred_coords and len(pred_coords) == 2:
                    scaled_x = (pred_coords[0] / 1000.0) * img_width
                    scaled_y = (pred_coords[1] / 1000.0) * img_height

                if tool_call_json.get('arguments', {}).get('action') == 'click':
                    current_action_acc = 1
                    bbox_data = json.loads(row[gt_col_name]) if isinstance(row[gt_col_name], str) else row[gt_col_name]
                    pts = bbox_data['objects'][0]['polygon']['ptList']
                    x_coords, y_coords = [p['x'] for p in pts], [p['y'] for p in pts]
                    if scaled_x is not None and min(x_coords) <= scaled_x <= max(x_coords) and min(y_coords) <= scaled_y <= max(y_coords):
                        current_detail_acc = 1
                    else: error_reason = "坐标偏差"
                else: error_reason = "动作错误"
            else: error_reason = "未找到标签"
        except Exception as e: error_reason = f"异常: {str(e)}"
        
        action_accuracies.append(current_action_acc)
        detail_accuracies.append(current_detail_acc)
        is_overall_correct = (current_action_acc == 1 and current_detail_acc == 1)
        overall_accuracies.append(1 if is_overall_correct else 0)

        if not is_overall_correct:
            error_row = row.copy()
            error_row['错误原因_ErrorReason'] = error_reason
            error_rows.append(error_row)

    f_action = sum(action_accuracies) / total_samples if total_samples > 0 else 0
    f_detail = sum(detail_accuracies) / sum(action_accuracies) if sum(action_accuracies) > 0 else 0
    f_overall = sum(overall_accuracies) / total_samples if total_samples > 0 else 0
    
    if len(error_rows) > 0:
        pd.DataFrame(error_rows).to_csv(os.path.join(dir_name, f"{file_name_no_ext}_errors.csv"), index=False, encoding='utf-8-sig')

    res_str = (f"\n=== 点击任务 (Click) 统计结果 ===\n"
               f"文件路径: {file_path}\n"
               f"有效样本数: {total_samples}\n"
               f"动作准确率: {f_action:.2%}\n"
               f"动作细节准确率: {f_detail:.2%}\n"
               f"总体准确率: {f_overall:.2%}\n")
    print(res_str)
    return res_str

def evaluate_type(file_path, col_name_pred, gt_col_name, error_suffix, filter_query):
    print(f"\n>>> [2] 开始进行【输入 Type】任务评测: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if filter_query: df = df[df['query'] != filter_query].copy()
            
        def _calc_action(row):
            try:
                match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', str(row.get(col_name_pred)), re.DOTALL)
                if match and json.loads(match.group(1)).get('arguments', {}).get('action') == 'type': return 1
            except: pass
            return 0

        def _calc_detail(row):
            try:
                match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', str(row.get(col_name_pred)), re.DOTALL)
                if match:
                    pred_txt = json.loads(match.group(1)).get('arguments', {}).get('text')
                    sim = difflib.SequenceMatcher(None, str(row.get(gt_col_name)).strip(), str(pred_txt).strip()).ratio()
                    return 1 if sim >= 0.5 else 0
            except: pass
            return 0

        df['动作准确率'] = df.apply(_calc_action, axis=1)
        df['动作细节准确率'] = df.apply(lambda r: _calc_detail(r) if r['动作准确率']==1 else 0, axis=1)
        df['整体准确率'] = df.apply(lambda r: 1 if r['动作准确率']==1 and r['动作细节准确率']==1 else 0, axis=1)
        
        num_detail, num_action = df['动作细节准确率'].sum(), df['动作准确率'].sum()
        
        res_str = (f"\n=== 输入任务 (Type) 统计结果 ===\n"
                   f"文件路径: {file_path}\n"
                   f"总样本数: {len(df)}\n"
                   f"动作准确率: {df['动作准确率'].mean():.2%} ({num_action}/{len(df)})\n"
                   f"动作细节准确率: {num_detail/num_action if num_action>0 else 0:.2%} ({num_detail}/{num_action})\n"
                   f"整体准确率: {df['整体准确率'].mean():.2%} ({df['整体准确率'].sum()}/{len(df)})\n")
        print(res_str)
        
        error_df = df[df['整体准确率'] == 0].copy()
        if not error_df.empty:
            output_path = os.path.join(os.path.split(file_path)[0], os.path.splitext(os.path.split(file_path)[1])[0] + error_suffix)
            error_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return res_str
    except Exception as e: 
        return f"输入任务处理失败: {e}\n"

def evaluate_swipe(csv_path, output_dir, col_name_pred, gt_col_name):
    print(f"\n>>> [3] 开始进行【滑动 Swipe】任务评测: {csv_path}")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    try:
        df = pd.read_csv(csv_path)
        def _get_dir(start, end):
            dx, dy = end[0] - start[0], end[1] - start[1]
            return ("向右" if dx > 0 else "向左") if abs(dx) > abs(dy) else ("向下" if dy > 0 else "向上")

        def _parse_gt(gt_json_str):
            areas = []
            try:
                gt_list = json.loads(gt_json_str)
                target = next((item for item in gt_list if "框选" in str(item.get("MarkTitle", ""))), None)
                mark_data = json.loads(target.get("MarkResult", "{}"))
                for obj in mark_data.get("objects", []):
                    areas.append({"polygon": np.array([[p["x"], p["y"]] for p in obj.get("polygon", {}).get("ptList", [])], dtype=np.int32), 
                                  "direction": obj.get("name", "") or obj.get("result", {}).get("框所对应的滑动方向", ""),
                                  "ref_width": mark_data.get("width", 0), "ref_height": mark_data.get("height", 0)})
            except: pass
            return areas

        total_samples = gt_swipe_count = model_swipe_count = correct_action_count = perfect_swipe_count = 0
        for index, row in df.iterrows():
            total_samples += 1
            gt_areas = _parse_gt(str(row.get(gt_col_name, '')))
            if len(gt_areas) > 0: gt_swipe_count += 1
            
            match = re.search(r"<tool_call>(.*?)</tool_call>", str(row.get(col_name_pred, '')), re.DOTALL)
            if match:
                try:
                    args = json.loads(match.group(1).strip()).get("arguments", {})
                    if args.get("action") == "swipe":
                        model_swipe_count += 1
                        if len(gt_areas) > 0:
                            correct_action_count += 1
                            s_pt, e_pt = args.get("coordinate"), args.get("coordinate2")
                            m_dir = _get_dir(s_pt, e_pt)
                            for area in gt_areas:
                                abs_x, abs_y = (s_pt[0]/1000.0)*area['ref_width'], (s_pt[1]/1000.0)*area['ref_height']
                                if m_dir == area['direction'] and cv2.pointPolygonTest(area['polygon'], (abs_x, abs_y), False) >= 0:
                                    perfect_swipe_count += 1
                                    break
                except: pass

        res_str = (f"\n=== 滑动任务 (Swipe) 统计结果 ===\n"
                   f"文件路径: {csv_path}\n"
                   f"总样本: {total_samples} | GT需滑动: {gt_swipe_count} | 预测滑动: {model_swipe_count}\n"
                   f"指标1 滑动动作准确率: {correct_action_count/gt_swipe_count*100 if gt_swipe_count else 0:.2f}%\n"
                   f"指标2 细节准确率(动作对前提下): {perfect_swipe_count/correct_action_count*100 if correct_action_count else 0:.2f}%\n"
                   f"指标3 总体滑动准确率: {perfect_swipe_count/gt_swipe_count*100 if gt_swipe_count else 0:.2f}%\n")
        print(res_str)
        return res_str
    except Exception as e:
        return f"滑动任务处理失败: {e}\n"


# ==============================================================================
# 🚀 主程序执行入口
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("                 🚀 单步导航评测脚本启动 🚀")
    print("=" * 60)

    all_summaries = []
    all_summaries.append("="*30 + " 评测报告 " + "="*30 + "\n")

    # 1. 执行点击评测
    if RUN_CLICK:
        res = evaluate_click(CLICK_CSV_PATH, MODEL_PRED_COL, CLICK_GT_COL, CLICK_IMG_COL)
        all_summaries.append(res)
        
    # 2. 执行输入评测
    if RUN_TYPE:
        res = evaluate_type(TYPE_CSV_PATH, MODEL_PRED_COL, TYPE_GT_COL, TYPE_ERROR_SUFFIX, TYPE_FILTER_QUERY)
        all_summaries.append(res)

    # 3. 执行滑动评测
    if RUN_SWIPE:
        res = evaluate_swipe(SWIPE_CSV_PATH, SWIPE_OUTPUT_FOLDER, MODEL_PRED_COL, SWIPE_GT_COL)
        all_summaries.append(res)

    # 保存到 TXT 文件
    try:
        output_dir = os.path.dirname(SUMMARY_TXT_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(SUMMARY_TXT_PATH, 'w', encoding='utf-8') as f:
            f.writelines(all_summaries)
        print(f"\n✅ 所有指标结果已成功保存至: {SUMMARY_TXT_PATH}")
    except Exception as e:
        print(f"\n❌ 保存TXT文件失败: {e}")

    print("\n✅ 所有勾选的评测任务均已运行完毕！")
