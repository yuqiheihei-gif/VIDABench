"""
    脚本二：模型推理 + 实时指标计算
    
    功能：读取 prepare_gt.py 预处理后的 _gt.csv 文件，
         逐行调用模型推理，同时实时计算 click_acc 指标，
         最终输出带有推理结果和评测指标的 CSV 与 Excel 汇总。

    数据流：
      prepare_gt.py 输出的 _gt.csv  →  本脚本  →  _result.csv + _metrics.xlsx
"""

import argparse
import json
import os
import re
import sys
import traceback
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import  AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# ================================
# 将当前目录加入路径，复用推理模块
# ================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


if torch.cuda.is_available():
    DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
else:
    DEVICES = ["cpu"]

print(DEVICES)
torch.set_num_threads(4)
USE_LOW_INSTRUCTION = False

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.append(current_dir)

_llm = None
_tokenizer = None

def move_to(device):
    global _llm, _tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"

def _init_model(model_name, IF_8B_4B=None):
    global _llm, _tokenizer

    _llm = AutoModelForImageTextToText.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )

    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)



import uuid
import mimetypes
import requests
from pathlib import Path

def save_img(url, save_dir):
    """
    主公请看：此函数将自动处理目录、识别后缀并保存图卷
    """
    # 1. 确保营寨（目录）已建成，parents=True 即使是多层目录也能一气呵成
    dest_path = Path(save_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        # 2. 深入敌阵获取图卷
        response = requests.get(url, timeout=10)
        # 兵贵神速，若状态码不对直接抛出异常，进入异常处理
        response.raise_for_status()
    except Exception as e:
        # 主公，若下载有失，臣建议返回 None 或抛出异常，以便主公定夺
        print(f"军情延误，下载失败: {e}")
        return None

    # 3. 审视图卷，定其名号
    content_type = response.headers.get('Content-Type', '')
    extension = mimetypes.guess_extension(content_type) or '.jpg'
    
    # 使用 uuid 确保名号唯一
    filename = f"{uuid.uuid4()}{extension}"
    file_full_path = dest_path / filename  # 路径拼接，既优雅又精准

    # 4. 录入库房
    file_full_path.write_bytes(response.content)

    # 5. 回报战果：返回完整的文件路径
    return str(file_full_path)
def get_qwen3_response_from_url(
    user_query: str,
    screenshot_url: str,
) -> tuple:
    """
    通过 HTTP URL 接收截图并调用 Qwen3-VL 模型生成响应
    
    返回: (model_output_json_str, status_code)
    """
    global _llm, _tokenizer
    
    try:
        image = save_img(screenshot_url, 'temp')


        prompt = f"Output the center point of the position corresponding to the following instruction: \n{user_query}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."  ## for point prediction, only output the coordinates in the format [x,y], and if infeasible, output [-1,-1]

            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process input
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_llm.device)
        
        # Generate
        generated_ids = _llm.generate(
            **inputs, 
            max_new_tokens=2048,
            temperature = 0,
            do_sample = False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Parse output
        raw_output = output_text[0]

        if os.path.exists(image):
            os.remove(image)

        return  raw_output

    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求异常: {str(e)}"}, 500
    except Exception as e:
        return {"error": f"处理异常: {str(e)}", "traceback": traceback.format_exc()}, 500


# ================================
# 指标计算工具函数
# ================================

def point_in_box(point: list, box: list) -> bool:
    """
    判断点是否落在矩形框内

    参数:
        point: [x, y] 预测点坐标
        box:   [x_min, y_min, x_max, y_max] 真值框坐标

    返回:
        bool: 点是否在框内
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def parse_point(text: str) -> list:
    """
    解析模型输出文本，提取归一化坐标点 [x, y]（范围 0~1）

    支持格式：
      - [x, y]           点坐标（0-1000 归一化）
      - [x1, y1, x2, y2] bbox 取中心点
      - [-1, -1]          不可行标记

    参数:
        text: 模型原始输出文本

    返回:
        [x, y] 归一化坐标（0~1），或 [-1, -1] 表示不可行，或 None 表示解析失败
    """
    pattern1 = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
    pattern2 = r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"
    pattern3 = r"\[\s*-?\d+\s*,\s*-?\d+\s*\],\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"

    text = text.strip()
    try:
        if re.fullmatch(pattern1, text, re.DOTALL):
            box = eval(text)
            point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        elif re.fullmatch(pattern2, text, re.DOTALL):
            point = eval(text)
        elif re.fullmatch(pattern3, text.replace(' ', ''), re.DOTALL):
            bbox = eval('[' + text + ']')
            point = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]
        else:
            # 兜底：尝试从文本中提取第一个 [x, y]
            point = list(map(int, text.split(']')[0].split('[')[1].split(',')))

        # 不可行标记
        if point == [-1, -1]:
            return [-1, -1]

        # 归一化到 0~1
        abs_x = float(point[0] / 1000)
        abs_y = float(point[1] / 1000)
        return [abs_x, abs_y]
    except Exception:
        return None


def decode_pred_point(pred_point: list, anno_width: float, anno_height: float) -> list:
    """
    将归一化预测点（0~1）转换为标注坐标系下的绝对坐标

    参数:
        pred_point:  [x, y] 归一化坐标（0~1）
        anno_width:  标注坐标系宽度
        anno_height: 标注坐标系高度

    返回:
        [x_abs, y_abs] 绝对坐标
    """
    return [pred_point[0] * anno_width, pred_point[1] * anno_height]


def clamp_point(point: list, width: float, height: float) -> tuple:
    """
    将点裁剪到 [0, width] x [0, height] 范围内

    返回:
        (clamped_point, is_out_of_bounds)
    """
    x, y = point
    clamped_x = max(0, min(x, width))
    clamped_y = max(0, min(y, height))
    is_oob = (x != clamped_x) or (y != clamped_y)
    return [clamped_x, clamped_y], is_oob


def evaluate_single_sample(
    pred_point: list,
    gt_boxes: list,
    anno_width: float,
    anno_height: float,
) -> dict:
    """
    对单个样本进行评测：判断预测点是否落在任一真值框内

    参数:
        pred_point:  [x, y] 归一化预测点（0~1）
        gt_boxes:    list[list[float]]，真值框列表
        anno_width:  标注坐标系宽度
        anno_height: 标注坐标系高度

    返回:
        dict: {
            "click_acc": bool,
            "pred_abs": [x, y],       # 标注坐标系下的绝对坐标
            "is_out_of_bounds": bool,
        }
    """
    # 将归一化坐标转为标注坐标系下的绝对坐标
    pred_abs = decode_pred_point(pred_point, anno_width, anno_height)

    # 裁剪越界点
    pred_abs, is_oob = clamp_point(pred_abs, anno_width, anno_height)

    # 判断点是否在任一真值框内
    click_correct = False
    for box in gt_boxes:
        if point_in_box(pred_abs, box):
            click_correct = True
            break

    return {
        "click_acc": click_correct,
        "pred_abs": pred_abs,
        "is_out_of_bounds": is_oob,
    }


# ================================
# 分类统计维度
# ================================
DIMENSIONS = [
    "页面复杂度", "目标元素类型", "query形式", "category_name_1",
    "定位类型", "页面元素大小", "页面中是否有目标元素？目前元素是否被遮挡？",
    "是否包含绝对位置的描述", "是否包含相对位置的描述",
    "是否包含形状的描述", "是否包含颜色的描述",
]


def compute_summary(df: pd.DataFrame, total_samples: int, failed_count: int, oob_count: int) -> tuple:
    """
    计算分类统计表与整体指标

    返回:
        (category_df, overall_df)
    """
    # 分类统计
    summary_list = []
    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        for name, group in df.groupby(dim):
            summary_list.append({
                "维度": dim,
                "类别": str(name),
                "样本数": len(group),
                "click_acc": round(group["click_acc"].mean(), 4),
            })
    category_df = pd.DataFrame(summary_list)

    # 整体指标
    overall_df = pd.DataFrame([{
        "metric": "整体准确率",
        "样本总数": total_samples,
        "click_acc": round(df["click_acc"].mean(), 4),
        "处理失败数": failed_count,
        "处理失败率": round(failed_count / max(total_samples, 1), 4),
        "越界数": oob_count,
        "越界率": round(oob_count / max(total_samples, 1), 4),
    }])

    return category_df, overall_df


# ================================
# 推理 + 评测主函数
# ================================

def infer_and_evaluate(
    input_gt_csv: str,
    output_result_csv: str,
    output_metrics_excel: str,
    model_path: str,
    url_column: str = "service_image",
    query_column: str = "修改instruction",
    target_type_column: str = "目标元素类型",
    v1: bool = True,
):
    """
    主函数：加载模型 → 逐行推理 → 实时评测 → 保存结果

    参数:
        input_gt_csv:          prepare_gt.py 输出的预处理 CSV
        output_result_csv:     推理结果 CSV 输出路径
        output_metrics_excel:  指标汇总 Excel 输出路径
        model_path:            模型路径
        url_column:            图片 URL 列名
        query_column:          用户指令列名
        target_type_column:    目标元素类型列名
        v1:                    是否为 v1 版本（非 text/icon 类型追加"图标"后缀）
    """
    # ---- 导入推理模块（延迟导入，避免无 GPU 时报错） ----
    # from infer_ui_venus import _init_model, move_to, get_qwen3_response_from_url, DEVICES

    # ---- 加载模型 ----
    print(f"🔧 加载模型: {model_path}")
    _init_model(model_path)
    move_to(DEVICES[0])
    print("✅ 模型加载完毕")

    # ---- 读取数据（支持断点续推） ----
    if os.path.exists(output_result_csv):
        print(f"🔄 检测到已有结果文件，启用断点续推: {output_result_csv}")
        df = pd.read_csv(output_result_csv)
    else:
        print(f"📖 读取预处理 GT 文件: {input_gt_csv}")
        df = pd.read_csv(input_gt_csv)

    # 确保必要列存在
    for col in ["model_grouding_ori", "model_grouding", "click_acc"]:
        if col not in df.columns:
            df[col] = "" if col != "click_acc" else False

    # 检查必要列
    required = [url_column, query_column, target_type_column, "gt_boxes_actual", "orig_width", "orig_height"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ CSV 中缺少列: {col}（请先运行 prepare_gt.py 进行预处理）")

    # 过滤有效定位类型
    if "定位类型" in df.columns:
        df_eval = df[df["定位类型"].isin(["无", "单目标定位"])].copy()
        print(f"📊 过滤后可评测样本数: {len(df_eval)} / {len(df)}")
    else:
        df_eval = df.copy()

    # ---- 统计变量 ----
    total_samples = len(df_eval)
    failed_count = 0
    oob_count = 0

    # ---- 逐行推理 + 评测 ----
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🚀Inference & Eval"):
        # 断点续推：若已有结果则跳过
        if pd.notna(row.get("model_grouding_ori")) and str(row["model_grouding_ori"]).strip() != "":
            continue

        user_query = str(row[query_column]).strip()
        image_url = str(row[url_column]).strip()
        target_type = str(row[target_type_column]).strip().lower()
        location_type = str(row.get("定位类型", "单目标定位")).strip()

        # ---- 1. 推理 ----
        if not user_query or not image_url or image_url.lower() == "nan":
            df.at[idx, "model_grouding_ori"] = json.dumps({"error": "Missing query or image URL"}, ensure_ascii=False)
            df.at[idx, "model_grouding"] = ""
            df.at[idx, "click_acc"] = False
            failed_count += 1
            continue

        # v1 模式下，非 text/icon 类型追加 "图标"
        query_for_model = user_query
        if target_type not in ["text", "icon"] and v1:
            query_for_model = f"{user_query} 图标"

        try:
            raw_output = get_qwen3_response_from_url(
                user_query=query_for_model,
                screenshot_url=image_url,
            )
            # 保存原始输出
            df.at[idx, "model_grouding_ori"] = (
                json.dumps(raw_output, ensure_ascii=False) if isinstance(raw_output, (dict, list)) else str(raw_output)
            )

            # 解析预测点
            pred_point = parse_point(str(raw_output))
            df.at[idx, "model_grouding"] = json.dumps(pred_point) if pred_point else ""

            print(f"[Row {idx}] Query: {query_for_model} | Output: {raw_output} | Point: {pred_point}")

        except Exception as e:
            error_info = {"error": str(e), "traceback": traceback.format_exc()}
            df.at[idx, "model_grouding_ori"] = json.dumps(error_info, ensure_ascii=False)
            df.at[idx, "model_grouding"] = ""
            pred_point = None
            failed_count += 1

        # ---- 2. 实时评测 ----
        if location_type == "无":
            # 定位类型为"无"：模型应输出空/不可行
            pred_str = str(df.at[idx, "model_grouding"]).strip()
            pred_is_empty = not pred_str or pred_str.lower() in ["nan", "null", "[]", "{}", ""]
            pred_is_infeasible = (pred_point == [-1, -1])
            df.at[idx, "click_acc"] = pred_is_empty or pred_is_infeasible
            print(f"[Row {idx}] 定位类型: 无 | 预测是否空/不可行: {df.at[idx, 'click_acc']}")
        elif pred_point is not None and pred_point != [-1, -1]:
            # 正常单目标定位
            gt_boxes_str = str(row["gt_boxes_actual"]).strip()
            anno_w = float(row["orig_width"])
            anno_h = float(row["orig_height"])

            if gt_boxes_str and gt_boxes_str.lower() not in ["nan", "null", ""]:
                try:
                    gt_boxes = json.loads(gt_boxes_str)
                    result = evaluate_single_sample(pred_point, gt_boxes, anno_w, anno_h)
                    
                    df.at[idx, "click_acc"] = result["click_acc"]
                    print(f'预测正确✅') if result["click_acc"] else print(f'预测错误❌')
                    if result["is_out_of_bounds"]:
                        oob_count += 1
                except Exception as e:
                    print(f"⚠️ 评测失败 Row {idx}: {e}")
                    df.at[idx, "click_acc"] = False
                    failed_count += 1
            else:
                df.at[idx, "click_acc"] = False
                print(f"⚠️ Row {idx} 真值框数据缺失或无效，无法评测")
                failed_count += 1
        else:
            df.at[idx, "click_acc"] = False
            print(f"⚠️ Row {idx} 预测点无效，无法评测")

        # 每 50 行保存一次，防止意外中断
        if idx % 50 == 0:
            df.to_csv(output_result_csv, index=False)

    # ---- 保存最终结果 CSV ----
    os.makedirs(os.path.dirname(output_result_csv) if os.path.dirname(output_result_csv) else ".", exist_ok=True)
    df.to_csv(output_result_csv, index=False)
    print(f"\n📁 推理结果已保存: {output_result_csv}")

    # ---- 计算汇总指标 ----
    # 仅对有效评测样本（定位类型为"无"或"单目标定位"）计算指标
    if "定位类型" in df.columns:
        df_metric = df[df["定位类型"].isin(["无", "单目标定位"])].copy()
    else:
        df_metric = df.copy()

    category_df, overall_df = compute_summary(df_metric, total_samples, failed_count, oob_count)

    # ---- 保存 Excel ----
    os.makedirs(os.path.dirname(output_metrics_excel) if os.path.dirname(output_metrics_excel) else ".", exist_ok=True)
    with pd.ExcelWriter(output_metrics_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_data", index=False)
        category_df.to_excel(writer, sheet_name="category_metrics", index=False)
        overall_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"📊 指标汇总已保存: {output_metrics_excel}")

    # ---- 输出统计 ----
    print(f"\n{'='*50}")
    print(f"📈 总体准确率: {df_metric['click_acc'].mean():.4f} ({df_metric['click_acc'].mean()*100:.2f}%)")
    print(f"📊 评测样本数: {total_samples}")
    print(f"❌ 处理失败数: {failed_count} ({failed_count / max(total_samples, 1):.2%})")
    print(f"⚠️ 点越界数:   {oob_count} ({oob_count / max(total_samples, 1):.2%})")
    if not category_df.empty:
        print(f"\n📋 分类统计预览：")
        print(category_df.round(4).to_string(index=False))
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理 + 实时指标计算")
    parser.add_argument("--input_gt_csv", type=str, default='./定位评测集/Alipay_Grounding_Valset_500.csv',
                        help="prepare_gt.py 输出的预处理 CSV 路径")
    parser.add_argument("--output_result_csv", type=str, default='./评测结果/results/500/Midtrain-30B-0224/inference_results.csv',
                        help="推理结果 CSV 输出路径")
    parser.add_argument("--output_metrics_excel", type=str, default=None,
                        help="指标汇总 Excel 路径（默认与 result_csv 同目录同名 .xlsx）")
    parser.add_argument("--model_path", type=str, default="/yuanxi_xui_agent_1/xui_mid_train/models/experiment/260224/checkpoint/30BA3B-v5-20260228-225812-HF",
                        help="模型路径")
    parser.add_argument("--url_column", type=str, default="service_image",
                        help="图片 URL 列名")
    parser.add_argument("--query_column", type=str, default="修改instruction",
                        help="用户指令列名")
    parser.add_argument("--target_type_column", type=str, default="目标元素类型",
                        help="目标元素类型列名")
    parser.add_argument("--v1", action="store_true", default=True,
                        help="是否为 v1 版本（非 text/icon 追加 '图标'）")

    args = parser.parse_args()

    # 若未指定 Excel 路径，自动生成
    if args.output_metrics_excel is None:
        base, _ = os.path.splitext(args.output_result_csv)
        args.output_metrics_excel = f"{base}_metrics.xlsx"
    
    # 若保存路径不存在，自动创建目录
    if not os.path.exists(os.path.dirname(args.output_result_csv)):
        os.makedirs(os.path.dirname(args.output_result_csv), exist_ok=True)
    if not os.path.exists(os.path.dirname(args.output_metrics_excel)):
        os.makedirs(os.path.dirname(args.output_metrics_excel), exist_ok=True)

    infer_and_evaluate(
        input_gt_csv=args.input_gt_csv,
        output_result_csv=args.output_result_csv,
        output_metrics_excel=args.output_metrics_excel,
        model_path=args.model_path,
        url_column=args.url_column,
        query_column=args.query_column,
        target_type_column=args.target_type_column,
        v1=args.v1,
    )
