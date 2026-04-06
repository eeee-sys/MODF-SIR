import argparse
import json
import os
import re
from collections import Counter, defaultdict


def extract_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_number(num_str):
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    epsilon = 1e-8
    rel_error = abs(float(pred) - float(target)) / (abs(float(target)) + epsilon)
    thresholds = []
    current = start
    while current <= end + interval / 2:
        thresholds.append(current)
        current += interval
    conditions = [rel_error < (1 - threshold) for threshold in thresholds]
    return sum(float(item) for item in conditions) / len(conditions)


def emer_ov_mc(reference, hypothesis):
    list_a = reference.split(",")
    list_b = hypothesis.split(",")
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def judge(reference, hypothesis):
    if "yes" in reference.lower() and "yes" in hypothesis.lower():
        return 1
    if "no" in reference.lower() and "no" in hypothesis.lower():
        return 1
    return 0


def reward_fn(output_ans, gt_ans, question_type):
    try:
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        if question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        if question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return mean_relative_accuracy(out_number, gt_number)
        if question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)
        if question_type == "judge":
            return judge(output_ans, gt_ans)
        return 0.0
    except Exception:
        return 0.0


def get_sample_id(sample):
    return f"{sample.get('Question', '')}||{sample.get('video_id', '')}"


def load_ground_truth(gt_file):
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_by_id = {}
    duplicate_ids = []
    for sample in data:
        sample_id = get_sample_id(sample)
        if sample_id in gt_by_id:
            duplicate_ids.append(sample_id)
        gt_by_id[sample_id] = sample
    return gt_by_id, duplicate_ids, len(data)


def load_predictions(pred_file):
    pred_by_id = {}
    order = []
    duplicate_ids = []

    with open(pred_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("id", "")
            if not sample_id:
                continue
            if sample_id in pred_by_id:
                duplicate_ids.append(sample_id)
            else:
                order.append(sample_id)
            pred_by_id[sample_id] = {
                "record": record,
                "line_idx": line_idx,
            }

    return pred_by_id, order, duplicate_ids


def evaluate_predictions(gt_by_id, pred_by_id, pred_order, raw_gt_count):
    final_output = []
    reward_sum = 0.0
    type_stats = defaultdict(lambda: {"reward_sum": 0.0, "count": 0})
    duration_stats = defaultdict(lambda: {"reward_sum": 0.0, "count": 0})
    unmatched_predictions = []

    for sample_id in pred_order:
        pred_item = pred_by_id[sample_id]["record"]
        gt_item = gt_by_id.get(sample_id)
        if gt_item is None:
            unmatched_predictions.append(
                {
                    "id": sample_id,
                    "line_idx": pred_by_id[sample_id]["line_idx"],
                }
            )
            continue

        gt_answer = gt_item.get("Answer", "")
        sample_type = gt_item.get("Type", "Unknown")
        video_duration = gt_item.get("video_duration", "Unknown")

        best_reward = 0.0
        final_pred_answer = ""
        
        history = pred_item.get("history", [])
        
        # 判断是否存在 history 且不为空
        if history and isinstance(history, list):
            for hist_item in history:
                if not isinstance(hist_item, dict):
                    continue
                
                raw_ans = hist_item.get("answer", "")
                pred_answer = extract_answer(raw_ans)
                if pred_answer == "":
                    pred_answer = raw_ans
                
                # Daily-omni 主要是 multiple choice 类型，传入对应 type 评判
                reward = reward_fn(pred_answer, gt_answer, "multiple choice")
                
                # 更新最高分
                if reward > best_reward:
                    best_reward = reward
                    final_pred_answer = pred_answer
                
                # 如果某一轮得分为 1.0 (完全答对)，则视为正确并退出当前样本的历史遍历
                if best_reward == 1.0:
                    break
        else:
            # 如果不存在 history，回退使用 best_answer 字段
            raw_prediction = pred_item.get("best_answer", "")
            pred_answer = extract_answer(raw_prediction)
            if pred_answer == "":
                pred_answer = raw_prediction
            
            best_reward = reward_fn(pred_answer, gt_answer, "multiple choice")
            final_pred_answer = pred_answer

        result = {
            "id": sample_id,
            "Question": gt_item.get("Question", pred_item.get("Question", "")),
            "video_id": gt_item.get("video_id", pred_item.get("video_id", "")),
            "Type": sample_type,
            "video_duration": video_duration,
            "gt_answer": gt_answer,
            "pred_answer": final_pred_answer,
            "reward": best_reward,
            "best_score": pred_item.get("best_score"),
            "use_grounder": pred_item.get("use_grounder"),
            "grounded_span": pred_item.get("grounded_span"),
            "raw_best_answer": pred_item.get("best_answer", ""),
        }
        final_output.append(result)
        reward_sum += best_reward
        type_stats[sample_type]["reward_sum"] += best_reward
        type_stats[sample_type]["count"] += 1
        duration_stats[video_duration]["reward_sum"] += best_reward
        duration_stats[video_duration]["count"] += 1

    final_acc = {"mean_acc": 0.0}
    if final_output:
        final_acc["mean_acc"] = float(reward_sum) / len(final_output)

    type_acc = {}
    for sample_type in sorted(type_stats.keys()):
        count = type_stats[sample_type]["count"]
        reward_total = type_stats[sample_type]["reward_sum"]
        type_acc[sample_type] = {
            "mean_acc": float(reward_total) / count if count else 0.0,
            "count": count,
        }

    duration_acc = {}
    for duration_name in sorted(duration_stats.keys()):
        count = duration_stats[duration_name]["count"]
        reward_total = duration_stats[duration_name]["reward_sum"]
        duration_acc[duration_name] = {
            "mean_acc": float(reward_total) / count if count else 0.0,
            "count": count,
        }

    matched_ids = {item["id"] for item in final_output}
    missing_predictions = []
    for sample_id, gt_item in gt_by_id.items():
        if sample_id not in matched_ids:
            missing_predictions.append(
                {
                    "id": sample_id,
                    "Question": gt_item.get("Question", ""),
                    "video_id": gt_item.get("video_id", ""),
                    "Type": gt_item.get("Type", "Unknown"),
                    "gt_answer": gt_item.get("Answer", ""),
                }
            )

    return {
        "results": final_output,
        "final_acc": [final_acc],
        "type_acc": type_acc,
        "duration_acc": duration_acc,
        "meta": {
            "raw_gt_count": raw_gt_count,
            "unique_gt_count": len(gt_by_id),
            "total_gt": len(gt_by_id),
            "total_predictions": len(pred_by_id),
            "matched_predictions": len(final_output),
            "unmatched_prediction_count": len(unmatched_predictions),
            "missing_prediction_count": len(missing_predictions),
            "unmatched_predictions": unmatched_predictions,
            "missing_predictions": missing_predictions,
        },
    }


def build_meta_summary(gt_duplicates, pred_duplicates, gt_by_id, pred_by_id):
    gt_type_counter = Counter(sample.get("Type", "Unknown") for sample in gt_by_id.values())
    gt_duration_counter = Counter(sample.get("video_duration", "Unknown") for sample in gt_by_id.values())
    return {
        "gt_duplicate_id_count": len(gt_duplicates),
        "gt_duplicate_ids": gt_duplicates,
        "prediction_duplicate_id_count": len(pred_duplicates),
        "prediction_duplicate_ids": pred_duplicates,
        "gt_type_distribution": dict(sorted(gt_type_counter.items())),
        "gt_video_duration_distribution": dict(sorted(gt_duration_counter.items())),
        "prediction_unique_id_count": len(pred_by_id),
    }


def parse_args():
    default_output = "eval_results/eval_results.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        default="path/to/your/pred/file",
    )
    parser.add_argument("--gt_file", default="Daily-Omni/qa.json")
    parser.add_argument("--output_file", default=default_output)
    return parser.parse_args()


def main():
    args = parse_args()

    gt_by_id, gt_duplicates, raw_gt_count = load_ground_truth(args.gt_file)
    pred_by_id, pred_order, pred_duplicates = load_predictions(args.pred_file)
    evaluation = evaluate_predictions(gt_by_id, pred_by_id, pred_order, raw_gt_count)
    evaluation["meta"].update(build_meta_summary(gt_duplicates, pred_duplicates, gt_by_id, pred_by_id))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation to: {args.output_file}")
    print(f"Matched predictions: {evaluation['meta']['matched_predictions']}")
    print(f"Missing predictions: {evaluation['meta']['missing_prediction_count']}")
    print(f"Unmatched predictions: {evaluation['meta']['unmatched_prediction_count']}")
    print(f"Final mean_acc: {evaluation['final_acc'][0]['mean_acc']:.6f}")
    for sample_type, stats in evaluation["type_acc"].items():
        print(f"{sample_type}: mean_acc={stats['mean_acc']:.6f}, count={stats['count']}")
    for duration_name, stats in evaluation["duration_acc"].items():
        print(f"{duration_name}: mean_acc={stats['mean_acc']:.6f}, count={stats['count']}")


if __name__ == "__main__":
    main()