import json
import re
import torch

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()

def emer_ov_mc(reference, hypothesis):
    list_a = reference.split(",")
    list_b = hypothesis.split(",")
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

def judge(reference, hypothesis):
    if "yes" in reference.lower()  and "yes" in hypothesis.lower():
        return 1
    elif "no" in reference.lower()  and "no" in hypothesis.lower():
        return 1
    else:
        return 0

def reward_fn(output_ans, gt_ans, question_type):
    try:
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)
        elif  question_type == "judge":
            return judge(output_ans, gt_ans)
        else:
            return 0.0
    except Exception as e:
        return 0.0

# Load results with duplicate protection
results = []
seen_ids = set()
total_result_lines = 0
duplicate_results_skipped = 0
bad_result_lines = 0

with open("path/to/your/pred/file", 'r') as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        total_result_lines += 1

        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            bad_result_lines += 1
            print(f"[Bad Result JSON] line {line_no}: {exc}")
            continue

        result_id = result.get("id")
        if result_id in seen_ids:
            duplicate_results_skipped += 1
            print(f"[Duplicate Result] line {line_no}: id={result_id} skipped")
            continue

        seen_ids.add(result_id)
        results.append(result)

# Load ground truth
with open('IntentBench/qa.json', 'r') as f:
    qa_data = json.load(f)

# Build lookup dictionaries
qa_by_id = {}
for qa in qa_data:
    key = qa.get("qid") or qa.get("video")
    if key:
        qa_by_id[key] = qa

# Evaluate
total_reward = 0
total_samples = 0
type_reward = {}
type_count = {}
missing_ground_truth = 0

for result in results:
    qa = qa_by_id.get(result["id"])
    if not qa:
        missing_ground_truth += 1
        continue

    gt = extract_answer(qa["solution"])
    problem_type = qa["problem_type"]
    
    best_reward = 0.0
    history = result.get("history", [])
    
    if history and isinstance(history, list):
        for hist_item in history:
            if not isinstance(hist_item, dict):
                continue
                
            raw_ans = hist_item.get("answer", "")
            pred = extract_answer(raw_ans)
            if pred == "":
                pred = raw_ans
            
            reward = reward_fn(pred, gt, problem_type)
            if reward > best_reward:
                best_reward = reward
                
            if best_reward == 1.0:
                break
    else:
        pred = extract_answer(result.get("best_answer", ""))
        if pred == "":
            pred = result.get("best_answer", "")
        best_reward = reward_fn(pred, gt, problem_type)

    total_reward += best_reward
    total_samples += 1

    type_name = qa["Type"]
    type_reward[type_name] = type_reward.get(type_name, 0) + best_reward
    type_count[type_name] = type_count.get(type_name, 0) + 1

# Output
print(f"QA Total Samples: {len(qa_data)}")
print(f"Result File Lines: {total_result_lines}")
print(f"Bad Result Lines Skipped: {bad_result_lines}")
print(f"Unique Result IDs: {len(results)}")
print(f"Duplicate Result IDs Skipped: {duplicate_results_skipped}")
print(f"Matched Ground Truth Samples: {total_samples}")
print(f"Missing Ground Truth Skipped: {missing_ground_truth}")
print(f"Overall Accuracy Denominator: {total_samples}")

overall_acc = (total_reward / total_samples) * 100 if total_samples > 0 else 0.0
print(f"Overall Accuracy: {overall_acc:.2f}%")
for type_name in sorted(type_reward.keys()):
    acc = (type_reward[type_name] / type_count[type_name]) * 100
    print(f"{type_name}: {acc:.2f}% ({type_count[type_name]} samples)")