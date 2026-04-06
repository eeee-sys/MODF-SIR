import argparse
import json
import os
import re
from collections import defaultdict


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


def stringify_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_answer_with_fallback(value):
    text = stringify_text(value).strip()
    if not text:
        return ""
    extracted = extract_answer(text)
    if extracted:
        return extracted.strip()
    return text


def load_predictions(pred_file):
    pred_by_id = {}
    order = []
    duplicate_ids = []
    malformed_lines = []
    missing_id_lines = []
    raw_line_count = 0
    blank_line_count = 0

    with open(pred_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw_line_count += 1
            line = line.strip()
            if not line:
                blank_line_count += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_lines.append(
                    {
                        "line_idx": line_idx,
                        "error": str(exc),
                    }
                )
                continue

            sample_id = stringify_text(record.get("id")).strip()
            if not sample_id:
                missing_id_lines.append({"line_idx": line_idx})
                continue

            if sample_id in pred_by_id:
                duplicate_ids.append(sample_id)
            else:
                order.append(sample_id)

            pred_by_id[sample_id] = {
                "record": record,
                "line_idx": line_idx,
            }

    return {
        "pred_by_id": pred_by_id,
        "order": order,
        "meta": {
            "raw_line_count": raw_line_count,
            "blank_line_count": blank_line_count,
            "prediction_unique_id_count": len(pred_by_id),
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_ids": duplicate_ids,
            "malformed_line_count": len(malformed_lines),
            "malformed_lines": malformed_lines,
            "missing_id_count": len(missing_id_lines),
            "missing_id_lines": missing_id_lines,
        },
    }


def build_skip_record(sample_id, line_idx, reason):
    return {
        "id": sample_id,
        "line_idx": line_idx,
        "reason": reason,
    }


def evaluate_single_prediction(raw_prediction, gt_answer, problem_type):
    if not stringify_text(raw_prediction).strip():
        return None, "missing_prediction"

    pred_answer = extract_answer_with_fallback(raw_prediction)
    if not pred_answer:
        return None, "empty_pred_answer_after_extraction"

    return {
        "pred_answer": pred_answer,
        "reward": reward_fn(pred_answer, gt_answer, problem_type),
        "raw_answer": raw_prediction,
    }, None


def evaluate_history_predictions(pred_item, gt_answer, problem_type):
    history = pred_item.get("history")
    total_iterations = len(history) if isinstance(history, list) else 0
    history_results = []
    evaluated_results = []
    ignored_history_entry_count = 0

    if history is None:
        history_state = "history_missing"
    elif not isinstance(history, list):
        history_state = "history_not_list"
    elif not history:
        history_state = "history_empty"
    else:
        history_state = "history_present"

    if not isinstance(history, list):
        return {
            "total_iterations": total_iterations,
            "history_state": history_state,
            "history_results": history_results,
            "evaluated_results": evaluated_results,
            "ignored_history_entry_count": ignored_history_entry_count,
        }

    for fallback_iter, history_item in enumerate(history, start=1):
        if not isinstance(history_item, dict):
            history_results.append(
                {
                    "iter": fallback_iter,
                    "pred_answer": "",
                    "reward": None,
                    "raw_answer": None,
                    "status": "ignored",
                    "reason": "history_item_not_dict",
                }
            )
            ignored_history_entry_count += 1
            continue

        iter_value = history_item.get("iter", fallback_iter)
        raw_answer = history_item.get("answer")
        entry = {
            "iter": iter_value,
            "raw_answer": raw_answer,
        }

        evaluated, error_reason = evaluate_single_prediction(raw_answer, gt_answer, problem_type)
        if error_reason is not None:
            entry.update(
                {
                    "pred_answer": "",
                    "reward": None,
                    "status": "ignored",
                    "reason": error_reason,
                }
            )
            history_results.append(entry)
            ignored_history_entry_count += 1
            continue

        entry.update(
            {
                "pred_answer": evaluated["pred_answer"],
                "reward": evaluated["reward"],
                "status": "evaluated",
            }
        )
        history_results.append(entry)
        evaluated_results.append(entry)

    if history_state == "history_present" and not evaluated_results:
        history_state = "history_no_usable_answers"

    return {
        "total_iterations": total_iterations,
        "history_state": history_state,
        "history_results": history_results,
        "evaluated_results": evaluated_results,
        "ignored_history_entry_count": ignored_history_entry_count,
    }


def select_best_result(results):
    best_result = None
    for result in results:
        if best_result is None or result["reward"] > best_result["reward"]:
            best_result = result
    return best_result


def evaluate_predictions(pred_by_id, pred_order):
    final_output = []
    reward_sum = 0.0
    domain_stats = defaultdict(lambda: {"reward_sum": 0.0, "count": 0})
    skipped_samples = []
    history_scored_sample_count = 0
    fallback_best_answer_sample_count = 0
    invalid_or_empty_history_sample_count = 0
    history_without_usable_answers_sample_count = 0
    ignored_history_entry_count = 0

    for sample_id in pred_order:
        pred_item = pred_by_id[sample_id]["record"]
        line_idx = pred_by_id[sample_id]["line_idx"]

        raw_solution = pred_item.get("solution")
        if not stringify_text(raw_solution).strip():
            skipped_samples.append(build_skip_record(sample_id, line_idx, "missing_solution"))
            continue

        gt_answer = extract_answer_with_fallback(raw_solution)
        if not gt_answer:
            skipped_samples.append(build_skip_record(sample_id, line_idx, "empty_gt_answer_after_extraction"))
            continue

        problem_type = pred_item.get("problem_type", "")
        domain = stringify_text(pred_item.get("domain")).strip() or "Unknown"
        raw_prediction = pred_item.get("best_answer")

        history_eval = evaluate_history_predictions(pred_item, gt_answer, problem_type)
        ignored_history_entry_count += history_eval["ignored_history_entry_count"]

        used_best_answer_fallback = False
        prediction_source = "history"
        matched_iters = []
        best_iter = None
        best_pred_answer = ""
        reward = 0.0

        if history_eval["evaluated_results"]:
            best_history_result = select_best_result(history_eval["evaluated_results"])
            reward = best_history_result["reward"]
            best_iter = best_history_result["iter"]
            best_pred_answer = best_history_result["pred_answer"]
            matched_iters = [
                item["iter"]
                for item in history_eval["evaluated_results"]
                if reward > 0 and item["reward"] == reward
            ]
            history_scored_sample_count += 1
        else:
            history_state = history_eval["history_state"]
            if history_state in {"history_missing", "history_not_list", "history_empty"}:
                invalid_or_empty_history_sample_count += 1
            elif history_state == "history_no_usable_answers":
                history_without_usable_answers_sample_count += 1

            fallback_result, fallback_error = evaluate_single_prediction(raw_prediction, gt_answer, problem_type)
            if fallback_error is not None:
                skipped_samples.append(
                    build_skip_record(
                        sample_id,
                        line_idx,
                        f"{history_state}_and_{fallback_error}",
                    )
                )
                continue

            used_best_answer_fallback = True
            prediction_source = "best_answer_fallback"
            reward = fallback_result["reward"]
            best_pred_answer = fallback_result["pred_answer"]
            fallback_best_answer_sample_count += 1

        result = {
            "id": sample_id,
            "domain": domain,
            "task_domain": pred_item.get("task_domain"),
            "task_type": pred_item.get("task_type"),
            "problem_type": problem_type,
            "gt_answer": gt_answer,
            "pred_answer": best_pred_answer,
            "reward": reward,
            "best_score": pred_item.get("best_score"),
            "raw_best_answer": raw_prediction,
            "prediction_source": prediction_source,
            "used_best_answer_fallback": used_best_answer_fallback,
            "total_iterations": history_eval["total_iterations"],
            "history_state": history_eval["history_state"],
            "history_evaluated_count": len(history_eval["evaluated_results"]),
            "matched_iters": matched_iters,
            "best_iter": best_iter,
            "best_pred_answer": best_pred_answer,
            "history_results": history_eval["history_results"],
        }
        final_output.append(result)
        reward_sum += reward
        domain_stats[domain]["reward_sum"] += reward
        domain_stats[domain]["count"] += 1

    final_acc = {"mean_acc": 0.0}
    if final_output:
        final_acc["mean_acc"] = float(reward_sum) / len(final_output)

    domain_acc = {}
    for domain in sorted(domain_stats.keys()):
        count = domain_stats[domain]["count"]
        reward_total = domain_stats[domain]["reward_sum"]
        domain_acc[domain] = {
            "mean_acc": float(reward_total) / count if count else 0.0,
            "count": count,
        }

    return {
        "results": final_output,
        "final_acc": [final_acc],
        "domain_acc": domain_acc,
        "meta": {
            "total_predictions": len(pred_by_id),
            "valid_sample_count": len(final_output),
            "skipped_sample_count": len(skipped_samples),
            "skipped_samples": skipped_samples,
            "history_scored_sample_count": history_scored_sample_count,
            "fallback_best_answer_sample_count": fallback_best_answer_sample_count,
            "invalid_or_empty_history_sample_count": invalid_or_empty_history_sample_count,
            "history_without_usable_answers_sample_count": history_without_usable_answers_sample_count,
            "ignored_history_entry_count": ignored_history_entry_count,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        default="/Users/mashang/党吉圣科研/HumanOmniV2/HumanOmniV2-main/src/open-r1-multimodal/eval_results/results_idea3_reviser_7b_worldsense.jsonl",
    )
    parser.add_argument(
        "--output_file",
        default="/Users/mashang/党吉圣科研/HumanOmniV2/HumanOmniV2-main/src/open-r1-multimodal/eval_results/eval_results_idea3_reviser_7b_worldsense_v2.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    loaded = load_predictions(args.pred_file)
    evaluation = evaluate_predictions(loaded["pred_by_id"], loaded["order"])
    evaluation["meta"].update(loaded["meta"])

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation to: {args.output_file}")
    print(f"Raw lines: {evaluation['meta']['raw_line_count']}")
    print(f"Unique predictions: {evaluation['meta']['prediction_unique_id_count']}")
    print(f"Valid samples: {evaluation['meta']['valid_sample_count']}")
    print(f"Skipped samples: {evaluation['meta']['skipped_sample_count']}")
    print(f"History-scored samples: {evaluation['meta']['history_scored_sample_count']}")
    print(f"Fallback best_answer samples: {evaluation['meta']['fallback_best_answer_sample_count']}")
    print(f"Invalid/empty history samples: {evaluation['meta']['invalid_or_empty_history_sample_count']}")
    print(
        "History without usable answers samples: "
        f"{evaluation['meta']['history_without_usable_answers_sample_count']}"
    )
    print(f"Ignored history entries: {evaluation['meta']['ignored_history_entry_count']}")
    print(f"Final mean_acc: {evaluation['final_acc'][0]['mean_acc']:.6f}")
    for domain, stats in evaluation["domain_acc"].items():
        print(f"{domain}: mean_acc={stats['mean_acc']:.6f}, count={stats['count']}")


if __name__ == "__main__":
    main()
