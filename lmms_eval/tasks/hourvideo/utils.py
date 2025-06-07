import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

VID_SUBDIR_NAME = "/local_datasets/HourVideo/videos"


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def hourvideo_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []

    for i in range(5):
        candidate = doc.get(f"answer_{i+1}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}\n{post_prompt}"

def hourvideo_doc_to_frame(doc, lmms_eval_specific_kwargs):
    """
    Convert a document to a list of frames.
    This is used for frame selection tasks.
    """
    video_path = doc["video_path"]
    video_path = os.path.join(VID_SUBDIR_NAME, video_path)
    max_num_frames = lmms_eval_specific_kwargs.get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def hourvideo_doc_to_visual_v(doc):
    with open(Path(__file__).parent / "hourvideo_val.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    video_path = os.path.join(VID_SUBDIR_NAME, f'{doc["video_uid"]}.mp4')
    return [video_path]


def hourvideo_doc_to_visual_i(doc):
    with open(Path(__file__).parent / "hourvideo_val.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    video_path = doc["video_path"]
    video_path = os.path.join(VID_SUBDIR_NAME, video_path)
    max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original hourvideo paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def evaluate_hourvideo(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def hourvideo_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["qid"]
    hourvideo_acc = {"id": id, "question_category": doc["task"], "answer": doc["correct_answer_label"], "parsed_pred": parsed_pred}
    return {
        "hourvideo_acc": hourvideo_acc,
        "submission": {
            id: pred,
        },
    }


def hourvideo_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_hourvideo(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]


def hourvideo_aggregate_results_for_submission(results, args):
    path = generate_submission_file("hourvideo_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")
