import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from math import isclose


def run_once(script_path, data_root, repo_root, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    start = time.perf_counter()
    subprocess.run([sys.executable, script_path, data_root], check=True, cwd=repo_root)
    end = time.perf_counter()
    return end - start


def sort_subject_data(subject_data):
    return sorted(subject_data, key=lambda d: d["file"])


def compare_float_list(a_list, b_list, rel_tol=1e-9, abs_tol=0.0):
    if len(a_list) != len(b_list):
        return False
    return all(isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(a_list, b_list))


def compare_dataset_info(base_info, optim_info):
    if base_info == optim_info:
        return True, ""
    messages = []
    for key in ["sampling_rate", "ch_names"]:
        if base_info.get(key) != optim_info.get(key):
            messages.append(f"{key} mismatch")
    for key in ["min", "max"]:
        if base_info.get(key) != optim_info.get(key):
            if isclose(base_info.get(key), optim_info.get(key), rel_tol=1e-9, abs_tol=0.0):
                messages.append(f"{key} mismatch (close)")
            else:
                messages.append(f"{key} mismatch")
    for key in ["mean", "std"]:
        if base_info.get(key) != optim_info.get(key):
            if compare_float_list(base_info.get(key, []), optim_info.get(key, [])):
                messages.append(f"{key} mismatch (close)")
            else:
                messages.append(f"{key} mismatch")
    if not messages:
        messages.append("dataset_info mismatch")
    return False, "; ".join(messages)


def compare_jsons(base_path, optim_path):
    with open(base_path, "r") as f:
        base_data = json.load(f)
    with open(optim_path, "r") as f:
        optim_data = json.load(f)

    base_info = base_data.get("dataset_info", {})
    optim_info = optim_data.get("dataset_info", {})
    info_match, info_message = compare_dataset_info(base_info, optim_info)

    base_subject_data = sort_subject_data(base_data.get("subject_data", []))
    optim_subject_data = sort_subject_data(optim_data.get("subject_data", []))
    data_match = base_subject_data == optim_subject_data

    message_parts = []
    if not info_match:
        message_parts.append(info_message)
    if not data_match:
        message_parts.append("subject_data mismatch")

    return info_match and data_match, "; ".join(message_parts)


def main():
    parser = argparse.ArgumentParser(description="Benchmark EEGMAT cross_json scripts.")
    parser.add_argument("--data-root", required=True, help="Path to the data root (contains EEGMAT/processed_data).")
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["baseline", "optim", "both"],
        default="both",
        help="Which script(s) to run.",
    )
    args = parser.parse_args()

    base_script = os.path.join(args.repo_root, "preprocessing", "EEGMAT", "cross_json_process.py")
    optim_script = os.path.join(args.repo_root, "preprocessing", "EEGMAT", "cross_json_process_optim.py")
    base_output = os.path.join(args.repo_root, "preprocessing", "EEGMAT", "cross_subject_json")
    optim_output = os.path.join(args.repo_root, "preprocessing", "EEGMAT", "cross_subject_json_optim")

    if not os.path.exists(base_script):
        raise FileNotFoundError(f"Missing baseline script: {base_script}")
    if not os.path.exists(optim_script):
        raise FileNotFoundError(f"Missing optimized script: {optim_script}")

    base_times = []
    if args.mode in ("baseline", "both"):
        print("Running baseline...")
        for i in range(args.runs):
            elapsed = run_once(base_script, args.data_root, args.repo_root, base_output)
            base_times.append(elapsed)
            print(f"  baseline run {i + 1}: {elapsed:.2f}s")

    optim_times = []
    if args.mode in ("optim", "both"):
        print("Running optimized...")
        for i in range(args.runs):
            elapsed = run_once(optim_script, args.data_root, args.repo_root, optim_output)
            optim_times.append(elapsed)
            print(f"  optim run {i + 1}: {elapsed:.2f}s")

    print("\nTiming summary (seconds):")
    if base_times:
        print(f"  baseline: {base_times}")
    if optim_times:
        print(f"  optim:    {optim_times}")

    if args.mode == "both":
        print("\nComparing outputs...")
        all_match = True
        for split in ["train.json", "val.json", "test.json"]:
            base_path = os.path.join(base_output, split)
            optim_path = os.path.join(optim_output, split)
            match, detail = compare_jsons(base_path, optim_path)
            status = "MATCH" if match else "MISMATCH"
            print(f"  {split}: {status}")
            if detail:
                print(f"    {detail}")
            all_match = all_match and match

        if all_match:
            print("\nAll outputs match.")
        else:
            print("\nOutputs differ. See mismatch details above.")
    else:
        print("\nComparison skipped (mode is not 'both').")


if __name__ == "__main__":
    main()
