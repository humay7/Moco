#!/usr/bin/env python3
"""
random_search_launcher.py

Runs random search over hyperparameters for tsp_ddpg_train.py by launching many
independent training runs as subprocesses. Each run writes a trial_summary.json
(containing best_val_last_best_reward), and this launcher collects them into a
leaderboard.jsonl while tracking the best run (minimizing val_last_best_reward).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


try:
    from sklearn.model_selection import ParameterSampler
    from scipy.stats import loguniform, uniform, randint
except Exception:
    ParameterSampler = None
    loguniform = uniform = randint = None


TRAIN_SCRIPT_DEFAULT = "tsp_ddpg_train.py"


DEFAULT_FIXED_ARGS: Dict[str, Any] = {

    "problem_size": 100,
    "task_batch_size": 128,
    "parallel_tasks_train": 128,
    "parallel_tasks_val": 128,
    "mlflow_uri": "logs/mlruns",
    "experiment_name": "ddpg_tsp_randomsearch",

  
    "outer_train_steps": 3000,
    "val_steps": 200,
    "patience": 5,

    # default values (can be overridden or tuned)
    "warmup_steps": 200,
    "episodes_per_batch": 4,
    "max_length": 50,
    "k": 20,
    "top_k": 32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # required paths
    p.add_argument("--val_path", type=str, required=True, help="Path to validation dataset used by tsp_ddpg_train.py")
    p.add_argument("--model_save_path", type=str, required=True, help="Root directory where trial folders will be created")

    # optional: location of training script
    p.add_argument("--train_script", type=str, default=TRAIN_SCRIPT_DEFAULT, help="Path to tsp_ddpg_train.py")

    # fixed args overrides
    p.add_argument("--problem_size", type=int, default=DEFAULT_FIXED_ARGS["problem_size"])
    p.add_argument("--task_batch_size", type=int, default=DEFAULT_FIXED_ARGS["task_batch_size"])
    p.add_argument("--parallel_tasks_train", type=int, default=DEFAULT_FIXED_ARGS["parallel_tasks_train"])
    p.add_argument("--parallel_tasks_val", type=int, default=DEFAULT_FIXED_ARGS["parallel_tasks_val"])

    p.add_argument("--mlflow_uri", type=str, default=DEFAULT_FIXED_ARGS["mlflow_uri"])
    p.add_argument("--experiment_name", type=str, default=DEFAULT_FIXED_ARGS["experiment_name"])

    # tuning budget knobs
    p.add_argument("--outer_train_steps", type=int, default=DEFAULT_FIXED_ARGS["outer_train_steps"])
    p.add_argument("--val_steps", type=int, default=DEFAULT_FIXED_ARGS["val_steps"])
    p.add_argument("--patience", type=int, default=DEFAULT_FIXED_ARGS["patience"])

    # defaults (used if not tuning / if sampler happens to pick them, or as baseline)
    p.add_argument("--warmup_steps", type=int, default=DEFAULT_FIXED_ARGS["warmup_steps"])
    p.add_argument("--episodes_per_batch", type=int, default=DEFAULT_FIXED_ARGS["episodes_per_batch"])
    p.add_argument("--max_length", type=int, default=DEFAULT_FIXED_ARGS["max_length"])
    p.add_argument("--k", type=int, default=DEFAULT_FIXED_ARGS["k"])
    p.add_argument("--top_k", type=int, default=DEFAULT_FIXED_ARGS["top_k"])

    # search control
    p.add_argument("--n_trials", type=int, default=30, help="How many random trials to run")
    p.add_argument("--base_seed", type=int, default=12345, help="Base seed; trial seed = base_seed + trial_idx")
    p.add_argument("--sampler_seed", type=int, default=0, help="Seed for ParameterSampler (reproducible sampling)")

    # device / env
    p.add_argument("--gpu_id", type=str, default=None, help="If set, export CUDA_VISIBLE_DEVICES to this value for each run")
    p.add_argument("--extra_env_json", type=str, default=None, help="JSON dict of extra env vars for subprocesses")

    # behavior
    p.add_argument("--skip_existing", action="store_true", help="Skip trial if a trial_summary.json already exists")
    p.add_argument("--fail_fast", action="store_true", help="Stop immediately if any trial fails")

    return p.parse_args()


def ensure_deps():
    if ParameterSampler is None or loguniform is None:
        raise RuntimeError(
            "Missing dependencies. Please install scikit-learn and scipy:\n"
            "  pip install scikit-learn scipy\n"
        )


def build_search_space() -> Dict[str, Any]:
    """
    Define the random search space. Values are distributions consumed by ParameterSampler.

    """
    ensure_deps()
    space = {
        # learning rates (log-uniform)
        "actor_lr": loguniform(1e-5, 3e-3),
        "critic_lr": loguniform(1e-5, 3e-3),

        # target smoothing
        "tau": loguniform(1e-3, 5e-2),


        "gamma": uniform(loc=0.95, scale=0.05), 

       
        "action_scale": uniform(loc=10, scale=10),         

        # delayed policy updates
        "policy_frequency": randint(1, 7), 


        # warmup steps for critic-only updates
        "warmup_steps": randint(0, 100),

        # number of episodes collected per outer step
        "episodes_per_batch": randint(4, 64),  

        # inner rollout length / budget K
        "max_length": randint(2, 10),  

    }
    return space


def build_cmd(train_script: str, args_dict: Dict[str, Any]) -> list:
    cmd = [sys.executable, train_script]
    for k, v in args_dict.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def find_newest_subdir(parent: Path) -> Optional[Path]:
    if not parent.exists():
        return None
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def trial_already_done(trial_root: Path) -> bool:
    if not trial_root.exists():
        return False
    for p in trial_root.glob("*/trial_summary.json"):
        if p.is_file():
            return True
    return False


def read_trial_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "trial_summary.json"
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def run_one_trial(
    train_script: str,
    fixed_args: Dict[str, Any],
    sampled_params: Dict[str, Any],
    trial_idx: int,
    trial_root: Path,
    base_seed: int,
    gpu_id: Optional[str],
    extra_env: Optional[Dict[str, str]],
    skip_existing: bool,
    fail_fast: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    trial_id = f"trial_{trial_idx:04d}"
    seed = int(base_seed + trial_idx)

    trial_root.mkdir(parents=True, exist_ok=True)

    if skip_existing and trial_already_done(trial_root):
        print(f"[skip] {trial_id}: already has trial_summary.json under {trial_root}")
        run_dir = find_newest_subdir(trial_root)
        if run_dir is not None and (run_dir / "trial_summary.json").exists():
            summary = read_trial_summary(run_dir)
            score = float(summary["best_val_last_best_reward"])
            row = {
                "trial_id": trial_id,
                "seed": seed,
                "score": score,
                "run_dir": str(run_dir),
                "params": sampled_params,
                "status": "skipped_existing",
                "timestamp": time.time(),
            }
            return row, run_dir
        return None, None

    # sampled params override fixed args
    run_args: Dict[str, Any] = dict(fixed_args)
    run_args.update(sampled_params)


    if "top_k" in run_args:
        run_args["top_k"] = max(1, int(run_args["top_k"]))
    if "k" in run_args:
        run_args["k"] = max(1, int(run_args["k"]))
    if "max_length" in run_args:
        run_args["max_length"] = max(1, int(run_args["max_length"]))
    if "episodes_per_batch" in run_args:
        run_args["episodes_per_batch"] = max(1, int(run_args["episodes_per_batch"]))
    if "warmup_steps" in run_args:
        run_args["warmup_steps"] = max(0, int(run_args["warmup_steps"]))

    run_args.update({
        "seed": seed,
        "trial_id": trial_id,
        "write_summary": True,
        "model_save_path": str(trial_root),  # training script will append uuid
    })

    cmd = build_cmd(train_script, run_args)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    print("================================================================================")
    print(f"[run] {trial_id} seed={seed}")
    print("[params]", json.dumps(sampled_params, indent=2, sort_keys=True))
    print("[cmd]", " ".join(cmd))
    t0 = time.time()

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[fail] {trial_id} returncode={e.returncode}")
        if fail_fast:
            raise
        return {
            "trial_id": trial_id,
            "seed": seed,
            "score": None,
            "run_dir": None,
            "params": sampled_params,
            "status": f"failed_returncode_{e.returncode}",
            "timestamp": time.time(),
        }, None

    dt = time.time() - t0

    run_dir = find_newest_subdir(trial_root)
    if run_dir is None:
        msg = f"[fail] {trial_id}: could not find uuid run dir in {trial_root}"
        print(msg)
        if fail_fast:
            raise RuntimeError(msg)
        return {
            "trial_id": trial_id,
            "seed": seed,
            "score": None,
            "run_dir": None,
            "params": sampled_params,
            "status": "failed_missing_run_dir",
            "timestamp": time.time(),
        }, None

    summary_path = run_dir / "trial_summary.json"
    if not summary_path.exists():
        msg = f"[fail] {trial_id}: missing {summary_path}"
        print(msg)
        if fail_fast:
            raise RuntimeError(msg)
        return {
            "trial_id": trial_id,
            "seed": seed,
            "score": None,
            "run_dir": str(run_dir),
            "params": sampled_params,
            "status": "failed_missing_summary",
            "timestamp": time.time(),
        }, run_dir

    summary = read_trial_summary(run_dir)
    score = float(summary["best_val_last_best_reward"])

    row = {
        "trial_id": trial_id,
        "seed": seed,
        "score": score,
        "run_dir": str(run_dir),
        "params": sampled_params,
        "status": "ok",
        "wall_time_sec": dt,
        "timestamp": time.time(),
    }
    print(f"[done] {trial_id}: score(best_val_last_best_reward)={score:.6f}  time={dt/60:.1f}min")
    return row, run_dir


def main():
    args = parse_args()
    ensure_deps()

    base_out = Path(args.model_save_path).expanduser().resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    fixed_args = dict(DEFAULT_FIXED_ARGS)
    fixed_args.update({
        "problem_size": args.problem_size,
        "task_batch_size": args.task_batch_size,
        "parallel_tasks_train": args.parallel_tasks_train,
        "parallel_tasks_val": args.parallel_tasks_val,
        "val_path": args.val_path,
        "mlflow_uri": args.mlflow_uri,
        "experiment_name": args.experiment_name,
        "outer_train_steps": args.outer_train_steps,
        "val_steps": args.val_steps,
        "patience": args.patience,
        # baselines (may be overridden by sampler now)
        "warmup_steps": args.warmup_steps,
        "episodes_per_batch": args.episodes_per_batch,
        "max_length": args.max_length,
        "k": args.k,
        "top_k": args.top_k,
    })

    extra_env = None
    if args.extra_env_json:
        extra_env = json.loads(args.extra_env_json)
        if not isinstance(extra_env, dict):
            raise ValueError("--extra_env_json must be a JSON dict")

    space = build_search_space()
    sampler = ParameterSampler(space, n_iter=args.n_trials, random_state=args.sampler_seed)

    leaderboard_path = base_out / "leaderboard.jsonl"

    best_score = None
    best_row = None

    print("================================================================================")
    print("[config]")
    print(" train_script:", args.train_script)
    print(" base_out:", str(base_out))
    print(" n_trials:", args.n_trials)
    print(" fixed_args:", json.dumps(fixed_args, indent=2))
    print(" search_space keys:", list(space.keys()))
    print("================================================================================")

    for i, sampled_params in enumerate(sampler):
        trial_id = f"trial_{i:04d}"
        trial_root = base_out / trial_id

        row, _ = run_one_trial(
            train_script=args.train_script,
            fixed_args=fixed_args,
            sampled_params=sampled_params,
            trial_idx=i,
            trial_root=trial_root,
            base_seed=args.base_seed,
            gpu_id=args.gpu_id,
            extra_env=extra_env,
            skip_existing=args.skip_existing,
            fail_fast=args.fail_fast,
        )

        if row is None:
            continue

        append_jsonl(leaderboard_path, row)

        score = row.get("score", None)
        if score is not None:
            if best_score is None or float(score) < float(best_score):
                best_score = float(score)
                best_row = row
                print(f"[best] new best score={best_score:.6f} at {best_row['trial_id']}  dir={best_row['run_dir']}")

    print("================================================================================")
    print("[done]")
    print(" leaderboard:", str(leaderboard_path))
    if best_row is None:
        print(" best: None (no successful trials)")
    else:
        print(" best trial_id:", best_row["trial_id"])
        print(" best score:", best_row["score"])
        print(" best run_dir:", best_row["run_dir"])
        print(" best params:", json.dumps(best_row["params"], indent=2))
    print("================================================================================")


if __name__ == "__main__":
    main()
