"""
Two checks for the checkpointing of train.py.

1) resume-test: end-to-end proof that "interrupt + resume" == "never interrupted".
   Runs train.py several times on a tiny CPU problem and compares the final checkpoint.pt bit by bit:
     base : fresh run, 1 epoch                      (initial data, shared by all the runs below)
     A    : base -> max_epochs uninterrupted
     A2   : same as A                               (control: proves the comparison is deterministic)
     B    : base -> killed (hard kill) in the middle of an epoch -> relaunched -> max_epochs
   Expected: A == A2 and A == B.
   The local search is disabled (--always_search false --redeem_only false) because it reseeds numpy with
   np.random.seed(None) (OS entropy) and would make even two uninterrupted runs different.

       python tools/check_checkpoint.py resume-test

2) inspect: consistency check of a real run folder (e.g. one of your cluster runs):
   checkpoint.pt vs epoch.txt / temperature.txt / run_state.json / train_data.pkl / test_data.pkl / model.pt / optimizer.pt

       python tools/check_checkpoint.py inspect checkpoint/square_N33/<exp_id>
"""
import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # needed to unpickle the datasets (src.envs classes)


# ----------------------------------------------------------------------------- comparison helpers


def same(a, b):
    """Recursive bitwise equality for tensors, numpy arrays, dicts, lists/tuples and plain values."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def dataset_key(data):
    return [(getattr(d, "features", None), getattr(d, "score", None)) for d in data]


def load_ckpt(run_dir):
    return torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)


def compare_checkpoints(a, b):
    checks = {
        "n_epoch": same(a["n_epoch"], b["n_epoch"]),
        "seed": same(a["seed"], b["seed"]),
        "temperature": same(a["temperature"], b["temperature"]),
        "run_state (counters, best_loss)": same(a["run_state"], b["run_state"]),
        "model weights": same(a["model"], b["model"]),
        "optimizer state (AdamW moments)": same(a["optimizer"], b["optimizer"]),
        "train_set": dataset_key(a["train_set"]) == dataset_key(b["train_set"]),
        "test_set": dataset_key(a["test_set"]) == dataset_key(b["test_set"]),
    }
    for name in a["rng"]:
        checks[f"RNG state: {name}"] = name in b["rng"] and same(a["rng"][name], b["rng"][name])
    return checks


def read_metrics(run_dir, skip=("searches_per_second",)):
    # searches_per_second is a timing, it is never identical between two runs
    with open(os.path.join(run_dir, "metrics.csv"), newline="") as f:
        return [{k: v for k, v in row.items() if k not in skip} for row in csv.DictReader(f)]


def print_checks(title, checks):
    print(f"\n{title}")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'DIFF'}] {name}")
    return all(checks.values())


# ----------------------------------------------------------------------------- resume-test


def resume_test(opts):
    root = os.path.abspath(opts.workdir or tempfile.mkdtemp(prefix="ckpt_resume_test_"))
    os.makedirs(root, exist_ok=True)
    exp_name = "resume_test"
    exp_dir = os.path.join(root, exp_name)
    common = [
        "--cpu", "true", "--seed", str(opts.seed), "--env_name", "square", "--N", "10", "--max_len", "100",
        "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
        "--max_steps", str(opts.max_steps), "--num_eval_steps", "50", "--batch_size", "16",
        "--gensize", "300", "--gen_batch_size", "100", "--num_samples_from_model", "400",
        "--pop_size", "200", "--ntest", "20",
        "--always_search", "false", "--redeem_only", "false", "--process_pool", "false", "--num_workers", "1",
        "--dump_path", root, "--exp_name", exp_name,
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # same thread count -> same float reductions
    env.pop("MODAL_EXP_ID", None)

    def launch(exp_id, max_epochs):
        log = open(os.path.join(root, f"{exp_id}_stdout.log"), "a")
        cmd = [sys.executable, "train.py", *common, "--exp_id", exp_id, "--max_epochs", str(max_epochs)]
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log

    def run(exp_id, max_epochs):
        p, log = launch(exp_id, max_epochs)
        code = p.wait()
        log.close()
        if code != 0:
            sys.exit(f"run {exp_id} failed (exit {code}), see {os.path.join(root, exp_id + '_stdout.log')}")

    def run_and_kill(exp_id, max_epochs, kill_epoch, marker):
        """Hard-kill the run when train.log shows `marker` for epoch `kill_epoch` (a simulated crash)."""
        log_path = os.path.join(exp_dir, exp_id, "train.log")
        offset = os.path.getsize(log_path)  # ignore the lines copied from base
        target = f"Epoch {kill_epoch} {marker}"
        p, log = launch(exp_id, max_epochs)
        while p.poll() is None:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                if target in f.read():
                    p.kill()  # TerminateProcess on Windows / SIGKILL on Linux: no cleanup, like a real crash
                    break
            time.sleep(0.05)
        p.wait()
        log.close()
        if p.returncode == 0:
            sys.exit(f"run {exp_id} finished before it could be killed: raise --max_steps")
        print(f"  killed {exp_id} at '{target}' (epoch.txt = {open(os.path.join(exp_dir, exp_id, 'epoch.txt')).read().strip()})")

    E = opts.max_epochs
    print(f"working directory: {root}")
    print("1/4 base run (initial data + 1 epoch)")
    run("base", 1)
    for name in ("A", "A2", "B"):
        shutil.copytree(os.path.join(exp_dir, "base"), os.path.join(exp_dir, name))
    print(f"2/4 A  : uninterrupted up to epoch {E}")
    run("A", E)
    print(f"3/4 A2 : uninterrupted up to epoch {E} (control)")
    run("A2", E)
    print(f"4/4 B  : killed during epoch {E - 1}, then resumed up to epoch {E}")
    run_and_kill("B", E, kill_epoch=E - 1, marker=opts.kill_at)
    run("B", E)

    a, a2, b = load_ckpt(os.path.join(exp_dir, "A")), load_ckpt(os.path.join(exp_dir, "A2")), load_ckpt(os.path.join(exp_dir, "B"))
    ok_control = print_checks("Control A vs A2 (two uninterrupted runs)", compare_checkpoints(a, a2))
    checks = compare_checkpoints(a, b)
    checks["metrics.csv rows (except searches_per_second)"] = read_metrics(os.path.join(exp_dir, "A")) == read_metrics(os.path.join(exp_dir, "B"))
    ok_resume = print_checks("Resume A vs B (uninterrupted vs killed + resumed)", checks)

    print()
    if not ok_control:
        print("INCONCLUSIVE: two uninterrupted runs already differ, so the setup is not deterministic.")
        sys.exit(2)
    if ok_resume:
        print("PASSED: the resumed run is bit-identical to the uninterrupted one.")
        if not opts.workdir:
            shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"FAILED: the resumed run differs. Files kept in {root}")
        sys.exit(1)


# ----------------------------------------------------------------------------- inspect


def inspect(run_dir):
    ckpt = load_ckpt(run_dir)
    p = lambda name: os.path.join(run_dir, name)
    params = pickle.load(open(p("params.pkl"), "rb")) if os.path.isfile(p("params.pkl")) else None
    print(f"checkpoint.pt: epoch {ckpt['n_epoch']}, seed {ckpt['seed']}, RNG streams saved: {', '.join(ckpt['rng'])}")

    checks = {}
    if os.path.isfile(p("epoch.txt")):
        checks["epoch.txt"] = int(open(p("epoch.txt")).read()) == ckpt["n_epoch"]
    if os.path.isfile(p("temperature.txt")):
        checks["temperature.txt"] = float(open(p("temperature.txt")).read()) == ckpt["temperature"]
    if os.path.isfile(p("run_state.json")):
        checks["run_state.json"] = json.load(open(p("run_state.json"))) == ckpt["run_state"]
    if os.path.isfile(p("train_data.pkl")):
        checks["train_data.pkl"] = dataset_key(pickle.load(open(p("train_data.pkl"), "rb"))) == dataset_key(ckpt["train_set"])
        checks["test_data.pkl"] = dataset_key(pickle.load(open(p("test_data.pkl"), "rb"))) == dataset_key(ckpt["test_set"])
    if os.path.isfile(p("model.pt")):
        checks["model.pt"] = same(torch.load(p("model.pt"), map_location="cpu"), ckpt["model"])
        checks["optimizer.pt"] = same(torch.load(p("optimizer.pt"), map_location="cpu"), ckpt["optimizer"])
    if os.path.isfile(p("metrics.csv")):
        with open(p("metrics.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        checks["metrics.csv last row"] = bool(rows) and int(rows[-1]["epoch"]) == ckpt["n_epoch"]

    ok = print_checks("checkpoint.pt vs the other files of the run", checks)
    if not ok:
        print(
            "\nA DIFF is expected only if the run was interrupted in the middle of an epoch: train() overwrites\n"
            "model.pt/optimizer.pt and update_datasets() overwrites the .pkl files before the epoch is over,\n"
            "while checkpoint.pt still holds the last completed epoch (the one a restart resumes from)."
        )
        if params is not None and getattr(params, "save_best", False):
            print("Note: this run uses --save_best true, so model.pt is the best model, not the last one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("resume-test", help="end-to-end interrupt/resume test on a tiny CPU run")
    t.add_argument("--max_epochs", type=int, default=3)
    t.add_argument("--max_steps", type=int, default=200, help="training steps per epoch")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kill_at", default="AFTER_TRAIN", help="log marker of the epoch where the run is killed (START / AFTER_TRAIN / AFTER_SAMPLE)")
    t.add_argument("--workdir", default=None, help="keep the runs here (default: temporary folder, deleted on success)")
    i = sub.add_parser("inspect", help="consistency check of an existing run folder")
    i.add_argument("run_dir")
    opts = parser.parse_args()
    if opts.cmd == "resume-test":
        resume_test(opts)
    else:
        inspect(opts.run_dir)
