import argparse
import csv
import ctypes
import gc
import json
import os
import pickle
import re
import subprocess
import sys
import time
from logging import getLogger

import psutil

from src.logger import create_logger

logger = getLogger()

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

MAX_WORKERS = 20


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, "train.log"), rank=0)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on Modal, the job id is the timestamp of the run.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, the job id is the timestamp of the run.
    exp_id = None
    modal_exp_id = os.environ.get("MODAL_EXP_ID")
    if modal_exp_id is not None:
        params.exp_id = modal_exp_id
    elif params.exp_id == "":
        chronos_job_id = os.environ.get("CHRONOS_JOB_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            exp_id = time.strftime("%Y_%m_%d_%H_%M_%S")
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def force_release_memory():
    gc.collect()
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def log_resources(label):
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    logger.info(f"[{label}] CPU: {cpu_percent:.1f}% | RAM: {rss_mb:.1f}MB")
 
RUN_STATE_FILE = "run_state.json"


def load_run_state(dump_path, default):
    """
    Counters of the run that must survive a restart (stored next to epoch.txt).
    """
    path = os.path.join(dump_path, RUN_STATE_FILE)
    state = dict(default)
    if os.path.isfile(path):
        with open(path, "r") as f:
            state.update(json.load(f))
        logger.info(f"resuming run state from {path}: {state}")
    return state


def save_run_state(dump_path, state):
    # write to a temporary file and rename it, so a crash never leaves a truncated file
    path = os.path.join(dump_path, RUN_STATE_FILE)
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=2)
    os.replace(path + ".tmp", path)


# num_local_searches dovrebbe essere sempre uguale... se lo è, togliere da csv (controlla)
METRIC_FIELDS = ["epoch", "mean", "median", "top_1_percentile", "max", "num_local_searches", "num_invalid", "searches_per_second", "flops"]


def write_important_metrics(metrics, epoch, metric_file, command=None):
    if metrics is not None:
        # the command does not fit in the csv: store it in a separate file next to it
        if command is not None:
            with open(os.path.join(os.path.dirname(metric_file), "command.txt"), "a") as f:
                f.write(f"{command}\n")
        write_header = not os.path.isfile(metric_file) or os.path.getsize(metric_file) == 0
        with open(metric_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            # fields not available (e.g. search stats for the initial dataset) are left empty
            writer.writerow({"epoch": epoch, **{k: metrics.get(k, "") for k in METRIC_FIELDS[1:]}})
