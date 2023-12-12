from rsl_rl.algorithms import *
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.runner import Runner

import copy
from datetime import datetime
import numpy as np
import optuna
import os
import random
import torch
from tune_cfg import samplers


ALGORITHM = PPO
ENVIRONMENT = "BipedalWalker-v3"
ENVIRONMENT_KWARGS = {}
EVAL_AGENTS = 64
EVAL_RUNS = 10
EVAL_STEPS = 1000
EXPERIMENT_DIR = os.environ.get("EXPERIMENT_DIRECTORY", "./")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", f"tune-{ALGORITHM.__name__}-{ENVIRONMENT}")
TRAIN_ITERATIONS = None
TRAIN_TIMEOUT = 60 * 15  # 10 minutes
TRAIN_RUNS = 3
TRAIN_SEED = None


def tune():
    assert TRAIN_RUNS == 1 or TRAIN_SEED is None, "If multiple runs are used, the seed must be None."

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{EXPERIMENT_DIR}/{EXPERIMENT_NAME}.db")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    try:
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, study_name=EXPERIMENT_NAME)
    except Exception:
        study = optuna.load_study(pruner=pruner, storage=storage, study_name=EXPERIMENT_NAME)

    study.optimize(objective, n_trials=100)


def seed(s=None):
    seed = int(datetime.now().timestamp() * 1e6) % 2**32 if s is None else s

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(trial):
    seed()
    agent_kwargs, env_kwargs, runner_kwargs = samplers[ALGORITHM.__name__](trial)

    evaluations = []
    for instantiation in range(TRAIN_RUNS):
        seed(TRAIN_SEED)

        env = GymEnv(ENVIRONMENT, gym_kwargs=ENVIRONMENT_KWARGS, **env_kwargs)
        agent = ALGORITHM(env, **agent_kwargs)
        runner = Runner(env, agent, **runner_kwargs)
        runner._learn_cb = [lambda _, stat: runner._log_progress(stat, prefix=f"learn {instantiation+1}/{TRAIN_RUNS}")]

        eval_env_kwargs = copy.deepcopy(env_kwargs)
        eval_env_kwargs["environment_count"] = EVAL_AGENTS
        eval_runner = Runner(
            GymEnv(ENVIRONMENT, gym_kwargs=ENVIRONMENT_KWARGS, **env_kwargs),
            agent,
            **runner_kwargs,
        )
        eval_runner._eval_cb = [
            lambda _, stat: runner._log_progress(stat, prefix=f"eval {instantiation+1}/{TRAIN_RUNS}")
        ]

        try:
            runner.learn(TRAIN_ITERATIONS, timeout=TRAIN_TIMEOUT)
        except Exception:
            raise optuna.TrialPruned()

        intermediate_evaluations = []
        for eval_run in range(EVAL_RUNS):
            eval_runner._eval_cb = [lambda _, stat: runner._log_progress(stat, prefix=f"eval {eval_run+1}/{EVAL_RUNS}")]

            seed()
            eval_runner.env.reset()
            intermediate_evaluations.append(eval_runner.evaluate(steps=EVAL_STEPS))
        eval = np.mean(intermediate_evaluations)

        trial.report(eval, instantiation)
        if trial.should_prune():
            raise optuna.TrialPruned()

        evaluations.append(eval)

    evaluation = np.mean(evaluations)

    return evaluation


if __name__ == "__main__":
    tune()
