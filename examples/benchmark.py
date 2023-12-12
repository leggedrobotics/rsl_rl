import numpy as np
import os
import torch
import wandb

from rsl_rl.algorithms import *
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.runner import Runner
from rsl_rl.runners.callbacks import make_wandb_cb

from hyperparams import hyperparams
from wandb_config import WANDB_API_KEY, WANDB_ENTITY


ALGORITHMS = [PPO, DPPO]
ENVIRONMENTS = ["BipedalWalker-v3"]
ENVIRONMENT_KWARGS = [{}]
EXPERIMENT_DIR = os.environ.get("EXPERIMENT_DIRECTORY", "./")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "benchmark")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RENDER_VIDEO = False
RETURN_EPOCHS = 100 # Number of epochs to average return over
LOG_WANDB = True
RUNS = 3
TRAIN_TIMEOUT = 60 * 10  # Training time (in seconds)
TRAIN_ENV_STEPS = None  # Number of training environment steps


os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def run(alg_class, env_name, env_kwargs={}):
    try:
        hp = hyperparams[alg_class.__name__][env_name]
    except KeyError:
        print("No hyperparameters found. Using default values.")
        hp = dict(agent_kwargs={}, env_kwargs={"environment_count": 1}, runner_kwargs={"num_steps_per_env": 1})

    agent_kwargs = dict(device=DEVICE, **hp["agent_kwargs"])
    env_kwargs = dict(name=env_name, gym_kwargs=env_kwargs, **hp["env_kwargs"])
    runner_kwargs = dict(device=DEVICE, **hp["runner_kwargs"])

    learn_steps = (
        None
        if TRAIN_ENV_STEPS is None
        else int(np.ceil(TRAIN_ENV_STEPS / (env_kwargs["environment_count"] * runner_kwargs["num_steps_per_env"])))
    )
    learn_timeout = None if TRAIN_TIMEOUT is None else TRAIN_TIMEOUT

    video_directory = f"{EXPERIMENT_DIR}/{EXPERIMENT_NAME}/videos/{env_name}/{alg_class.__name__}"
    save_video_cb = (
        lambda ep, file: wandb.log({f"video-{ep}": wandb.Video(file, fps=4, format="mp4")}) if LOG_WANDB else None
    )
    env = GymEnv(**env_kwargs, draw=RENDER_VIDEO, draw_cb=save_video_cb, draw_directory=video_directory)
    agent = alg_class(env, **agent_kwargs)

    config = dict(
        agent_kwargs=agent_kwargs,
        env_kwargs=env_kwargs,
        learn_steps=learn_steps,
        learn_timeout=learn_timeout,
        runner_kwargs=runner_kwargs,
    )
    wandb_learn_config = dict(
        config=config,
        entity=WANDB_ENTITY,
        group=f"{alg_class.__name__}_{env_name}",
        project="rsl_rl-benchmark",
        tags=[alg_class.__name__, env_name, "train"],
    )

    runner = Runner(env, agent, **runner_kwargs)
    runner._learn_cb = [lambda *args, **kwargs: Runner._log(*args, prefix=f"{alg_class.__name__}_{env_name}", **kwargs)]
    if LOG_WANDB:
        runner._learn_cb.append(make_wandb_cb(wandb_learn_config))

    runner.learn(iterations=learn_steps, timeout=learn_timeout, return_epochs=RETURN_EPOCHS)

    env.close()


def main():
    for algorithm in ALGORITHMS:
        for i, env_name in enumerate(ENVIRONMENTS):
            env_kwargs = ENVIRONMENT_KWARGS[i]

            for _ in range(RUNS):
                run(algorithm, env_name, env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
