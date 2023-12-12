import os

from rsl_rl.algorithms import *
from rsl_rl.env import VecEnv
from rsl_rl.runners.callbacks import (
    make_final_cb,
    make_first_cb,
    make_interval_cb,
    make_save_model_onnx_cb,
)
from rsl_rl.runners.runner import Runner
from rsl_rl.storage import *


def make_legacy_save_model_cb(directory):
    def cb(runner, stat):
        data = {}

        if hasattr(runner.env, "_persistent_data"):
            data["env_data"] = runner.env._persistent_data

        path = os.path.join(directory, "model_{}.pt".format(stat["current_iteration"]))
        runner.save(path, data=data)

    return cb


class LeggedGymRunner(Runner):
    """Runner for legged_gym environments."""

    mappings = [
        ("init_noise_std", "actor_noise_std"),
        ("clip_param", "clip_ratio"),
        ("desired_kl", "target_kl"),
        ("entropy_coef", "entropy_coeff"),
        ("lam", "gae_lambda"),
        ("max_grad_norm", "gradient_clip"),
        ("num_learning_epochs", None),
        ("num_mini_batches", "batch_count"),
        ("use_clipped_value_loss", None),
        ("value_loss_coef", "value_coeff"),
    ]

    @staticmethod
    def _hook_env(env: VecEnv):
        old_step = env.step

        def step_hook(*args, **kwargs):
            result = old_step(*args, **kwargs)

            if len(result) == 4:
                obs, rewards, dones, env_info = result
            elif len(result) == 5:
                obs, _, rewards, dones, env_info = result
            else:
                raise ValueError("Invalid number of return values from env.step().")

            return obs, rewards, dones.float(), env_info

        env.step = step_hook

        return env

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        env = self._hook_env(env)
        self.cfg = train_cfg["runner"]

        alg_class = eval(self.cfg["algorithm_class_name"])
        if "policy_class_name" in self.cfg:
            print("WARNING: ignoring deprecated parameter 'runner.policy_class_name'.")

        alg_cfg = train_cfg["algorithm"]
        alg_cfg.update(train_cfg["policy"])

        if "activation" in alg_cfg:
            print(
                "WARNING: using deprecated parameter 'activation'. Use 'actor_activations' and 'critic_activations' instead."
            )
            alg_cfg["actor_activations"] = [alg_cfg["activation"] for _ in range(len(alg_cfg["actor_hidden_dims"]))]
            alg_cfg["actor_activations"] += ["linear"]
            alg_cfg["critic_activations"] = [alg_cfg["activation"] for _ in range(len(alg_cfg["critic_hidden_dims"]))]
            alg_cfg["critic_activations"] += ["linear"]
            del alg_cfg["activation"]

        for old, new in self.mappings:
            if old not in alg_cfg:
                continue

            if new is None:
                print(f"WARNING: ignoring deprecated parameter '{old}'.")
                del alg_cfg[old]
                continue

            print(f"WARNING: using deprecated parameter '{old}'. Use '{new}' instead.")
            alg_cfg[new] = alg_cfg[old]
            del alg_cfg[old]

        agent: Agent = alg_class(env, device=device, **train_cfg["algorithm"])

        callbacks = []
        evaluation_callbacks = []

        evaluation_callbacks.append(lambda *args: Runner._log_progress(*args, prefix="eval"))

        if log_dir and "save_interval" in self.cfg:
            callbacks.append(make_first_cb(make_legacy_save_model_cb(log_dir)))
            callbacks.append(make_interval_cb(make_legacy_save_model_cb(log_dir), self.cfg["save_interval"]))

        if log_dir:
            callbacks.append(Runner._log)
            callbacks.append(make_final_cb(make_legacy_save_model_cb(log_dir)))
            callbacks.append(make_final_cb(make_save_model_onnx_cb(log_dir)))
            # callbacks.append(make_first_cb(lambda *_: store_code_state(log_dir, self._git_status_repos)))
        else:
            callbacks.append(Runner._log_progress)

        super().__init__(
            env,
            agent,
            learn_cb=callbacks,
            evaluation_cb=evaluation_callbacks,
            device=device,
            num_steps_per_env=self.cfg["num_steps_per_env"],
        )

        self._iteration_time = 0.0

    def learn(self, *args, num_learning_iterations=None, init_at_random_ep_len=None, **kwargs):
        if num_learning_iterations is not None:
            print("WARNING: using deprecated parameter 'num_learning_iterations'. Use 'iterations' instead.")
            kwargs["iterations"] = num_learning_iterations

        if init_at_random_ep_len is not None:
            print("WARNING: ignoring deprecated parameter 'init_at_random_ep_len'.")

        super().learn(*args, **kwargs)
