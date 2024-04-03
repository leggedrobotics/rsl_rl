from __future__ import annotations
import copy
from datetime import timedelta
import numpy as np
import os
import time
import torch
from typing import Any, Callable, Dict, List, Tuple, TypedDict, Union

import rsl_rl
from rsl_rl.storage.storage import Dataset
from rsl_rl.algorithms import Agent
from rsl_rl.env import VecEnv


class EpisodeStatistics(TypedDict):
    """The statistics of an episode."""

    # Time it took to collect samples for the current interation.
    collection_time: Union[int, None]
    # The counter of the current interation.
    current_iteration: int
    # The number of the final iteration of the current run.
    final_iteration: int
    # The number of the first iteration of the current run.
    first_iteration: int
    # Environment information about the current interation.
    info: list
    # The lengths of the episodes.
    lengths: Union[List[int], None]
    # The loss of the current interation.
    loss: Union[dict, None]
    # The returns of the episodes.
    returns: Union[List[float], None]
    # The total time it took to run the current interation.
    total_time: Union[int, None]
    # The time it took to update the agent.
    update_time: Union[int, None]


Callback = Callable[[EpisodeStatistics], None]


class Runner:
    """The runner class for running an agent in an environment.

    This class is responsible for running an agent in an environment. It is responsible for collecting data from the
    environment, updating the agent, and evaluating the agent. It also provides a number of callbacks that can be used
    to log and visualize the training progress.
    """

    _dataset: Dataset
    _episode_statistics: EpisodeStatistics
    _num_steps_per_env: int

    def __init__(
        self,
        environment: VecEnv,
        agent: Agent,
        device: str = "cpu",
        evaluation_cb: List[Callback] = None,
        learn_cb: List[Callback] = None,
        observation_history_length: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            environment (rsl_rl.env.VecEnv): The environment to run the agent in.
            agent (rsl_rl.algorithms.agent): The RL agent to run.
            device (str): The device to run on.
            evaluation_cb (List[Callable[[dict], None]], optional): A list of callbacks that are called after each round
                of evaluation.
            learn_cb (List[Callable[[dict], None]], optional): A list of callbacks that are called after each round of
                learning.
            observation_history_length: The number of observations to concatenate into a single observation.
        """
        self.env = environment
        self.agent = agent
        self.device = device
        self._obs_hist_len = observation_history_length
        self._learn_cb = learn_cb if learn_cb else []
        self._eval_cb = evaluation_cb if evaluation_cb else []

        self._set_kwarg(kwargs, "num_steps_per_env", default=1)

        self._current_learning_iteration = 0
        self._git_status_repos = [rsl_rl.__file__]

        self.to(self.device)

        self._stored_dataset = []  # For computing observation history over multiple steps.

    def add_git_repo_to_log(self, repo_file_path):
        self._git_status_repos.append(repo_file_path)

    def eval_mode(self):
        """Sets the agent to evaluation mode."""
        self.agent.eval_mode()

    def evaluate(self, steps: int, return_epochs: int = 100) -> float:
        """Evaluates the agent for a number of steps.

        Args:
            steps (int): The number of steps to evaluate the agent for.
            return_epochs (int): The number of epochs over which to aggregate the return. Defaults to 100.
        Returns:
            The mean return of the agent.
        """
        cumulative_rewards = []
        current_cumulative_rewards = torch.zeros(self.env.num_envs, dtype=torch.float)
        current_episode_lengths = torch.zeros(self.env.num_envs, dtype=torch.int)
        episode_lengths = []

        self.eval_mode()

        policy = self.get_inference_policy()
        obs, env_info = self.env.get_observations()

        with torch.inference_mode():
            for step in range(steps):
                actions = policy(obs.clone(), copy.deepcopy(env_info))
                obs, rewards, dones, env_info, episode_statistics = self.evaluate_step(obs, env_info, actions)

                dones_idx = dones.nonzero().cpu()
                current_cumulative_rewards += rewards.clone().cpu()
                current_episode_lengths += 1
                cumulative_rewards.extend(current_cumulative_rewards[dones_idx].squeeze(1).cpu().tolist())
                episode_lengths.extend(current_episode_lengths[dones_idx].squeeze(1).cpu().tolist())
                current_cumulative_rewards[dones_idx] = 0.0
                current_episode_lengths[dones_idx] = 0

                episode_statistics["current_iteration"] = step
                episode_statistics["final_iteration"] = steps
                episode_statistics["lengths"] = episode_lengths[-return_epochs:]
                episode_statistics["returns"] = cumulative_rewards[-return_epochs:]

                for cb in self._eval_cb:
                    cb(self, episode_statistics)

        cumulative_rewards.extend(current_cumulative_rewards.cpu().tolist())
        mean_return = np.mean(cumulative_rewards)

        return mean_return

    def evaluate_step(
        self, observations=None, environment_info=None, actions=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """Evaluates the agent for a single step.

        Args:
            observations (torch.Tensor): The observations to evaluate the agent for.
            environment_info (Dict[str, Any]): The environment information for the observations.
            actions (torch.Tensor): The actions to evaluate the agent for.
        Returns:
            A tuple containing the observations, rewards, dones, environment information, and episode statistics after
            the evaluation step.
        """
        episode_statistics = {
            "current_actions": None,
            "current_dones": None,
            "current_iteration": 0,
            "current_observations": None,
            "current_rewards": None,
            "final_iteration": 0,
            "first_iteration": 0,
            "info": [],
            "lengths": [],
            "returns": [],
            "timeout": None,
            "total_time": None,
        }

        self.eval_mode()

        with torch.inference_mode():
            obs, env_info = self.env.get_observations() if observations is None else (observations, environment_info)

        with torch.inference_mode():
            start = time.time()

            actions = self.get_inference_policy()(obs.clone(), copy.deepcopy(env_info)) if actions is None else actions
            obs, rewards, dones, env_info = self.env.step(actions.clone())

            self.agent.register_terminations(dones.nonzero().reshape(-1))

            end = time.time()

            if "episode" in env_info:
                episode_statistics["info"].append(env_info["episode"])
            episode_statistics["current_actions"] = actions
            episode_statistics["current_dones"] = dones
            episode_statistics["current_observations"] = obs
            episode_statistics["current_rewards"] = rewards
            episode_statistics["total_time"] = end - start

        return obs, rewards, dones, env_info, episode_statistics

    def get_inference_policy(self, device=None):
        self.eval_mode()

        return self.agent.get_inference_policy(device)

    def learn(
        self, iterations: Union[int, None] = None, timeout: Union[int, None] = None, return_epochs: int = 100
    ) -> None:
        """Runs a number of learning iterations.

        Args:
            iterations (int): The number of iterations to run.
            timeout (int): Optional number of seconds after which to terminate training. Defaults to None.
            return_epochs (int): The number of epochs over which to aggregate the return. Defaults to 100.
        """
        assert iterations is not None or timeout is not None

        self._episode_statistics = {
            "collection_time": None,
            "current_actions": None,
            "current_iteration": self._current_learning_iteration,
            "current_observations": None,
            "final_iteration": self._current_learning_iteration + iterations if iterations is not None else None,
            "first_iteration": self._current_learning_iteration,
            "info": [],
            "lengths": [],
            "loss": {},
            "returns": [],
            "storage_initialized": False,
            "timeout": timeout,
            "total_time": None,
            "training_time": 0,
            "update_time": None,
        }
        self._current_episode_lengths = torch.zeros(self.env.num_envs, dtype=torch.float)
        self._current_cumulative_rewards = torch.zeros(self.env.num_envs, dtype=torch.float)

        self.train_mode()

        self._obs, self._env_info = self.env.get_observations()
        while True:
            if self._learning_should_terminate():
                break

            # Collect data
            start = time.time()

            with torch.inference_mode():
                self._dataset = []

                for _ in range(self._num_steps_per_env):
                    self._collect()

                self._episode_statistics["lengths"] = self._episode_statistics["lengths"][-return_epochs:]
                self._episode_statistics["returns"] = self._episode_statistics["returns"][-return_epochs:]

            self._episode_statistics["collection_time"] = time.time() - start

            # Update agent

            start = time.time()

            self._update()

            self._episode_statistics["update_time"] = time.time() - start

            # Housekeeping

            self._episode_statistics["total_time"] = (
                self._episode_statistics["collection_time"] + self._episode_statistics["update_time"]
            )
            self._episode_statistics["training_time"] += self._episode_statistics["total_time"]

            if self.agent.initialized:
                self._episode_statistics["current_iteration"] += 1

            terminate = False
            for cb in self._learn_cb:
                terminate = (cb(self, self._episode_statistics) == False) or terminate

            if terminate:
                break

            self._episode_statistics["info"].clear()
            self._current_learning_iteration = self._episode_statistics["current_iteration"]

    def _collect(self) -> None:
        """Runs a single step in the environment to collect a transition and stores it in the dataset.

        This method runs a single step in the environment to collect a transition and stores it in the dataset. If the
        agent is not initialized, random actions are drawn from the action space. Furthermore, the method gathers
        statistics about the episode and stores them in the episode statistics dictionary of the runner.
        """
        if self.agent.initialized:
            actions, data = self.agent.draw_actions(self._obs, self._env_info)
        else:
            actions, data = self.agent.draw_random_actions(self._obs, self._env_info)

        next_obs, rewards, dones, next_env_info = self.env.step(actions)

        self._dataset.append(
            self.agent.process_transition(
                self._obs.clone(),
                copy.deepcopy(self._env_info),
                actions.clone(),
                rewards.clone(),
                next_obs.clone(),
                copy.deepcopy(next_env_info),
                dones.clone(),
                copy.deepcopy(data),
            )
        )

        self.agent.register_terminations(dones.nonzero().reshape(-1))

        self._obs, self._env_info = next_obs, next_env_info

        # Gather statistics
        if "episode" in self._env_info:
            self._episode_statistics["info"].append(self._env_info["episode"])
        dones_idx = (dones + next_env_info["time_outs"]).nonzero().cpu()
        self._current_episode_lengths += 1
        self._current_cumulative_rewards += rewards.cpu()

        completed_lengths = self._current_episode_lengths[dones_idx][:, 0].cpu()
        completed_returns = self._current_cumulative_rewards[dones_idx][:, 0].cpu()
        self._episode_statistics["lengths"].extend(completed_lengths.tolist())
        self._episode_statistics["returns"].extend(completed_returns.tolist())
        self._current_episode_lengths[dones_idx] = 0.0
        self._current_cumulative_rewards[dones_idx] = 0.0

        self._episode_statistics["current_actions"] = actions
        self._episode_statistics["current_observations"] = self._obs
        self._episode_statistics["sample_count"] = self.agent.storage.sample_count

    def _learning_should_terminate(self):
        """Checks whether the learning should terminate.

        Termination is triggered if the number of iterations or the timeout is reached.

        Returns:
            Whether the learning should terminate.
        """
        if (
            self._episode_statistics["final_iteration"] is not None
            and self._episode_statistics["current_iteration"] >= self._episode_statistics["final_iteration"]
        ):
            return True

        if (
            self._episode_statistics["timeout"] is not None
            and self._episode_statistics["training_time"] >= self._episode_statistics["timeout"]
        ):
            return True

        return False

    def _update(self) -> None:
        """Updates the agent using the collected data."""
        loss = self.agent.update(self._dataset)
        self._dataset = []

        if not self.agent.initialized:
            return

        self._episode_statistics["loss"] = loss
        self._episode_statistics["storage_initialized"] = True

    def load(self, path: str) -> Any:
        """Restores the agent and runner state from a file."""
        content = torch.load(path, map_location=self.device)

        assert "agent" in content
        assert "data" in content
        assert "iteration" in content

        self.agent.load_state_dict(content["agent"])
        self._current_learning_iteration = content["iteration"]

        return content["data"]

    def save(self, path: str, data: Any = None) -> None:
        """Saves the agent and runner state to a file."""
        content = {
            "agent": self.agent.state_dict(),
            "data": data,
            "iteration": self._current_learning_iteration,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(content, path)

    def export_onnx(self, path: str) -> None:
        """Exports the agent's policy network to ONNX format."""
        model, args, kwargs = self.agent.export_onnx()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.onnx.export(model, args, path, **kwargs)

    def to(self, device) -> Runner:
        """Sets the device of the runner and its components."""
        self.device = device

        self.agent.to(device)

        try:
            self.env.to(device)
        except AttributeError:
            pass

        return self

    def train_mode(self):
        """Sets the agent to training mode."""
        self.agent.train_mode()

    def _set_kwarg(self, args, key, default=None, private=True):
        setattr(self, f"_{key}" if private else key, args[key] if key in args else default)

    def _log_progress(self, stat, clear_line=True, prefix=""):
        """Logs the progress of the runner."""
        if not hasattr(self, "_iteration_times"):
            self._iteration_times = []

        self._iteration_times = (self._iteration_times + [stat["total_time"]])[-100:]
        average_total_time = np.mean(self._iteration_times)

        if stat["final_iteration"] is not None:
            first_iteration = stat["first_iteration"]
            final_iteration = stat["final_iteration"]
            current_iteration = stat["current_iteration"]
            final_run_iteration = final_iteration - first_iteration
            remaining_iterations = final_iteration - current_iteration

            remaining_iteration_time = remaining_iterations * average_total_time
            iteration_completion_percentage = 100 * (current_iteration - first_iteration) / final_run_iteration
        else:
            remaining_iteration_time = np.inf
            iteration_completion_percentage = 0

        if stat["timeout"] is not None:
            training_time = stat["training_time"]
            timeout = stat["timeout"]

            remaining_timeout_time = stat["timeout"] - stat["training_time"]
            timeout_completion_percentage = 100 * stat["training_time"] / stat["timeout"]
        else:
            remaining_timeout_time = np.inf
            timeout_completion_percentage = 0

        if remaining_iteration_time > remaining_timeout_time:
            completion_percentage = timeout_completion_percentage
            remaining_time = remaining_timeout_time
            step_string = f"({int(training_time)}s / {timeout}s)"
        else:
            completion_percentage = iteration_completion_percentage
            remaining_time = remaining_iteration_time
            step_string = f"({current_iteration} / {final_iteration})"

        prefix = f"[{prefix}] " if prefix else ""
        progress = "".join(["#" if i <= int(completion_percentage) else "_" for i in range(10, 101, 5)])
        remaining_time_string = str(timedelta(seconds=int(np.ceil(remaining_time))))
        print(
            f"{prefix}{progress} {step_string} [{completion_percentage:.1f}%, {1/average_total_time:.2f}it/s, {remaining_time_string} ETA]",
            end="\r" if clear_line else "\n",
        )

    def _log(self, stat, prefix=""):
        """Logs the progress and statistics of the runner."""
        current_iteration = stat["current_iteration"]

        collection_time = stat["collection_time"]
        update_time = stat["update_time"]
        total_time = stat["total_time"]
        collection_percentage = 100 * collection_time / total_time
        update_percentage = 100 * update_time / total_time

        if prefix == "":
            prefix = "learn" if stat["storage_initialized"] else "init"
        self._log_progress(stat, clear_line=False, prefix=prefix)
        print(
            f"iteration time:\t{total_time:.4f}s (collection: {collection_time:.2f}s [{collection_percentage:.1f}%], update: {update_time:.2f}s [{update_percentage:.1f}%])"
        )

        mean_reward = sum(stat["returns"]) / len(stat["returns"]) if len(stat["returns"]) > 0 else 0.0
        mean_steps = sum(stat["lengths"]) / len(stat["lengths"]) if len(stat["lengths"]) > 0 else 0.0
        total_steps = current_iteration * self.env.num_envs * self._num_steps_per_env
        sample_count = stat["sample_count"]
        print(f"avg. reward:\t{mean_reward:.4f}")
        print(f"avg. steps:\t{mean_steps:.4f}")
        print(f"stored samples:\t{sample_count}")
        print(f"total steps:\t{total_steps}")

        for key, value in stat["loss"].items():
            print(f"{key} loss:\t{value:.4f}")

        for key, value in self.agent._bm_report().items():
            mean, count = value
            print(f"BM {key}:\t{mean/1000000.0:.4f}ms ({count} calls, total {mean*count/1000000.0:.4f}ms)")

        self.agent._bm_flush()
