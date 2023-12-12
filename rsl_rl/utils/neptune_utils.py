import os

from torch.utils.tensorboard import SummaryWriter
from legged_gym.utils import class_to_dict

try:
    import neptune.new as neptune
except ModuleNotFoundError:
    raise ModuleNotFoundError("neptune-client is required to log to Neptune.")


class NeptuneLogger:
    def __init__(self, project, token):
        self.run = neptune.init(project=project, api_token=token)

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.run["runner_cfg"] = runner_cfg
        self.run["policy_cfg"] = policy_cfg
        self.run["alg_cfg"] = alg_cfg
        self.run["env_cfg"] = class_to_dict(env_cfg)


class NeptuneSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["neptune_project"]
        except KeyError:
            raise KeyError("Please specify neptune_project in the runner config, e.g. legged_gym.")

        try:
            token = os.environ["NEPTUNE_API_TOKEN"]
        except KeyError:
            raise KeyError(
                "Neptune api token not found. Please run or add to ~/.bashrc: export NEPTUNE_API_TOKEN=YOUR_API_TOKEN"
            )

        try:
            entity = os.environ["NEPTUNE_USERNAME"]
        except KeyError:
            raise KeyError(
                "Neptune username not found. Please run or add to ~/.bashrc: export NEPTUNE_USERNAME=YOUR_USERNAME"
            )

        neptune_project = entity + "/" + project

        self.neptune_logger = NeptuneLogger(neptune_project, token)

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        self.neptune_logger.run["log_dir"].log(run_name)

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.neptune_logger.run[self._map_path(tag)].log(scalar_value, step=global_step)

    def stop(self):
        self.neptune_logger.run.stop()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.neptune_logger.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        self.neptune_logger.run["model/saved_model_" + str(iter)].upload(model_path)
