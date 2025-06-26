from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.optim as optim

from robot_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticEstimator,
)


class ProbeAlg:
    policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstimator

    def __init__(
        self,
        policy,
        probes,
        num_learning_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        oracle: bool = False,
        probe_type: Literal["linear", "sae"] = "linear",
        **kwargs
    ) -> None:
        self.device = device
        self.policy = policy.to(self.device)
        self.probes = {k:v.to(self.device) for k,v in probes.items()}
        self.probe_type = probe_type
        self.learning_rate = learning_rate
        params = []
        for v in self.probes.values():
            p_list = []
            for param in v.parameters():
                p_list.append(param)
            #params.append(*[param for param in v.parameters()])
            params += p_list
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.num_learning_epochs = num_learning_epochs

        self.latents = {}
        self.hooks = {}

        def get_latents(name: str) -> Callable:
            def hook(module, input, output) -> None:
                self.latents[name] = output.detach()
            return hook
        
        if oracle:
            for name, module in self.policy.estimator.named_children():
                self.hooks[name] = module.register_forward_hook(get_latents(name))
        else:
            for name, module in self.policy.actor.named_children():
                self.hooks[name] = module.register_forward_hook(get_latents(name))


    def test_mode(self) -> None:
        self.policy.test()

    #def train_mode(self):
    #    self.policy.train()

    def act(self, obs) -> tuple:
        actions = self.policy.act_inference(obs)
        return actions
    
    def estimate(self, obs) -> torch.tensor:
        if isinstance(self.policy, ActorCriticEstimator):
            return self.policy.estimate(obs)
        else:
            raise ValueError(f"Policy class {type(self.policy)} does not have an estimator.")
    
    def extract_latents(self) -> dict:
        return self.latents

    def update(
            self,
            latents: dict,
            dones: torch.Tensor,
            target: torch.Tensor | None,
            probe_idx: list = [],
            names: list[str] = ["Probe", "Obs", "Next_Obs"]
    ) -> tuple:
        error_dict = {}
        loss_dict = {}
        for i in range(self.num_learning_epochs):
            loss = 0
            if self.probe_type == "sae":
                sparsity_loss = 0
            for k, v in self.probes.items():
                if self.probe_type == "linear":
                    estimate = v(latents[k].clone())
                elif self.probe_type == "sae":
                    target = latents[k].clone()
                    estimate, sparsity = v(target)
                    sparsity_loss += sparsity.mean()
                se = (target - estimate).pow(2)
                se = torch.where(dones, 0, se) # no loss on reset
                loss += se.mean()

                if self.probe_type == "linear" and i == self.num_learning_epochs - 1:
                    errors = {}
                    for j in range(len(probe_idx)-1):
                        errors[names[j]] = se[:, probe_idx[j]:probe_idx[j+1]].pow(0.5).mean(dim=0)
                    error_dict[k] = errors
            
            if self.probe_type == "sae":
                loss += sparsity_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict["loss"] = loss.item()
        loss_dict["mse"] = loss.item()
        if self.probe_type == "sae":
            loss_dict["sparsity"] = sparsity_loss.item()
            loss_dict["mse"] -= sparsity_loss.item()

        return loss_dict, error_dict