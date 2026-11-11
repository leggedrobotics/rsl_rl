# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ExecuTorch policy export."""

from __future__ import annotations

import pathlib
import sys
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models import CNNModel, MLPModel, RNNModel
from rsl_rl.utils import export_policy_to_executorch

NUM_ENVS = 8
OBS_DIM = 6
NUM_ACTIONS = 4


def _make_policy(model_type: str, distribution: str = "GaussianDistribution") -> MLPModel:
    """Create a policy whose observation normalization and batch norm statistics differ from the identity."""
    obs = TensorDict({"policy": 3.0 * torch.randn(NUM_ENVS, OBS_DIM) + 1.0}, batch_size=[NUM_ENVS])
    obs_groups = {"actor": ["policy"]}
    kwargs: dict[str, object] = {
        "hidden_dims": [16],
        "obs_normalization": True,
        "distribution_cfg": {"class_name": distribution},
    }
    if model_type == "mlp":
        policy = MLPModel(obs, obs_groups, "actor", NUM_ACTIONS, **kwargs)
    elif model_type in ("gru", "lstm"):
        policy = RNNModel(obs, obs_groups, "actor", NUM_ACTIONS, rnn_type=model_type, rnn_hidden_dim=8, **kwargs)
    else:
        obs["image"] = torch.randn(NUM_ENVS, 3, 8, 8)
        obs_groups = {"actor": ["policy", "image"]}
        cnn_cfg = {"image": {"output_channels": [4], "kernel_size": 3, "activation": "elu", "norm": "batch"}}
        policy = CNNModel(obs, obs_groups, "actor", NUM_ACTIONS, cnn_cfg=cnn_cfg, **kwargs)
        for module in policy.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-1.0, 1.0)  # type: ignore
                module.running_var.uniform_(0.5, 2.0)  # type: ignore
    policy.update_normalization(obs)
    return policy


@pytest.mark.parametrize("model_type", ["mlp", "gru", "lstm", "cnn"])
@pytest.mark.parametrize(
    "distribution", ["GaussianDistribution", "HeteroscedasticGaussianDistribution", "BetaDistribution"]
)
@pytest.mark.parametrize("backend", ["xnnpack", "portable"])
def test_export_matches_policy(model_type: str, distribution: str, backend: str, tmp_path: pathlib.Path) -> None:
    """The program should match the ONNX model, keep its input and output names, and use the requested backend."""
    runtime = pytest.importorskip("executorch.runtime")
    policy = _make_policy(model_type, distribution)
    reference = policy.as_onnx().eval()
    inputs = [torch.randn_like(x) for x in reference.get_dummy_inputs()]
    expected = reference(*inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    export_policy_to_executorch(policy, str(tmp_path), filename="actor.pte", backend=backend)

    program_bytes = (tmp_path / "actor.pte").read_bytes()
    assert (b"XnnpackBackend" in program_bytes) == (backend == "xnnpack")
    program = runtime.Runtime.get().load_program(program_bytes)
    outputs = program.load_method("forward").execute(inputs)

    assert program.load_method("input_names").execute([]) == reference.input_names
    assert program.load_method("output_names").execute([]) == reference.output_names
    assert len(outputs) == len(reference.output_names) == len(expected)
    for output, target in zip(outputs, expected):
        torch.testing.assert_close(output, target, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_export_policy_on_gpu(tmp_path: pathlib.Path) -> None:
    """A policy that is still on the GPU after training should export to a program that matches it."""
    runtime = pytest.importorskip("executorch.runtime")
    policy = _make_policy("gru").to("cuda")
    reference = policy.as_onnx().eval()
    inputs = [torch.randn_like(x, device="cuda") for x in reference.get_dummy_inputs()]
    expected = [x.cpu() for x in reference(*inputs)]
    export_policy_to_executorch(policy, str(tmp_path))

    program = runtime.Runtime.get().load_program(str(tmp_path / "policy.pte"))
    outputs = program.load_method("forward").execute([x.cpu() for x in inputs])

    assert len(outputs) == len(expected)
    for output, target in zip(outputs, expected):
        torch.testing.assert_close(output, target, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"backend": "vulkan"}, "Unknown ExecuTorch backend"), ({"filename": "policy.onnx"}, "must end with '.pte'")],
)
def test_invalid_arguments_raise(kwargs: dict[str, str], match: str, tmp_path: pathlib.Path) -> None:
    """An unknown backend or a filename without the .pte suffix should raise before anything is written."""
    with pytest.raises(ValueError, match=match):
        export_policy_to_executorch(_make_policy("mlp"), str(tmp_path / "out"), **kwargs)
    assert not (tmp_path / "out").exists()


def test_default_export_of_max_global_pool(tmp_path: pathlib.Path) -> None:
    """With default arguments, a CNN policy with max global pooling should export to policy.pte with XNNPACK."""
    runtime = pytest.importorskip("executorch.runtime")
    obs = TensorDict({"image": torch.randn(NUM_ENVS, 1, 8, 8)}, batch_size=[NUM_ENVS])
    cnn_cfg = {"image": {"output_channels": [4], "kernel_size": 3, "global_pool": "max"}}
    policy = CNNModel(obs, {"actor": ["image"]}, "actor", NUM_ACTIONS, cnn_cfg=cnn_cfg, hidden_dims=[16])
    reference = policy.as_onnx().eval()
    inputs = [torch.randn_like(x) for x in reference.get_dummy_inputs()]
    export_policy_to_executorch(policy, str(tmp_path / "new" / "folder"))

    program_bytes = (tmp_path / "new" / "folder" / "policy.pte").read_bytes()
    assert b"XnnpackBackend" in program_bytes
    (output,) = runtime.Runtime.get().load_program(program_bytes).load_method("forward").execute(inputs)
    torch.testing.assert_close(output, reference(*inputs), atol=1e-5, rtol=1e-5)


def test_missing_executorch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A missing executorch package should raise a ModuleNotFoundError that says so."""
    monkeypatch.setitem(sys.modules, "executorch", None)
    with pytest.raises(ModuleNotFoundError, match="executorch package is required"):
        export_policy_to_executorch(_make_policy("mlp"), str(tmp_path))
