# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the CNNModel."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import CNNModel
from rsl_rl.modules.cnn import _compute_output_dim, _compute_padding

NUM_ENVS = 2
OBS_DIM_1D = 6
IMG_H, IMG_W = 16, 16
IMG_CHANNELS = 3
NUM_ACTIONS = 4

CNN_CFG = {
    "output_channels": [8, 16],
    "kernel_size": 3,
    "stride": 1,
    "activation": "elu",
}


def _make_cnn_model(**kwargs: object) -> tuple[CNNModel, TensorDict]:
    obs = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, OBS_DIM_1D),
            "image": torch.randn(NUM_ENVS, IMG_CHANNELS, IMG_H, IMG_W),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {"actor": ["policy", "image"]}
    defaults: dict[str, object] = dict(
        hidden_dims=[32, 32],
        cnn_cfg={"image": CNN_CFG},
    )
    defaults.update(kwargs)
    model = CNNModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


class TestCNNLatentConcatenation:
    """Tests for correct combination of 1D and CNN-encoded 2D features in get_latent."""

    def test_latent_contains_1d_and_cnn_features(self) -> None:
        """get_latent output should have dimension = 1D obs dim + CNN latent dim."""
        model, obs = _make_cnn_model()
        latent = model.get_latent(obs)

        expected_dim = model.obs_dim + model.cnn_latent_dim
        assert latent.shape == (NUM_ENVS, expected_dim)

    def test_1d_portion_matches_normalized_obs(self) -> None:
        """The first obs_dim columns of the latent should equal the normalized 1D observation."""
        model, obs = _make_cnn_model()
        latent = model.get_latent(obs)

        expected_1d = model.obs_normalizer(obs["policy"])
        assert torch.allclose(latent[:, : model.obs_dim], expected_1d, atol=1e-6)

    def test_cnn_portion_changes_with_image(self) -> None:
        """Changing only the image input should change the CNN portion of the latent."""
        model, obs = _make_cnn_model()
        latent_before = model.get_latent(obs).detach().clone()

        obs["image"] = torch.randn_like(obs["image"])
        latent_after = model.get_latent(obs).detach()

        cnn_start = model.obs_dim
        assert not torch.allclose(latent_before[:, cnn_start:], latent_after[:, cnn_start:], atol=1e-6)


class TestCNNOutputDimComputation:
    """Tests for the spatial dimension math in CNN layers."""

    def test_no_padding_shrinks_output(self) -> None:
        """Without padding, a kernel > 1 should reduce spatial dimensions."""
        h_out, w_out = _compute_output_dim((16, 16), kernel=3, stride=1, dilation=1, padding=(0, 0))
        assert h_out == 14
        assert w_out == 14

    def test_same_padding_preserves_dim(self) -> None:
        """With computed 'same' padding, output should equal input for stride=1."""
        pad = _compute_padding((16, 16), kernel=3, stride=1, dilation=1)
        h_out, w_out = _compute_output_dim((16, 16), kernel=3, stride=1, dilation=1, padding=pad)
        assert h_out == 16
        assert w_out == 16

    def test_stride_reduces_output(self) -> None:
        """Stride > 1 should reduce spatial dimensions."""
        h_out, w_out = _compute_output_dim((16, 16), kernel=3, stride=2, dilation=1, padding=(1, 1))
        assert h_out == 8
        assert w_out == 8

    def test_computed_dim_matches_actual_tensor(self) -> None:
        """The computed output_dim of a CNN should match the actual output tensor shape."""
        model, obs = _make_cnn_model()
        cnn = model.cnns["image"]

        actual_output = cnn(obs["image"])
        assert actual_output.shape == (NUM_ENVS, cnn.output_dim)

    def test_max_pool_halves_output(self) -> None:
        """Max pooling should roughly halve spatial dimensions."""
        h_out, w_out = _compute_output_dim((16, 16), kernel=3, stride=1, dilation=1, padding=(1, 1), is_max_pool=True)
        assert h_out == 8
        assert w_out == 8


class TestCNNSharing:
    """Tests for sharing CNN encoders between models."""

    def test_shared_cnns_are_same_object(self) -> None:
        """When passing cnns to a second model, they should be the exact same nn.Module objects."""
        model_a, obs = _make_cnn_model()

        model_b = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            cnns=model_a.cnns,
        )

        for name in model_a.cnns:
            assert model_a.cnns[name] is model_b.cnns[name], f"CNN '{name}' is not shared (different object)"

    def test_shared_cnns_produce_identical_latent_cnn_portion(self) -> None:
        """Two models sharing CNNs should produce the same CNN latent for the same input."""
        model_a, obs = _make_cnn_model()
        model_b = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[64],
            cnns=model_a.cnns,
        )

        latent_a = model_a.get_latent(obs)
        latent_b = model_b.get_latent(obs)

        cnn_a = latent_a[:, model_a.obs_dim :]
        cnn_b = latent_b[:, model_b.obs_dim :]
        assert torch.allclose(cnn_a, cnn_b, atol=1e-6)

    def test_shared_cnn_gradient_reaches_both_models(self) -> None:
        """A gradient through one model's output should flow to the shared CNN parameters."""
        model_a, obs = _make_cnn_model()
        model_b = CNNModel(
            obs,
            {"actor": ["policy", "image"]},
            "actor",
            1,
            hidden_dims=[32],
            cnns=model_a.cnns,
        )

        output_b = model_b(obs)
        output_b.sum().backward()

        for name, param in model_a.cnns.named_parameters():
            assert param.grad is not None, f"Shared CNN param '{name}' has no gradient after backward through model_b"


class TestCNNModelJITExport:
    """Tests for JIT export fidelity of CNN models."""

    def test_jit_export_model(self) -> None:
        """JIT-exported CNN model should produce deterministic outputs matching the original."""
        model, obs = _make_cnn_model(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        original_output = model(obs).detach()

        jit_model = torch.jit.script(model.as_jit())
        obs_1d = obs["policy"]
        obs_2d = [obs["image"]]
        jit_output = jit_model(obs_1d, obs_2d)

        assert torch.allclose(original_output, jit_output, atol=1e-5), "JIT stochastic CNN export should match original"


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
class TestCNNModelONNXExport:
    """Tests for ONNX export fidelity of CNN models."""

    def test_onnx_export_model(self) -> None:
        """ONNX-exported CNN model should be a valid ONNX graph with correct I/O names."""
        model, _obs = _make_cnn_model(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)

            assert [i.name for i in loaded.graph.input] == ["obs", "image"]
            assert [o.name for o in loaded.graph.output] == ["actions"]
