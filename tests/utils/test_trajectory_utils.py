# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for trajectory padding and unpadding utilities."""

import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories, unpad_trajectories


class TestSplitAndPad:
    def test_split_and_pad_tensor_simple(self) -> None:
        """Test basic functionality with a standard Tensor.

        Scenario:
            - 2 Environments, 5 timesteps.
            - Env 0: Trajectory A (len 2), Trajectory B (len 3).
            - Env 1: Trajectory C (len 5, never done until force-finish).
        """
        n_steps, n_envs, feature_dim = 5, 2, 5

        data = torch.arange(n_steps * n_envs * feature_dim).float().reshape(n_steps, n_envs, feature_dim)

        # Dones
        dones = torch.zeros(n_steps, n_envs, dtype=torch.bool)
        dones[1, 0] = True

        padded_traj, masks = split_and_pad_trajectories(data, dones)

        # Expected Trajectories:
        # 1. Env 0, part 1: data[0:2, 0] -> Length 2
        # 2. Env 0, part 2: data[2:5, 0] -> Length 3
        # 3. Env 1, part 1: data[0:5, 1] -> Length 5

        # Verify Shapes
        # Max length is 5 (from Env 1). Number of trajs is 3.
        assert padded_traj.shape == (n_steps, 3, feature_dim)
        assert masks.shape == (n_steps, 3)

        # Check Traj 1 (Length 2)
        assert torch.all(masks[0:2, 0])
        assert not torch.any(masks[2:, 0])  # Padding
        assert torch.allclose(padded_traj[0:2, 0], data[0:2, 0])
        assert torch.all(padded_traj[2:, 0] == 0)  # Zero padded

        # Check Traj 2 (Length 3)
        assert torch.all(masks[0:3, 1])
        assert torch.allclose(padded_traj[0:3, 1], data[2:5, 0])

        # Check Traj 3 (Length 5)
        assert torch.all(masks[:, 2])
        assert torch.allclose(padded_traj[:, 2], data[:, 1])

    def test_split_and_pad_tensor_high_dim(self) -> None:
        """Test split and pad with higher dimensional tensors (e.g. images).

        Scenario:
            - Shape: [Time, Envs, C, H, W] -> [5, 2, 3, 4, 4]
        """
        n_steps, n_envs = 5, 2
        extra_dims = (3, 4, 4)  # C, H, W

        # Create data
        data = torch.randn(n_steps, n_envs, *extra_dims)

        dones = torch.zeros(n_steps, n_envs, dtype=torch.bool)
        dones[1, 0] = True

        padded_traj, masks = split_and_pad_trajectories(data, dones)

        # Expected shape: [5, 3, 3, 4, 4]
        # Max length is 5. Number of trajs is 3.
        assert padded_traj.shape == (n_steps, 3, *extra_dims)
        assert masks.shape == (n_steps, 3)

        # Verify data
        # Traj 1 (Env 0, Part 1)
        assert torch.allclose(padded_traj[0:2, 0], data[0:2, 0])
        # Traj 2 (Env 0, Part 2)
        assert torch.allclose(padded_traj[0:3, 1], data[2:5, 0])
        # Traj 3 (Env 1, Part 1)
        assert torch.allclose(padded_traj[:, 2], data[:, 1])

    def test_split_and_pad_tensordict(self) -> None:
        """Test that TensorDicts are handled recursively and shapes match.

        Scenario:
            - 2 Environments (n_envs), 5 timesteps (n_steps).
            - Env 0 splits into len 2 and len 3.
            - Env 1 remains len 5.
        """
        n_steps, n_envs = 5, 2
        obs_0 = torch.randn(n_steps, n_envs, 4)
        obs_1 = torch.randn(n_steps, n_envs, 2, 2)

        td = TensorDict({"obs_0": obs_0, "obs_1": obs_1}, batch_size=(n_steps, n_envs))

        dones = torch.zeros(n_steps, n_envs, dtype=torch.bool)
        dones[1, 0] = True

        padded_td, _ = split_and_pad_trajectories(td, dones)

        # Expect 3 trajectories: Env0(Len2), Env0(Len3), Env1(Len5)
        expected_trajs = 3
        expected_len = 5

        assert isinstance(padded_td, TensorDict)
        assert padded_td.shape == (expected_len, expected_trajs)
        assert padded_td["obs_0"].shape == (expected_len, expected_trajs, 4)
        assert padded_td["obs_1"].shape == (expected_len, expected_trajs, 2, 2)

        # Check integrity of one specific value
        assert torch.allclose(padded_td["obs_0"][:3, 1], obs_0[2:, 0])


class TestUnpad:
    def test_unpad_tensor_identity(self) -> None:
        """Test unpad with no padding (Masks all True)."""
        n_steps, n_traj, feature_dim = 5, 2, 3
        data = torch.randn(n_steps, n_traj, feature_dim)
        masks = torch.ones(n_steps, n_traj, dtype=torch.bool)

        # Logic: transpose(1,0) -> [n_traj, n_steps, feature_dim] -> flatten -> [10, feature_dim]
        # view(-1, n_steps, feature_dim) -> [2, 5, feature_dim]
        # transpose(1,0) -> [5, 2, feature_dim]
        output = unpad_trajectories(data, masks)

        assert output.shape == data.shape
        assert torch.allclose(output, data)

    def test_unpad_tensor_high_dim_identity(self) -> None:
        """Test unpad with no padding on high-dimensional tensors (e.g. images).

        Scenario:
            - Shape: [Time, Trajectories, C, H, W] -> [5, 2, 3, 4, 4]
            - Masks all True.
        """
        n_steps, n_traj = 5, 2
        extra_dims = (3, 4, 4)
        data = torch.randn(n_steps, n_traj, *extra_dims)
        masks = torch.ones(n_steps, n_traj, dtype=torch.bool)

        output = unpad_trajectories(data, masks)

        assert output.shape == data.shape
        assert torch.allclose(output, data)

    def test_unpad_round_trip(self) -> None:
        """Test that `unpad_trajectories` correctly reconstructs the original tensor after `split_and_pad_trajectories`.

        Scenario:
            - n_steps=5, n_envs=1.
            - Split into len 2 and len 3 trajectories.
        """
        n_steps, n_envs, feature_dim = 5, 1, 2
        data = torch.randn(n_steps, n_envs, feature_dim)

        dones = torch.zeros(n_steps, n_envs, dtype=torch.bool)
        dones[1, 0] = True

        # Split and Pad
        padded_traj, masks = split_and_pad_trajectories(data, dones)
        assert padded_traj.shape[0] == 5

        # Unpad
        output = unpad_trajectories(padded_traj, masks)

        # Should reconstruct original shape [5, 1, 2]
        assert output.shape == data.shape

        # Verify data is restored correctly
        assert torch.allclose(output, data)

    def test_unpad_tensordict_round_trip(self) -> None:
        """Test round trip with TensorDict (split -> unpad -> original)."""
        n_steps, n_envs = 5, 2
        obs_0 = torch.randn(n_steps, n_envs, 4)
        obs_1 = torch.randn(n_steps, n_envs, 2, 2)

        td = TensorDict({"obs_0": obs_0, "obs_1": obs_1}, batch_size=(n_steps, n_envs))

        dones = torch.zeros(n_steps, n_envs, dtype=torch.bool)
        dones[1, 0] = True

        # Split and Pad
        padded_td, masks = split_and_pad_trajectories(td, dones)

        # Unpad
        output_td = unpad_trajectories(padded_td, masks)

        # Verify reconstruction
        assert output_td.shape == td.shape
        assert torch.allclose(output_td["obs_0"], obs_0)
        assert torch.allclose(output_td["obs_1"], obs_1)
