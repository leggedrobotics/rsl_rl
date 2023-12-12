import torch
import unittest

from rsl_rl.utils.recurrency import trajectories_to_transitions, transitions_to_trajectories


class TrajectoryConversionTest(unittest.TestCase):
    def test_basic_conversion(self):
        input = torch.rand(128, 24)
        dones = (torch.rand(128, 24) > 0.8).float()

        trajectories, data = transitions_to_trajectories(input, dones)
        transitions = trajectories_to_transitions(trajectories, data)

        self.assertTrue(torch.allclose(input, transitions))

    def test_2d_observations(self):
        input = torch.rand(128, 24, 32)
        dones = (torch.rand(128, 24) > 0.8).float()

        trajectories, data = transitions_to_trajectories(input, dones)
        transitions = trajectories_to_transitions(trajectories, data)

        self.assertTrue(torch.allclose(input, transitions))

    def test_batch_first(self):
        input = torch.rand(128, 24, 32)
        dones = (torch.rand(128, 24) > 0.8).float()

        trajectories, data = transitions_to_trajectories(input, dones, batch_first=True)
        transitions = trajectories_to_transitions(trajectories, data)

        self.assertTrue(torch.allclose(input, transitions))
