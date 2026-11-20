# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel, QCNNModel
from .mlp_model import MLPModel, QMLPModel
from .mlp_model_equivariant import EquivariantMLPModel
from .rnn_model import RNNModel

__all__ = [
    "CNNModel",
    "EquivariantMLPModel",
    "MLPModel",
    "QCNNModel",
    "QMLPModel",
    "RNNModel",
]
