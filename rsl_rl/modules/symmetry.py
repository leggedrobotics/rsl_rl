# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


def resolve_symmetry_config(alg_cfg, env):
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: The algorithm configuration dictionary.
        env: The environment.

    Returns:
        The resolved algorithm configuration dictionary.
    """

    # if using symmetry then pass the environment config object
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        # this is used by the symmetry function for handling different observation terms
        alg_cfg["symmetry_cfg"]["_env"] = env
    return alg_cfg
