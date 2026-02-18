# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx configuration for the RSL-RL documentation."""

from __future__ import annotations

import os
import sys

# Make the package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

project = "RSL-RL"
author = "The RSL-RL Developers"
copyright = "2021-2026, ETH Zurich and NVIDIA CORPORATION"
html_title = "RSL-RL"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme = "furo"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Keep docs builds lightweight in CI by mocking heavy optional dependencies.
autodoc_mock_imports = [
    "torch",
    "tensordict",
    "onnx",
    "onnxruntime",
    "wandb",
    "neptune",
]
