#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="2.0.2",
    packages=find_packages(),
    author="ETH Zurich, NVIDIA CORPORATION",
    maintainer="Nikita Rudin, David Hoeller",
    maintainer_email="rudinn@ethz.ch",
    url="https://github.com/leggedrobotics/rsl_rl",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
    ],
)
