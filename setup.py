from setuptools import setup, find_packages

setup(
    name="rsl_rl",
    version="1.0.2",
    packages=find_packages(),
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "GitPython", "gym", "numpy", "onnx", "tensorboard", "torch", "torchvision", "wandb",
    ],
)
