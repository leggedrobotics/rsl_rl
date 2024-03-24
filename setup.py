from setuptools import setup, find_packages

setup(
    name="rsl_rl",
    version="1.0.2",
    packages=find_packages(),
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "GitPython",
        "gym[all]>=0.26.0",
        "numpy>=1.24.4",
        "onnx>=1.14.0",
        "tensorboard>=2.10.0",
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "wandb>=0.15.4",
    ],
)
