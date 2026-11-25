# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import os
import tempfile
import torch
from torch import nn


def export_policy_to_executorch(
    policy: nn.Module, path: str, filename: str = "policy.pte", backend: str = "xnnpack"
) -> None:
    """Export a policy into an ExecuTorch program file.

    The program takes the same inputs and returns the same outputs, in the same order, as the ONNX export. Their names
    are stored in the program as the constant methods ``input_names`` and ``output_names``.

    It requires the ``executorch`` package. Version 1.6.0 requires exactly torch 2.15.0 and Python 3.11 or newer, so
    installing it can replace the ``torch`` of a training environment. If that is a problem, export in a separate
    environment. The runtime parses a program in native code, so load only programs from sources you trust.

    Args:
        policy: The policy model to export, for example ``runner.alg.get_policy()``. It must provide ``as_onnx()``.
        path: Directory to save the program to.
        filename: Name of the program file. It must end with ``.pte``.
        backend: ``"xnnpack"`` for optimized CPU kernels, ``"portable"`` for the reference CPU kernels, ``"cuda"`` for
            NVIDIA GPUs, or ``"tensorrt"`` for NVIDIA GPUs through TensorRT. The CUDA backend needs ``executorch`` 1.6
            or newer, a CUDA build of ``torch``, and a GPU. It also writes the weights to ``<name>.ptd`` in ``path``,
            where ``<name>`` is ``filename`` without ``.pte``, and the program has to be loaded together with that file.
            ExecuTorch rejects a name it cannot use for that file. A CUDA program contains native code built for the
            GPU, CPU architecture, and system libraries of the exporting machine, so export on a Linux machine like the
            one that runs the policy, with a CUDA toolkit and a C++ compiler installed. Loading a CUDA program runs that
            code. It runs on GPU 0, so choose the GPU with ``CUDA_VISIBLE_DEVICES``, and keep GPU 0 as the current
            device of the process that loads and runs it. The TensorRT backend runs on Linux and needs
            ``torch_tensorrt`` 2.15 or newer, installed with ``pip install "torch_tensorrt[executorch]>=2.15"``, a CUDA
            13 build of ``torch``, and a GPU. It builds the engines on GPU 0, even when the calling process uses another
            GPU, and the program runs on GPU 0. So choose the GPU with ``CUDA_VISIBLE_DEVICES``, and on a shared machine
            export in a separate process started with it. The engines load only with the same TensorRT version on the
            same kind of GPU, so export on the machine that runs the policy. To load the program in Python, import
            ``torch_tensorrt_executorch_runtime`` before loading it, which adds the TensorRT backend to the ExecuTorch
            runtime. In C++, link ``executorch::backend_tensorrt`` from the ``executorch_backend_tensorrt`` CMake
            package that ``torch_tensorrt_executorch_runtime`` ships, together with ExecuTorch's ``backend_cuda`` and
            ``kernels_optimized``.

    Raises:
        ValueError: If the backend is unknown or the filename does not end with ``.pte``.
        ModuleNotFoundError: If the ``executorch`` package, or for ``"tensorrt"`` the ``torch_tensorrt`` package, is
            not installed.
    """
    if backend not in ("xnnpack", "portable", "cuda", "tensorrt"):
        raise ValueError(f"Unknown ExecuTorch backend: {backend}. Choose 'xnnpack', 'portable', 'cuda', or 'tensorrt'.")
    if not filename.endswith(".pte"):
        raise ValueError(f"ExecuTorch program filename must end with '.pte': {filename}")
    model = policy.as_onnx()
    if backend == "tensorrt" and not all(importlib.util.find_spec(name) for name in ("torch_tensorrt", "executorch")):
        raise ModuleNotFoundError(
            "torch_tensorrt and executorch packages are required for the TensorRT backend:"
            ' pip install "torch_tensorrt[executorch]>=2.15"'
        )
    if importlib.util.find_spec("executorch") is None:
        raise ModuleNotFoundError("executorch package is required to export a policy to ExecuTorch.")

    model.eval()
    os.makedirs(path, exist_ok=True)
    constant_methods = {"input_names": model.input_names, "output_names": model.output_names}

    if backend == "tensorrt":
        import torch_tensorrt  # type: ignore

        with torch.cuda.device(0), tempfile.TemporaryDirectory() as cache_dir:
            model.to("cuda")
            inputs = [x.to("cuda") for x in model.get_dummy_inputs()]
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(model, tuple(inputs)),
                inputs=inputs,
                min_block_size=1,
                disable_tf32=True,
                timing_cache_path=os.path.join(cache_dir, "timing_cache.bin"),
            )
            torch_tensorrt.save(
                compiled, os.path.join(path, filename), output_format="executorch", constant_methods=constant_methods
            )
        return

    model.to("cpu")
    from torch._decomp import get_decompositions

    from executorch.exir import to_edge_transform_and_lower  # type: ignore

    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # type: ignore

        partitioner = [XnnpackPartitioner()]
    elif backend == "portable":
        partitioner = []
    else:
        from executorch.backends.cuda.cuda_backend import CudaBackend  # type: ignore
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner  # type: ignore

        partitioner = [
            CudaPartitioner([
                CudaBackend.generate_method_name_compile_spec("forward"),
                CudaBackend.generate_data_name_compile_spec(filename.removesuffix(".pte")),
            ])
        ]

    exported = torch.export.export(model, model.get_dummy_inputs())  # type: ignore
    exported = exported.run_decompositions(get_decompositions([torch.ops.aten.adaptive_max_pool2d]))
    program = to_edge_transform_and_lower(
        exported,
        partitioner=partitioner,
        constant_methods=constant_methods,
    ).to_executorch()

    with open(os.path.join(path, filename), "wb") as f:
        program.write_to_file(f)
    if backend == "cuda":
        program.write_tensor_data_to_file(path)
