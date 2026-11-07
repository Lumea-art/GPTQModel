# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fuse the q/k/v projections into one quantized GEMM.

A decode step issues three small GEMMs for attention: q is wide, k and v are
narrow, and the narrow ones lose most of the card's memory bandwidth because
there is not enough work per launch. vLLM and sglang avoid that by storing q, k
and v as one projection and issuing a single GEMM over the concatenated weights.

Marlin already tiles N in blocks and its repacked layout is linear in N, so the
already-repacked tensors can simply be concatenated:

    cat(repack(q), repack(k), repack(v)) == repack(cat(q, k, v))

verified bit-exact by ``tests/test_fused_qkv.py``. That means fusion needs no
re-quantization and no access to the pre-repack weights: it is a load-time
concatenation plus a split on the output.

Measured on an RTX 5090 Laptop, GPTQ w4a16 group 128, cold, weight read only,
Qwen3-8B shapes (4096 -> 4096 + 1024 + 1024):

    isolated harness     three calls 45-52us   one fused 21.5us   2.1-2.4x
    real module pair     three calls 42-52us   one fused 24-32us   1.65-1.78x

The harness number is the shape-only upper bound; the module pair keeps the
three parts resident, which costs some of it.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..qlinear import BaseQuantLinear


class FusedQKVLinear:
    """One quantized GEMM behind the q/k/v projections.

    This owns a single inner linear of width ``sum(split_sizes)`` and returns the
    parts on ``forward``. It is deliberately not an ``nn.Module``: it holds the
    same tensors the parts did and the caller decides how to route them.
    """

    def __init__(self, inner: BaseQuantLinear, split_sizes: Sequence[int], names: Sequence[str]):
        self.inner = inner
        self.split_sizes = tuple(int(size) for size in split_sizes)
        self.names = tuple(names)
        if len(self.split_sizes) != len(self.names):
            raise ValueError("split_sizes and names must have the same length")
        if sum(self.split_sizes) != inner.out_features:
            raise ValueError(
                f"split sizes sum to {sum(self.split_sizes)}, inner out_features is {inner.out_features}"
            )

    @property
    def in_features(self) -> int:
        return self.inner.in_features

    @property
    def out_features(self) -> int:
        return self.inner.out_features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(torch.split(self.inner(x), self.split_sizes, dim=1))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward(x)

    @classmethod
    def from_modules(cls, modules: Sequence[BaseQuantLinear]) -> "FusedQKVLinear":
        """Build the fused projection from already-processed q/k/v modules."""
        if len(modules) < 2:
            raise ValueError("at least two modules are needed to fuse")
        _validate_same_shape_and_quantization(modules)

        first = modules[0]
        constructor = _fused_constructor(first)
        inner = constructor(sum(module.out_features for module in modules))
        # post_init() sizes the scratch workspace and the repack metadata for the
        # fused shape; the packed tensors it produces from random data are then
        # replaced by the concatenation below.
        inner.post_init()
        _adopt_packed_tensors(inner, modules)

        return cls(
            inner=inner,
            split_sizes=[module.out_features for module in modules],
            names=[type(module).__name__ for module in modules],
        )


def _fingerprint(module: BaseQuantLinear) -> tuple:
    """Everything that must match for the concatenation to be meaningful."""
    return (
        type(module),
        module.bits,
        getattr(module, "group_size", None),
        getattr(module, "desc_act", None),
        getattr(module, "sym", None),
        module.in_features,
        getattr(module, "bias", None) is not None,
    )


def _validate_same_shape_and_quantization(modules: Sequence[BaseQuantLinear]) -> None:
    first = modules[0]
    expected = _fingerprint(first)
    for module in modules[1:]:
        if _fingerprint(module) != expected:
            raise ValueError(
                "fused projections must share backend, bits, group_size, desc_act, "
                f"sym, in_features and bias: {expected} != {_fingerprint(module)}"
            )
    package = type(first).__module__
    if "marlin" not in package:
        raise NotImplementedError(
            "fused q/k/v concatenation is implemented for the marlin backend only; "
            f"got {type(first).__name__}"
        )


def _fused_constructor(first: BaseQuantLinear):
    """The same quant linear class, built at the fused width."""
    from ..qlinear.marlin import MarlinLinear

    def build(out_features: int) -> MarlinLinear:
        return MarlinLinear(
            bits=first.bits,
            group_size=first.group_size,
            desc_act=first.desc_act,
            sym=first.sym,
            in_features=first.in_features,
            out_features=out_features,
            bias=False,
            dtype=getattr(first, "compute_dtype", torch.float16),
            backend=first.backend,
        ).to(device=first.qweight.device, dtype=getattr(first, "compute_dtype", torch.float16))

    return build


def _adopt_packed_tensors(inner: BaseQuantLinear, modules: Sequence[BaseQuantLinear]) -> None:
    """Concatenate the repacked tensors along N; the layout is linear in N."""
    inner.qweight.data = torch.cat([module.qweight.data for module in modules], dim=1).contiguous()
    inner.scales.data = torch.cat([module.scales.data for module in modules], dim=1).contiguous()
    inner.qzeros.data = torch.cat([module.qzeros.data for module in modules], dim=1).contiguous()
