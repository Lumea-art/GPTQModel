# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fusing q/k/v into one marlin GEMM, and the layout property it relies on."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fused import FusedQKVLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.marlin import gptq_marlin_gemm


K = 4096
NQ = 4096
NKV = 1024
GROUP = 128
DTYPE = torch.float16

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="marlin needs a CUDA device"
)


def _make(k: int, n: int, seed: int, device: str) -> MarlinLinear:
    module = MarlinLinear(
        bits=4,
        group_size=GROUP,
        desc_act=False,
        sym=True,
        in_features=k,
        out_features=n,
        bias=False,
        dtype=DTYPE,
        backend=BACKEND.MARLIN,
    ).to(device=device, dtype=DTYPE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    module.qweight.data = torch.randint(
        0, 2**31 - 1, module.qweight.shape, generator=generator, dtype=torch.int32
    ).to(device)
    module.scales.data = (
        torch.rand(module.scales.shape, generator=generator, dtype=torch.float32) * 0.02 + 1e-3
    ).to(device).to(DTYPE)
    module.qzeros.data = torch.full(
        module.qzeros.shape, -2004318072, dtype=torch.int32, device=device
    )  # 0x88888888: the symmetric 4-bit zero point, 8 per field
    return module


def _gemm(module: MarlinLinear, x: torch.Tensor) -> torch.Tensor:
    return gptq_marlin_gemm(
        x,
        None,
        module.qweight,
        None,
        module.scales,
        None,
        module.qzeros,
        module.g_idx,
        module.g_idx_sort_indices,
        module.workspace,
        module.weight_type,
        size_m=x.shape[0],
        size_n=module.out_features,
        size_k=module.in_features,
        is_k_full=module.is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=module.fp32,
        is_zp_float=False,
    )


def _dequantize(qweight: torch.Tensor, scales: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """GPTQ symmetric layout -> fp32 weights, for a reference outside the kernel."""
    packed = qweight.cpu()
    scale = scales.cpu().float()
    packed_k, n = packed.shape
    per_int32 = 32 // bits
    k = packed_k * per_int32
    values = torch.zeros(k, n, dtype=torch.float32)
    mask = (1 << bits) - 1
    for index in range(per_int32):
        values[index::per_int32] = ((packed >> (bits * index)) & mask).float()
    values = values - (1 << (bits - 1))
    groups = k // GROUP
    return (values.reshape(groups, GROUP, n) * scale.reshape(groups, 1, n)).reshape(k, n)


def test_repack_then_concat_equals_concat_then_repack():
    """The whole approach rests on this: the repacked layout is linear in N."""
    device = "cuda:0"
    parts = [_make(K, NQ, 1, device), _make(K, NKV, 2, device), _make(K, NKV, 3, device)]
    raw = [
        (m.qweight.data.clone(), m.scales.data.clone(), m.qzeros.data.clone()) for m in parts
    ]
    for module in parts:
        module.post_init()

    fused = _make(K, NQ + NKV + NKV, 99, device)
    fused.qweight.data = torch.cat([w for w, _, _ in raw], dim=1).contiguous()
    fused.scales.data = torch.cat([s for _, s, _ in raw], dim=1).contiguous()
    fused.qzeros.data = torch.cat([z for _, _, z in raw], dim=1).contiguous()
    fused.post_init()

    assert torch.equal(torch.cat([m.qweight.data for m in parts], dim=1), fused.qweight)
    assert torch.equal(torch.cat([m.scales.data for m in parts], dim=1), fused.scales)
    assert torch.equal(torch.cat([m.qzeros.data for m in parts], dim=1), fused.qzeros)


@requires_cuda
def test_fused_output_matches_the_parts():
    device = "cuda:0"
    parts = [_make(K, NQ, 1, device), _make(K, NKV, 2, device), _make(K, NKV, 3, device)]
    # post_init() replaces qweight/scales with their repacked form, so keep the
    # GPTQ layout for the reference before that happens.
    raw = [(m.qweight.data.cpu().clone(), m.scales.data.cpu().clone()) for m in parts]
    for module in parts:
        module.post_init()

    fused = FusedQKVLinear.from_modules(parts)
    x = torch.randn(8, K, dtype=DTYPE, device=device)

    separate = torch.cat([_gemm(module, x) for module in parts], dim=1).float().cpu()
    combined = torch.cat(fused(x), dim=1).float().cpu()

    reference = x.float().cpu() @ torch.cat(
        [_dequantize(weight, scale) for weight, scale in raw], dim=1
    )
    separate_err = (separate - reference).abs().max()
    fused_err = (combined - reference).abs().max()
    assert torch.allclose(separate_err, fused_err, rtol=0.05, atol=1e-5), (
        f"fused error {fused_err} drifted from parts error {separate_err}"
    )
    assert (combined - separate).abs().max() < 2e-2


@requires_cuda
def test_split_sizes_come_back_in_order():
    device = "cuda:0"
    parts = [_make(K, NQ, 1, device), _make(K, NKV, 2, device), _make(K, NKV, 3, device)]
    for module in parts:
        module.post_init()
    fused = FusedQKVLinear.from_modules(parts)
    q, k, v = fused(torch.randn(1, K, dtype=DTYPE, device=device))
    assert (q.shape[1], k.shape[1], v.shape[1]) == (NQ, NKV, NKV)


@requires_cuda
def test_mismatched_parts_are_rejected():
    device = "cuda:0"
    parts = [_make(K, NQ, 1, device), _make(K, NKV, 2, device)]
    for module in parts:
        module.post_init()
    parts[1].group_size = 64
    with pytest.raises(ValueError, match="must share"):
        FusedQKVLinear.from_modules(parts)


@requires_cuda
def test_short_input_mismatch_is_rejected():
    device = "cuda:0"
    module = _make(K, NQ, 1, device)
    module.post_init()
    with pytest.raises(ValueError, match="at least two"):
        FusedQKVLinear.from_modules([module])
