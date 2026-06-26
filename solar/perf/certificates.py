from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from solar.common.utils import parse_einsum_equation


def _prod(shape: List[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def gemm_output_external_elements(m: int, n: int, k: int, C_elems: float) -> float:
    if C_elems <= 0:
        return 0.0
    boundary = (m * k) + (k * n) + (m * n)
    lb = (2.0 * m * n * k / math.sqrt(C_elems)) + (m * n)
    return float(max(boundary, lb))


def gemm_input_traffic_elements(m: int, n: int, k: int, C_elems: float) -> float:
    if C_elems <= 0:
        return 0.0
    boundary = (m * k) + (k * n)
    lb = 2.0 * m * n * k / math.sqrt(C_elems)
    return float(max(boundary, lb))


def conv_demm_dinh_5term_elements(
    B: int,
    K: int,
    C_in: int,
    W: int,
    H: int,
    R: int,
    S: int,
    C_elems: float,
    sigma_w: int = 1,
    sigma_h: int = 1,
) -> float:
    """Demmel-Dinh direct-convolution I/O lower bound in elements.

    Parameters follow the proposal notation: batch ``B``, output channels ``K``,
    input channels ``C_in``, output spatial ``W``/``H``, filter ``R``/``S``, and
    strides ``sigma_w``/``sigma_h``.  The returned value is the conservative
    five-term maximum from Demmel-Dinh Thm. 1 Eq. (4) used for M0 tests.
    """
    if C_elems <= 0:
        return 0.0
    macs = B * K * W * H * C_in * R * S
    output_elems = B * K * W * H
    input_elems = sigma_w * sigma_h * B * C_in * W * H
    filter_elems = C_in * K * R * S
    linear_term = macs / C_elems
    sqrt_term = B * K * W * H * C_in * math.sqrt(
        (R * S * sigma_w * sigma_h) / C_elems
    )
    return float(max(output_elems, input_elems, filter_elems, linear_term, sqrt_term))


def attention_saha_ye_elements(N: int, d: int, C_elems: float) -> float:
    if C_elems < d * d or C_elems <= 0:
        return 0.0
    return float((N * N * d * d) / (2.0 * C_elems))


def _shape_gemm_dims(layer: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    shapes = layer.get("tensor_shapes") or {}
    inputs = shapes.get("inputs") or []
    outputs = shapes.get("outputs") or []
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    a, b, out = inputs[0], inputs[1], outputs[0]
    if not (isinstance(a, list) and isinstance(b, list) and isinstance(out, list)):
        return None
    if len(a) < 2 or len(b) < 2 or len(out) < 2:
        return None
    try:
        k_a = int(a[-1])
        k_b = int(b[-2])
        n = int(b[-1])
        if k_a != k_b or int(out[-1]) != n:
            return None
        m = _prod([int(x) for x in out[:-1]])
        return int(m), int(n), int(k_a)
    except Exception:
        return None


def _einsum_gemm_dims(layer: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    equation = str(layer.get("einsum_equation", "") or "")
    operands, output = parse_einsum_equation(equation)
    if len(operands) != 2 or len(output) < 2:
        return None
    out_set = set(output)
    shared = (set(operands[0]) & set(operands[1])) - out_set
    if len(shared) != 1:
        return None
    return _shape_gemm_dims(layer)


def _gemm_dims(layer: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    return _einsum_gemm_dims(layer) or _shape_gemm_dims(layer)


def _external_input_elems(layer: Dict[str, Any]) -> int:
    if layer.get("input_is_intermediate"):
        return 0
    return int(layer.get("input_elements", 0))


def evaluate_certificates(
    analysis: Dict[str, Any],
    C_elems: float,
    bytes_per_element: float,
) -> Dict[str, Any]:
    certificates: List[Dict[str, Any]] = []
    extra_elems = 0.0
    for layer_id, layer in (analysis.get("layers") or {}).items():
        dims = _gemm_dims(layer)
        cert: Dict[str, Any] = {
            "layer_id": layer_id,
            "archetype": "GENERIC",
            "admissible": False,
            "bound_elements": 0.0,
            "bound_bytes": 0,
            "subsumed_boundary_elements": 0,
            "subsumed_boundary_bytes": 0,
            "extra_elements": 0.0,
            "extra_bytes": 0,
            "fallback_reason": "no admissible certificate",
        }
        if dims is not None:
            m, n, k = dims
            input_internal = bool(layer.get("input_is_intermediate"))
            output_internal = bool(layer.get("output_is_intermediate"))
            external_inputs = _external_input_elems(layer)
            output_elems = int(layer.get("output_elements", 0))
            cert.update({"archetype": "GEMM", "cert_shape": {"m": m, "n": n, "k": k}})
            if input_internal:
                cert["fallback_reason"] = "internal matmul input"
            elif output_internal:
                bound = gemm_input_traffic_elements(m, n, k, C_elems)
                subsumed = external_inputs
                extra = max(0.0, bound - subsumed)
                cert.update({"admissible": True, "mode": "input_traffic", "bound_elements": bound, "subsumed_boundary_elements": subsumed, "extra_elements": extra, "fallback_reason": ""})
                extra_elems += extra
            else:
                bound = gemm_output_external_elements(m, n, k, C_elems)
                subsumed = external_inputs + output_elems
                extra = max(0.0, bound - subsumed)
                cert.update({"admissible": True, "mode": "output_external", "bound_elements": bound, "subsumed_boundary_elements": subsumed, "extra_elements": extra, "fallback_reason": ""})
                extra_elems += extra

        cert["bound_bytes"] = int(cert["bound_elements"] * bytes_per_element)
        cert["subsumed_boundary_bytes"] = int(cert["subsumed_boundary_elements"] * bytes_per_element)
        cert["extra_bytes"] = int(cert["extra_elements"] * bytes_per_element)
        certificates.append(cert)

    return {
        "extra_dram_elements": float(extra_elems),
        "extra_dram_bytes": int(extra_elems * bytes_per_element),
        "certificates": certificates,
    }
