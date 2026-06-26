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
    """Derive (m, n, k) from the einsum equation by axis role, not position.

    Reading roles from the equation (rather than assuming the second operand is
    ``[K, N]``) is required because PyTorch ``nn.Linear`` stores its weight as
    ``[out, in] = [N, K]`` (equation ``...K,NK->...N``).  A positional reader
    mistakes ``N`` for the contraction axis and drops every Linear to GENERIC.

    Roles: ``k`` is the single shared non-output axis (the contraction); ``n`` is
    the axis present in operand B and the output but absent from A; ``m`` is the
    product of all remaining output axes (which folds any batch dims into ``m``,
    matching the prior positional behavior for batched matmul).
    """
    equation = str(layer.get("einsum_equation", "") or "")
    operands, output = parse_einsum_equation(equation)
    if len(operands) != 2 or len(output) < 2:
        return None
    a_axes, b_axes = operands[0], operands[1]
    out_set = set(output)
    shared = (set(a_axes) & set(b_axes)) - out_set
    if len(shared) != 1:
        return None
    k_axis = next(iter(shared))

    # n: an axis contributed by B that survives to the output and is not in A.
    n_candidates = [c for c in b_axes if c in out_set and c not in set(a_axes)]
    if len(n_candidates) != 1:
        return None
    n_axis = n_candidates[0]

    shapes = layer.get("tensor_shapes") or {}
    inputs = shapes.get("inputs") or []
    outputs = shapes.get("outputs") or []
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    a_sh, b_sh, out_sh = inputs[0], inputs[1], outputs[0]
    if not (isinstance(a_sh, list) and isinstance(b_sh, list) and isinstance(out_sh, list)):
        return None
    if len(a_sh) != len(a_axes) or len(b_sh) != len(b_axes) or len(out_sh) != len(output):
        return None
    try:
        a_map = dict(zip(a_axes, (int(x) for x in a_sh)))
        b_map = dict(zip(b_axes, (int(x) for x in b_sh)))
        out_map = dict(zip(output, (int(x) for x in out_sh)))
        k = a_map[k_axis]
        n = b_map[n_axis]
        m = 1
        for axis in output:
            if axis != n_axis:
                m *= out_map[axis]
    except Exception:
        return None
    if m <= 0 or n <= 0 or k <= 0:
        return None
    return int(m), int(n), int(k)


def _gemm_dims(layer: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    # Convolutions carry a matmul-shaped (im2col) einsum but must be dispatched
    # to the CONV certificate, not GEMM.  Exclude conv types so conv
    # classification is not pre-empted.
    if str(layer.get("type", "") or "") in ("conv1d", "conv2d", "conv3d"):
        return None
    return _einsum_gemm_dims(layer) or _shape_gemm_dims(layer)


def _external_input_elems(layer: Dict[str, Any]) -> int:
    if layer.get("input_is_intermediate"):
        return 0
    return int(layer.get("input_elements", 0))


# Convolution charge policy.  Demmel-Dinh is a proven I/O lower bound for the
# *direct / implicit-GEMM* convolution families.  Winograd/FFT backends may move
# strictly less, so charging the direct-conv floor against an unknown backend can
# break the SOL lower-bound contract.  Because the backend is not recoverable
# from the traced graph, the default is "compulsory_only": classify the CONV
# archetype and report the Demmel-Dinh bound as a diagnostic, but charge 0 extra
# (always a valid floor).  Set to "direct_gemm" only when every charged conv is
# known to run a direct/implicit-GEMM kernel (e.g. the Claim-2 analytical study),
# in which case the certified floor is charged like the GEMM certificate.
CONV_CHARGE_POLICY = "compulsory_only"


def _conv_dims(layer: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Map a standard (groups=1) conv layer to Demmel-Dinh parameters.

    Returns a dict with the products the 5-term bound actually uses
    (``W*H`` output-spatial, ``R*S`` filter-spatial, ``sigma_w*sigma_h``
    stride), collapsed so the same helper serves conv1d/2d/3d.  The stride
    product is recovered from the input/output spatial-size ratio because the
    analysis layer does not carry ``module_args``/stride.  Returns ``None`` for
    grouped/depthwise/transpose convs (they fall through to GENERIC = safe 0).
    """
    layer_type = str(layer.get("type", "") or "")
    if layer_type not in ("conv1d", "conv2d", "conv3d"):
        return None
    shapes = layer.get("tensor_shapes") or {}
    inputs = shapes.get("inputs") or []
    outputs = shapes.get("outputs") or []
    if len(inputs) < 2 or len(outputs) < 1:
        return None
    act, weight, out = inputs[0], inputs[1], outputs[0]
    if not (isinstance(act, list) and isinstance(weight, list) and isinstance(out, list)):
        return None
    if len(act) < 3 or len(weight) < 3 or len(out) < 3:
        return None
    try:
        B = int(out[0])
        K = int(out[1])
        C_in = int(weight[1])
        # groups=1 only: weight in-channels must equal activation channels.
        if int(act[1]) != C_in or int(weight[0]) != K:
            return None
        out_spatial = _prod([int(x) for x in out[2:]])
        in_spatial = _prod([int(x) for x in act[2:]])
        filter_spatial = _prod([int(x) for x in weight[2:]])
        if out_spatial <= 0 or in_spatial <= 0 or filter_spatial <= 0:
            return None
        # sigma_w*sigma_h captured as the input/output spatial ratio.
        sigma_prod = in_spatial / out_spatial
        # Collapse to the 2D signature via products (W=1, R=1, sigma_w=1).
        return {
            "B": B, "K": K, "C_in": C_in,
            "W": 1, "H": out_spatial,
            "R": 1, "S": filter_spatial,
            "sigma_w": 1, "sigma_h": sigma_prod,
        }
    except Exception:
        return None


def evaluate_certificates(
    analysis: Dict[str, Any],
    C_elems: float,
    bytes_per_element: float,
) -> Dict[str, Any]:
    certificates: List[Dict[str, Any]] = []
    extra_elems = 0.0
    for layer_id, layer in (analysis.get("layers") or {}).items():
        dims = _gemm_dims(layer)
        conv = _conv_dims(layer) if dims is None else None
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
        elif conv is not None:
            input_internal = bool(layer.get("input_is_intermediate"))
            external_inputs = _external_input_elems(layer)
            output_elems = int(layer.get("output_elements", 0))
            output_internal = bool(layer.get("output_is_intermediate"))
            bound = conv_demm_dinh_5term_elements(
                B=conv["B"], K=conv["K"], C_in=conv["C_in"],
                W=conv["W"], H=conv["H"], R=conv["R"], S=conv["S"],
                C_elems=C_elems, sigma_w=conv["sigma_w"], sigma_h=conv["sigma_h"],
            )
            cert.update({"archetype": "CONV", "cert_shape": {k: conv[k] for k in ("B", "K", "C_in", "H", "S")}})
            if input_internal:
                # Input re-fetch certificate needs external read operands.
                cert.update({"bound_elements": bound, "fallback_reason": "internal conv input"})
            else:
                # Boundary already counted: external inputs (+ output write if external).
                subsumed = external_inputs + (0 if output_internal else output_elems)
                # Direct/implicit-GEMM floor is charged only under that policy;
                # default compulsory_only keeps the contract safe vs Winograd/FFT.
                charged = CONV_CHARGE_POLICY == "direct_gemm"
                extra = max(0.0, bound - subsumed) if charged else 0.0
                cert.update({
                    "admissible": True,
                    "mode": "conv_direct_gemm" if charged else "conv_compulsory_only",
                    "bound_elements": bound,
                    "subsumed_boundary_elements": subsumed,
                    "extra_elements": extra,
                    "fallback_reason": "" if charged else "winograd-safe: bound reported, not charged",
                })
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
