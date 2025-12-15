from __future__ import annotations

# IMPORTANT: These IDs are intended to be shared with C.
# Keep in sync with c_physics/signal_kernel_ids.h

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    id: str  # single char
    label: str
    notes: str


KERNELS: list[KernelSpec] = [
    KernelSpec(id="I", label="IDENTITY", notes="v"),
    KernelSpec(id="D", label="DEADZONE", notes="v = 0 if |v|<dz else v"),
    KernelSpec(id="C", label="CLAMP", notes="v clamped to [lo, hi]"),
    KernelSpec(id="S", label="SMOOTHSTEP", notes="cheap s-curve shaping"),
    KernelSpec(id="E", label="EXPO", notes="expo shaping"),
]


def kernel_ids() -> list[str]:
    return [k.id for k in KERNELS]


def kernel_label(kid: str) -> str:
    s = str(kid)[:1]
    for k in KERNELS:
        if k.id == s:
            return k.label
    return "UNKNOWN"


def normalize_kernel_id(kid: object, default: str = "I") -> str:
    s = str(kid)[:1] if isinstance(kid, str) and kid else str(default)[:1]
    ids = set(kernel_ids())
    return s if s in ids else str(default)[:1]
