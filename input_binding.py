from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import input_graph


@dataclass(frozen=True)
class BindingOffer:
    """A suggested binding target.

    This is intentionally UI-agnostic. Menu code can turn offers into choices.
    """

    label: str
    mapping: dict[str, Any]


def offers_for_axis_binding(*, cfg: dict[str, Any], axis_index: int, sign: int | None = None) -> list[BindingOffer]:
    """Return a small set of smart binding offers for a given physical axis.

    Today:
    - Always includes a raw axis offer (optionally signed).
    - If the controller graph maps signals to channels that depend on this axis,
      offer binding to those channels (future: “bind to channel/signal” workflow).

    This is the "reasoning home" so we can get smarter without contaminating menu code.
    """
    out: list[BindingOffer] = []

    raw: dict[str, Any] = {"type": "axis", "axis": int(axis_index)}
    if sign is not None:
        try:
            s = int(sign)
        except Exception:
            s = 0
        if s in (-1, 1):
            raw["sign"] = int(s)
    out.append(BindingOffer(label="RAW AXIS", mapping=raw))

    for ch in input_graph.iter_channels_involving_axis(cfg=cfg, axis_index=int(axis_index)):
        out.append(BindingOffer(label=f"CHANNEL {int(ch)} (via controller graph)", mapping={"type": "channel", "index": int(ch)}))

    return out


def direction_pair_of(key: str) -> str | None:
    k = str(key)
    pairs = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }
    return pairs.get(k)


def is_direction_key(key: str) -> bool:
    return str(key) in ("up", "down", "left", "right")
