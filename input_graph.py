from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GraphEvalContext:
    axes: dict[int, float]
    buttons: set[int]
    hats: dict[int, tuple[int, int]]


def ensure_controller_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure cfg has flight_controls.controller.{features,signals,channels,state} skeleton."""
    if not isinstance(cfg.get("flight_controls", None), dict):
        cfg["flight_controls"] = {}
    fc = cfg["flight_controls"]
    if not isinstance(fc.get("controller", None), dict):
        fc["controller"] = {}
    ctrl = fc["controller"]
    if not isinstance(ctrl.get("features", None), dict):
        ctrl["features"] = {}
    if not isinstance(ctrl.get("signals", None), dict):
        ctrl["signals"] = {}
    if not isinstance(ctrl.get("channels", None), dict):
        ctrl["channels"] = {}
    if not isinstance(ctrl.get("state", None), dict):
        ctrl["state"] = {}
    st = ctrl["state"]
    if not isinstance(st.get("last_feature", None), str):
        st["last_feature"] = ""
    if not isinstance(st.get("last_signal", None), str):
        st["last_signal"] = ""
    return cfg


def _feature_value(spec: Any, ctx: GraphEvalContext) -> float:
    if not isinstance(spec, dict):
        return 0.0
    t = spec.get("type")
    if t == "axis":
        a = spec.get("axis")
        if not isinstance(a, int):
            return 0.0
        return float(ctx.axes.get(int(a), 0.0))
    if t == "button":
        b = spec.get("button")
        if not isinstance(b, int):
            return 0.0
        return 1.0 if int(b) in ctx.buttons else 0.0
    if t == "hat":
        h = spec.get("hat")
        x = spec.get("x")
        y = spec.get("y")
        if not (isinstance(h, int) and isinstance(x, int) and isinstance(y, int)):
            return 0.0
        now = ctx.hats.get(int(h), (0, 0))
        return 1.0 if now == (int(x), int(y)) else 0.0
    return 0.0


def _iter_signal_refs(node: dict[str, Any]) -> Iterable[str]:
    # References are strings like "feat:<id>" or "sig:<id>" (or plain signal id for back-compat).
    for k in ("in", "a", "b"):
        v = node.get(k, None)
        if isinstance(v, str):
            yield v


def eval_controller_signal(
    *,
    ctrl: dict[str, Any],
    sid: str,
    ctx: GraphEvalContext,
    _memo: dict[str, float] | None = None,
    _visiting: set[str] | None = None,
) -> float:
    """Evaluate a signal id under flight_controls.controller.signals.

    Cycles return 0.0.
    """
    sigs = ctrl.get("signals") if isinstance(ctrl, dict) else None
    feats = ctrl.get("features") if isinstance(ctrl, dict) else None
    if not (isinstance(sigs, dict) and isinstance(feats, dict)):
        return 0.0

    memo = _memo if _memo is not None else {}
    visiting = _visiting if _visiting is not None else set()

    key = str(sid)
    if key in memo:
        return float(memo[key])
    if key in visiting:
        return 0.0

    visiting.add(key)
    node = sigs.get(key)
    v = 0.0

    def _eval_ref(ref: Any) -> float:
        if not isinstance(ref, str):
            return 0.0
        if ref.startswith("feat:"):
            fid = ref[len("feat:") :]
            return float(_feature_value(feats.get(str(fid)), ctx))
        if ref.startswith("sig:"):
            return float(eval_controller_signal(ctrl=ctrl, sid=ref[len("sig:") :], ctx=ctx, _memo=memo, _visiting=visiting))
        # Back-compat: treat plain strings as signal ids.
        return float(eval_controller_signal(ctrl=ctrl, sid=str(ref), ctx=ctx, _memo=memo, _visiting=visiting))

    if isinstance(node, dict):
        op = str(node.get("op", ""))
        if op == "raw":
            fid = str(node.get("feature", ""))
            v = float(_feature_value(feats.get(fid), ctx))
        elif op == "const":
            try:
                v = float(node.get("value", 0.0))
            except Exception:
                v = 0.0
        elif op == "scale":
            v = float(_eval_ref(node.get("in"))) * float(node.get("gain", 1.0))
        elif op == "add":
            v = float(_eval_ref(node.get("a"))) + float(_eval_ref(node.get("b")))
        elif op == "sub":
            v = float(_eval_ref(node.get("a"))) - float(_eval_ref(node.get("b")))
        elif op == "clamp":
            vv = float(_eval_ref(node.get("in")))
            lo = float(node.get("lo", -1.0))
            hi = float(node.get("hi", 1.0))
            if lo > hi:
                lo, hi = hi, lo
            v = float(max(lo, min(hi, vv)))
        elif op == "deadzone":
            vv = float(_eval_ref(node.get("in")))
            dzv = float(node.get("dz", 0.08))
            v = 0.0 if abs(float(vv)) < float(dzv) else float(vv)

    visiting.discard(key)
    memo[key] = float(v)
    return float(v)


def eval_controller_channel_overrides(
    *,
    cfg: dict[str, Any],
    ctx: GraphEvalContext,
    clamp: bool = True,
) -> dict[int, float]:
    """Evaluate controller graph and return channel overrides for channels[0..7]."""
    fc = cfg.get("flight_controls") if isinstance(cfg, dict) else None
    ctrl = fc.get("controller") if isinstance(fc, dict) else None
    if not isinstance(ctrl, dict):
        return {}

    feats = ctrl.get("features")
    sigs = ctrl.get("signals")
    chmap = ctrl.get("channels")
    if not (isinstance(feats, dict) and isinstance(sigs, dict) and isinstance(chmap, dict)):
        return {}

    overrides: dict[int, float] = {}
    memo: dict[str, float] = {}
    visiting: set[str] = set()

    def _cl(v: float) -> float:
        if not bool(clamp):
            return float(v)
        return float(max(-1.0, min(1.0, float(v))))

    for k, spec in chmap.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if not (0 <= int(idx) < 8):
            continue
        if not isinstance(spec, dict):
            continue
        src = spec.get("source")
        if src == "const":
            try:
                vv = float(spec.get("value", 0.0))
            except Exception:
                vv = 0.0
            overrides[int(idx)] = _cl(vv)
        elif src == "signal":
            sid = spec.get("id")
            vv = float(eval_controller_signal(ctrl=ctrl, sid=str(sid), ctx=ctx, _memo=memo, _visiting=visiting))
            overrides[int(idx)] = _cl(vv)

    return overrides


def iter_signal_dependency_feature_ids(*, ctrl: dict[str, Any], sid: str) -> set[str]:
    """Return feature ids reachable from a signal (best-effort)."""
    sigs = ctrl.get("signals") if isinstance(ctrl, dict) else None
    if not isinstance(sigs, dict):
        return set()

    out: set[str] = set()
    seen_sigs: set[str] = set()

    def _walk_sig(s: str) -> None:
        key = str(s)
        if key in seen_sigs:
            return
        seen_sigs.add(key)
        node = sigs.get(key)
        if not isinstance(node, dict):
            return
        op = str(node.get("op", ""))
        if op == "raw":
            fid = node.get("feature")
            if isinstance(fid, str) and fid:
                out.add(str(fid))
        for ref in _iter_signal_refs(node):
            if ref.startswith("feat:"):
                out.add(ref[len("feat:") :])
            elif ref.startswith("sig:"):
                _walk_sig(ref[len("sig:") :])
            else:
                _walk_sig(ref)

    _walk_sig(str(sid))
    return out


def iter_channels_involving_axis(*, cfg: dict[str, Any], axis_index: int) -> list[int]:
    """Heuristic: return channel indices whose mapped signal depends on an axis feature."""
    fc = cfg.get("flight_controls") if isinstance(cfg, dict) else None
    ctrl = fc.get("controller") if isinstance(fc, dict) else None
    if not isinstance(ctrl, dict):
        return []

    feats = ctrl.get("features")
    chmap = ctrl.get("channels")
    if not (isinstance(feats, dict) and isinstance(chmap, dict)):
        return []

    axis_feats: set[str] = set()
    for fid, spec in feats.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("type") != "axis":
            continue
        a = spec.get("axis")
        if isinstance(a, int) and int(a) == int(axis_index):
            axis_feats.add(str(fid))

    if not axis_feats:
        return []

    out: list[int] = []
    for k, spec in chmap.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if not (0 <= idx < 8):
            continue
        if not isinstance(spec, dict):
            continue
        if spec.get("source") != "signal":
            continue
        sid = spec.get("id")
        if not isinstance(sid, str):
            continue
        deps = iter_signal_dependency_feature_ids(ctrl=ctrl, sid=sid)
        if deps.intersection(axis_feats):
            out.append(int(idx))

    return sorted(list(set(out)))
