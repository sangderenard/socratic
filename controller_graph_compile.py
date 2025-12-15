from __future__ import annotations

"""Compile controller signals+channels config into a C-friendly graph JSON.

This is intentionally a *compiler* step: it turns the editable config shape in
`joystick.json` into a normalized, explicit, topologically-ordered DAG that a
C-side executor can evaluate efficiently per input-event batch.

Scope (current):
- Consume `flight_controls.controller.signals` and `flight_controls.controller.channels`.
- Support ops used in the workbench today: const, add, sub, kernel, 2dseek, 2dflightstick.
- Resolve args[] specs into explicit input/signal references.

This file does NOT implement the C executor; it produces a stable JSON artifact
that the C executor can load.
"""

from dataclasses import dataclass
from typing import Any, Literal
import json
import os
import sys
import traceback
import time


GraphValueType = Literal["f1", "f2"]


# Keep these numeric codes aligned with c_physics/signal_kernel_abi.h (GP_SIGSEL_*).
# We embed both a human label and the numeric code so the C side can be strict.
_SIGSEL_BY_STATE: dict[str, int] = {
    # buttons
    "raw": 1,
    "D": 2,
    "+": 3,
    "-": 4,
    "H": 5,
    "2": 6,
    "2H": 7,
    "T": 8,
    "t": 9,
    "2t": 10,
    # timers (normalized minutes)
    "hold_s": 11,
    "last_hold_s": 12,
    "last_hold_pulse": 13,
    # axes
    "axis": 0,
}


def _now_utc_ms() -> int:
    return int(time.time() * 1000.0)


def _controller_block(cfg: dict[str, Any]) -> dict[str, Any]:
    fc = cfg.get("flight_controls")
    if not isinstance(fc, dict):
        raise ValueError("missing flight_controls")
    ctrl = fc.get("controller")
    if not isinstance(ctrl, dict):
        raise ValueError("missing flight_controls.controller")
    return ctrl


def _signals_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    ctrl = _controller_block(cfg)
    sigs = ctrl.get("signals")
    if not isinstance(sigs, dict):
        return {}
    return sigs


def _channels_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    ctrl = _controller_block(cfg)
    ch = ctrl.get("channels")
    if not isinstance(ch, dict):
        return {}
    return ch


def _axis_invert_pref(cfg: dict[str, Any], *, device: str, kind: str, item_id: int) -> bool:
    # Device-level preference stored in the editable config (not per-signal).
    # Schema:
    # cfg["flight_controls"]["controller"]["device_prefs"][device]["axis_invert"]["<id>"] = bool
    if str(kind) not in ("axis", "mouse_motion", "motion"):
        return False
    try:
        ctrl = _controller_block(cfg)
    except Exception:
        return False
    dp = ctrl.get("device_prefs")
    if not isinstance(dp, dict):
        return False
    dblk = dp.get(str(device))
    if not isinstance(dblk, dict):
        return False
    amap = dblk.get("axis_invert")
    if not isinstance(amap, dict):
        return False
    return bool(amap.get(str(int(item_id)), False))


def _axis_calib_pref(cfg: dict[str, Any], *, device: str, kind: str, item_id: int) -> dict[str, Any]:
    # Device-level calibration stored in the editable config (not per-signal).
    # Schema:
    # cfg["flight_controls"]["controller"]["device_prefs"][device]["axis_calib"]["<id>"] = {
    #   "cap_min": <float>, "cap_max": <float>, "trim": <float>, "deadzone": <float>
    # }
    if str(kind) not in ("axis", "mouse_motion", "motion"):
        return {}
    try:
        ctrl = _controller_block(cfg)
    except Exception:
        return {}
    dp = ctrl.get("device_prefs")
    if not isinstance(dp, dict):
        return {}
    dblk = dp.get(str(device))
    if not isinstance(dblk, dict):
        return {}
    cmap = dblk.get("axis_calib")
    if not isinstance(cmap, dict):
        return {}
    node = cmap.get(str(int(item_id)))
    return dict(node) if isinstance(node, dict) else {}


def _normalize_dim(node: dict[str, Any]) -> int:
    try:
        d = int(node.get("dim", 1))
    except Exception:
        d = 1
    return 2 if d == 2 else 1


def _op_required_args(op: str) -> int:
    if op == "const":
        return 0
    if op in ("add", "sub"):
        return 2
    if op in ("2dseek", "2dflightstick"):
        return 2
    if op in ("2dsumclamp",):
        # variadic: uses args as (x0,y0,x1,y1,...) pairs; minimum one pair
        return 2
    # kernel and default passthrough
    return 1


def _op_is_variadic(op: str) -> bool:
    return str(op) in ("2dsumclamp",)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class InputKey:
    # Normalized unique key for an input source.
    device: str
    kind: str  # axis|button
    id: int
    state: str


def _parse_arg_spec(spec: dict[str, Any]) -> tuple[str, Any]:
    """Return (ref_kind, ref_payload).

    - ("input", InputKey)
    - ("signal", (sid, comp))
    """
    src = str(spec.get("source", ""))
    if src == "input":
        dev = str(spec.get("device", ""))
        kind = str(spec.get("kind", ""))
        state = str(spec.get("state", "axis"))
        if dev == "joystick":
            if kind not in ("axis", "button"):
                raise ValueError(f"unsupported input spec: {spec}")
        elif dev == "keyboard":
            if kind not in ("key", "axis"):
                raise ValueError(f"unsupported input spec: {spec}")
        elif dev == "mouse":
            if kind not in ("mouse_motion",):
                raise ValueError(f"unsupported input spec: {spec}")
        else:
            raise ValueError(f"unsupported input device: {spec}")
        try:
            item_id = int(spec.get("id", 0))
        except Exception:
            item_id = 0
        return "input", InputKey(device=dev, kind=kind, id=int(item_id), state=state)

    if src == "signal":
        sid = str(spec.get("id", ""))
        comp = str(spec.get("comp", ""))
        comp2 = comp if comp else None
        if not sid:
            raise ValueError(f"invalid signal spec: {spec}")
        return "signal", (sid, comp2)

    raise ValueError(f"unknown arg spec source: {spec}")


def compile_controller_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compile config into an explicit graph JSON.

    Output schema (v1):
    {
      "version": 1,
      "generated_utc_ms": <int>,
      "inputs": [ {"iid":0, "device":"joystick", "kind":"axis|button", "id":<int>, "state":"...", "sigsel":<int>} ...],
      "nodes": [ {"nid":0, "op":"...", "dim":1|2, "args":[{"ref":"input","iid":...}|{"ref":"node","nid":...}], ... } ... ],
      "signals": {"sid": {"nid":..., "dim":...} ...},
      "channels": {"0": {"sid":"...", "dim":...} ...}
    }

    Notes:
    - Each signal compiles to exactly one node (even if dim=2).
    - Inputs are de-duplicated.
    - Nodes are topologically ordered by construction.
    """

    sigs = _signals_dict(cfg)
    # stable iteration order
    sids = sorted([str(k) for k in sigs.keys()])

    # Input table (dedup)
    input_ids: dict[InputKey, int] = {}
    inputs_out: list[dict[str, Any]] = []

    def _get_input_iid(key: InputKey) -> int:
        got = input_ids.get(key)
        if isinstance(got, int):
            return int(got)
        iid = int(len(inputs_out))
        sigsel = int(_SIGSEL_BY_STATE.get(str(key.state), 0))
        invert = bool(_axis_invert_pref(cfg, device=str(key.device), kind=str(key.kind), item_id=int(key.id)))
        calib = _axis_calib_pref(cfg, device=str(key.device), kind=str(key.kind), item_id=int(key.id))
        cap_min = calib.get("cap_min")
        cap_max = calib.get("cap_max")
        trim = calib.get("trim")
        deadzone = calib.get("deadzone")
        item: dict[str, Any] = {
            "iid": int(iid),
            "device": str(key.device),
            "kind": str(key.kind),
            "id": int(key.id),
            "state": str(key.state),
            "sigsel": int(sigsel),
            "invert": bool(invert),
        }
        if cap_min is not None:
            item["cap_min"] = float(cap_min)
        if cap_max is not None:
            item["cap_max"] = float(cap_max)
        if trim is not None:
            item["trim"] = float(trim)
        if deadzone is not None:
            item["deadzone"] = float(deadzone)
        inputs_out.append(item)
        input_ids[key] = int(iid)
        return int(iid)

    # Node compilation (memoize by signal id)
    node_ids: dict[str, int] = {}
    nodes_out: list[dict[str, Any]] = []

    def _compile_signal(sid: str) -> int:
        got = node_ids.get(str(sid))
        if isinstance(got, int):
            return int(got)
        node = sigs.get(str(sid))
        if not isinstance(node, dict):
            raise ValueError(f"signal node not a dict: {sid}")

        op = str(node.get("op", "const"))
        dim = _normalize_dim(node)
        req = _op_required_args(op)

        args_in = node.get("args")
        args_list: list[Any] = list(args_in) if isinstance(args_in, list) else []
        if _op_is_variadic(op):
            req = max(int(req), int(len(args_list)))
            # For 2D variadic ops, keep args count even (x/y pairs).
            if int(req) % 2 == 1:
                req += 1
        while len(args_list) < int(req):
            args_list.append(None)

        args_out: list[dict[str, Any]] = []
        for i in range(int(req)):
            spec = args_list[i]
            if spec is None:
                # allow unbound arg; C executor can treat as 0.
                args_out.append({"ref": "imm", "value": 0.0})
                continue
            if not isinstance(spec, dict):
                raise ValueError(f"invalid arg spec for {sid}[{i}]: {spec}")
            ref_kind, payload = _parse_arg_spec(spec)
            if ref_kind == "input":
                iid = _get_input_iid(payload)
                args_out.append({"ref": "input", "iid": int(iid)})
            else:
                s2, comp2 = payload
                nid_dep = _compile_signal(str(s2))
                args_out.append({"ref": "node", "nid": int(nid_dep), "comp": (str(comp2) if comp2 else "x")})

        out_node: dict[str, Any] = {
            "nid": int(len(nodes_out)),
            "op": op,
            "dim": int(dim),
            "args": args_out,
        }

        if op == "const":
            out_node["value"] = _as_float(node.get("value", 0.0), 0.0)
        if op == "kernel":
            # kernel nodes can still be represented as an input-derived node.
            try:
                out_node["signal_id"] = int(node.get("signal_id"))
            except Exception:
                out_node["signal_id"] = 0
            out_node["state"] = str(node.get("state", ""))
            out_node["sigsel"] = int(_SIGSEL_BY_STATE.get(str(out_node["state"]), 0))

        nodes_out.append(out_node)
        node_ids[str(sid)] = int(out_node["nid"])
        return int(out_node["nid"])

    # Compile all signals
    signals_out: dict[str, Any] = {}
    for sid in sids:
        nid = _compile_signal(str(sid))
        dim = _normalize_dim(sigs[str(sid)])
        signals_out[str(sid)] = {"nid": int(nid), "dim": int(dim)}

    # Channels mapping (as-is, no massaging)
    channels_in = _channels_dict(cfg)
    channels_out: dict[str, Any] = {}
    for k, spec in channels_in.items():
        try:
            ch_idx = int(k)
        except Exception:
            continue
        if not isinstance(spec, dict):
            continue
        if str(spec.get("source")) != "signal":
            continue
        sid = str(spec.get("id", ""))
        if sid not in signals_out:
            continue
        channels_out[str(ch_idx)] = {"sid": str(sid), "dim": int(signals_out[str(sid)]["dim"])}

    return {
        "version": 1,
        "generated_utc_ms": int(_now_utc_ms()),
        "inputs": inputs_out,
        "nodes": nodes_out,
        "signals": signals_out,
        "channels": channels_out,
    }


def finalize_compiled_graph(compiled: dict[str, Any], *, mixer_routes: dict[str, str] | None) -> dict[str, Any]:
    """Merge an (out_channel -> in_channel) routing table onto a compiled graph.

    This is stage-2 compilation: it does not add new ops; it just defines how
    channels are routed into the final output channel buffers.

    Output keeps the original compiled graph plus:
    - mixer.routes: explicit remaps
    - final_channels: resolved output channels with sid/dim and source channel
    """
    base_channels = compiled.get("channels")
    if not isinstance(base_channels, dict):
        base_channels = {}

    routes_in = mixer_routes or {}
    routes: dict[str, str] = {}
    for out_ch, in_ch in routes_in.items():
        try:
            o = str(int(out_ch))
            i = str(int(in_ch))
        except Exception:
            continue
        if o not in base_channels:
            continue
        if i not in base_channels:
            continue
        if o == i:
            continue
        routes[o] = i

    final_channels: dict[str, Any] = {}
    for out_ch in sorted(base_channels.keys(), key=lambda s: int(s) if str(s).isdigit() else 10_000):
        src_ch = routes.get(str(out_ch), str(out_ch))
        spec = base_channels.get(str(src_ch))
        if not isinstance(spec, dict):
            continue
        final_channels[str(out_ch)] = {
            "from": str(src_ch),
            "sid": str(spec.get("sid", "")),
            "dim": int(spec.get("dim", 1)),
        }

    out = dict(compiled)
    out["final_generated_utc_ms"] = int(_now_utc_ms())
    out["mixer"] = {
        "routes": routes,
        "outputs": sorted(list(base_channels.keys()), key=lambda s: int(s) if str(s).isdigit() else 10_000),
    }
    out["final_channels"] = final_channels

    # Enforce channel0 shape: menu/navigation expects a 2D vector.
    if "0" in out.get("final_channels", {}):
        try:
            d0 = int(out["final_channels"]["0"].get("dim", 1))
        except Exception:
            d0 = 1
        if int(d0) != 2:
            raise ValueError("channel 0 must be dim=2 (DirOR)")
    return out


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _load_mixer_routes(mixer_path: str) -> dict[str, str] | None:
    if not mixer_path:
        return None
    if not os.path.exists(mixer_path):
        return None
    try:
        with open(str(mixer_path), "r", encoding="utf-8") as f:
            m = json.load(f)
        if not isinstance(m, dict):
            return None
        rr = m.get("routes") if "routes" in m else m
        if not isinstance(rr, dict):
            return None
        out: dict[str, str] = {}
        for k, v in rr.items():
            try:
                out[str(int(k))] = str(int(v))
            except Exception:
                continue
        return out
    except Exception:
        return None


def try_build_final_graph(
    *,
    joystick_path: str = "joystick.json",
    mixer_path: str = "channel_mixer.json",
    compiled_out_path: str = "controller_graph_compiled.json",
    final_out_path: str = "controller_graph_final.json",
) -> tuple[bool, str, dict[str, Any] | None, dict[str, Any] | None]:
    """Stage-1 compile + stage-2 mixer finalize.

    Returns (ok, status_msg, compiled_dict, final_dict).
    Soft-fails: never raises.
    """
    try:
        with open(str(joystick_path), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return False, "joystick config is not a JSON object", None, None

        compiled = compile_controller_graph(cfg)
        if compiled_out_path:
            _atomic_write_json(str(compiled_out_path), compiled)

        mixer_routes = _load_mixer_routes(str(mixer_path))
        final_graph = finalize_compiled_graph(compiled, mixer_routes=mixer_routes)
        if final_out_path:
            _atomic_write_json(str(final_out_path), final_graph)

        return True, "ACTIVE (compile ok)", compiled, final_graph
    except Exception as e:
        # Emit diagnostics to stderr for future debugging.
        try:
            print("[controller_graph_compile] try_build_final_graph failed", file=sys.stderr, flush=True)
            print(f"  joystick_path={joystick_path}", file=sys.stderr, flush=True)
            print(f"  mixer_path={mixer_path}", file=sys.stderr, flush=True)
            print(f"  compiled_out_path={compiled_out_path}", file=sys.stderr, flush=True)
            print(f"  final_out_path={final_out_path}", file=sys.stderr, flush=True)
            if isinstance(e, json.JSONDecodeError):
                print(
                    f"  JSONDecodeError: {e.msg} (line {e.lineno}, col {e.colno})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"  {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass

        msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}"
        if len(msg) > 180:
            msg = msg[:177] + "..."
        return False, msg, None, None


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Compile joystick controller signals+channels into a C-friendly graph JSON")
    ap.add_argument("--in", dest="inp", default="joystick.json", help="Input joystick config JSON")
    ap.add_argument("--out", dest="out", default="controller_graph_compiled.json", help="Output compiled graph JSON")
    ap.add_argument("--mixer", dest="mixer", default="", help="Optional channel mixer JSON (routes) to merge")
    ap.add_argument("--final", dest="final", default="", help="Optional final graph output JSON (after merging mixer)")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out = compile_controller_graph(cfg)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {args.out} (nodes={len(out['nodes'])}, inputs={len(out['inputs'])}, channels={len(out['channels'])})")

    if str(args.final):
        mixer_routes: dict[str, str] | None = None
        if str(args.mixer):
            try:
                with open(str(args.mixer), "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, dict):
                    rr = m.get("routes") if "routes" in m else m
                    if isinstance(rr, dict):
                        mixer_routes = {str(k): str(v) for k, v in rr.items()}
            except Exception:
                mixer_routes = None
        final_graph = finalize_compiled_graph(out, mixer_routes=mixer_routes)
        with open(str(args.final), "w", encoding="utf-8") as f:
            json.dump(final_graph, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.final} (final_channels={len(final_graph.get('final_channels', {}))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
