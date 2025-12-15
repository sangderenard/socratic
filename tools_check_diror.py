from __future__ import annotations

import json

import controller_graph_compile


def main() -> int:
    with open("joystick.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    g = controller_graph_compile.compile_controller_graph(cfg)
    ch0 = g.get("channels", {}).get("0")
    if not isinstance(ch0, dict):
        raise SystemExit("missing channels[0]")

    print("channels[0] =", ch0)

    # A quick sanity check: channel 0 must be dim=2.
    dim = int(ch0.get("dim", 0))
    if dim != 2:
        raise SystemExit(f"channel0 dim must be 2, got {dim}")

    # Ensure DirOR exists
    sigs = g.get("signals", {})
    if not isinstance(sigs, dict) or "DirOR" not in sigs:
        raise SystemExit("DirOR not compiled")

    print("DirOR compiled as", sigs["DirOR"])
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
