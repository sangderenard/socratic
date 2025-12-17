# c_menu

Experimental menu/UI support code.

## Goal
Provide a **C++ layout/data module** that emits numeric buffers for UI composition (rect arrays, draw ops, hit targets) while keeping rendering and event routing on the Python side.

This folder starts with the first UI archetype you requested:

### Pixel waveform “signal window”
A small widget that renders **raw, frame-by-frame** signal history as pixels.

Key properties:
- No smoothing/shaping here (that belongs to the channel mixer).
- “Tearing is OK”: consumers read the latest ring buffer state.
- The widget compiles to **pixel ops / texture keys / byte spans**, not pygame/OpenGL calls.

## Important dependency: recurring sampler in the C controller backend
A waveform needs a stable cadence. The cleanest model is:
- C controller backend already has a fixed-rate tick (e.g. the controller engine `gp_ctl_start(rate_hz, ...)`).
- Each tick, it can sample a chosen scalar (wheel signal index or channel) and append to a waveform ring.

This is likely incomplete today. The intent of the API in this folder is:
1. Define the ring buffer + pixelization output in a self-contained way.
2. Later, attach a sampler hook from the controller engine tick loop.

## API sketch
See:
- `workbench_rows_abi.{h,cpp}` (C ABI for rasterizing legacy workbench rows)

Next integration step (not done yet):
