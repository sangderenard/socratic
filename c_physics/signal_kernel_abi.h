#pragma once

// ABI for the (planned) input/signal kernel.
// This header is intentionally small and stable.
//
// IMPORTANT:
// - Keep struct packing and fields in sync with Python ctypes
// - Intended to be used across the DLL boundary

#include <stdint.h>

#include "signal_status_flags.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- Input events (Python -> C) ----

// Device IDs (suggested; not enforced by ABI)
#define GP_DEV_KEYBOARD  1u
#define GP_DEV_MOUSE     2u
#define GP_DEV_JOYSTICK  3u

// Event kinds (suggested; not enforced by ABI)
#define GP_EV_AXIS       1u
#define GP_EV_BUTTON     2u
#define GP_EV_HAT        3u
#define GP_EV_KEY        4u
#define GP_EV_MOUSE_MOTION 5u
#define GP_EV_MOUSE_BUTTON 6u
#define GP_EV_MOUSE_WHEEL  7u

#pragma pack(push, 1)

typedef struct GP_InputEvent {
  uint64_t t_mono_ns;   // monotonic timestamp at observation
  uint32_t device;      // GP_DEV_* or user-defined
  uint16_t kind;        // GP_EV_* or user-defined
  uint16_t id;          // axis/button/hat/key code, etc.
  float v0;             // primary value (axis, button 0/1, mouse dx)
  float v1;             // secondary value (hat y, mouse dy)
  uint32_t flags;       // reserved (e.g. modifier bits, device-specific)
} GP_InputEvent;

// ---- Output signal frames (C -> Python) ----

typedef struct GP_SignalFrame {
  uint64_t t_emit_ns;   // when this frame was emitted (mono ns)
  uint32_t signal_id;   // stable numeric ID (hash or index defined by kernel)
  uint32_t flags;       // SIGF_* status flags
  float value;          // signal value (-1..1 typical)
  float hold_s;         // 0 if not down; else duration since down edge (seconds)
  uint32_t aux;         // reserved (e.g. click count, toggle state)
} GP_SignalFrame;

// ---- Signal selection (for binary derived signals) ----
// Selector codes used by gp_sigk_peek_sel. These are intentionally numeric
// (not chars) so we can represent multi-token states like "2H" and "2t".
#define GP_SIGSEL_RAW                0u
#define GP_SIGSEL_AXIS               1u
#define GP_SIGSEL_DOWN               2u
#define GP_SIGSEL_DOWN_EDGE          3u
#define GP_SIGSEL_UP_EDGE            4u
#define GP_SIGSEL_HOLD               5u
#define GP_SIGSEL_DOUBLE_EDGE        6u
#define GP_SIGSEL_DOUBLE_HOLD        7u
#define GP_SIGSEL_TOGGLED            8u
#define GP_SIGSEL_TOGGLE_EDGE        9u
#define GP_SIGSEL_DOUBLE_TOGGLE_EDGE 10u

// Timer-valued selectors (values are normalized to minutes, clamped to [0..1])
#define GP_SIGSEL_HOLD_S             11u  // while down: hold_s / 60
#define GP_SIGSEL_LAST_HOLD_S        12u  // last completed hold duration / 60
#define GP_SIGSEL_LAST_HOLD_PULSE    13u  // pulse on UP_EDGE if release had HOLD: last_hold_s / 60

#pragma pack(pop)

#ifdef __cplusplus
}
#endif
