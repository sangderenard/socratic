#pragma once

// ABI for the controller engine (compiled graph executor).
//
// Goals:
// - Stable packed structs across the DLL boundary
// - Dimension-agnostic output layout via (dim, stride, offset)
// - Lock-free-ish output ring reads (single-writer, multi-reader)
//
// IMPORTANT:
// - Keep struct packing and fields in sync with Python ctypes
// - Intended to be used across the DLL boundary

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)

// Binary graph blob header.
// Stored little-endian.
#define GP_CTL_BLOB_MAGIC 0x4C544347u  // 'GCTL'
#define GP_CTL_BLOB_VERSION 1u

typedef struct GP_CtlBlobHeader {
  uint32_t magic;       // GP_CTL_BLOB_MAGIC
  uint32_t version;     // GP_CTL_BLOB_VERSION
  uint32_t input_count;
  uint32_t node_count;
  uint32_t arg_count;
  uint32_t output_count;
  uint32_t _reserved0;
  uint32_t _reserved1;
} GP_CtlBlobHeader;

// Input descriptor (scalar).
// Mirrors controller_backend._apply_input_calib() behavior.
typedef struct GP_CtlInputDesc {
  uint32_t iid;      // input index used by node args
  uint32_t signal_id; // compose_signal_id(device, kind, id)
  uint32_t sigsel;   // GP_SIGSEL_* selector
  uint32_t flags;    // bit0: invert
  float cap_min;     // if cap_max>cap_min, normalize to [-1..1]
  float cap_max;
  float trim;        // [-0.95..0.95], 0 to disable
  float deadzone;    // [0..1], 0 to disable
} GP_CtlInputDesc;

// Node op codes.
// Keep aligned with controller_graph_compile.py _CTL_OP.
#define GP_CTL_OP_PASSTHROUGH 0u
#define GP_CTL_OP_CONST 1u
#define GP_CTL_OP_ADD 2u
#define GP_CTL_OP_SUB 3u
#define GP_CTL_OP_KERNEL 4u
#define GP_CTL_OP_2DSEEK 10u
#define GP_CTL_OP_2DFLIGHTSTICK 11u
#define GP_CTL_OP_2DSUMCLAMP 12u

// Node arg reference.
// ref:
// - 0: input (index=iid)
// - 1: node  (index=nid, comp selects x/y for scalar reads from dim=2)
// - 2: imm   (use imm)
typedef struct GP_CtlArg {
  uint8_t ref;
  uint8_t comp;     // 0: x/default, 1: y
  uint16_t _pad0;
  int32_t index;    // iid or nid
  float imm;
} GP_CtlArg;

typedef struct GP_CtlNodeDesc {
  uint32_t nid;
  uint16_t op;
  uint16_t dim;      // 1 or 2 (reserved for N)
  uint32_t argc;
  uint32_t args_offset; // into args[]
  float value;       // const value
  uint32_t signal_id; // for kernel op
  uint32_t sigsel;    // for kernel op
} GP_CtlNodeDesc;

// Final outputs published by the controller engine.
// offset is in float elements into the packed output vector.
// stride is elements between adjacent components (usually 1).
typedef struct GP_CtlOutputDesc {
  uint32_t channel;
  uint16_t dim;
  uint16_t stride;
  uint32_t offset;
  uint32_t nid; // node that produces this output
} GP_CtlOutputDesc;

// Meta about the active engine instance.
typedef struct GP_CtlMeta {
  uint32_t output_count;
  uint32_t total_floats;
  uint32_t ring_capacity;
  uint32_t tick_hz;
  uint64_t write_seq;
} GP_CtlMeta;

// ---------------- Hook dispatch (prototype) ----------------
// The controller engine can emit "hook" events based on watched channel values.
// Intended usage:
// - Python registers watches (channel + component + threshold + edge type)
// - C tick thread detects edges and enqueues GP_CtlHookEvent into a ring
// - A Python "parked runner" thread blocks on gp_ctl_hookq_wait() and drains events

// Hook watch flags.
// If multiple are set, multiple edges can trigger.
#define GP_CTL_HOOK_ON_RISE  (1u << 0)
#define GP_CTL_HOOK_ON_FALL  (1u << 1)
#define GP_CTL_HOOK_LEVEL    (1u << 2) // emit every tick while above threshold

// Hook source selector.
// Default is to watch a controller output channel component.
// If GP_CTL_HOOK_SRC_SIGNAL is set, the watch reads directly from the signal kernel
// using (signal_id, sigsel) and ignores (channel, comp).
#define GP_CTL_HOOK_SRC_SIGNAL (1u << 8)

// A watch describes how to turn a channel component into hook events.
// - comp: which component to monitor for vector outputs (0=x, 1=y, ...)
// - threshold: compare against value
// - hysteresis: to avoid chatter around threshold (0 disables)
typedef struct GP_CtlHookWatch {
  uint32_t hook_id;
  uint32_t channel;
  uint16_t comp;
  uint16_t flags;
  float threshold;
  float hysteresis;
  uint32_t signal_id;
  uint32_t sigsel;
} GP_CtlHookWatch;

// Hook event emitted by the controller engine.
// value is the monitored scalar component that triggered the event.
typedef struct GP_CtlHookEvent {
  uint64_t t_ns;
  uint32_t hook_id;
  uint32_t channel;
  uint16_t comp;
  uint16_t flags; // which condition triggered (rise/fall/level)
  float value;
  float _pad0;
} GP_CtlHookEvent;

typedef struct GP_CtlHookQMeta {
  uint32_t capacity;
  uint32_t _pad0;
  uint64_t write_seq;
} GP_CtlHookQMeta;

// ---------------- Passthrough outputs (reserved channel IDs) ----------------
// A passthrough output appends a scalar to the packed output vector.
// It is identified by a stable numeric channel and reads directly from the
// signal kernel via (signal_id, sigsel).
//
// This is intended to "button up" the black box: gameplay bindings can bind to
// numeric channels (often in a reserved high range) even when no explicit
// controller-graph signal was authored for that input.

typedef struct GP_CtlPassthruDesc {
  uint32_t channel;   // numeric controller channel id (stable)
  uint32_t signal_id; // compose_signal_id(device, kind, id)
  uint32_t sigsel;    // GP_SIGSEL_* selector
  uint32_t flags;     // reserved for future (invert, etc)
} GP_CtlPassthruDesc;

// ---------------- Signal wheel (per-signal history + sticky flags) ----------------
// A "signal" here is a scalar slot in the controller engine's published signal space.
// In this repo, the primary signal space is the packed output scalars (total_floats).
// Optional pass-through scalars can be appended for raw signal-kernel sources.

// Sticky flags for signals.
#define GP_WHEEL_F_HOT     (1u << 0) // saw nonzero activity since last clear
#define GP_WHEEL_F_CHANGED (1u << 1) // saw value change since last clear

typedef struct GP_WheelSample {
  uint64_t t_ns;
  float value;
  uint32_t _pad0;
} GP_WheelSample;

typedef struct GP_WheelMeta {
  uint32_t signal_count;   // total scalar slots
  uint32_t history_len;    // samples per signal
  uint64_t tick_seq;       // monotonic tick counter (wakes waiters)
} GP_WheelMeta;

// ---------------- Timers (prototype) ----------------
// Timers are backend-owned periodic pulses emitted by the controller engine.
//
// Implementation notes:
// - The controller engine runs at a fixed tick rate (tick_hz).
// - Each timer is evaluated in tick units and writes a button-like state
//   into the signal kernel each tick.
// - Menus can watch these as normal signal-kernel signals (e.g. GP_SIGSEL_DOWN).

// Timer flags.
#define GP_CTL_TIMER_REPEAT (1u << 0) // repeat forever (default)

typedef struct GP_CtlTimerDesc {
  uint32_t timer_id;      // user-defined logical id
  uint32_t period_ticks;  // >=1 ticks per cycle
  uint32_t duty_ticks;    // >=1 ticks high within the cycle
  uint32_t phase_ticks;   // [0..period_ticks-1] start offset
  uint32_t flags;         // GP_CTL_TIMER_*
  uint32_t _reserved0;
} GP_CtlTimerDesc;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif
