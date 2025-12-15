#include "geodesic_physics.h"

#include <string.h>

// A minimal signal-kernel implementation intended as a live test harness:
// - Accepts input events (currently button/key semantics are meaningful)
// - Maintains per-signal state: down/hold/double/toggle
// - Allows Python to query an immediate GP_SignalFrame (gp_sigk_peek)
//
// This is intentionally small; it will evolve into the fixed-rate scheduler.

#ifndef SIGK_MAX_SIGNALS
#define SIGK_MAX_SIGNALS 512u
#endif

// Timing thresholds (ns)
#ifndef SIGK_DOUBLE_WINDOW_NS
#define SIGK_DOUBLE_WINDOW_NS (300000000ull) // 0.30s
#endif
#ifndef SIGK_HOLD_THRESHOLD_NS
#define SIGK_HOLD_THRESHOLD_NS (350000000ull) // 0.35s
#endif

// Internal pulse bits (subset of SIGF_* that should clear after being reported)
#define SIGK_P_DOWN_EDGE           SIGF_DOWN_EDGE
#define SIGK_P_UP_EDGE             SIGF_UP_EDGE
#define SIGK_P_DOUBLE_EDGE         SIGF_DOUBLE_EDGE
#define SIGK_P_TOGGLE_EDGE         SIGF_TOGGLE_EDGE
#define SIGK_P_DOUBLE_TOGGLE_EDGE  SIGF_DOUBLE_TOGGLE_EDGE

typedef struct SigK_State {
  uint32_t signal_id;
  uint8_t in_use;
  uint8_t down;
  uint8_t toggled;
  uint8_t dbl_pending;

  uint64_t down_since_ns;
  uint64_t last_down_ns;

  // Duration (seconds) of the last completed down->up interval.
  float last_hold_s;
  // True if the most recent release qualified as a HOLD at release.
  uint8_t last_release_had_hold;

  uint32_t pulse_flags;
  float last_value;
} SigK_State;

static SigK_State g_states[SIGK_MAX_SIGNALS];

static uint32_t hash_u32(uint32_t x) {
  // cheap reversible-ish mix
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static SigK_State* get_state(uint32_t signal_id, int create) {
  const uint32_t h = hash_u32(signal_id);
  uint32_t idx = h % SIGK_MAX_SIGNALS;
  for (uint32_t probe = 0; probe < SIGK_MAX_SIGNALS; ++probe) {
    SigK_State* st = &g_states[idx];
    if (!st->in_use) {
      if (!create) return NULL;
      memset(st, 0, sizeof(*st));
      st->in_use = 1;
      st->signal_id = signal_id;
      return st;
    }
    if (st->signal_id == signal_id) return st;
    idx = (idx + 1u) % SIGK_MAX_SIGNALS;
  }
  return NULL;
}

static uint32_t compose_signal_id_u32(uint32_t device, uint32_t kind, uint32_t id) {
  return ((device & 0xFFu) << 24) | ((kind & 0xFFu) << 16) | (id & 0xFFFFu);
}

GP_EXPORT void gp_sigk_reset(void) {
  memset(g_states, 0, sizeof(g_states));
}

static void apply_button_like(SigK_State* st, uint64_t t_ns, int is_down, float value) {
  const int was_down = (st->down != 0);

  if (is_down && !was_down) {
    st->pulse_flags |= SIGK_P_DOWN_EDGE;

    // double detection: second down within window
    if (st->last_down_ns != 0 && (t_ns - st->last_down_ns) <= SIGK_DOUBLE_WINDOW_NS) {
      st->pulse_flags |= SIGK_P_DOUBLE_EDGE;
      st->dbl_pending = 1;
    } else {
      st->dbl_pending = 0;
    }

    st->last_down_ns = t_ns;
    st->down_since_ns = t_ns;
    st->down = 1;

    // toggle on each press edge; annotate whether it was double
    st->toggled = (uint8_t)(!st->toggled);
    st->pulse_flags |= SIGK_P_TOGGLE_EDGE;
    if ((st->pulse_flags & SIGK_P_DOUBLE_EDGE) != 0) {
      st->pulse_flags |= SIGK_P_DOUBLE_TOGGLE_EDGE;
    }
  } else if (!is_down && was_down) {
    // Capture the just-finished hold duration.
    if (st->down_since_ns != 0 && t_ns >= st->down_since_ns) {
      const uint64_t held_ns = (t_ns - st->down_since_ns);
      st->last_hold_s = (float)((double)held_ns * 1e-9);
      st->last_release_had_hold = (held_ns >= SIGK_HOLD_THRESHOLD_NS) ? 1u : 0u;
    } else {
      st->last_hold_s = 0.0f;
      st->last_release_had_hold = 0u;
    }
    st->pulse_flags |= SIGK_P_UP_EDGE;
    st->down = 0;
    st->down_since_ns = 0;
    st->dbl_pending = 0;
  }

  st->last_value = value;
}

GP_EXPORT void gp_sigk_push_events(const GP_InputEvent* ev, uint32_t count) {
  if (!ev || count == 0) return;

  for (uint32_t i = 0; i < count; ++i) {
    const GP_InputEvent* e = &ev[i];
    const uint32_t sid = compose_signal_id_u32(e->device, (uint32_t)e->kind, (uint32_t)e->id);
    SigK_State* st = get_state(sid, 1);
    if (!st) continue;

    // For now only button/key/mouse-button are treated as button-like.
    if (e->kind == GP_EV_BUTTON || e->kind == GP_EV_KEY || e->kind == GP_EV_MOUSE_BUTTON) {
      const int down = (e->v0 >= 0.5f) ? 1 : 0;
      const float value = down ? 1.0f : 0.0f;
      apply_button_like(st, e->t_mono_ns, down, value);
    } else {
      // For other kinds, just store the instantaneous value.
      st->last_value = e->v0;
    }
  }
}

GP_EXPORT int gp_sigk_peek(uint64_t now_ns, uint32_t signal_id, GP_SignalFrame* out) {
  if (!out) return 0;
  SigK_State* st = get_state(signal_id, 0);
  if (!st) return 0;

  uint32_t flags = 0;
  float hold_s = 0.0f;

  if (st->down) {
    flags |= SIGF_DOWN;
    if (st->down_since_ns != 0 && now_ns >= st->down_since_ns) {
      const uint64_t held_ns = (now_ns - st->down_since_ns);
      hold_s = (float)((double)held_ns * 1e-9);
      if (held_ns >= SIGK_HOLD_THRESHOLD_NS) {
        flags |= SIGF_HOLD;
      }
      if (st->dbl_pending && held_ns >= SIGK_HOLD_THRESHOLD_NS) {
        flags |= SIGF_DOUBLE_HOLD;
      }
    }
  }

  if (st->toggled) {
    flags |= SIGF_TOGGLED;
  }

  // Pulses persist until gp_sigk_clear_pulses() is called.
  // This allows multiple queries per frame without consuming edges.
  flags |= st->pulse_flags;

  out->t_emit_ns = now_ns;
  out->signal_id = signal_id;
  out->flags = flags;
  out->value = st->down ? 1.0f : st->last_value;
  out->hold_s = hold_s;
  // aux: last completed hold duration in milliseconds (for UI/debug)
  double ms = (double)st->last_hold_s * 1000.0;
  if (ms < 0.0) ms = 0.0;
  if (ms > 4294967295.0) ms = 4294967295.0;
  out->aux = (uint32_t)(ms + 0.5);

  return 1;
}

GP_EXPORT void gp_sigk_clear_pulses(void) {
  for (uint32_t i = 0; i < SIGK_MAX_SIGNALS; ++i) {
    SigK_State* st = &g_states[i];
    if (!st->in_use) continue;
    st->pulse_flags = 0;
    // last_release_had_hold is only used to gate the release pulse.
    // Clear it with pulses so it doesn't persist across frames.
    st->last_release_had_hold = 0u;
  }
}

GP_EXPORT void gp_sigk_sigtobutton(uint64_t now_ns, uint32_t out_button_signal_id, float value, float epsilon) {
  SigK_State* st = get_state(out_button_signal_id, 1);
  if (!st) return;

  // Interpret value > epsilon as pressed.
  const int down = (value > epsilon) ? 1 : 0;
  const float v = down ? 1.0f : 0.0f;
  apply_button_like(st, now_ns, down, v);
}

// ---------------- Signal operator helpers (prototype) ----------------

GP_EXPORT void gp_sigop_2dseek(float a, float b, float* out_x, float* out_y) {
  if (out_x) *out_x = a;
  if (out_y) *out_y = b;
}

GP_EXPORT void gp_sigop_2dflightstick(float a, float b, float* out_x, float* out_y) {
  // Conventional flight stick: X = roll, Y = pitch; pulling back usually means pitch up.
  // Many joystick Y axes report up as negative, so invert the second axis here.
  if (out_x) *out_x = a;
  if (out_y) *out_y = -b;
}

static float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

GP_EXPORT void gp_sigop_2dstereocontrolsurface(float x, float y, float* roll_left, float* roll_right, float* pitch_up, float* pitch_down) {
  // Interpret x as roll command (-1..1), y as pitch command (-1..1).
  // Split into unilateral channels for downstream linkage logic.
  const float rl = clamp01(-x);
  const float rr = clamp01(+x);
  const float pu = clamp01(+y);
  const float pd = clamp01(-y);
  if (roll_left) *roll_left = rl;
  if (roll_right) *roll_right = rr;
  if (pitch_up) *pitch_up = pu;
  if (pitch_down) *pitch_down = pd;
}

GP_EXPORT int gp_sigk_peek_sel(uint64_t now_ns, uint32_t signal_id, uint32_t selector, GP_SignalFrame* out) {
  if (!out) return 0;
  GP_SignalFrame tmp;
  if (!gp_sigk_peek(now_ns, signal_id, &tmp)) return 0;

  SigK_State* st = get_state(signal_id, 0);

  float v = tmp.value;
  const uint32_t f = tmp.flags;
  switch (selector) {
    case GP_SIGSEL_RAW:
    case GP_SIGSEL_AXIS:
      v = tmp.value;
      break;
    case GP_SIGSEL_DOWN:
      v = (f & SIGF_DOWN) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_DOWN_EDGE:
      v = (f & SIGF_DOWN_EDGE) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_UP_EDGE:
      v = (f & SIGF_UP_EDGE) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_HOLD:
      v = (f & SIGF_HOLD) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_DOUBLE_EDGE:
      v = (f & SIGF_DOUBLE_EDGE) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_DOUBLE_HOLD:
      v = (f & SIGF_DOUBLE_HOLD) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_TOGGLED:
      v = (f & SIGF_TOGGLED) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_TOGGLE_EDGE:
      v = (f & SIGF_TOGGLE_EDGE) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_DOUBLE_TOGGLE_EDGE:
      v = (f & SIGF_DOUBLE_TOGGLE_EDGE) ? 1.0f : 0.0f;
      break;
    case GP_SIGSEL_HOLD_S: {
      // Normalize to minutes in [0..1]
      float x = tmp.hold_s / 60.0f;
      if (x < 0.0f) x = 0.0f;
      if (x > 1.0f) x = 1.0f;
      v = x;
      break;
    }
    case GP_SIGSEL_LAST_HOLD_S: {
      float x = (st ? st->last_hold_s : 0.0f) / 60.0f;
      if (x < 0.0f) x = 0.0f;
      if (x > 1.0f) x = 1.0f;
      v = x;
      break;
    }
    case GP_SIGSEL_LAST_HOLD_PULSE: {
      // Pulse concurrent with UP_EDGE only if the release had HOLD.
      if ((f & SIGF_UP_EDGE) != 0 && st && st->last_release_had_hold) {
        float x = st->last_hold_s / 60.0f;
        if (x < 0.0f) x = 0.0f;
        if (x > 1.0f) x = 1.0f;
        v = x;
      } else {
        v = 0.0f;
      }
      break;
    }
    default:
      v = tmp.value;
      break;
  }

  *out = tmp;
  out->value = v;
  return 1;
}
