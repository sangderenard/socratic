#include "geodesic_physics.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
  #include <time.h>
  #include <errno.h>
  #include <semaphore.h>
  #include <stdatomic.h>
#endif

// ---------------- Cross-platform atomics/fences ----------------

#ifdef _WIN32
  #define GP_ATOMIC_U64 volatile LONGLONG
  #define GP_ATOMIC_U32 volatile LONG

  static inline uint64_t gp_atomic_u64_load(GP_ATOMIC_U64* p) {
    return (uint64_t)InterlockedCompareExchange64((volatile LONGLONG*)p, 0, 0);
  }
  static inline void gp_atomic_u64_store(GP_ATOMIC_U64* p, uint64_t v) {
    InterlockedExchange64((volatile LONGLONG*)p, (LONGLONG)v);
  }
  static inline uint32_t gp_atomic_u32_load(GP_ATOMIC_U32* p) {
    return (uint32_t)InterlockedCompareExchange(p, 0, 0);
  }
  static inline void gp_atomic_u32_store(GP_ATOMIC_U32* p, uint32_t v) {
    InterlockedExchange(p, (LONG)v);
  }
  static inline void gp_fence_release(void) { MemoryBarrier(); }
  static inline void gp_fence_acquire(void) { MemoryBarrier(); }
#else
  #define GP_ATOMIC_U64 _Atomic uint64_t
  #define GP_ATOMIC_U32 _Atomic uint32_t

  static inline uint64_t gp_atomic_u64_load(GP_ATOMIC_U64* p) {
    return atomic_load_explicit(p, memory_order_acquire);
  }
  static inline void gp_atomic_u64_store(GP_ATOMIC_U64* p, uint64_t v) {
    atomic_store_explicit(p, v, memory_order_release);
  }
  static inline uint32_t gp_atomic_u32_load(GP_ATOMIC_U32* p) {
    return atomic_load_explicit(p, memory_order_acquire);
  }
  static inline void gp_atomic_u32_store(GP_ATOMIC_U32* p, uint32_t v) {
    atomic_store_explicit(p, v, memory_order_release);
  }
  static inline void gp_fence_release(void) { atomic_thread_fence(memory_order_release); }
  static inline void gp_fence_acquire(void) { atomic_thread_fence(memory_order_acquire); }
#endif

// Single loaded graph instance.

typedef struct GP_CtlGraph {
  GP_CtlInputDesc* inputs;
  uint32_t input_count;

  GP_CtlNodeDesc* nodes;
  uint32_t node_count;

  GP_CtlArg* args;
  uint32_t arg_count;

  GP_CtlOutputDesc* outputs;
  uint32_t output_count;

  uint32_t total_floats;
  uint32_t tick_hz;
} GP_CtlGraph;

static GP_CtlGraph g_graph;

// Passthrough outputs appended after compiled graph outputs.
// Each passthrough adds exactly one scalar to the packed output vector.
static GP_CtlPassthruDesc* g_passthru = NULL;
static uint32_t g_passthru_count = 0;
static uint32_t g_passthru_cap = 0;

static uint32_t ctl_total_floats(void) {
  return g_graph.total_floats + g_passthru_count;
}

static uint32_t ctl_total_outputs(void) {
  return g_graph.output_count + g_passthru_count;
}

// Reusable scratch (single-writer).
static float* g_in_vals = NULL;
static float* g_node_out_1d = NULL;
static float* g_node_out_2d = NULL;
static uint16_t* g_node_dim = NULL;
static float* g_tick_values = NULL; // total_floats

// Output ring (single-writer).
static float* g_ring_values = NULL;      // ring_capacity * total_floats
static uint64_t* g_ring_t_ns = NULL;     // ring_capacity
static GP_ATOMIC_U64* g_ring_seq = NULL; // ring_capacity
static uint32_t g_ring_capacity = 0;
static uint32_t g_ring_mask = 0;

#ifdef _WIN32
static GP_ATOMIC_U64 g_write_seq = 0;
static HANDLE g_ctl_thread = NULL;
static GP_ATOMIC_U32 g_ctl_run = 0;
#else
static GP_ATOMIC_U64 g_write_seq = 0;
static pthread_t g_ctl_thread;
static int g_ctl_thread_valid = 0;
static GP_ATOMIC_U32 g_ctl_run = 0;
#endif

// Hook watches + hook queue (single-writer: controller tick thread).
static GP_CtlHookWatch* g_hooks = NULL;
static float* g_hook_prev_above = NULL; // 0/1 per hook
static uint32_t g_hook_count = 0;

static GP_CtlHookEvent* g_hookq_events = NULL;
static GP_ATOMIC_U64* g_hookq_seq = NULL;
static uint32_t g_hookq_capacity = 0;
static uint32_t g_hookq_mask = 0;
static GP_ATOMIC_U64 g_hookq_write_seq = 0;

#ifdef _WIN32
static HANDLE g_hookq_sem = NULL;
#else
static sem_t g_hookq_sem;
static int g_hookq_sem_inited = 0;
#endif

static void ctl_free_graph(void) {
  free(g_graph.inputs);
  free(g_graph.nodes);
  free(g_graph.args);
  free(g_graph.outputs);
  memset(&g_graph, 0, sizeof(g_graph));
}

static void ctl_free_passthru(void) {
  free(g_passthru);
  g_passthru = NULL;
  g_passthru_count = 0;
  g_passthru_cap = 0;
}

static void ctl_free_scratch(void) {
  free(g_in_vals);
  free(g_node_out_1d);
  free(g_node_out_2d);
  free(g_node_dim);
  free(g_tick_values);
  g_in_vals = NULL;
  g_node_out_1d = NULL;
  g_node_out_2d = NULL;
  g_node_dim = NULL;
  g_tick_values = NULL;
}

static int ctl_alloc_scratch(void) {
  ctl_free_scratch();
  const uint32_t tf = ctl_total_floats();
  if (g_graph.input_count == 0 || g_graph.node_count == 0 || tf == 0) return 0;
  g_in_vals = (float*)malloc((size_t)g_graph.input_count * sizeof(float));
  g_node_out_1d = (float*)malloc((size_t)g_graph.node_count * sizeof(float));
  g_node_out_2d = (float*)malloc((size_t)g_graph.node_count * 2u * sizeof(float));
  g_node_dim = (uint16_t*)malloc((size_t)g_graph.node_count * sizeof(uint16_t));
  g_tick_values = (float*)malloc((size_t)tf * sizeof(float));
  if (!g_in_vals || !g_node_out_1d || !g_node_out_2d || !g_node_dim || !g_tick_values) {
    ctl_free_scratch();
    return 0;
  }
  memset(g_in_vals, 0, (size_t)g_graph.input_count * sizeof(float));
  memset(g_node_out_1d, 0, (size_t)g_graph.node_count * sizeof(float));
  memset(g_node_out_2d, 0, (size_t)g_graph.node_count * 2u * sizeof(float));
  memset(g_node_dim, 0, (size_t)g_graph.node_count * sizeof(uint16_t));
  memset(g_tick_values, 0, (size_t)tf * sizeof(float));
  return 1;
}

static void ctl_free_ring(void) {
  free(g_ring_values);
  free(g_ring_t_ns);
  free((void*)g_ring_seq);
  g_ring_values = NULL;
  g_ring_t_ns = NULL;
  g_ring_seq = NULL;
  g_ring_capacity = 0;
  g_ring_mask = 0;
}

static void ctl_free_hooks(void) {
  free(g_hooks);
  free(g_hook_prev_above);
  g_hooks = NULL;
  g_hook_prev_above = NULL;
  g_hook_count = 0;
}

static void ctl_free_hookq(void) {
  free(g_hookq_events);
  free((void*)g_hookq_seq);
  g_hookq_events = NULL;
  g_hookq_seq = NULL;
  g_hookq_capacity = 0;
  g_hookq_mask = 0;
  gp_atomic_u64_store(&g_hookq_write_seq, 0);

#ifdef _WIN32
  if (g_hookq_sem) {
    CloseHandle(g_hookq_sem);
    g_hookq_sem = NULL;
  }
#else
  if (g_hookq_sem_inited) {
    sem_destroy(&g_hookq_sem);
    g_hookq_sem_inited = 0;
  }
#endif
}

#ifdef _WIN32
static uint64_t gp_now_ns(void) {
  static LARGE_INTEGER freq = {0};
  static int inited = 0;
  if (!inited) {
    QueryPerformanceFrequency(&freq);
    inited = 1;
  }
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  // Convert to ns with 128-bit intermediate.
  const uint64_t num = (uint64_t)c.QuadPart * 1000000000ull;
  const uint64_t den = (uint64_t)freq.QuadPart;
  return den ? (num / den) : 0ull;
}
#else
static uint64_t gp_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

static void gp_sleep_yield(void) {
#ifdef _WIN32
  Sleep(0);
#else
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000000L; // 1ms
  nanosleep(&ts, NULL);
#endif
}

static float clamp11(float v) {
  if (v < -1.0f) return -1.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static float apply_input_calib(float v_raw, const GP_CtlInputDesc* inp, uint32_t kind_code) {
  // Only apply to axis-like kinds.
  if (!(kind_code == GP_EV_AXIS || kind_code == GP_EV_MOUSE_MOTION)) {
    return v_raw;
  }

  float v = v_raw;

  // cap normalization
  const float lo = inp->cap_min;
  const float hi = inp->cap_max;
  if (hi > lo + 1e-6f) {
    const float mid = (lo + hi) * 0.5f;
    const float span = (hi - lo) * 0.5f;
    if (span > 1e-6f) {
      v = (v - mid) / span;
    }
  }
  v = clamp11(v);

  // trim remap
  float t = inp->trim;
  if (t != 0.0f) {
    if (t < -0.95f) t = -0.95f;
    if (t > 0.95f) t = 0.95f;
    const float v0 = v - t;
    float denom = (v0 >= 0.0f) ? (1.0f - t) : (1.0f + t);
    if (denom < 1e-6f) denom = 1e-6f;
    v = v0 / denom;
    v = clamp11(v);
  }

  // deadzone
  const float dz = inp->deadzone;
  if (dz > 0.0f && (v > -dz && v < dz)) {
    v = 0.0f;
  }

  return v;
}

static uint32_t signal_kind_from_signal_id(uint32_t signal_id) {
  // signal_id encoding matches compose_signal_id_u32 in signal_kernel.c:
  // ((device&0xFF)<<24)|((kind&0xFF)<<16)|(id&0xFFFF)
  return (signal_id >> 16) & 0xFFu;
}

static float read_input_scalar(uint64_t now_ns, const GP_CtlInputDesc* inp) {
  GP_SignalFrame fr;
  memset(&fr, 0, sizeof(fr));
  const uint32_t kind_code = signal_kind_from_signal_id(inp->signal_id);

  int ok = 0;
  ok = gp_sigk_peek_sel(now_ns, inp->signal_id, inp->sigsel, &fr);
  float v = ok ? fr.value : 0.0f;

  if ((inp->flags & 1u) != 0 && (kind_code == GP_EV_AXIS || kind_code == GP_EV_MOUSE_MOTION)) {
    v = -v;
  }

  v = apply_input_calib(v, inp, kind_code);
  return v;
}

static float arg_scalar(const float* in_vals, const float* node_out_1d, const float* node_out_2d, const uint16_t* node_dim, const GP_CtlArg* a) {
  if (!a) return 0.0f;
  switch (a->ref) {
    case 2: // imm
      return a->imm;
    case 0: { // input
      const int32_t iid = a->index;
      if (iid < 0) return 0.0f;
      return in_vals[(uint32_t)iid];
    }
    case 1: { // node
      const int32_t nid = a->index;
      if (nid < 0) return 0.0f;
      const uint32_t u = (uint32_t)nid;
      const uint16_t d = node_dim[u];
      if (d == 2) {
        const float x = node_out_2d[u * 2u + 0u];
        const float y = node_out_2d[u * 2u + 1u];
        return (a->comp == 1) ? y : x;
      }
      return node_out_1d[u];
    }
    default:
      return 0.0f;
  }
}

static int ctl_eval_once(uint64_t now_ns, float* out_values) {
  if (!g_graph.outputs || !g_graph.nodes || !g_graph.inputs) return 0;

  const uint32_t ic = g_graph.input_count;
  const uint32_t nc = g_graph.node_count;

  if (!g_in_vals || !g_node_out_1d || !g_node_out_2d || !g_node_dim) return 0;

  float* in_vals = g_in_vals;
  float* node_out_1d = g_node_out_1d;
  float* node_out_2d = g_node_out_2d;
  uint16_t* node_dim = g_node_dim;

  memset(in_vals, 0, (size_t)ic * sizeof(float));
  memset(node_out_1d, 0, (size_t)nc * sizeof(float));
  memset(node_out_2d, 0, (size_t)nc * 2u * sizeof(float));
  memset(node_dim, 0, (size_t)nc * sizeof(uint16_t));

  // 1) read inputs
  for (uint32_t i = 0; i < ic; ++i) {
    in_vals[i] = read_input_scalar(now_ns, &g_graph.inputs[i]);
  }

  // 2) eval nodes (topological by construction)
  for (uint32_t i = 0; i < nc; ++i) {
    const GP_CtlNodeDesc* n = &g_graph.nodes[i];
    const uint16_t dim = (uint16_t)n->dim;
    node_dim[i] = dim;

    const GP_CtlArg* args = NULL;
    if (n->argc > 0 && n->args_offset < g_graph.arg_count) {
      args = &g_graph.args[n->args_offset];
    }

    if (dim == 2) {
      const float a0 = (n->argc > 0) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[0]) : 0.0f;
      const float a1 = (n->argc > 1) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[1]) : 0.0f;
      float x = a0;
      float y = a1;

      if (n->op == GP_CTL_OP_2DFLIGHTSTICK) {
        gp_sigop_2dflightstick(a0, a1, &x, &y);
      } else if (n->op == GP_CTL_OP_2DSEEK) {
        gp_sigop_2dseek(a0, a1, &x, &y);
      } else if (n->op == GP_CTL_OP_2DSUMCLAMP) {
        float sx = 0.0f;
        float sy = 0.0f;
        uint32_t k = 0;
        while (k + 1u < n->argc) {
          sx += arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[k]);
          sy += arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[k + 1u]);
          k += 2u;
        }
        x = clamp11(sx);
        y = clamp11(sy);
      }

      node_out_2d[i * 2u + 0u] = x;
      node_out_2d[i * 2u + 1u] = y;
    } else {
      float v = 0.0f;
      if (n->op == GP_CTL_OP_CONST) {
        v = n->value;
      } else if (n->op == GP_CTL_OP_KERNEL) {
        GP_SignalFrame fr;
        memset(&fr, 0, sizeof(fr));
        int ok = gp_sigk_peek_sel(now_ns, n->signal_id, n->sigsel, &fr);
        v = ok ? fr.value : 0.0f;
      } else if (n->op == GP_CTL_OP_ADD) {
        const float a = (n->argc > 0) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[0]) : 0.0f;
        const float b = (n->argc > 1) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[1]) : 0.0f;
        v = a + b;
      } else if (n->op == GP_CTL_OP_SUB) {
        const float a = (n->argc > 0) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[0]) : 0.0f;
        const float b = (n->argc > 1) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[1]) : 0.0f;
        v = a - b;
      } else {
        v = (n->argc > 0) ? arg_scalar(in_vals, node_out_1d, node_out_2d, node_dim, &args[0]) : 0.0f;
      }
      node_out_1d[i] = v;
    }
  }

  // 3) resolve outputs into packed float vector
  for (uint32_t i = 0; i < g_graph.output_count; ++i) {
    const GP_CtlOutputDesc* o = &g_graph.outputs[i];
    const uint32_t off = o->offset;
    const uint16_t dim = o->dim;
    const uint32_t nid = o->nid;
    if (off + (uint32_t)dim > g_graph.total_floats) continue;

    if (dim == 2 && nid < nc) {
      out_values[off + 0u] = node_out_2d[nid * 2u + 0u];
      out_values[off + 1u] = node_out_2d[nid * 2u + 1u];
    } else if (dim == 1 && nid < nc) {
      out_values[off] = node_out_1d[nid];
    } else {
      // Future N-D: fill zeros.
      for (uint32_t k = 0; k < (uint32_t)dim; ++k) {
        out_values[off + k] = 0.0f;
      }
    }
  }

  return 1;
}

static int ctl_ring_init(uint32_t ring_capacity) {
  const uint32_t tf = ctl_total_floats();
  if (tf == 0 || ctl_total_outputs() == 0) return 0;

  // Require power-of-two for mask indexing.
  uint32_t cap = ring_capacity;
  if (cap < 64u) cap = 64u;
  // round up to pow2
  uint32_t p = 1u;
  while (p < cap) p <<= 1u;
  cap = p;

  ctl_free_ring();

  const size_t values_bytes = (size_t)cap * (size_t)tf * sizeof(float);
  g_ring_values = (float*)malloc(values_bytes);
  g_ring_t_ns = (uint64_t*)malloc((size_t)cap * sizeof(uint64_t));
  g_ring_seq = (GP_ATOMIC_U64*)malloc((size_t)cap * sizeof(*g_ring_seq));
  if (!g_ring_values || !g_ring_t_ns || !g_ring_seq) {
    ctl_free_ring();
    return 0;
  }
  memset(g_ring_values, 0, values_bytes);
  memset(g_ring_t_ns, 0, (size_t)cap * sizeof(uint64_t));
  memset((void*)g_ring_seq, 0, (size_t)cap * sizeof(*g_ring_seq));
  g_ring_capacity = cap;
  g_ring_mask = cap - 1u;

  gp_atomic_u64_store(&g_write_seq, 0);

  return 1;
}

static void ctl_ring_write(uint64_t t_ns, const float* values) {
  if (!g_ring_values || !g_ring_seq || !g_ring_t_ns || g_ring_capacity == 0) return;

  const uint64_t next_seq = gp_atomic_u64_load(&g_write_seq) + 1ull;
  const uint32_t idx = (uint32_t)(next_seq & (uint64_t)g_ring_mask);

  const uint32_t tf = ctl_total_floats();
  float* dst = &g_ring_values[(size_t)idx * (size_t)tf];
  memcpy(dst, values, (size_t)tf * sizeof(float));
  g_ring_t_ns[idx] = t_ns;

  gp_fence_release();
  gp_atomic_u64_store(&g_ring_seq[idx], next_seq);
  gp_atomic_u64_store(&g_write_seq, next_seq);
}

static int ctl_hookq_init(uint32_t ring_capacity) {
  // power-of-two capacity
  uint32_t cap = ring_capacity;
  if (cap < 64u) cap = 64u;
  uint32_t p = 1u;
  while (p < cap) p <<= 1u;
  cap = p;

  ctl_free_hookq();
  g_hookq_events = (GP_CtlHookEvent*)malloc((size_t)cap * sizeof(GP_CtlHookEvent));
  g_hookq_seq = (GP_ATOMIC_U64*)malloc((size_t)cap * sizeof(*g_hookq_seq));
  if (!g_hookq_events || !g_hookq_seq) {
    ctl_free_hookq();
    return 0;
  }
  memset(g_hookq_events, 0, (size_t)cap * sizeof(GP_CtlHookEvent));
  memset((void*)g_hookq_seq, 0, (size_t)cap * sizeof(*g_hookq_seq));
  g_hookq_capacity = cap;
  g_hookq_mask = cap - 1u;
  gp_atomic_u64_store(&g_hookq_write_seq, 0);

#ifdef _WIN32
  g_hookq_sem = CreateSemaphoreA(NULL, 0, 0x7fffffff, NULL);
  if (!g_hookq_sem) {
    ctl_free_hookq();
    return 0;
  }
#else
  if (sem_init(&g_hookq_sem, 0, 0) != 0) {
    ctl_free_hookq();
    return 0;
  }
  g_hookq_sem_inited = 1;
#endif

  return 1;
}

static void ctl_hookq_write(const GP_CtlHookEvent* ev) {
  if (!g_hookq_events || !g_hookq_seq || g_hookq_capacity == 0) return;
  if (!ev) return;

  const uint64_t next_seq = gp_atomic_u64_load(&g_hookq_write_seq) + 1ull;
  const uint32_t idx = (uint32_t)(next_seq & (uint64_t)g_hookq_mask);
  g_hookq_events[idx] = *ev;
  gp_fence_release();
  gp_atomic_u64_store(&g_hookq_seq[idx], next_seq);
  gp_atomic_u64_store(&g_hookq_write_seq, next_seq);

#ifdef _WIN32
  if (g_hookq_sem) ReleaseSemaphore(g_hookq_sem, 1, NULL);
#else
  if (g_hookq_sem_inited) sem_post(&g_hookq_sem);
#endif
}

static float ctl_get_channel_comp_value(const float* packed_values, uint32_t channel, uint16_t comp) {
  if (!packed_values || !g_graph.outputs) return 0.0f;
  for (uint32_t i = 0; i < g_graph.output_count; ++i) {
    const GP_CtlOutputDesc* o = &g_graph.outputs[i];
    if (o->channel != channel) continue;
    const uint16_t dim = o->dim;
    if (dim == 0) return 0.0f;
    const uint16_t c = (comp < dim) ? comp : 0;
    const uint32_t idx = o->offset + (uint32_t)c * (uint32_t)o->stride;
    if (idx >= g_graph.total_floats) return 0.0f;
    return packed_values[idx];
  }
  return 0.0f;
}

static float ctl_get_watch_value(uint64_t now_ns, const GP_CtlHookWatch* w, const float* packed_values) {
  if (!w) return 0.0f;
  if (w->flags & (uint16_t)GP_CTL_HOOK_SRC_SIGNAL) {
    GP_SignalFrame fr;
    memset(&fr, 0, sizeof(fr));
    const uint32_t sid = w->signal_id;
    const uint32_t sel = w->sigsel;
    int ok = gp_sigk_peek_sel(now_ns, sid, sel, &fr);
    if (!ok) return 0.0f;
    return fr.value;
  }
  return ctl_get_channel_comp_value(packed_values, w->channel, w->comp);
}

static void ctl_eval_hooks(uint64_t now_ns, const float* packed_values) {
  if (!g_hooks || g_hook_count == 0 || !g_hook_prev_above) return;
  if (!g_hookq_events) {
    // lazy init
    (void)ctl_hookq_init(1024u);
  }

  for (uint32_t i = 0; i < g_hook_count; ++i) {
    const GP_CtlHookWatch* w = &g_hooks[i];
    const float v = ctl_get_watch_value(now_ns, w, packed_values);
    const float thr = w->threshold;
    const float h = (w->hysteresis > 0.0f) ? w->hysteresis : 0.0f;
    const int was_above = (g_hook_prev_above[i] > 0.5f) ? 1 : 0;
    int is_above = 0;
    if (was_above) {
      is_above = (v >= (thr - h)) ? 1 : 0;
    } else {
      is_above = (v >= thr) ? 1 : 0;
    }

    uint16_t trig = 0;
    if (!was_above && is_above && (w->flags & GP_CTL_HOOK_ON_RISE)) trig |= (uint16_t)GP_CTL_HOOK_ON_RISE;
    if (was_above && !is_above && (w->flags & GP_CTL_HOOK_ON_FALL)) trig |= (uint16_t)GP_CTL_HOOK_ON_FALL;
    if (is_above && (w->flags & GP_CTL_HOOK_LEVEL)) trig |= (uint16_t)GP_CTL_HOOK_LEVEL;

    if (trig) {
      GP_CtlHookEvent ev;
      memset(&ev, 0, sizeof(ev));
      ev.t_ns = now_ns;
      ev.hook_id = w->hook_id;
      ev.channel = w->channel;
      ev.comp = w->comp;
      ev.flags = trig;
      ev.value = v;
      ctl_hookq_write(&ev);
    }

    g_hook_prev_above[i] = is_above ? 1.0f : 0.0f;
  }
}

// ---------------- Signal wheel (history + sticky flags) ----------------

static GP_WheelSample* g_wheel_samples = NULL;  // [signal_count * history_len]
static uint64_t* g_wheel_write_seq = NULL;      // [signal_count]
static uint32_t* g_wheel_flags = NULL;          // [signal_count] sticky bits
static float* g_wheel_prev = NULL;              // [signal_count] last value
static uint32_t g_wheel_signal_count = 0;
static uint32_t g_wheel_history_len = 0;
static uint32_t g_wheel_mask = 0;
static GP_ATOMIC_U64 g_wheel_tick_seq = 0;

#ifdef _WIN32
static HANDLE g_wheel_sem = NULL;
#else
static sem_t g_wheel_sem;
static int g_wheel_sem_inited = 0;
#endif

static void ctl_free_wheel(void) {
  if (g_wheel_samples) {
    free(g_wheel_samples);
    g_wheel_samples = NULL;
  }
  if (g_wheel_write_seq) {
    free(g_wheel_write_seq);
    g_wheel_write_seq = NULL;
  }
  if (g_wheel_flags) {
    free(g_wheel_flags);
    g_wheel_flags = NULL;
  }
  if (g_wheel_prev) {
    free(g_wheel_prev);
    g_wheel_prev = NULL;
  }
  g_wheel_signal_count = 0;
  g_wheel_history_len = 0;
  g_wheel_mask = 0;
  gp_atomic_u64_store(&g_wheel_tick_seq, 0ull);

#ifdef _WIN32
  if (g_wheel_sem) {
    CloseHandle(g_wheel_sem);
    g_wheel_sem = NULL;
  }
#else
  if (g_wheel_sem_inited) {
    sem_destroy(&g_wheel_sem);
    g_wheel_sem_inited = 0;
  }
#endif
}

static uint32_t ctl_next_pow2_u32(uint32_t v) {
  if (v <= 1u) return 1u;
  v -= 1u;
  v |= v >> 1u;
  v |= v >> 2u;
  v |= v >> 4u;
  v |= v >> 8u;
  v |= v >> 16u;
  return v + 1u;
}

static int ctl_wheel_init(uint32_t signal_count, uint32_t history_len) {
  ctl_free_wheel();
  if (signal_count == 0u) return 0;

  uint32_t h = (history_len == 0u) ? 32u : history_len;
  // Use power-of-two ring for cheap masking.
  h = ctl_next_pow2_u32(h);
  if (h < 2u) h = 2u;

  const size_t samples_n = (size_t)signal_count * (size_t)h;
  g_wheel_samples = (GP_WheelSample*)malloc(samples_n * sizeof(GP_WheelSample));
  g_wheel_write_seq = (uint64_t*)malloc((size_t)signal_count * sizeof(uint64_t));
  g_wheel_flags = (uint32_t*)malloc((size_t)signal_count * sizeof(uint32_t));
  g_wheel_prev = (float*)malloc((size_t)signal_count * sizeof(float));
  if (!g_wheel_samples || !g_wheel_write_seq || !g_wheel_flags || !g_wheel_prev) {
    ctl_free_wheel();
    return 0;
  }
  memset(g_wheel_samples, 0, samples_n * sizeof(GP_WheelSample));
  memset(g_wheel_write_seq, 0, (size_t)signal_count * sizeof(uint64_t));
  memset(g_wheel_flags, 0, (size_t)signal_count * sizeof(uint32_t));
  memset(g_wheel_prev, 0, (size_t)signal_count * sizeof(float));

  g_wheel_signal_count = signal_count;
  g_wheel_history_len = h;
  g_wheel_mask = h - 1u;
  gp_atomic_u64_store(&g_wheel_tick_seq, 0ull);

#ifdef _WIN32
  g_wheel_sem = CreateSemaphoreA(NULL, 0, 0x7fffffff, NULL);
  if (!g_wheel_sem) {
    ctl_free_wheel();
    return 0;
  }
#else
  if (sem_init(&g_wheel_sem, 0, 0) != 0) {
    ctl_free_wheel();
    return 0;
  }
  g_wheel_sem_inited = 1;
#endif

  return 1;
}

static inline uint32_t ctl_atomic_u32_exchange(uint32_t* p, uint32_t v) {
#ifdef _WIN32
  return (uint32_t)InterlockedExchange((volatile LONG*)p, (LONG)v);
#else
  return __atomic_exchange_n(p, v, __ATOMIC_ACQ_REL);
#endif
}

static inline uint32_t ctl_atomic_u32_fetch_or(uint32_t* p, uint32_t v) {
#ifdef _WIN32
  return (uint32_t)InterlockedOr((volatile LONG*)p, (LONG)v);
#else
  return __atomic_fetch_or(p, v, __ATOMIC_RELAXED);
#endif
}

static void ctl_wheel_publish(uint64_t now_ns, const float* values, uint32_t n_values) {
  if (!g_wheel_samples || !g_wheel_write_seq || !g_wheel_flags || !g_wheel_prev) return;
  if (!values) return;
  const uint32_t n = g_wheel_signal_count;
  const uint32_t m = (n_values < n) ? n_values : n;

  const float eps_hot = 1e-6f;
  const float eps_chg = 1e-6f;

  for (uint32_t i = 0; i < m; ++i) {
    const float v = values[i];
    const float pv = g_wheel_prev[i];
    g_wheel_prev[i] = v;

    const uint64_t next_seq = g_wheel_write_seq[i] + 1ull;
    const uint32_t idx = (uint32_t)(next_seq & (uint64_t)g_wheel_mask);
    GP_WheelSample* s = &g_wheel_samples[(size_t)i * (size_t)g_wheel_history_len + (size_t)idx];
    s->t_ns = now_ns;
    s->value = v;
    s->_pad0 = 0u;

    g_wheel_write_seq[i] = next_seq;

    uint32_t f = 0u;
    if (v > eps_hot || v < -eps_hot) f |= (uint32_t)GP_WHEEL_F_HOT;
    const float dv = v - pv;
    if (dv > eps_chg || dv < -eps_chg) f |= (uint32_t)GP_WHEEL_F_CHANGED;
    if (f) (void)ctl_atomic_u32_fetch_or(&g_wheel_flags[i], f);
  }

  const uint64_t tnext = gp_atomic_u64_load(&g_wheel_tick_seq) + 1ull;
  gp_atomic_u64_store(&g_wheel_tick_seq, tnext);
#ifdef _WIN32
  if (g_wheel_sem) ReleaseSemaphore(g_wheel_sem, 1, NULL);
#else
  if (g_wheel_sem_inited) sem_post(&g_wheel_sem);
#endif
}

GP_EXPORT void gp_ctl_reset(void) {
  gp_ctl_stop();
  ctl_free_ring();
  ctl_free_hookq();
  ctl_free_hooks();
  ctl_free_wheel();
  ctl_free_passthru();
  ctl_free_scratch();
  ctl_free_graph();
}

GP_EXPORT int gp_ctl_load_graph_file(const char* path_utf8) {
  if (!path_utf8) return 0;

  gp_ctl_stop();
  ctl_free_ring();
  ctl_free_passthru();
  ctl_free_scratch();
  ctl_free_graph();

  FILE* f = fopen(path_utf8, "rb");
  if (!f) return 0;

  GP_CtlBlobHeader hdr;
  if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (hdr.magic != GP_CTL_BLOB_MAGIC || hdr.version != GP_CTL_BLOB_VERSION) {
    fclose(f);
    return 0;
  }

  if (hdr.input_count > 65536u || hdr.node_count > 65536u || hdr.arg_count > 1000000u || hdr.output_count > 65536u) {
    fclose(f);
    return 0;
  }

  g_graph.input_count = hdr.input_count;
  g_graph.node_count = hdr.node_count;
  g_graph.arg_count = hdr.arg_count;
  g_graph.output_count = hdr.output_count;

  if (g_graph.input_count) {
    g_graph.inputs = (GP_CtlInputDesc*)malloc((size_t)g_graph.input_count * sizeof(GP_CtlInputDesc));
    if (!g_graph.inputs) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
    if (fread(g_graph.inputs, sizeof(GP_CtlInputDesc), (size_t)g_graph.input_count, f) != (size_t)g_graph.input_count) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
  }

  if (g_graph.node_count) {
    g_graph.nodes = (GP_CtlNodeDesc*)malloc((size_t)g_graph.node_count * sizeof(GP_CtlNodeDesc));
    if (!g_graph.nodes) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
    if (fread(g_graph.nodes, sizeof(GP_CtlNodeDesc), (size_t)g_graph.node_count, f) != (size_t)g_graph.node_count) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
  }

  if (g_graph.arg_count) {
    g_graph.args = (GP_CtlArg*)malloc((size_t)g_graph.arg_count * sizeof(GP_CtlArg));
    if (!g_graph.args) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
    if (fread(g_graph.args, sizeof(GP_CtlArg), (size_t)g_graph.arg_count, f) != (size_t)g_graph.arg_count) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
  }

  if (g_graph.output_count) {
    g_graph.outputs = (GP_CtlOutputDesc*)malloc((size_t)g_graph.output_count * sizeof(GP_CtlOutputDesc));
    if (!g_graph.outputs) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
    if (fread(g_graph.outputs, sizeof(GP_CtlOutputDesc), (size_t)g_graph.output_count, f) != (size_t)g_graph.output_count) {
      fclose(f);
      ctl_free_graph();
      return 0;
    }
  }

  fclose(f);

  // Compute packed float layout (offsets) and total_floats.
  uint32_t off = 0;
  for (uint32_t i = 0; i < g_graph.output_count; ++i) {
    g_graph.outputs[i].stride = 1u;
    g_graph.outputs[i].offset = off;
    const uint32_t d = (uint32_t)g_graph.outputs[i].dim;
    off += (d > 0u ? d : 1u);
  }
  g_graph.total_floats = off;

  // Allocate reusable scratch buffers for ticking.
  if (!ctl_alloc_scratch()) {
    ctl_free_graph();
    return 0;
  }

  // Initialize the wheel over packed scalar outputs.
  // History length is a power-of-two for cheap masking.
  (void)ctl_wheel_init(ctl_total_floats(), 64u);

  return 1;
}

GP_EXPORT int gp_ctl_get_meta(GP_CtlMeta* out_meta) {
  if (!out_meta) return 0;
  memset(out_meta, 0, sizeof(*out_meta));
  out_meta->output_count = ctl_total_outputs();
  out_meta->total_floats = ctl_total_floats();
  out_meta->ring_capacity = g_ring_capacity;
  out_meta->tick_hz = g_graph.tick_hz;
  out_meta->write_seq = gp_atomic_u64_load(&g_write_seq);
  return 1;
}

GP_EXPORT uint32_t gp_ctl_get_outputs(GP_CtlOutputDesc* out_arr, uint32_t cap) {
  if (!out_arr || cap == 0) return 0;
  const uint32_t base_n = g_graph.output_count;
  const uint32_t extra_n = g_passthru_count;
  const uint32_t total_n = base_n + extra_n;
  if (total_n == 0u) return 0;

  uint32_t w = 0u;
  if (g_graph.outputs && base_n > 0u) {
    uint32_t take = base_n;
    if (take > cap) take = cap;
    if (take > 0u) {
      memcpy(out_arr, g_graph.outputs, (size_t)take * sizeof(GP_CtlOutputDesc));
      w += take;
    }
  }
  if (w < cap && g_passthru && extra_n > 0u) {
    const uint32_t base_off = g_graph.total_floats;
    while (w < cap) {
      const uint32_t i = w - base_n;
      if (i >= extra_n) break;
      GP_CtlOutputDesc d;
      memset(&d, 0, sizeof(d));
      d.channel = g_passthru[i].channel;
      d.dim = 1u;
      d.stride = 1u;
      d.offset = base_off + i;
      d.nid = 0u;
      out_arr[w] = d;
      ++w;
    }
  }
  return w;
}

GP_EXPORT uint64_t gp_ctl_get_write_seq(void) {
  return gp_atomic_u64_load(&g_write_seq);
}

GP_EXPORT int gp_ctl_peek_seq(uint64_t seq, float* out_values, uint32_t cap_values, uint64_t* out_t_ns) {
  const uint32_t tf = ctl_total_floats();
  if (!out_values || cap_values < tf) return 0;
  if (!g_ring_values || !g_ring_seq || !g_ring_t_ns || g_ring_capacity == 0) return 0;

  const uint64_t latest = gp_ctl_get_write_seq();
  if (seq == 0 || seq > latest) return 0;
  if (latest - seq >= (uint64_t)g_ring_capacity) {
    // Overwritten.
    return 0;
  }

  const uint32_t idx = (uint32_t)(seq & (uint64_t)g_ring_mask);

  // Copy and verify slot seq to avoid torn reads.
  for (int attempt = 0; attempt < 2; ++attempt) {
    const uint64_t slot_seq = gp_atomic_u64_load(&g_ring_seq[idx]);
    if (slot_seq != seq) {
      return 0;
    }
    gp_fence_acquire();
    memcpy(out_values, &g_ring_values[(size_t)idx * (size_t)tf], (size_t)tf * sizeof(float));
    gp_fence_acquire();
    const uint64_t slot_seq2 = gp_atomic_u64_load(&g_ring_seq[idx]);
    if (slot_seq2 == seq) {
      if (out_t_ns) *out_t_ns = g_ring_t_ns[idx];
      return 1;
    }
  }
  return 0;
}

GP_EXPORT int gp_ctl_peek_latest(float* out_values, uint32_t cap_values, uint64_t* out_seq, uint64_t* out_t_ns) {
  const uint64_t seq = gp_ctl_get_write_seq();
  if (out_seq) *out_seq = seq;
  return gp_ctl_peek_seq(seq, out_values, cap_values, out_t_ns);
}

GP_EXPORT int gp_ctl_step_once(uint64_t now_ns) {
  if (!g_graph.outputs || g_graph.total_floats == 0) return 0;
  if (!g_ring_values) {
    // default ring if user didn't start thread.
    if (!ctl_ring_init(1024u)) return 0;
  }

  if (!g_tick_values) return 0;
  const uint32_t tf = ctl_total_floats();
  memset(g_tick_values, 0, (size_t)tf * sizeof(float));
  const int ok = ctl_eval_once(now_ns, g_tick_values);
  if (ok) {
    ctl_eval_hooks(now_ns, g_tick_values);
    // Fill passthrough scalar region (after compiled packed outputs).
    if (g_passthru && g_passthru_count > 0u) {
      const uint32_t base_off = g_graph.total_floats;
      GP_SignalFrame fr;
      memset(&fr, 0, sizeof(fr));
      for (uint32_t i = 0; i < g_passthru_count; ++i) {
        const GP_CtlPassthruDesc* p = &g_passthru[i];
        int okp = gp_sigk_peek_sel(now_ns, p->signal_id, p->sigsel, &fr);
        g_tick_values[base_off + i] = okp ? fr.value : 0.0f;
      }
    }
    ctl_wheel_publish(now_ns, g_tick_values, tf);
    ctl_ring_write(now_ns, g_tick_values);
  }
  // Pulses persist in the kernel until explicitly cleared. Since the controller
  // engine is the authoritative evaluator, it also owns pulse lifetime.
  // Clearing here prevents repeated "edge" reads across subsequent ticks.
  gp_sigk_clear_pulses();
  return ok;
}

#ifdef _WIN32
static DWORD WINAPI ctl_thread_main(LPVOID unused) {
  (void)unused;
  const uint32_t hz = (g_graph.tick_hz > 0u) ? g_graph.tick_hz : 240u;
  const uint64_t period_ns = (uint64_t)(1000000000ull / (uint64_t)hz);
  uint64_t next_ns = gp_now_ns();

  while (gp_atomic_u32_load(&g_ctl_run) != 0u) {
    const uint64_t now_ns = gp_now_ns();
    if (now_ns >= next_ns) {
      uint32_t steps = 0;
      while (gp_now_ns() >= next_ns && steps < 8u) {
        gp_ctl_step_once(next_ns);
        next_ns += period_ns;
        ++steps;
      }
    } else {
      gp_sleep_yield();
    }
  }
  return 0;
}
#else
static void* ctl_thread_main(void* unused) {
  (void)unused;
  const uint32_t hz = (g_graph.tick_hz > 0u) ? g_graph.tick_hz : 240u;
  const uint64_t period_ns = (uint64_t)(1000000000ull / (uint64_t)hz);
  uint64_t next_ns = gp_now_ns();

  while (gp_atomic_u32_load(&g_ctl_run) != 0u) {
    const uint64_t now_ns = gp_now_ns();
    if (now_ns >= next_ns) {
      uint32_t steps = 0;
      while (gp_now_ns() >= next_ns && steps < 8u) {
        gp_ctl_step_once(next_ns);
        next_ns += period_ns;
        ++steps;
      }
    } else {
      gp_sleep_yield();
    }
  }
  return NULL;
}
#endif

GP_EXPORT int gp_ctl_start(uint32_t tick_hz, uint32_t ring_capacity) {
  if (!g_graph.outputs || g_graph.total_floats == 0) return 0;
  if (tick_hz == 0) tick_hz = 240u;
  g_graph.tick_hz = tick_hz;

  if (!g_ring_values) {
    if (!ctl_ring_init(ring_capacity ? ring_capacity : 2048u)) return 0;
  }

  if (!g_hookq_events) {
    (void)ctl_hookq_init(1024u);
  }

#ifdef _WIN32
  if (g_ctl_thread && gp_atomic_u32_load(&g_ctl_run) != 0u) {
    return 1;
  }

  gp_atomic_u32_store(&g_ctl_run, 1u);
  g_ctl_thread = CreateThread(NULL, 0, ctl_thread_main, NULL, 0, NULL);
  if (!g_ctl_thread) {
    gp_atomic_u32_store(&g_ctl_run, 0u);
    return 0;
  }
  return 1;
#else
  if (g_ctl_thread_valid && gp_atomic_u32_load(&g_ctl_run) != 0u) {
    return 1;
  }
  gp_atomic_u32_store(&g_ctl_run, 1u);
  if (pthread_create(&g_ctl_thread, NULL, ctl_thread_main, NULL) != 0) {
    gp_atomic_u32_store(&g_ctl_run, 0u);
    g_ctl_thread_valid = 0;
    return 0;
  }
  g_ctl_thread_valid = 1;
  return 1;
#endif
}

GP_EXPORT void gp_ctl_stop(void) {
#ifdef _WIN32
  if (g_ctl_thread) {
    gp_atomic_u32_store(&g_ctl_run, 0u);
    WaitForSingleObject(g_ctl_thread, 2000);
    CloseHandle(g_ctl_thread);
    g_ctl_thread = NULL;
  }
#else
  if (g_ctl_thread_valid) {
    gp_atomic_u32_store(&g_ctl_run, 0u);
    pthread_join(g_ctl_thread, NULL);
    g_ctl_thread_valid = 0;
  }
#endif
}

GP_EXPORT int gp_ctl_is_running(void) {
#ifdef _WIN32
  return (g_ctl_thread && gp_atomic_u32_load(&g_ctl_run) != 0u) ? 1 : 0;
#else
  return (g_ctl_thread_valid && gp_atomic_u32_load(&g_ctl_run) != 0u) ? 1 : 0;
#endif
}

GP_EXPORT void gp_ctl_hooks_clear(void) {
  ctl_free_hooks();
}

GP_EXPORT int gp_ctl_hooks_add(const GP_CtlHookWatch* w) {
  if (!w) return 0;
  const uint32_t n = g_hook_count + 1u;
  GP_CtlHookWatch* nw = (GP_CtlHookWatch*)realloc(g_hooks, (size_t)n * sizeof(GP_CtlHookWatch));
  float* np = (float*)realloc(g_hook_prev_above, (size_t)n * sizeof(float));
  if (!nw || !np) {
    // If realloc failed, leave existing buffers intact.
    if (nw) g_hooks = nw;
    if (np) g_hook_prev_above = np;
    return 0;
  }
  g_hooks = nw;
  g_hook_prev_above = np;
  g_hooks[g_hook_count] = *w;
  g_hook_prev_above[g_hook_count] = 0.0f;
  g_hook_count = n;
  return 1;
}

GP_EXPORT int gp_ctl_hookq_get_meta(GP_CtlHookQMeta* out_meta) {
  if (!out_meta) return 0;
  memset(out_meta, 0, sizeof(*out_meta));
  out_meta->capacity = g_hookq_capacity;
  out_meta->write_seq = gp_atomic_u64_load(&g_hookq_write_seq);
  return 1;
}

GP_EXPORT uint64_t gp_ctl_hookq_get_write_seq(void) {
  return gp_atomic_u64_load(&g_hookq_write_seq);
}

GP_EXPORT int gp_ctl_hookq_peek_seq(uint64_t seq, GP_CtlHookEvent* out_ev) {
  if (!out_ev) return 0;
  if (!g_hookq_events || !g_hookq_seq || g_hookq_capacity == 0) return 0;

  const uint64_t latest = gp_ctl_hookq_get_write_seq();
  if (seq == 0 || seq > latest) return 0;
  if (latest - seq >= (uint64_t)g_hookq_capacity) return 0;

  const uint32_t idx = (uint32_t)(seq & (uint64_t)g_hookq_mask);
  for (int attempt = 0; attempt < 2; ++attempt) {
    const uint64_t slot_seq = gp_atomic_u64_load(&g_hookq_seq[idx]);
    if (slot_seq != seq) return 0;
    gp_fence_acquire();
    *out_ev = g_hookq_events[idx];
    gp_fence_acquire();
    if (gp_atomic_u64_load(&g_hookq_seq[idx]) == seq) return 1;
  }
  return 0;
}

GP_EXPORT int gp_ctl_hookq_peek_latest(GP_CtlHookEvent* out_ev, uint64_t* out_seq) {
  const uint64_t seq = gp_ctl_hookq_get_write_seq();
  if (out_seq) *out_seq = seq;
  return gp_ctl_hookq_peek_seq(seq, out_ev);
}

GP_EXPORT int gp_ctl_hookq_wait(uint64_t last_seen_seq, uint32_t timeout_ms, uint64_t* out_seq) {
  if (!out_seq) return 0;

  const uint64_t w0 = gp_ctl_hookq_get_write_seq();
  if (w0 > last_seen_seq) {
    *out_seq = w0;
    return 1;
  }

  if (!g_hookq_events) {
    *out_seq = w0;
    return 0;
  }

#ifdef _WIN32
  if (!g_hookq_sem) {
    *out_seq = w0;
    return 0;
  }
  DWORD rc = WaitForSingleObject(g_hookq_sem, (DWORD)timeout_ms);
  (void)rc;
#else
  if (!g_hookq_sem_inited) {
    *out_seq = w0;
    return 0;
  }
  if (timeout_ms == 0) {
    *out_seq = w0;
    return 0;
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t add_ns = (uint64_t)timeout_ms * 1000000ull;
  ts.tv_sec += (time_t)(add_ns / 1000000000ull);
  ts.tv_nsec += (long)(add_ns % 1000000000ull);
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec += 1;
    ts.tv_nsec -= 1000000000L;
  }
  while (sem_timedwait(&g_hookq_sem, &ts) != 0) {
    if (errno == EINTR) continue;
    break;
  }
#endif

  const uint64_t w1 = gp_ctl_hookq_get_write_seq();
  *out_seq = w1;
  return (w1 > last_seen_seq) ? 1 : 0;
}

// ---------------- Passthrough outputs ----------------

GP_EXPORT void gp_ctl_passthru_clear(void) {
  // Must be configured while stopped.
  if (gp_ctl_is_running()) return;
  ctl_free_passthru();
  (void)ctl_alloc_scratch();
  (void)ctl_wheel_init(ctl_total_floats(), 64u);
}

GP_EXPORT int gp_ctl_passthru_add(const GP_CtlPassthruDesc* desc) {
  if (!desc) return 0;
  if (gp_ctl_is_running()) return 0;

  // Ignore duplicates by channel.
  for (uint32_t i = 0; i < g_passthru_count; ++i) {
    if (g_passthru[i].channel == desc->channel) return 1;
  }

  if (g_passthru_count + 1u > g_passthru_cap) {
    uint32_t new_cap = (g_passthru_cap == 0u) ? 16u : (g_passthru_cap * 2u);
    GP_CtlPassthruDesc* p2 = (GP_CtlPassthruDesc*)realloc(g_passthru, (size_t)new_cap * sizeof(GP_CtlPassthruDesc));
    if (!p2) return 0;
    g_passthru = p2;
    g_passthru_cap = new_cap;
  }

  g_passthru[g_passthru_count] = *desc;
  g_passthru_count += 1u;

  (void)ctl_alloc_scratch();
  (void)ctl_wheel_init(ctl_total_floats(), 64u);
  return 1;
}

// ---------------- Signal wheel exports ----------------

GP_EXPORT int gp_ctl_wheel_get_meta(GP_WheelMeta* out_meta) {
  if (!out_meta) return 0;
  memset(out_meta, 0, sizeof(*out_meta));
  out_meta->signal_count = g_wheel_signal_count;
  out_meta->history_len = g_wheel_history_len;
  out_meta->tick_seq = gp_atomic_u64_load(&g_wheel_tick_seq);
  return (g_wheel_signal_count > 0u && g_wheel_history_len > 0u) ? 1 : 0;
}

GP_EXPORT uint64_t gp_ctl_wheel_get_tick_seq(void) {
  return gp_atomic_u64_load(&g_wheel_tick_seq);
}

GP_EXPORT int gp_ctl_wheel_wait(uint64_t last_tick_seq, uint32_t timeout_ms, uint64_t* out_tick_seq) {
  const uint64_t cur = gp_ctl_wheel_get_tick_seq();
  if (out_tick_seq) *out_tick_seq = cur;
  if (cur > last_tick_seq) return 1;

#ifdef _WIN32
  if (!g_wheel_sem) return 0;
  DWORD ms = (timeout_ms == 0u) ? 0u : (DWORD)timeout_ms;
  DWORD rc = WaitForSingleObject(g_wheel_sem, ms);
  (void)rc;
#else
  if (!g_wheel_sem_inited) return 0;
  if (timeout_ms == 0u) {
    (void)sem_trywait(&g_wheel_sem);
  } else {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ns = (uint64_t)ts.tv_nsec + (uint64_t)timeout_ms * 1000000ull;
    ts.tv_sec += (time_t)(ns / 1000000000ull);
    ts.tv_nsec = (long)(ns % 1000000000ull);
    (void)sem_timedwait(&g_wheel_sem, &ts);
  }
#endif

  const uint64_t cur2 = gp_ctl_wheel_get_tick_seq();
  if (out_tick_seq) *out_tick_seq = cur2;
  return (cur2 > last_tick_seq) ? 1 : 0;
}

GP_EXPORT int gp_ctl_wheel_peek_latest(uint32_t signal_idx, GP_WheelSample* out_sample, uint64_t* out_write_seq) {
  if (!out_sample) return 0;
  if (!g_wheel_samples || !g_wheel_write_seq) return 0;
  if (signal_idx >= g_wheel_signal_count || g_wheel_history_len == 0u) return 0;

  const uint64_t seq = g_wheel_write_seq[signal_idx];
  if (out_write_seq) *out_write_seq = seq;
  if (seq == 0ull) {
    memset(out_sample, 0, sizeof(*out_sample));
    return 1;
  }
  const uint32_t idx = (uint32_t)(seq & (uint64_t)g_wheel_mask);
  *out_sample = g_wheel_samples[(size_t)signal_idx * (size_t)g_wheel_history_len + (size_t)idx];
  return 1;
}

GP_EXPORT int gp_ctl_wheel_peek_seq(uint32_t signal_idx, uint64_t write_seq, GP_WheelSample* out_sample) {
  if (!out_sample) return 0;
  if (!g_wheel_samples || !g_wheel_write_seq) return 0;
  if (signal_idx >= g_wheel_signal_count || g_wheel_history_len == 0u) return 0;
  const uint64_t latest = g_wheel_write_seq[signal_idx];
  if (write_seq == 0ull || write_seq > latest) return 0;
  if (latest - write_seq >= (uint64_t)g_wheel_history_len) return 0;
  const uint32_t idx = (uint32_t)(write_seq & (uint64_t)g_wheel_mask);
  *out_sample = g_wheel_samples[(size_t)signal_idx * (size_t)g_wheel_history_len + (size_t)idx];
  return 1;
}

GP_EXPORT int gp_ctl_wheel_drain_hot(uint32_t* out_indices, uint32_t max_indices, uint32_t* out_count) {
  if (out_count) *out_count = 0u;
  if (!g_wheel_flags || g_wheel_signal_count == 0u) return 0;
  uint32_t n = 0u;
  for (uint32_t i = 0; i < g_wheel_signal_count; ++i) {
    if (out_indices && n >= max_indices) break;
    const uint32_t prev = ctl_atomic_u32_exchange(&g_wheel_flags[i], 0u);
    if (prev & (uint32_t)GP_WHEEL_F_HOT) {
      if (out_indices) {
        out_indices[n] = i;
      }
      ++n;
    }
  }
  if (out_count) *out_count = n;
  return 1;
}
