#include "geodesic_physics.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// MSVC doesn't support C99 VLAs, so lifecycle code uses fixed-size stack buffers.
// This is a safety cap for temporary vectors used by merge/shatter/condense.
#ifndef GP_MAX_D
#define GP_MAX_D 128u
#endif

// Forward declarations (avoid implicit declarations in C compilers).
static void compute_force(GP_Header* h, const double* p, const double* v, double dt, double* Ftmp);
static void update_temps(GP_Header* h, const double* v, double dt);
static void project_tangent(const double* p, const double* v, double* out, uint32_t d);
static void rebuild_bond_adjacency_from_springs(GP_Header* h);

// Layout: [GP_Header][padding->8]
//         pos (2*n*d doubles)                 // ping-pong
//         vel (2*n*d doubles)                 // ping-pong
//         mass (n doubles)
//         charge (n doubles)
//         temps (n doubles)
//         hist_pos (history_cap*n*d doubles)  // ring buffer frames (optional)
//         hist_vel (history_cap*n*d doubles)  // ring buffer frames (optional)
//         hist_dt  (history_cap doubles)      // per-frame dt (optional)
//         pairs (pair_count * GP_Pair)        // all i<j
//         bond_bits (pair_count bytes)        // 0/1 adjacency for pairs
//         degree (n uint32)                   // per-node bond degree
//         scratch (5*n*d doubles)             // F0, F1, v_mid, p_mid, v_new
//         springs (2*spring_cap * GP_Spring) // ping-pong (front_idx/sim_idx)

static size_t align8(size_t x) { return (x + 7u) & ~(size_t)7u; }

static double* ptr_pos(GP_Header* h) {
    uint8_t* base = (uint8_t*)h;
    size_t off = align8(sizeof(GP_Header));
    return (double*)(base + off);
}

static double* ptr_vel(GP_Header* h) {
    double* pos = ptr_pos(h);
    return pos + (size_t)2u * (size_t)h->n * (size_t)h->d;
}

static double* ptr_mass(GP_Header* h) {
    double* vel = ptr_vel(h);
    return vel + (size_t)2u * (size_t)h->n * (size_t)h->d;
}

static double* ptr_charge(GP_Header* h) {
    double* m = ptr_mass(h);
    return m + (size_t)h->n;
}

static double* ptr_temps(GP_Header* h) {
    double* q = ptr_charge(h);
    return q + (size_t)h->n;
}

static double* ptr_hist_pos(GP_Header* h) {
    double* t = ptr_temps(h);
    return t + (size_t)h->n;
}

static double* ptr_hist_vel(GP_Header* h) {
    double* hp = ptr_hist_pos(h);
    return hp + (size_t)h->history_cap * (size_t)h->n * (size_t)h->d;
}

static double* ptr_hist_dt(GP_Header* h) {
    double* hv = ptr_hist_vel(h);
    return hv + (size_t)h->history_cap * (size_t)h->n * (size_t)h->d;
}

static GP_Pair* ptr_pairs(GP_Header* h) {
    uint8_t* base = (uint8_t*)h;
    double* dtp = ptr_hist_dt(h);
    uint8_t* p = (uint8_t*)(dtp + (size_t)h->history_cap);
    (void)base;
    return (GP_Pair*)p;
}

static uint8_t* ptr_bond_bits(GP_Header* h) {
    GP_Pair* pairs = ptr_pairs(h);
    return (uint8_t*)(pairs + (size_t)h->pair_count);
}

static uint32_t* ptr_degree(GP_Header* h) {
    uint8_t* bits = ptr_bond_bits(h);
    uint8_t* p = bits + (size_t)h->pair_count;
    p = (uint8_t*)(((uintptr_t)p + 7u) & ~(uintptr_t)7u);
    return (uint32_t*)p;
}

static double* ptr_scratch(GP_Header* h) {
    uint32_t* deg = ptr_degree(h);
    uint8_t* p = (uint8_t*)(deg + (size_t)h->n);
    p = (uint8_t*)(((uintptr_t)p + 7u) & ~(uintptr_t)7u);
    return (double*)p;
}

static GP_Spring* ptr_springs_base(GP_Header* h) {
    double* s = ptr_scratch(h);
    uint8_t* p = (uint8_t*)(s + (size_t)5u * (size_t)h->n * (size_t)h->d);
    return (GP_Spring*)p;
}

static GP_Spring* ptr_springs_buf(GP_Header* h, uint32_t buf_idx) {
    GP_Spring* base = ptr_springs_base(h);
    const uint32_t bi = buf_idx & 1u;
    return base + (size_t)bi * (size_t)h->spring_cap;
}

static GP_Spring* ptr_springs_front(GP_Header* h) {
    return ptr_springs_buf(h, h->front_idx);
}

static GP_Spring* ptr_springs_sim(GP_Header* h) {
    return ptr_springs_buf(h, h->sim_idx);
}

size_t gp_required_bytes(uint32_t n, uint32_t d, uint32_t spring_cap) {
    return gp_required_bytes_ex(n, d, spring_cap, 0u);
}

size_t gp_required_bytes_ex(uint32_t n, uint32_t d, uint32_t spring_cap, uint32_t history_cap) {
    size_t bytes = 0;
    bytes += sizeof(GP_Header);
    bytes = align8(bytes);
    bytes += (size_t)2u * (size_t)n * (size_t)d * sizeof(double); // pos
    bytes += (size_t)2u * (size_t)n * (size_t)d * sizeof(double); // vel
    bytes += (size_t)n * sizeof(double); // mass
    bytes += (size_t)n * sizeof(double); // charge
    bytes += (size_t)n * sizeof(double); // temps
    // history (optional)
    if (history_cap) {
        bytes += (size_t)history_cap * (size_t)n * (size_t)d * sizeof(double); // hist_pos
        bytes += (size_t)history_cap * (size_t)n * (size_t)d * sizeof(double); // hist_vel
        bytes += (size_t)history_cap * sizeof(double); // hist_dt
    }
    // all-pairs edge list + adjacency + per-node degrees
    {
        const uint64_t pc = ((uint64_t)n * (uint64_t)(n - 1u)) / 2u;
        bytes += (size_t)pc * sizeof(GP_Pair);
        bytes += (size_t)pc * sizeof(uint8_t);
        bytes = align8(bytes);
        bytes += (size_t)n * sizeof(uint32_t);
        bytes = align8(bytes);
    }
    bytes += (size_t)5u * (size_t)n * (size_t)d * sizeof(double); // scratch
    // double-buffer springs so readers can safely access a stable front list
    // while the simulation mutates its own list.
    bytes += (size_t)2u * (size_t)spring_cap * sizeof(GP_Spring);
    return bytes;
}

int gp_init(void* state_mem, uint32_t n, uint32_t d, uint32_t spring_cap) {
    return gp_init_ex(state_mem, n, d, spring_cap, 0u);
}

int gp_init_ex(void* state_mem, uint32_t n, uint32_t d, uint32_t spring_cap, uint32_t history_cap) {
    if (!state_mem || n == 0 || d < 2) return 0;
    GP_Header* h = (GP_Header*)state_mem;
    memset(h, 0, gp_required_bytes_ex(n, d, spring_cap, history_cap));
    h->magic = 0x48505047u; // 'GPPH'
    h->version = 9u;
    h->n = n;
    h->n_active = n;
    h->n_active_sim = n;
    h->d = d;
    h->spring_cap = spring_cap;
    h->spring_count = 0u;
    h->spring_count_sim = 0u;
    h->history_cap = history_cap;
    h->history_len = 0u;
    h->history_head = 0u;
    h->history_seq = 0u;
    h->front_idx = 0u;
    h->sim_idx = 1u;
    h->swap_requested = 0u;
    h->swap_seq = 0u;

    // all-pairs edge list
    {
        const uint64_t pc64 = ((uint64_t)n * (uint64_t)(n - 1u)) / 2u;
        if (pc64 > 0xFFFFFFFFu) return 0;
        h->pair_count = (uint32_t)pc64;
    }

    // bond defaults: disabled
    h->bonds_enabled = 0u;
    h->bond_max_per_node = 0u;
    h->ionic_bonds = 0u;
    h->ionic_valence = 1u;
    h->bond_link_angle = 0.0;
    h->bond_shear_ratio = 2.0;
    h->bond_k = 1.0;

    h->k_spring = 1.0;
    h->k_coulomb = 0.0;
    h->G = 0.0;
    h->damping = 0.0;
    h->softening = 1e-6;

    h->south_enabled = 0u;
    h->south_axis = -1;
    h->south_strength = 0.0;

    h->lorentz_k = 0.0;
    h->Bx = 0.0;
    h->By = 0.0;
    h->Bz = 1.0;

    h->temp_ambient = 0.0;
    h->temp_conv = 0.0;
    h->temp_noise = 0.0;
    h->temp_heat_gain = 1.0;

    // Lifecycle defaults: disabled unless params are set.
    h->max_speed = 0.0;
    h->radius_scale = 0.0;
    h->mass_ref_cuberoot = 1.0;
    h->collide_gain = 0.0;
    h->merge_speed_frac = 0.4;
    h->merge_size_frac = 0.5;
    h->shatter_speed_frac = 0.8;
    h->shatter_size_frac = 0.5;
    h->shatter_k_max = 8u;
    h->collision_pair_limit = 0u;
    h->mass_vapor_thresh = 0.0;
    h->condense_temp_thresh = 0.0;
    h->condense_chunk_mass = 0.0;
    h->vapor_mass = 0.0;
    h->vapor_charge = 0.0;
    h->accel_shatter_thresh = 0.0;
    h->accel_shatter_fraction = 0.25;
    h->accel_shatter_k = 3u;
    h->accel_shatter_kick = 0.25;

    h->rng_state = 0x12345678u;

    // initialize the full edge list (pairs) once; clear adjacency & degree.
    {
        GP_Pair* pairs = ptr_pairs(h);
        uint8_t* bits = ptr_bond_bits(h);
        uint32_t* deg = ptr_degree(h);
        const uint32_t N = h->n;
        uint32_t idx = 0u;
        for (uint32_t i = 0u; i < N; i++) {
            for (uint32_t j = i + 1u; j < N; j++) {
                pairs[idx].i = i;
                pairs[idx].j = j;
                bits[idx] = 0u;
                idx++;
            }
            deg[i] = 0u;
        }
    }

    // initialize masses to 1 to avoid divide-by-zero surprises
    double* m = ptr_mass(h);
    for (uint32_t i = 0; i < n; i++) m[i] = 1.0;

    // initialize temps to ambient
    double* t = ptr_temps(h);
    for (uint32_t i = 0; i < n; i++) t[i] = h->temp_ambient;

    // history dt defaults to 0
    if (h->history_cap) {
        double* hdt = ptr_hist_dt(h);
        for (uint32_t i = 0; i < h->history_cap; i++) hdt[i] = 0.0;
    }

    return 1;
}

int gp_set_south(void* state_mem, uint32_t south_enabled, double south_strength, int32_t south_axis) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->south_enabled = south_enabled ? 1u : 0u;
    h->south_strength = south_strength;
    h->south_axis = south_axis;
    return 1;
}

static uint32_t history_age_to_index_raw(const GP_Header* h, uint32_t age) {
    if (!h->history_cap) return 0xFFFFFFFFu;
    if (age >= h->history_len) return 0xFFFFFFFFu;
    const uint32_t cap = h->history_cap;
    const uint32_t head = h->history_head;
    return (uint32_t)((head + cap - (age % cap)) % cap);
}

static void history_push(GP_Header* h, const double* p_sim, const double* v_sim, double dt) {
    if (!h->history_cap) return;
    const uint32_t cap = h->history_cap;
    uint32_t head = 0u;
    if (h->history_len == 0u) {
        head = 0u;
        h->history_len = 1u;
    } else {
        head = (h->history_head + 1u) % cap;
        if (h->history_len < cap) h->history_len += 1u;
    }
    h->history_head = head;

    const size_t nd = (size_t)h->n * (size_t)h->d;
    const size_t off = (size_t)head * nd;
    double* hp = ptr_hist_pos(h);
    double* hv = ptr_hist_vel(h);
    double* hdt = ptr_hist_dt(h);
    memcpy(hp + off, p_sim, nd * sizeof(double));
    memcpy(hv + off, v_sim, nd * sizeof(double));
    hdt[head] = dt;
    h->history_seq += 1u;
}

int gp_set_n_active(void* state_mem, uint32_t n_active) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    if (n_active == 0u) n_active = 1u;
    if (n_active > h->n) n_active = h->n;
    // Update both published and simulation counts.
    h->n_active = n_active;
    h->n_active_sim = n_active;
    return 1;
}

int gp_set_lifecycle_params(
    void* state_mem,
    double max_speed,
    double radius_scale,
    double mass_ref_cuberoot,
    double collide_gain,
    double merge_speed_frac,
    double merge_size_frac,
    double shatter_speed_frac,
    double shatter_size_frac,
    uint32_t shatter_k_max,
    uint32_t collision_pair_limit,
    double mass_vapor_thresh,
    double condense_temp_thresh,
    double condense_chunk_mass,
    double accel_shatter_thresh,
    double accel_shatter_fraction,
    uint32_t accel_shatter_k,
    double accel_shatter_kick
) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->max_speed = (max_speed >= 0.0) ? max_speed : 0.0;
    h->radius_scale = (radius_scale >= 0.0) ? radius_scale : 0.0;
    h->mass_ref_cuberoot = (mass_ref_cuberoot > 0.0) ? mass_ref_cuberoot : 1.0;
    h->collide_gain = (collide_gain >= 0.0) ? collide_gain : 0.0;
    h->merge_speed_frac = merge_speed_frac;
    h->merge_size_frac = merge_size_frac;
    h->shatter_speed_frac = shatter_speed_frac;
    h->shatter_size_frac = shatter_size_frac;
    h->shatter_k_max = (shatter_k_max < 2u) ? 2u : shatter_k_max;
    h->collision_pair_limit = collision_pair_limit;
    h->mass_vapor_thresh = (mass_vapor_thresh >= 0.0) ? mass_vapor_thresh : 0.0;
    h->condense_temp_thresh = condense_temp_thresh;
    h->condense_chunk_mass = (condense_chunk_mass >= 0.0) ? condense_chunk_mass : 0.0;
    h->accel_shatter_thresh = (accel_shatter_thresh >= 0.0) ? accel_shatter_thresh : 0.0;
    h->accel_shatter_fraction = accel_shatter_fraction;
    h->accel_shatter_k = (accel_shatter_k < 2u) ? 2u : accel_shatter_k;
    h->accel_shatter_kick = accel_shatter_kick;
    return 1;
}

int gp_set_bond_params(
    void* state_mem,
    uint32_t bonds_enabled,
    uint32_t bond_max_per_node,
    double bond_link_angle,
    double bond_shear_ratio,
    double bond_k,
    uint32_t ionic_bonds,
    uint32_t ionic_valence
) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->bonds_enabled = bonds_enabled ? 1u : 0u;
    h->bond_max_per_node = bond_max_per_node;
    h->bond_link_angle = bond_link_angle;
    h->bond_shear_ratio = (bond_shear_ratio <= 0.0) ? 1.0 : bond_shear_ratio;
    h->bond_k = bond_k;
    h->ionic_bonds = ionic_bonds ? 1u : 0u;
    h->ionic_valence = (ionic_valence == 0u) ? 1u : ionic_valence;
    return 1;
}

static uint32_t pair_index(uint32_t i, uint32_t j, uint32_t n) {
    // Requires i<j.
    // idx = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    //     = i*(2n - i - 1)/2 + (j - i - 1)
    return (uint32_t)(((uint64_t)i * (uint64_t)(2u * n - i - 1u)) / 2u + (uint64_t)(j - i - 1u));
}

int gp_set_params(void* state_mem, double k_spring, double k_coulomb, double G, double damping, double softening) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->k_spring = k_spring;
    h->k_coulomb = k_coulomb;
    h->G = G;
    h->damping = damping;
    h->softening = (softening > 0.0) ? softening : 1e-12;
    return 1;
}

int gp_set_lorentz(void* state_mem, double lorentz_k, double Bx, double By, double Bz) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->lorentz_k = lorentz_k;
    h->Bx = Bx;
    h->By = By;
    h->Bz = Bz;
    return 1;
}

int gp_set_temp_params(void* state_mem, double temp_ambient, double temp_conv, double temp_noise, double temp_heat_gain) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->temp_ambient = temp_ambient;
    h->temp_conv = (temp_conv >= 0.0) ? temp_conv : 0.0;
    h->temp_noise = (temp_noise >= 0.0) ? temp_noise : 0.0;
    h->temp_heat_gain = (temp_heat_gain >= 0.0) ? temp_heat_gain : 0.0;
    return 1;
}

int gp_seed(void* state_mem, uint32_t seed) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    h->rng_state = (seed != 0u) ? seed : 0x12345678u;
    return 1;
}

static uint32_t rng_next_u32(GP_Header* h) {
    // xorshift32
    uint32_t x = h->rng_state;
    if (x == 0u) x = 0x12345678u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    h->rng_state = x;
    return x;
}

static double rng_u01(GP_Header* h) {
    // [0,1)
    uint32_t x = rng_next_u32(h);
    return (double)x / 4294967296.0;
}

static double rng_normal_approx(GP_Header* h) {
    // Approximate N(0,1) via Irwin–Hall: sum_{k=1..12} U - 6
    double s = 0.0;
    for (int i = 0; i < 12; i++) s += rng_u01(h);
    return s - 6.0;
}

int gp_set_springs(void* state_mem, const GP_Spring* springs, uint32_t spring_count) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0;
    if (spring_count > h->spring_cap) return 0;
    GP_Spring* dst0 = ptr_springs_buf(h, 0u);
    GP_Spring* dst1 = ptr_springs_buf(h, 1u);
    if (spring_count && springs) {
        memcpy(dst0, springs, (size_t)spring_count * sizeof(GP_Spring));
        memcpy(dst1, springs, (size_t)spring_count * sizeof(GP_Spring));
    }
    h->spring_count = spring_count;
    h->spring_count_sim = spring_count;

    rebuild_bond_adjacency_from_springs(h);
    return 1;
}

static void normalize_vec(double* v, uint32_t d) {
    double s = 0.0;
    for (uint32_t k = 0; k < d; k++) s += v[k] * v[k];
    double inv = 1.0 / sqrt(fmax(s, 1e-24));
    for (uint32_t k = 0; k < d; k++) v[k] *= inv;
}

static double clampd(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double radius_from_mass(GP_Header* h, double m) {
    const double base = fmax(m, 1e-6);
    const double denom = (h->mass_ref_cuberoot > 1e-12) ? h->mass_ref_cuberoot : 1.0;
    if (!(h->radius_scale > 0.0)) return 0.0;
    return h->radius_scale * cbrt(base) / denom;
}

static void rand_tangent_unit(GP_Header* h, const double* p, double* out, uint32_t d) {
    // Random Gaussian vector, projected to tangent, normalized.
    for (uint32_t k = 0; k < d; k++) out[k] = rng_normal_approx(h);
    project_tangent(p, out, out, d);
    normalize_vec(out, d);
}

static void dirichlet_alpha1(GP_Header* h, double* w, uint32_t k) {
    // Dirichlet(alpha=1) by sampling Exp(1) and normalizing.
    double sum = 0.0;
    for (uint32_t i = 0; i < k; i++) {
        double u = rng_u01(h);
        if (u < 1e-12) u = 1e-12;
        double e = -log(u);
        w[i] = e;
        sum += e;
    }
    if (!(sum > 0.0)) {
        const double inv = 1.0 / (double)k;
        for (uint32_t i = 0; i < k; i++) w[i] = inv;
        return;
    }
    const double inv = 1.0 / sum;
    for (uint32_t i = 0; i < k; i++) w[i] *= inv;
}

static uint32_t clamp_u32(uint32_t x, uint32_t lo, uint32_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

typedef struct PairSpeed {
    uint32_t i;
    uint32_t j;
    double speed;
} PairSpeed;

static int pair_speed_cmp_desc(const void* a, const void* b) {
    const PairSpeed* pa = (const PairSpeed*)a;
    const PairSpeed* pb = (const PairSpeed*)b;
    if (pa->speed < pb->speed) return 1;
    if (pa->speed > pb->speed) return -1;
    return 0;
}

static void remove_springs_touching(GP_Header* h, uint8_t* removed_mask, uint32_t old_n_active) {
    // Filter spring list to remove any edge touching removed nodes.
    if (!h->spring_count_sim) return;
    GP_Spring* sp = ptr_springs_sim(h);
    uint32_t out = 0u;
    for (uint32_t e = 0u; e < h->spring_count_sim; e++) {
        uint32_t a = sp[e].i;
        uint32_t b = sp[e].j;
        if (a < old_n_active && removed_mask[a]) continue;
        if (b < old_n_active && removed_mask[b]) continue;
        sp[out++] = sp[e];
    }
    h->spring_count_sim = out;
}

static void remap_springs(GP_Header* h, const int32_t* remap, uint32_t old_n_active) {
    if (!h->spring_count_sim) return;
    GP_Spring* sp = ptr_springs_sim(h);
    uint32_t out = 0u;
    for (uint32_t e = 0u; e < h->spring_count_sim; e++) {
        uint32_t a = sp[e].i;
        uint32_t b = sp[e].j;
        if (a >= old_n_active || b >= old_n_active) continue;
        int32_t na = remap[a];
        int32_t nb = remap[b];
        if (na < 0 || nb < 0) continue;
        sp[e].i = (uint32_t)na;
        sp[e].j = (uint32_t)nb;
        sp[out++] = sp[e];
    }
    h->spring_count_sim = out;
}

static void rebuild_bond_adjacency_from_springs(GP_Header* h) {
    if (!h->bonds_enabled) return;
    if (!h->pair_count) return;
    uint8_t* bits = ptr_bond_bits(h);
    uint32_t* deg = ptr_degree(h);
    memset(bits, 0, (size_t)h->pair_count * sizeof(uint8_t));
    for (uint32_t i = 0; i < h->n; i++) deg[i] = 0u;
    GP_Spring* sp = ptr_springs_sim(h);
    for (uint32_t e = 0; e < h->spring_count_sim; e++) {
        uint32_t i = sp[e].i;
        uint32_t j = sp[e].j;
        if (i >= h->n || j >= h->n || i == j) continue;
        uint32_t a = (i < j) ? i : j;
        uint32_t b = (i < j) ? j : i;
        uint32_t idx = pair_index(a, b, h->n);
        if (idx >= h->pair_count) continue;
        if (!bits[idx]) {
            bits[idx] = 1u;
            deg[a] += 1u;
            deg[b] += 1u;
        }
    }
}

static void compact_nodes(
    GP_Header* h,
    double* p_sim,
    double* v_sim,
    uint8_t* keep_mask,
    uint32_t old_n_active,
    uint32_t* out_new_n_active,
    int32_t* out_remap
) {
    double* mass = ptr_mass(h);
    double* charge = ptr_charge(h);
    double* temps = ptr_temps(h);
    const uint32_t d = h->d;
    uint32_t w = 0u;
    for (uint32_t i = 0u; i < old_n_active; i++) {
        if (!keep_mask[i]) {
            out_remap[i] = -1;
            continue;
        }
        out_remap[i] = (int32_t)w;
        if (w != i) {
            memcpy(p_sim + (size_t)w * d, p_sim + (size_t)i * d, (size_t)d * sizeof(double));
            memcpy(v_sim + (size_t)w * d, v_sim + (size_t)i * d, (size_t)d * sizeof(double));
            mass[w] = mass[i];
            charge[w] = charge[i];
            temps[w] = temps[i];
        }
        w++;
    }
    *out_new_n_active = w;
}

static void lifecycle_accel_shatter(GP_Header* h, double* p_sim, double* v_sim, double dt, double* acc_buf) {
    if (!(h->accel_shatter_thresh > 0.0)) return;
    const uint32_t n = h->n;
    uint32_t n_active = h->n_active_sim;
    if (n_active < 1u) return;
    if (n_active >= n) {
        // no room to spawn shards; still allow parent checks but skip spawns
    }

    compute_force(h, p_sim, v_sim, dt, acc_buf);

    double* mass = ptr_mass(h);
    double* charge = ptr_charge(h);
    double* temps = ptr_temps(h);
    const uint32_t d = h->d;

    const double temp_live = h->temp_ambient;
    const double hot_temp = fmax(0.0, temp_live);
    const double cold_temp = fmax(0.0, -temp_live);
    const double hot_gain = 1.0 + 0.5 * hot_temp;
    const double cold_gain = 1.0 + 0.4 * cold_temp;

    double hot_prob = fmin(0.98, 0.10 * hot_temp);
    if (hot_temp > 0.05 && hot_prob < 0.02) hot_prob = 0.02;
    if (hot_temp <= 0.05) hot_prob = 0.0;

    const double split_frac = clampd(h->accel_shatter_fraction, 0.05, 0.8);
    const uint32_t k = clamp_u32(h->accel_shatter_k, 2u, 1024u);

    // Iterate over current active nodes; appends go at the end.
    for (uint32_t idx = 0u; idx < n_active; idx++) {
        const double m_v = mass[idx];
        if (!(m_v > 1e-6)) continue;

        // mass-dependent threshold scaling
        const double mass_scale = fmax(m_v / fmax((double)(h->mass_ref_cuberoot * h->mass_ref_cuberoot * h->mass_ref_cuberoot), 1e-6), 0.25);
        const double frag_gain = 1.0 + 0.75 * sqrt(mass_scale);
        const double eff_thresh = h->accel_shatter_thresh * cold_gain / fmax((hot_gain * frag_gain), 1e-12);

        const double* ai = acc_buf + (size_t)idx * d;
        double acc_mag2 = 0.0;
        for (uint32_t kk = 0; kk < d; kk++) acc_mag2 += ai[kk] * ai[kk];
        const double acc_mag = sqrt(fmax(acc_mag2, 0.0));

        double overdrive = 0.0;
        if (eff_thresh > 1e-12) overdrive = fmax((acc_mag / eff_thresh) - 1.0, 0.0);
        double over_prob = clampd(0.15 * overdrive, 0.0, 0.65);

        const int rand_hot = (hot_prob > 0.0) && (rng_u01(h) < hot_prob);
        const int rand_over = (over_prob > 0.0) && (rng_u01(h) < over_prob);
        const int hit = (acc_mag > eff_thresh) || rand_hot || rand_over;
        if (!hit) continue;

        // Capacity check (need k new nodes).
        if (h->n_active_sim + k > h->n) continue;

        const double q_v = charge[idx];
        const double m_split = m_v * split_frac;
        if (!(m_split > 1e-6)) continue;
        const double m_remain = fmax(1e-6, m_v - m_split);
        const double m_share = m_split / (double)k;

        const double q_split = q_v * split_frac;
        const double sign_q = (q_split >= 0.0) ? 1.0 : -1.0;
        const double abs_q = fabs(q_split);

        // Compute charge shares.
        double weights_stack[1024];
        double* weights = weights_stack;
        if (k > 1024u) {
            weights = (double*)malloc((size_t)k * sizeof(double));
            if (!weights) continue;
        }
        if (abs_q < 1e-12) {
            for (uint32_t kk = 0; kk < k; kk++) weights[kk] = 0.0;
        } else {
            dirichlet_alpha1(h, weights, k);
            for (uint32_t kk = 0; kk < k; kk++) weights[kk] = weights[kk] * abs_q * sign_q;
            // exact conservation
            double sumq = 0.0;
            for (uint32_t kk = 0; kk < k; kk++) sumq += weights[kk];
            weights[0] += (q_split - sumq);
        }

        const double* p_v = p_sim + (size_t)idx * d;
        const double* v_v = v_sim + (size_t)idx * d;

        // Acceleration tangent direction.
        double acc_tan[64];
        for (uint32_t kk = 0; kk < d; kk++) acc_tan[kk] = ai[kk];
        project_tangent(p_v, acc_tan, acc_tan, d);
        double acc_tan_n2 = 0.0;
        for (uint32_t kk = 0; kk < d; kk++) acc_tan_n2 += acc_tan[kk] * acc_tan[kk];
        if (acc_tan_n2 < 1e-12) {
            rand_tangent_unit(h, p_v, acc_tan, d);
        } else {
            normalize_vec(acc_tan, d);
        }

        const double offset_scale = 0.3 * radius_from_mass(h, m_v);

        for (uint32_t kk = 0; kk < k; kk++) {
            double jitter[64];
            rand_tangent_unit(h, p_v, jitter, d);
            double dir_vec[64];
            for (uint32_t t = 0; t < d; t++) dir_vec[t] = acc_tan[t] + 0.4 * jitter[t];
            normalize_vec(dir_vec, d);

            double p_new[64];
            for (uint32_t t = 0; t < d; t++) p_new[t] = p_v[t] + offset_scale * dir_vec[t];
            normalize_vec(p_new, d);

            double v_new_local[64];
            for (uint32_t t = 0; t < d; t++) v_new_local[t] = v_v[t] + dir_vec[t] * (h->accel_shatter_kick * acc_mag);
            project_tangent(p_new, v_new_local, v_new_local, d);

            const uint32_t out_idx = h->n_active_sim;
            memcpy(p_sim + (size_t)out_idx * d, p_new, (size_t)d * sizeof(double));
            memcpy(v_sim + (size_t)out_idx * d, v_new_local, (size_t)d * sizeof(double));
            mass[out_idx] = m_share;
            charge[out_idx] = (abs_q < 1e-12) ? 0.0 : weights[kk];
            temps[out_idx] = temps[idx];
            h->n_active_sim += 1u;
        }

        // Update parent.
        mass[idx] = m_remain;
        charge[idx] = q_v - q_split;

        if (weights != weights_stack) free(weights);
        // n_active used for loop bound; do not iterate over freshly appended nodes.
    }
}

static void lifecycle_collisions(GP_Header* h, double* p_sim, double* v_sim) {
    if (!(h->collide_gain > 0.0)) return;
    uint32_t n_active = h->n_active_sim;
    if (n_active < 2u) return;

    const double max_speed = h->max_speed;
    if (!(max_speed > 0.0)) return;
    const double v_merge = h->merge_speed_frac * max_speed;
    const double v_shatter = h->shatter_speed_frac * max_speed;

    // Build candidate pair list.
    PairSpeed* pairs = NULL;
    uint32_t pair_n = 0u;
    const uint32_t limit = h->collision_pair_limit;
    if (limit > 0u && n_active <= limit) {
        const uint64_t pc64 = ((uint64_t)n_active * (uint64_t)(n_active - 1u)) / 2u;
        if (pc64 > 0xFFFFFFFFu) return;
        pair_n = (uint32_t)pc64;
        pairs = (PairSpeed*)malloc((size_t)pair_n * sizeof(PairSpeed));
        if (!pairs) return;
        uint32_t idx = 0u;
        for (uint32_t i = 0u; i < n_active; i++) {
            for (uint32_t j = i + 1u; j < n_active; j++) {
                pairs[idx].i = i;
                pairs[idx].j = j;
                pairs[idx].speed = 0.0;
                idx++;
            }
        }
    } else if (h->spring_count_sim) {
        pair_n = h->spring_count_sim;
        pairs = (PairSpeed*)malloc((size_t)pair_n * sizeof(PairSpeed));
        if (!pairs) return;
        GP_Spring* sp = ptr_springs_sim(h);
        for (uint32_t e = 0u; e < pair_n; e++) {
            pairs[e].i = sp[e].i;
            pairs[e].j = sp[e].j;
            pairs[e].speed = 0.0;
        }
    } else {
        return;
    }

    const uint32_t d = h->d;
    double* mass = ptr_mass(h);
    double* charge = ptr_charge(h);
    double* temps = ptr_temps(h);

    // Precompute speeds for sorting.
    for (uint32_t k = 0u; k < pair_n; k++) {
        const uint32_t i = pairs[k].i;
        const uint32_t j = pairs[k].j;
        if (i >= n_active || j >= n_active || i == j) {
            pairs[k].speed = -1.0;
            continue;
        }
        const double* vi = v_sim + (size_t)i * d;
        const double* vj = v_sim + (size_t)j * d;
        double s2 = 0.0;
        for (uint32_t kk = 0; kk < d; kk++) {
            double dv = vi[kk] - vj[kk];
            s2 += dv * dv;
        }
        pairs[k].speed = sqrt(fmax(s2, 0.0));
    }

    qsort(pairs, (size_t)pair_n, sizeof(PairSpeed), pair_speed_cmp_desc);

    // Track removals and spawned shards (in old index space).
    uint8_t* remove_mask = (uint8_t*)calloc((size_t)n_active, 1u);
    if (!remove_mask) {
        free(pairs);
        return;
    }

    // Shards are appended after compaction.
    typedef struct Shard {
        double p[GP_MAX_D];
        double v[GP_MAX_D];
        double m;
        double q;
        double t;
    } Shard;
    // Lifecycle uses fixed-size temporary vectors.
    if (d > GP_MAX_D) {
        free(remove_mask);
        free(pairs);
        return;
    }
    Shard* shards = NULL;
    uint32_t shard_n = 0u;
    uint32_t shard_cap = 0u;

    // Radii cache.
    double* radii = (double*)malloc((size_t)n_active * sizeof(double));
    if (!radii) {
        free(remove_mask);
        free(pairs);
        return;
    }
    for (uint32_t i = 0u; i < n_active; i++) radii[i] = radius_from_mass(h, mass[i]);

    for (uint32_t k = 0u; k < pair_n; k++) {
        const uint32_t i = pairs[k].i;
        const uint32_t j = pairs[k].j;
        if (i >= n_active || j >= n_active || i == j) continue;
        if (remove_mask[i] || remove_mask[j]) continue;

        const double* pi = p_sim + (size_t)i * d;
        const double* pj = p_sim + (size_t)j * d;
        double dist2 = 0.0;
        for (uint32_t kk = 0; kk < d; kk++) {
            double diff = pi[kk] - pj[kk];
            dist2 += diff * diff;
        }
        const double dist = sqrt(fmax(dist2, 0.0));
        const double r_sum = (radii[i] + radii[j]) * h->collide_gain;
        if (!(dist < r_sum)) continue;

        const double v_rel = pairs[k].speed;
        const double m_i = mass[i];
        const double m_j = mass[j];
        const double size_ratio = fmin(m_i, m_j) / fmax(fmax(m_i, m_j), 1e-12);

        if ((v_rel <= v_merge) || (size_ratio >= h->merge_size_frac)) {
            // Merge: keep i, remove j.
            const double m_new = m_i + m_j;
            const double q_new = charge[i] + charge[j];
            const double w_i = m_i / fmax(m_new, 1e-12);
            const double w_j = 1.0 - w_i;
            double p_new[GP_MAX_D];
            for (uint32_t kk = 0u; kk < d; kk++) p_new[kk] = w_i * pi[kk] + w_j * pj[kk];
            normalize_vec(p_new, d);
            const double* vi = v_sim + (size_t)i * d;
            const double* vj = v_sim + (size_t)j * d;
            double v_new[GP_MAX_D];
            for (uint32_t kk = 0u; kk < d; kk++) v_new[kk] = w_i * vi[kk] + w_j * vj[kk];
            project_tangent(p_new, v_new, v_new, d);
            memcpy(p_sim + (size_t)i * d, p_new, (size_t)d * sizeof(double));
            memcpy(v_sim + (size_t)i * d, v_new, (size_t)d * sizeof(double));
            mass[i] = m_new;
            charge[i] = q_new;
            // temperature: weighted blend
            temps[i] = w_i * temps[i] + w_j * temps[j];
            remove_mask[j] = 1u;
        } else if ((v_rel >= v_shatter) && (size_ratio <= h->shatter_size_frac)) {
            // Shatter: remove victim (larger mass), spawn shards.
            const uint32_t victim = (m_i >= m_j) ? i : j;
            const uint32_t attacker = (victim == i) ? j : i;
            if (remove_mask[victim]) continue;
            const double m_v = mass[victim];
            const double q_v = charge[victim];
            const double t_v = temps[victim];

            uint32_t k_shards = 2u + (uint32_t)floor((v_rel / max_speed) * 2.0);
            if (k_shards < 2u) k_shards = 2u;
            k_shards = clamp_u32(k_shards, 2u, h->shatter_k_max);

            // Capacity check: worst case append all shards.
            if (h->n_active_sim + k_shards > h->n) continue;

            const double m_share = m_v / (double)k_shards;
            const double sign_q = (q_v >= 0.0) ? 1.0 : -1.0;
            const double abs_q = fabs(q_v);
            double weights_stack[1024];
            double* q_share = weights_stack;
            if (k_shards > 1024u) {
                q_share = (double*)malloc((size_t)k_shards * sizeof(double));
                if (!q_share) continue;
            }
            if (abs_q < 1e-12) {
                for (uint32_t kk = 0; kk < k_shards; kk++) q_share[kk] = 0.0;
            } else {
                dirichlet_alpha1(h, q_share, k_shards);
                for (uint32_t kk = 0; kk < k_shards; kk++) q_share[kk] = q_share[kk] * abs_q * sign_q;
                double sumq = 0.0;
                for (uint32_t kk = 0; kk < k_shards; kk++) sumq += q_share[kk];
                q_share[0] += (q_v - sumq);
            }

            const double* p_v = p_sim + (size_t)victim * d;
            const double* v_att = v_sim + (size_t)attacker * d;
            double dir_att[GP_MAX_D];
            for (uint32_t kk = 0u; kk < d; kk++) dir_att[kk] = v_att[kk];
            project_tangent(p_v, dir_att, dir_att, d);
            double da2 = 0.0;
            for (uint32_t kk = 0u; kk < d; kk++) da2 += dir_att[kk] * dir_att[kk];
            if (da2 < 1e-12) {
                rand_tangent_unit(h, p_v, dir_att, d);
            } else {
                normalize_vec(dir_att, d);
            }

            // Ensure shard storage.
            if (shard_n + k_shards > shard_cap) {
                uint32_t new_cap = (shard_cap == 0u) ? 16u : shard_cap;
                while (new_cap < shard_n + k_shards) new_cap *= 2u;
                Shard* ns = (Shard*)realloc(shards, (size_t)new_cap * sizeof(Shard));
                if (!ns) {
                    if (q_share != weights_stack) free(q_share);
                    break;
                }
                shards = ns;
                shard_cap = new_cap;
            }

            for (uint32_t kk = 0u; kk < k_shards; kk++) {
                double jitter[GP_MAX_D];
                rand_tangent_unit(h, p_v, jitter, d);
                const double offset = 0.5 * radii[victim];
                double p_new[GP_MAX_D];
                for (uint32_t tkk = 0u; tkk < d; tkk++) p_new[tkk] = p_v[tkk] + offset * jitter[tkk];
                normalize_vec(p_new, d);

                const double* v_v = v_sim + (size_t)victim * d;
                const double v_bias = v_rel * 0.25;
                double v_new[GP_MAX_D];
                for (uint32_t tkk = 0u; tkk < d; tkk++) v_new[tkk] = v_v[tkk] + dir_att[tkk] * v_bias;
                project_tangent(p_new, v_new, v_new, d);

                memcpy(shards[shard_n].p, p_new, (size_t)d * sizeof(double));
                memcpy(shards[shard_n].v, v_new, (size_t)d * sizeof(double));
                shards[shard_n].m = m_share;
                shards[shard_n].q = (abs_q < 1e-12) ? 0.0 : q_share[kk];
                shards[shard_n].t = t_v;
                shard_n++;
            }

            remove_mask[victim] = 1u;
            if (q_share != weights_stack) free(q_share);
        }
    }

    free(pairs);
    free(radii);

    if (!shard_n) {
        // Only removals.
    }

    // If nothing removed and no shards, done.
    int any_removed = 0;
    for (uint32_t i = 0u; i < n_active; i++) {
        if (remove_mask[i]) { any_removed = 1; break; }
    }
    if (!any_removed && shard_n == 0u) {
        free(remove_mask);
        free(shards);
        return;
    }

    // Keep mask is inverse of remove.
    uint8_t* keep_mask = (uint8_t*)malloc((size_t)n_active);
    int32_t* remap = (int32_t*)malloc((size_t)n_active * sizeof(int32_t));
    if (!keep_mask || !remap) {
        free(keep_mask);
        free(remap);
        free(remove_mask);
        free(shards);
        return;
    }
    for (uint32_t i = 0u; i < n_active; i++) keep_mask[i] = (remove_mask[i] ? 0u : 1u);

    // Remove springs touching removed nodes.
    remove_springs_touching(h, remove_mask, n_active);

    uint32_t new_n_active = 0u;
    compact_nodes(h, p_sim, v_sim, keep_mask, n_active, &new_n_active, remap);
    remap_springs(h, remap, n_active);
    h->n_active_sim = new_n_active;

    // Append shards without springs.
    for (uint32_t si = 0u; si < shard_n; si++) {
        if (h->n_active_sim >= h->n) break;
        const uint32_t out_idx = h->n_active_sim;
        memcpy(p_sim + (size_t)out_idx * d, shards[si].p, (size_t)d * sizeof(double));
        memcpy(v_sim + (size_t)out_idx * d, shards[si].v, (size_t)d * sizeof(double));
        mass[out_idx] = shards[si].m;
        charge[out_idx] = shards[si].q;
        temps[out_idx] = shards[si].t;
        h->n_active_sim += 1u;
    }

    // Rebuild adjacency if needed.
    rebuild_bond_adjacency_from_springs(h);

    free(keep_mask);
    free(remap);
    free(remove_mask);
    free(shards);
}

static void lifecycle_vaporize_small(GP_Header* h, double* p_sim, double* v_sim) {
    if (!(h->mass_vapor_thresh > 0.0)) return;
    uint32_t n_active = h->n_active_sim;
    if (n_active < 1u) return;
    double* mass = ptr_mass(h);
    double* charge = ptr_charge(h);
    double* temps = ptr_temps(h);

    uint8_t* keep_mask = (uint8_t*)malloc((size_t)n_active);
    int32_t* remap = (int32_t*)malloc((size_t)n_active * sizeof(int32_t));
    if (!keep_mask || !remap) {
        free(keep_mask);
        free(remap);
        return;
    }
    uint8_t any = 0u;
    for (uint32_t i = 0u; i < n_active; i++) {
        if (mass[i] < h->mass_vapor_thresh) {
            keep_mask[i] = 0u;
            any = 1u;
        } else {
            keep_mask[i] = 1u;
        }
    }
    if (!any) {
        free(keep_mask);
        free(remap);
        return;
    }

    // Accumulate vapor pool.
    for (uint32_t i = 0u; i < n_active; i++) {
        if (!keep_mask[i]) {
            h->vapor_mass += fmax(mass[i], 0.0);
            h->vapor_charge += charge[i];
        }
    }

    // Build removed mask for spring filtering.
    uint8_t* removed = (uint8_t*)malloc((size_t)n_active);
    if (!removed) {
        free(keep_mask);
        free(remap);
        return;
    }
    for (uint32_t i = 0u; i < n_active; i++) removed[i] = (keep_mask[i] ? 0u : 1u);
    remove_springs_touching(h, removed, n_active);

    uint32_t new_n_active = 0u;
    compact_nodes(h, p_sim, v_sim, keep_mask, n_active, &new_n_active, remap);
    remap_springs(h, remap, n_active);
    h->n_active_sim = new_n_active;

    // Fill temps for moved nodes already handled; but ensure ambient for any remaining mismatch.
    for (uint32_t i = 0u; i < h->n_active_sim; i++) {
        if (!isfinite(temps[i])) temps[i] = h->temp_ambient;
    }

    rebuild_bond_adjacency_from_springs(h);

    free(removed);
    free(keep_mask);
    free(remap);
}

static void lifecycle_condense_vapor(GP_Header* h, double* p_sim, double* v_sim) {
    if (!(h->condense_chunk_mass > 0.0)) return;
    if (!(h->vapor_mass > 1e-9)) return;
    const double ambient = h->temp_ambient;
    if (ambient > h->condense_temp_thresh) return;
    if (h->n_active_sim >= h->n) return;
    const uint32_t d = h->d;
    if (d > GP_MAX_D) return;

    const double cold_gain = 1.0 + fmax(0.0, -ambient) * 1.5;
    double chunk = h->condense_chunk_mass * cold_gain;
    chunk = fmin(chunk, h->vapor_mass);
    if (!(chunk > 0.0)) return;

    const double q_chunk = (h->vapor_mass > 0.0) ? (h->vapor_charge * (chunk / h->vapor_mass)) : 0.0;
    h->vapor_mass -= chunk;
    h->vapor_charge -= q_chunk;

    uint32_t n_active = h->n_active_sim;
    double p_new[GP_MAX_D];
    if (n_active > 0u) {
        const uint32_t anchor = (uint32_t)(rng_next_u32(h) % n_active);
        const double* p_anchor = p_sim + (size_t)anchor * d;
        double jitter[GP_MAX_D];
        rand_tangent_unit(h, p_anchor, jitter, d);
        const double offset = 0.4 * radius_from_mass(h, chunk);
        for (uint32_t kk = 0u; kk < d; kk++) p_new[kk] = p_anchor[kk] + offset * jitter[kk];
        normalize_vec(p_new, d);
    } else {
        for (uint32_t kk = 0u; kk < d; kk++) p_new[kk] = rng_normal_approx(h);
        normalize_vec(p_new, d);
    }
    double v_new[GP_MAX_D];
    for (uint32_t kk = 0u; kk < d; kk++) v_new[kk] = 0.0;

    const uint32_t out_idx = h->n_active_sim;
    memcpy(p_sim + (size_t)out_idx * d, p_new, (size_t)d * sizeof(double));
    memcpy(v_sim + (size_t)out_idx * d, v_new, (size_t)d * sizeof(double));
    double* mass = ptr_mass(h);
    double* charge = ptr_charge(h);
    double* temps = ptr_temps(h);
    mass[out_idx] = chunk;
    charge[out_idx] = q_chunk;
    temps[out_idx] = ambient;
    h->n_active_sim += 1u;
}

static void project_tangent(const double* p, const double* v, double* out, uint32_t d) {
    double dot = 0.0;
    for (uint32_t k = 0; k < d; k++) dot += v[k] * p[k];
    for (uint32_t k = 0; k < d; k++) out[k] = v[k] - dot * p[k];
}

static void expmap(const double* p, const double* v_step, double* out, uint32_t d) {
    // out = cos(|v|) p + sin(|v|) v_hat
    double s = 0.0;
    for (uint32_t k = 0; k < d; k++) s += v_step[k] * v_step[k];
    double vn = sqrt(s);
    if (vn < 1e-12 || !isfinite(vn)) {
        // small-step series: p + v - 0.5|v|^2 p
        for (uint32_t k = 0; k < d; k++) out[k] = p[k] + v_step[k] - 0.5 * s * p[k];
        normalize_vec(out, d);
        return;
    }
    double inv = 1.0 / vn;
    double c = cos(vn);
    double sn = sin(vn);
    for (uint32_t k = 0; k < d; k++) {
        double vhat = v_step[k] * inv;
        out[k] = c * p[k] + sn * vhat;
    }
    normalize_vec(out, d);
}

static void force_zero(double* F, uint32_t n, uint32_t d) {
    memset(F, 0, (size_t)n * (size_t)d * sizeof(double));
}

static void add_pair_force(
    double* Fi,
    double* Fj,
    const double* pi,
    const double* pj,
    double strength,
    double softening,
    double core_radius,
    double core_stiffness,
    uint32_t d
) {
    // Match the Torch path semantics:
    //   dist = raw_dist + softening
    //   base force ~ strength * diff / dist^3
    // plus hard-core repulsion when raw_dist < core_radius:
    //   repulsion_mag = core_stiffness * (core_radius - raw_dist)
    //   repulsion force = repulsion_mag * diff / raw_dist

    double diff_buf[64];
    // d is small in our current usage (D==3 for GL mode), but keep safe for modest D.
    // If someone runs with large D, fall back to recomputing diff.
    int use_buf = (d <= 64);

    double dist2 = 0.0;
    for (uint32_t k = 0; k < d; k++) {
        double diff = pi[k] - pj[k];
        if (use_buf) diff_buf[k] = diff;
        dist2 += diff * diff;
    }
    double raw = sqrt(dist2);
    double dist = raw + softening;
    if (dist < 1e-12) dist = 1e-12;
    double inv = 1.0 / (dist * dist * dist);
    double scale = strength * inv;

    double repulsion_mag = 0.0;
    if (core_stiffness > 0.0 && core_radius > 0.0 && raw < core_radius) {
        repulsion_mag = core_stiffness * (core_radius - raw);
    }
    double inv_raw = (raw > 1e-12) ? (1.0 / raw) : 0.0;

    for (uint32_t k = 0; k < d; k++) {
        double diff = use_buf ? diff_buf[k] : (pi[k] - pj[k]);
        double f = scale * diff;
        if (repulsion_mag != 0.0) {
            f += repulsion_mag * inv_raw * diff;
        }
        Fi[k] += f;
        Fj[k] -= f;
    }
}

static void add_spring_force(double* Fi, double* Fj, const double* pi, const double* pj, double rest, double k_s, uint32_t d) {
    // energy 0.5*k*(theta-rest)^2, theta = acos(dot)
    double dot = 0.0;
    for (uint32_t k = 0; k < d; k++) dot += pi[k] * pj[k];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    double theta = acos(dot);
    double sin_t = sin(theta);
    if (sin_t < 1e-9) return;
    double err = (theta - rest);
    double mag = -k_s * err;

    // unit tangent at pi pointing toward pj: u_i = (pj - dot*pi)/sin(theta)
    // unit tangent at pj pointing toward pi: u_j = (pi - dot*pj)/sin(theta)
    double inv_sin = 1.0 / sin_t;
    for (uint32_t k = 0; k < d; k++) {
        double ui = (pj[k] - dot * pi[k]) * inv_sin;
        double uj = (pi[k] - dot * pj[k]) * inv_sin;
        Fi[k] += mag * ui;
        Fj[k] += mag * uj;
    }
}

static void compute_force(GP_Header* h, const double* p, const double* v, double dt, double* Ftmp) {
    const uint32_t n = h->n_active_sim;
    const uint32_t d = h->d;

    force_zero(Ftmp, n, d);

    const double* mass = ptr_mass(h);
    const double* charge = ptr_charge(h);
    const double* temps = ptr_temps(h);
    const double k_c = h->k_coulomb;
    const double G = h->G;
    const double soft = h->softening;

    // Match Torch defaults used by _geodesic_forces.
    const double core_radius = 0.06;
    const double core_stiffness = 800.0;

    // Pairwise coulomb + gravity in ambient space (chord distance).
    if ((k_c != 0.0) || (G != 0.0)) {
        for (uint32_t i = 0; i < n; i++) {
            const double* pi = p + (size_t)i * d;
            double* Fi = Ftmp + (size_t)i * d;
            for (uint32_t j = i + 1; j < n; j++) {
                const double* pj = p + (size_t)j * d;
                double* Fj = Ftmp + (size_t)j * d;
                double strength = 0.0;
                if (k_c != 0.0) strength += k_c * charge[i] * charge[j];
                if (G != 0.0) strength += -G * mass[i] * mass[j];
                if (strength != 0.0) {
                    add_pair_force(Fi, Fj, pi, pj, strength, soft, core_radius, core_stiffness, d);
                }
            }
        }
    }

    // Springs (geodesic angle)
    const uint32_t sc = h->spring_count_sim;
    const GP_Spring* sp = ptr_springs_sim(h);
    const double k_base = h->k_spring;
    if (sc) {
        for (uint32_t e = 0; e < sc; e++) {
            uint32_t i = sp[e].i;
            uint32_t j = sp[e].j;
            if (i >= n || j >= n || i == j) continue;
            const double* pi = p + (size_t)i * d;
            const double* pj = p + (size_t)j * d;
            double* Fi = Ftmp + (size_t)i * d;
            double* Fj = Ftmp + (size_t)j * d;
            double k_s = (sp[e].k != 0.0) ? sp[e].k : k_base;
            add_spring_force(Fi, Fj, pi, pj, sp[e].rest_angle, k_s, d);
        }
    }

    // Lorentz-like term (3D only): a_force += lorentz_k * (v x B)
    // Applied before mass/inertia divide to match Torch ordering (where Lorentz is added to F).
    if (h->lorentz_k != 0.0 && d >= 3) {
        const double lk = h->lorentz_k;
        const double Bx = h->Bx;
        const double By = h->By;
        const double Bz = h->Bz;
        for (uint32_t i = 0; i < n; i++) {
            const double* vi = v + (size_t)i * d;
            double* Fi = Ftmp + (size_t)i * d;
            double vx = vi[0], vy = vi[1], vz = vi[2];
            // v x B
            double cx = vy * Bz - vz * By;
            double cy = vz * Bx - vx * Bz;
            double cz = vx * By - vy * Bx;
            Fi[0] += lk * cx;
            Fi[1] += lk * cy;
            Fi[2] += lk * cz;
        }
    }

    // Mass-as-inertia (Torch): a = F / m, then tangent projection.
    for (uint32_t i = 0; i < n; i++) {
        const double* pi = p + (size_t)i * d;
        double* Fi = Ftmp + (size_t)i * d;

        double mi = mass[i];
        if (!(mi > 1e-12) || !isfinite(mi)) mi = 1e-12;
        double inv_m = 1.0 / mi;
        for (uint32_t k = 0; k < d; k++) Fi[k] *= inv_m;

        double dot = 0.0;
        for (uint32_t k = 0; k < d; k++) dot += Fi[k] * pi[k];
        for (uint32_t k = 0; k < d; k++) Fi[k] = Fi[k] - dot * pi[k];
    }

    // Thermal noise (Torch-like but per-node): a += sqrt(dt) * temp_noise * |T_i| * N(0,1)
    // Noise is tangent-projected at p.
    if (h->temp_noise > 0.0 && dt > 0.0) {
        double scale_dt = sqrt(dt) * h->temp_noise;
        for (uint32_t i = 0; i < n; i++) {
            double Ti = temps[i];
            double amp = fabs(Ti) * scale_dt;
            if (amp == 0.0) continue;
            const double* pi = p + (size_t)i * d;
            double* Fi = Ftmp + (size_t)i * d;

            // generate random vector and project to tangent
            double dot = 0.0;
            for (uint32_t k = 0; k < d; k++) {
                double r = rng_normal_approx(h);
                Fi[k] += amp * r;
            }
            // project tangent in-place for just the added noise component is tricky;
            // do a full projection for simplicity.
            for (uint32_t k = 0; k < d; k++) dot += Fi[k] * pi[k];
            for (uint32_t k = 0; k < d; k++) Fi[k] = Fi[k] - dot * pi[k];
        }
    }

    // Per-frame damping semantics (Torch force_at): lambda = damping / dt_eff.
    const double damp = h->damping;
    if (damp > 0.0) {
        double damp_clamped = damp;
        if (damp_clamped < 0.0) damp_clamped = 0.0;
        if (damp_clamped > 1.0) damp_clamped = 1.0;
        double dt_eff = (dt > 1e-12) ? dt : 1e-12;
        double lambda_d = damp_clamped / dt_eff;
        for (uint32_t i = 0; i < n; i++) {
            const double* vi = v + (size_t)i * d;
            double* Fi = Ftmp + (size_t)i * d;
            for (uint32_t k = 0; k < d; k++) Fi[k] -= lambda_d * vi[k];
        }
    }

    // South-pole field (Torch force_at): F += south_strength * tang(dir, p)
    if (h->south_enabled && h->south_strength != 0.0) {
        int32_t axis = h->south_axis;
        if (axis < 0) axis = (int32_t)d - 1;
        if (axis >= 0 && (uint32_t)axis < d) {
            const double strength = h->south_strength;
            for (uint32_t i = 0; i < n; i++) {
                const double* pi = p + (size_t)i * d;
                double* Fi = Ftmp + (size_t)i * d;
                // dir is constant: -1 on selected axis, 0 elsewhere
                double dot = -pi[(uint32_t)axis];
                for (uint32_t k = 0; k < d; k++) {
                    double dirk = (k == (uint32_t)axis) ? -1.0 : 0.0;
                    double tangk = dirk - dot * pi[k];
                    Fi[k] += strength * tangk;
                }
            }
        }
    }
}

static void update_temps(GP_Header* h, const double* v, double dt) {
    if (dt <= 0.0) return;
    const uint32_t n = h->n_active_sim;
    const uint32_t d = h->d;
    double* temps = ptr_temps(h);
    const double* mass = ptr_mass(h);

    const double ambient = h->temp_ambient;
    const double conv = h->temp_conv;
    const double heat_gain = h->temp_heat_gain;

    // Damping-heating uses the same lambda as our per-frame damping.
    double damp = h->damping;
    if (damp < 0.0) damp = 0.0;
    if (damp > 1.0) damp = 1.0;
    double lambda_d = (damp > 0.0) ? (damp / fmax(dt, 1e-12)) : 0.0;

    for (uint32_t i = 0; i < n; i++) {
        const double* vi = v + (size_t)i * d;
        double v2 = 0.0;
        for (uint32_t k = 0; k < d; k++) v2 += vi[k] * vi[k];
        double speed = sqrt(v2);

        double Ti = temps[i];
        if (conv > 0.0) {
            // convection proportional to interface speed
            Ti += (-conv * speed * (Ti - ambient)) * dt;
        } else {
            // weak relax toward ambient if conv==0 but someone still wants ambient behavior
            Ti += (-(Ti - ambient) * 0.0) * dt;
        }

        if (heat_gain > 0.0 && lambda_d > 0.0) {
            // crude: convert damping energy loss into heat.
            // energy lost per unit mass ~ lambda * |v|^2 * dt.
            // use mass as heat capacity: heavier nodes heat less.
            double mi = mass[i];
            if (!(mi > 1e-12) || !isfinite(mi)) mi = 1e-12;
            double dT_heat = heat_gain * (lambda_d * v2 * dt) / mi;
            Ti += dT_heat;
        }
        temps[i] = Ti;
    }
}

static double clamp01(double x) {
    if (x > 1.0) return 1.0;
    if (x < -1.0) return -1.0;
    return x;
}

static double angle_between(const double* a, const double* b, uint32_t d) {
    // Robust angular separation. Positions are intended to be unit vectors,
    // but lifecycle/init/transient numerics can drift; guard against that so
    // bond logic cannot spuriously treat far points as coincident.
    double dot = 0.0;
    double na2 = 0.0;
    double nb2 = 0.0;
    for (uint32_t k = 0; k < d; k++) {
        double ak = a[k];
        double bk = b[k];
        dot += ak * bk;
        na2 += ak * ak;
        nb2 += bk * bk;
    }
    double denom = sqrt(fmax(na2 * nb2, 0.0));
    if (!(denom > 1e-12) || !isfinite(denom)) {
        // If either vector is degenerate, treat as maximally separated.
        return 3.141592653589793;
    }
    double c = dot / denom;
    c = clamp01(c);
    return acos(c);
}

static void update_bonds(GP_Header* h, const double* p, const double* v, double dt) {
    (void)v;
    (void)dt;
    if (!h->bonds_enabled) return;
    if (h->pair_count == 0u) return;
    if (!(h->bond_link_angle > 0.0)) return;

    const uint32_t n_active = h->n_active_sim;
    const uint32_t n_total = h->n;
    const uint32_t d = h->d;
    GP_Spring* sp = ptr_springs_sim(h);
    uint32_t sc = h->spring_count_sim;
    const uint32_t cap = h->spring_cap;
    uint8_t* bits = ptr_bond_bits(h);
    uint32_t* deg = ptr_degree(h);
    const double* charge = ptr_charge(h);

    const double shear = (h->bond_shear_ratio > 0.0) ? h->bond_shear_ratio : 1.0;

    // Break bonds that are overstretched.
    uint32_t e = 0u;
    while (e < sc) {
        uint32_t i = sp[e].i;
        uint32_t j = sp[e].j;
        if (i >= n_active || j >= n_active || i == j) {
            sp[e] = sp[sc - 1u];
            sc--;
            continue;
        }
        const double* pi = p + (size_t)i * d;
        const double* pj = p + (size_t)j * d;
        double theta = angle_between(pi, pj, d);
        if (theta > sp[e].rest_angle * shear) {
            uint32_t a = (i < j) ? i : j;
            uint32_t b = (i < j) ? j : i;
            uint32_t idx = pair_index(a, b, n_total);
            if (idx < h->pair_count) bits[idx] = 0u;
            if (deg[a] > 0u) deg[a] -= 1u;
            if (deg[b] > 0u) deg[b] -= 1u;
            sp[e] = sp[sc - 1u];
            sc--;
            continue;
        }
        e++;
    }

    // Form new bonds from the all-pairs list.
    if (sc >= cap) {
        h->spring_count_sim = sc;
        return;
    }

    const GP_Pair* pairs = ptr_pairs(h);
    const uint32_t pc = h->pair_count;
    const uint32_t max_per = h->bond_max_per_node;
    const uint32_t ionic = h->ionic_bonds ? 1u : 0u;
    const uint32_t valence = (h->ionic_valence == 0u) ? 1u : h->ionic_valence;
    const double link = h->bond_link_angle;
    const double k_new = h->bond_k;

    for (uint32_t idx = 0u; idx < pc && sc < cap; idx++) {
        if (bits[idx]) continue;
        uint32_t i = pairs[idx].i;
        uint32_t j = pairs[idx].j;
        if (i >= n_active || j >= n_active || i == j) continue;

        if (ionic) {
            if (charge[i] * charge[j] >= 0.0) continue;
            if (deg[i] >= valence || deg[j] >= valence) continue;
        } else if (max_per) {
            if (deg[i] >= max_per || deg[j] >= max_per) continue;
        }

        const double* pi = p + (size_t)i * d;
        const double* pj = p + (size_t)j * d;
        double theta = angle_between(pi, pj, d);
        if (theta < link) {
            sp[sc].i = i;
            sp[sc].j = j;
            sp[sc].rest_angle = theta;
            sp[sc].k = k_new;
            sc++;
            bits[idx] = 1u;
            deg[i] += 1u;
            deg[j] += 1u;
        }
    }

    h->spring_count_sim = sc;
}

uintptr_t gp_get_springs_ptr(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return (uintptr_t)0;
    return (uintptr_t)ptr_springs_front(h);
}

uint32_t gp_get_spring_count(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->spring_count;
}

uint32_t gp_get_spring_cap(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->spring_cap;
}

void gp_step(void* state_mem, double dt, uint32_t steps) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return;
    if (!(dt > 0.0) || steps == 0) return;

    const uint32_t n = h->n;
    const uint32_t d = h->d;

    double* pos = ptr_pos(h);
    double* vel = ptr_vel(h);

    size_t nd = (size_t)n * (size_t)d;
    double* scratch = ptr_scratch(h);
    double* F0 = scratch + 0u * nd;
    double* F1 = scratch + 1u * nd;
    double* v_mid = scratch + 2u * nd;
    double* p_mid = scratch + 3u * nd;
    double* v_new = scratch + 4u * nd;

    for (uint32_t s = 0; s < steps; s++) {
        const uint32_t n_active = h->n_active_sim;
        const size_t nd_active = (size_t)n_active * (size_t)d;
        // Advance simulation only in sim_idx; do not mutate front_idx here.
        // This lets the simulation run "unread" without forcing a front swap.
        uint32_t sim = h->sim_idx & 1u;
        double* p_sim = pos + (size_t)sim * nd;
        double* v_sim = vel + (size_t)sim * nd;
        const double* p = p_sim;
        const double* v = v_sim;

        // We'll compute p_new into F0 (reuse after force is consumed) and v_new into v_new.
        double* p_new = F0;

        compute_force(h, p, v, dt, F0);

        // v_mid = project_tangent(v + 0.5*dt*F0, p)
        for (uint32_t i = 0; i < n_active; i++) {
            const double* pi = p + (size_t)i * d;
            const double* vi = v + (size_t)i * d;
            const double* ai = F0 + (size_t)i * d;
            // write into v_mid first (as the temp) then re-project in place
            double* vmi = v_mid + (size_t)i * d;
            for (uint32_t k = 0; k < d; k++) vmi[k] = vi[k] + 0.5 * dt * ai[k];
            project_tangent(pi, vmi, vmi, d);
        }

        // p_mid = expmap(p, 0.5*dt*v_mid)
        for (uint32_t i = 0; i < n_active; i++) {
            const double* pi = p + (size_t)i * d;
            const double* vmi = v_mid + (size_t)i * d;
            // use v_new as a per-node temp step vector (dt-scaled)
            double* stepv = v_new + (size_t)i * d;
            for (uint32_t k = 0; k < d; k++) stepv[k] = 0.5 * dt * vmi[k];
            expmap(pi, stepv, p_mid + (size_t)i * d, d);
        }

        compute_force(h, p_mid, v_mid, dt, F1);

        // v_new = project_tangent(v + dt*F1, p_mid) (then reproject at p_new later)
        for (uint32_t i = 0; i < n_active; i++) {
            const double* pmi = p_mid + (size_t)i * d;
            const double* vi = v + (size_t)i * d;
            const double* a1 = F1 + (size_t)i * d;
            double* vni = v_new + (size_t)i * d;
            for (uint32_t k = 0; k < d; k++) vni[k] = vi[k] + dt * a1[k];
            project_tangent(pmi, vni, vni, d);
        }

        // p_new = expmap(p, dt*v_mid)
        for (uint32_t i = 0; i < n_active; i++) {
            const double* pi = p + (size_t)i * d;
            const double* vmi = v_mid + (size_t)i * d;
            double* stepv = p_mid + (size_t)i * d; // reuse p_mid row as temp step
            for (uint32_t k = 0; k < d; k++) stepv[k] = dt * vmi[k];
            expmap(pi, stepv, p_new + (size_t)i * d, d);
        }

        // final tangent projection at new position
        for (uint32_t i = 0; i < n_active; i++) {
            const double* pni = p_new + (size_t)i * d;
            const double* vni = v_new + (size_t)i * d;
            project_tangent(pni, vni, v_sim + (size_t)i * d, d);
        }

        // Commit new positions into the sim buffer.
        memcpy(p_sim, p_new, nd_active * sizeof(double));

        // Temperature evolution (uses updated velocity state).
        update_temps(h, v_sim, dt);

        // Push history snapshot for ghosting/collision processing.
        history_push(h, p_sim, v_sim, dt);

        // Lifecycle consequences (splitting/merging/vapor/condense) run in the hot loop.
        // These operate on the sim buffer and update n_active_sim.
        lifecycle_accel_shatter(h, p_sim, v_sim, dt, F1);
        lifecycle_collisions(h, p_sim, v_sim);
        lifecycle_vaporize_small(h, p_sim, v_sim);
        lifecycle_condense_vapor(h, p_sim, v_sim);

        // If the reader asked for a swap, perform it only at a step boundary.
        if (h->swap_requested) {
            // Bond forming/breaking is tied to the reader-driven publish boundary.
            // This keeps bonds consistent with the published front buffer, and
            // avoids spending cycles when the simulation runs unread.
            update_bonds(h, p_sim, v_sim, dt);

            uint32_t new_front = sim;
            uint32_t new_sim = new_front ^ 1u;

            // Move simulation to the other buffer so the front buffer can be read
            // without being written to. This copy happens only on reads.
            memcpy(pos + (size_t)new_sim * nd, pos + (size_t)new_front * nd, nd * sizeof(double));
            memcpy(vel + (size_t)new_sim * nd, vel + (size_t)new_front * nd, nd * sizeof(double));

            // Springs are also double-buffered: publish the sim list and
            // copy it into the next sim buffer so the front buffer remains read-only.
            h->spring_count = h->spring_count_sim;
            if (h->spring_cap) {
                memcpy(
                    ptr_springs_buf(h, new_sim),
                    ptr_springs_buf(h, new_front),
                    (size_t)h->spring_cap * sizeof(GP_Spring)
                );
            }
            h->spring_count_sim = h->spring_count;

            h->front_idx = new_front;
            h->sim_idx = new_sim;
            // Publish the latest active count along with the buffer swap.
            h->n_active = h->n_active_sim;
            h->swap_requested = 0u;
            h->swap_seq += 1u;
        }
    }

}

uint32_t gp_get_history_cap(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->history_cap;
}

uint32_t gp_get_history_len(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->history_len;
}

uint32_t gp_get_history_head(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->history_head;
}

uint32_t gp_get_history_seq(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->history_seq;
}

uint32_t gp_history_age_to_index(void* state_mem, uint32_t age) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0xFFFFFFFFu;
    return history_age_to_index_raw(h, age);
}

uintptr_t gp_get_history_pos_ptr(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return (uintptr_t)0;
    if (!h->history_cap) return (uintptr_t)0;
    return (uintptr_t)ptr_hist_pos(h);
}

uintptr_t gp_get_history_vel_ptr(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return (uintptr_t)0;
    if (!h->history_cap) return (uintptr_t)0;
    return (uintptr_t)ptr_hist_vel(h);
}

uintptr_t gp_get_history_dt_ptr(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return (uintptr_t)0;
    if (!h->history_cap) return (uintptr_t)0;
    return (uintptr_t)ptr_hist_dt(h);
}

static void lerp_vec(const double* a, const double* b, double t, double* out, uint32_t d) {
    const double u = 1.0 - t;
    for (uint32_t k = 0; k < d; k++) out[k] = u * a[k] + t * b[k];
}

static double clamp01d(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static double clamp11(double x) {
    if (x < -1.0) return -1.0;
    if (x > 1.0) return 1.0;
    return x;
}

static double norm2_vec(const double* v, uint32_t d) {
    double s = 0.0;
    for (uint32_t k = 0; k < d; k++) s += v[k] * v[k];
    return s;
}

static double dot_vec(const double* a, const double* b, uint32_t d) {
    double s = 0.0;
    for (uint32_t k = 0; k < d; k++) s += a[k] * b[k];
    return s;
}

static void nlerp_on_sphere(const double* a, const double* b, double t, double* out, uint32_t d) {
    // Normalized linear interpolation; avoids leaving the sphere during collision sampling.
    const double u = 1.0 - t;
    for (uint32_t k = 0; k < d; k++) out[k] = u * a[k] + t * b[k];
    double n2 = norm2_vec(out, d);
    if (n2 > 1e-24) {
        const double invn = 1.0 / sqrt(n2);
        for (uint32_t k = 0; k < d; k++) out[k] *= invn;
    }
}

static double segment_segment_dist2(const double* p0, const double* p1, const double* q0, const double* q1, uint32_t d) {
    // Closest distance between two segments in R^d.
    // Based on standard 2x2 system with clamping, using dot products of
    // direction vectors u, v and offset w0.
    double a = 0.0, b = 0.0, c = 0.0, d0 = 0.0, e0 = 0.0;
    for (uint32_t k = 0; k < d; k++) {
        const double uk = p1[k] - p0[k];
        const double vk = q1[k] - q0[k];
        const double w0k = p0[k] - q0[k];
        a += uk * uk;
        b += uk * vk;
        c += vk * vk;
        d0 += uk * w0k;
        e0 += vk * w0k;
    }
    const double denom = a * c - b * b;

    double sN, sD = denom;
    double tN, tD = denom;

    const double eps = 1e-12;
    if (denom < eps) {
        // parallel
        sN = 0.0;
        sD = 1.0;
        tN = e0;
        tD = c;
    } else {
        sN = (b * e0 - c * d0);
        tN = (a * e0 - b * d0);
        if (sN < 0.0) {
            sN = 0.0;
            tN = e0;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e0 + b;
            tD = c;
        }
    }

    if (tN < 0.0) {
        tN = 0.0;
        if (-d0 < 0.0) {
            sN = 0.0;
        } else if (-d0 > a) {
            sN = sD;
        } else {
            sN = -d0;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if ((-d0 + b) < 0.0) {
            sN = 0.0;
        } else if ((-d0 + b) > a) {
            sN = sD;
        } else {
            sN = (-d0 + b);
            sD = a;
        }
    }

    const double sc = (fabs(sN) < eps) ? 0.0 : (sN / sD);
    const double tc = (fabs(tN) < eps) ? 0.0 : (tN / tD);

    double dist2 = 0.0;
    for (uint32_t k = 0; k < d; k++) {
        const double uk = p1[k] - p0[k];
        const double vk = q1[k] - q0[k];
        const double w0k = p0[k] - q0[k];
        const double dk = w0k + sc * uk - tc * vk;
        dist2 += dk * dk;
    }
    return dist2;
}

uint32_t gp_detect_point_overlaps(
    void* state_mem,
    uint32_t age_old,
    uint32_t age_new,
    double threshold,
    GP_BallisticContact* out,
    uint32_t out_cap
) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    if (!h->history_cap || h->history_len < 2u) return 0u;
    if (!(threshold > 0.0) || !out || out_cap == 0u) return 0u;

    const uint32_t idx_old = history_age_to_index_raw(h, age_old);
    const uint32_t idx_new = history_age_to_index_raw(h, age_new);
    if (idx_old == 0xFFFFFFFFu || idx_new == 0xFFFFFFFFu) return 0u;

    const uint32_t n_active = h->n_active;
    const uint32_t d = h->d;
    const size_t nd = (size_t)h->n * (size_t)h->d;
    const double* hp = ptr_hist_pos(h);
    const double* hdt = ptr_hist_dt(h);
    const double* p_old = hp + (size_t)idx_old * nd;
    const double* p_new = hp + (size_t)idx_new * nd;

    double dt = hdt[idx_new];
    if (!(dt > 0.0)) dt = 1.0;

    const double thr2 = threshold * threshold;
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < n_active; i++) {
        const double* pi0 = p_old + (size_t)i * d;
        const double* pi1 = p_new + (size_t)i * d;
        for (uint32_t j = i + 1u; j < n_active; j++) {
            const double* pj0 = p_old + (size_t)j * d;
            const double* pj1 = p_new + (size_t)j * d;

            double r0_2 = 0.0;
            double dr2 = 0.0;
            double r0_dot_dr = 0.0;
            for (uint32_t k = 0; k < d; k++) {
                const double r0 = pi0[k] - pj0[k];
                const double r1 = pi1[k] - pj1[k];
                const double dr = r1 - r0;
                r0_2 += r0 * r0;
                dr2 += dr * dr;
                r0_dot_dr += r0 * dr;
            }

            double t = 0.0;
            if (dr2 > 1e-18) {
                t = clamp01d(-r0_dot_dr / dr2);
            }
            const double dist2 = r0_2 + 2.0 * t * r0_dot_dr + t * t * dr2;
            if (dist2 <= thr2) {
                if (count < out_cap) {
                    out[count].i = i;
                    out[count].j = j;
                    out[count].t = t;
                    out[count].dist = sqrt(dist2);
                    out[count].rel_speed = sqrt(dr2) / dt;
                }
                count++;
                if (count >= out_cap) return out_cap;
            }
        }
    }
    return count;
}

uint32_t gp_detect_point_overlaps_geodesic(
    void* state_mem,
    uint32_t age_old,
    uint32_t age_new,
    double angle_threshold,
    uint32_t samples,
    GP_BallisticContact* out,
    uint32_t out_cap
) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    if (!h->history_cap || h->history_len < 2u) return 0u;
    if (!(angle_threshold > 0.0) || !out || out_cap == 0u) return 0u;

    const uint32_t idx_old = history_age_to_index_raw(h, age_old);
    const uint32_t idx_new = history_age_to_index_raw(h, age_new);
    if (idx_old == 0xFFFFFFFFu || idx_new == 0xFFFFFFFFu) return 0u;

    if (samples < 2u) samples = 2u;
    if (samples > 64u) samples = 64u;

    const uint32_t n_active = h->n_active;
    const uint32_t d = h->d;
    const size_t nd = (size_t)h->n * (size_t)h->d;
    const double* hp = ptr_hist_pos(h);
    const double* hv = ptr_hist_vel(h);
    const double* hdt = ptr_hist_dt(h);
    const double* p_old = hp + (size_t)idx_old * nd;
    const double* p_new = hp + (size_t)idx_new * nd;
    const double* v_old = hv + (size_t)idx_old * nd;
    const double* v_new = hv + (size_t)idx_new * nd;

    double dt = hdt[idx_new];
    if (!(dt > 0.0)) dt = 1.0;

    const double cos_thr = cos(angle_threshold);

    double* pi = (double*)malloc((size_t)d * sizeof(double));
    double* pj = (double*)malloc((size_t)d * sizeof(double));
    if (!pi || !pj) {
        free(pi);
        free(pj);
        return 0u;
    }

    uint32_t count = 0u;
    for (uint32_t i = 0u; i < n_active; i++) {
        const double* pi0 = p_old + (size_t)i * d;
        const double* pi1 = p_new + (size_t)i * d;
        const double* vi0 = v_old + (size_t)i * d;
        const double* vi1 = v_new + (size_t)i * d;
        for (uint32_t j = i + 1u; j < n_active; j++) {
            const double* pj0 = p_old + (size_t)j * d;
            const double* pj1 = p_new + (size_t)j * d;
            const double* vj0 = v_old + (size_t)j * d;
            const double* vj1 = v_new + (size_t)j * d;

            double best_dot = -2.0;
            double best_t = 0.0;
            for (uint32_t si = 0u; si < samples; si++) {
                const double t = (double)si / (double)(samples - 1u);
                nlerp_on_sphere(pi0, pi1, t, pi, d);
                nlerp_on_sphere(pj0, pj1, t, pj, d);
                const double dot = clamp11(dot_vec(pi, pj, d));
                if (dot > best_dot) {
                    best_dot = dot;
                    best_t = t;
                }
                if (best_dot >= cos_thr) break;
            }

            if (best_dot >= cos_thr) {
                if (count < out_cap) {
                    // Estimate relative speed at the sampled time from interpolated velocities.
                    double vri2 = 0.0;
                    for (uint32_t k = 0; k < d; k++) {
                        const double vi = (1.0 - best_t) * vi0[k] + best_t * vi1[k];
                        const double vj = (1.0 - best_t) * vj0[k] + best_t * vj1[k];
                        const double dv = vi - vj;
                        vri2 += dv * dv;
                    }
                    out[count].i = i;
                    out[count].j = j;
                    out[count].t = best_t;
                    // dist stores angular separation (radians) for the geodesic variant.
                    out[count].dist = acos(clamp11(best_dot));
                    out[count].rel_speed = sqrt(vri2);
                }
                count++;
                if (count >= out_cap) {
                    free(pi);
                    free(pj);
                    return out_cap;
                }
            }
        }
    }

    free(pi);
    free(pj);
    return count;
}

uint32_t gp_detect_spring_proximity(
    void* state_mem,
    uint32_t age_old,
    uint32_t age_new,
    double threshold,
    uint32_t samples,
    GP_SegmentContact* out,
    uint32_t out_cap
) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    if (!h->history_cap || h->history_len < 2u) return 0u;
    if (!(threshold > 0.0) || !out || out_cap == 0u) return 0u;

    const uint32_t idx_old = history_age_to_index_raw(h, age_old);
    const uint32_t idx_new = history_age_to_index_raw(h, age_new);
    if (idx_old == 0xFFFFFFFFu || idx_new == 0xFFFFFFFFu) return 0u;

    const uint32_t d = h->d;
    const size_t nd = (size_t)h->n * (size_t)h->d;
    const double* hp = ptr_hist_pos(h);
    const double* p_old = hp + (size_t)idx_old * nd;
    const double* p_new = hp + (size_t)idx_new * nd;

    const GP_Spring* sp = ptr_springs_front(h);
    const uint32_t sc = h->spring_count;
    if (sc < 2u) return 0u;
    if (samples < 2u) samples = 2u;

    const double thr2 = threshold * threshold;
    uint32_t count = 0u;
    double* a0 = (double*)malloc((size_t)d * sizeof(double));
    double* a1 = (double*)malloc((size_t)d * sizeof(double));
    double* b0 = (double*)malloc((size_t)d * sizeof(double));
    double* b1 = (double*)malloc((size_t)d * sizeof(double));
    if (!a0 || !a1 || !b0 || !b1) {
        free(a0);
        free(a1);
        free(b0);
        free(b1);
        return 0u;
    }

    for (uint32_t sa = 0u; sa < sc; sa++) {
        const uint32_t ai = sp[sa].i;
        const uint32_t aj = sp[sa].j;
        for (uint32_t sb = sa + 1u; sb < sc; sb++) {
            const uint32_t bi = sp[sb].i;
            const uint32_t bj = sp[sb].j;

            double best_d2 = 1e300;
            double best_t = 0.0;
            for (uint32_t si = 0u; si < samples; si++) {
                const double t = (double)si / (double)(samples - 1u);

                lerp_vec(p_old + (size_t)ai * d, p_new + (size_t)ai * d, t, a0, d);
                lerp_vec(p_old + (size_t)aj * d, p_new + (size_t)aj * d, t, a1, d);
                lerp_vec(p_old + (size_t)bi * d, p_new + (size_t)bi * d, t, b0, d);
                lerp_vec(p_old + (size_t)bj * d, p_new + (size_t)bj * d, t, b1, d);

                const double d2 = segment_segment_dist2(a0, a1, b0, b1, d);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_t = t;
                }
                if (best_d2 <= thr2) break;
            }

            if (best_d2 <= thr2) {
                if (count < out_cap) {
                    out[count].sa = sa;
                    out[count].sb = sb;
                    out[count].t = best_t;
                    out[count].dist = sqrt(best_d2);
                }
                count++;
                if (count >= out_cap) {
                    free(a0);
                    free(a1);
                    free(b0);
                    free(b1);
                    return out_cap;
                }
            }
        }
    }
    free(a0);
    free(a1);
    free(b0);
    free(b1);
    return count;
}

void gp_request_swap(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return;
    h->swap_requested = 1u;
}

uint32_t gp_get_swap_seq(void* state_mem) {
    GP_Header* h = (GP_Header*)state_mem;
    if (!h || h->magic != 0x48505047u) return 0u;
    return h->swap_seq;
}
