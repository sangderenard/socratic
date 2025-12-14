#pragma once

#include <stdint.h>
#include <stddef.h>

#include <inttypes.h>

#ifdef _WIN32
  #define GP_EXPORT __declspec(dllexport)
#else
  #define GP_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
typedef struct GP_Header {
    uint32_t magic;        // 'GPPH' = 0x48505047
  uint32_t version;      // 9
    uint32_t n;            // node capacity (allocated nodes)
    // Active node counts:
    // - n_active is the published/front count (safe for readers using front_idx).
    // - n_active_sim is the writer/simulation count (may differ until a swap).
    uint32_t n_active;     // published active node count (<= n)
    uint32_t n_active_sim; // simulation active node count (<= n)
    uint32_t d;            // dimension
    uint32_t spring_cap;      // capacity (number of springs)
    // Spring counts:
    // - spring_count is the published/front count (safe for readers using front_idx).
    // - spring_count_sim is the writer/simulation count (may differ until a swap).
    uint32_t spring_count;     // published spring count
    uint32_t spring_count_sim; // simulation spring count
    // History ring buffer (optional, preallocated)
    // - Stores full-capacity node state snapshots for the simulation thread to
    //   support ghosting and downstream collision processing.
    // - Frames are indexed by [history_head] as newest.
    uint32_t history_cap;   // allocated history frames (0 disables history)
    uint32_t history_len;   // valid frames (<= history_cap)
    uint32_t history_head;  // newest frame index in [0, history_cap)
    uint32_t history_seq;   // increments each time a frame is pushed
  // Front/back protocol:
  // - Reader reads only from front_idx.
  // - Writer advances the simulation in sim_idx (kept != front_idx).
  // - Reader requests a swap; writer performs it only at the end of a step,
  //   publishing the latest state as front and moving the simulation to the
  //   other buffer by copying state once.
  volatile uint32_t front_idx;       // 0 or 1 (read buffer)
  volatile uint32_t sim_idx;         // 0 or 1 (writer buffer)
  volatile uint32_t swap_requested;  // set by reader, cleared by writer
  volatile uint32_t swap_seq;        // increments when writer completes a swap
  // All-pairs edge list for bond logic (covers total connectivity)
  uint32_t pair_count;              // n*(n-1)/2 for capacity n (0 if not allocated)

  // Bond forming/breaking params
  uint32_t bonds_enabled;           // 0/1
  uint32_t bond_max_per_node;       // 0 => unlimited (subject to spring_cap)
  uint32_t ionic_bonds;             // 0/1 (if enabled, require opposite charges)
  uint32_t ionic_valence;           // per-node max bonds when ionic_bonds=1
  double bond_link_angle;           // form if current angle < bond_link_angle
  double bond_shear_ratio;          // break if current angle > rest_angle * bond_shear_ratio
  double bond_k;                    // spring k for newly formed bonds

    // params
    double k_spring;
    double k_coulomb;
    double G;
    double damping;
    double softening;

    // South-pole field (Torch: F += south_strength * (dir - dot(dir,p)*p))
    uint32_t south_enabled; // 0/1
    int32_t south_axis;     // axis index in [0,d), negative => (d-1)
    uint32_t _pad_south;
    double south_strength;

    // Lorentz-like term (3D only for now): a += lorentz_k * (v x B)
    double lorentz_k;
    double Bx;
    double By;
    double Bz;

    // Temperature model (per-node temps live in the state block)
    // - Thermal noise: a += sqrt(dt) * temp_noise * |T_i| * N(0,1) (tangent-projected)
    // - Convection: dT/dt = -temp_conv * |v| * (T - temp_ambient)
    // - Damping heats: dT += temp_heat_gain * (lambda_d * |v|^2 * dt) / max(m, 1e-6)
    double temp_ambient;
    double temp_conv;
    double temp_noise;
    double temp_heat_gain;

    // Lifecycle / collision / phase-change parameters.
    // These are used to keep all "physics consequences" inside gp_step.
    double max_speed;            // scales merge/shatter thresholds
    double radius_scale;         // radius = radius_scale*cbrt(m)/mass_ref_cuberoot
    double mass_ref_cuberoot;    // normalization for radii

    double collide_gain;         // >0 enables collision handling
    double merge_speed_frac;     // v_rel <= frac*max_speed triggers merge
    double merge_size_frac;      // size_ratio >= frac triggers merge
    double shatter_speed_frac;   // v_rel >= frac*max_speed triggers shatter
    double shatter_size_frac;    // size_ratio <= frac allows shatter
    uint32_t shatter_k_max;      // max shards on impact
    uint32_t collision_pair_limit; // if n_active_sim <= limit, test all pairs; else test spring pairs

    double mass_vapor_thresh;    // masses below this vaporize into the pool
    double condense_temp_thresh; // ambient <= thresh enables condensation
    double condense_chunk_mass;  // base condensation chunk per step
    double vapor_mass;           // accumulated vapor mass
    double vapor_charge;         // accumulated vapor charge

    double accel_shatter_thresh;    // acceleration threshold for atomization
    double accel_shatter_fraction;  // fraction of mass/charge to split into shards
    uint32_t accel_shatter_k;       // shards count for accel shatter
    double accel_shatter_kick;      // velocity kick factor for accel shatter

    // RNG state for thermal noise
    uint32_t rng_state;
    uint32_t _pad1;
} GP_Header;

typedef struct GP_Pair {
  uint32_t i;
  uint32_t j;
} GP_Pair;

typedef struct GP_Spring {
    uint32_t i;
    uint32_t j;
    double rest_angle;
    double k;
} GP_Spring;

typedef struct GP_Segment {
  uint32_t i;
  uint32_t j;
} GP_Segment;

typedef struct GP_BallisticContact {
  uint32_t i;
  uint32_t j;
  double t;         // [0,1] interpolation fraction between (age_old -> age_new)
  double dist;      // closest Euclidean distance
  double rel_speed; // |d/dt (p_i - p_j)| at frame scale
} GP_BallisticContact;

typedef struct GP_SegmentContact {
  uint32_t sa;  // spring index A
  uint32_t sb;  // spring index B
  double t;     // sampled interpolation fraction where min was observed
  double dist;  // closest Euclidean distance between segments
} GP_SegmentContact;

// ---------------- Weapon / Projectile simulator ABI (prototype) ----------------
// This is a separate packed-buffer protocol that is intended to be filled from
// Python via ctypes, processed in-place by a DLL, and read back by Python.

#define GP_WPN_MAGIC 0x514E5057u  // 'WPNQ'

typedef struct GP_WpnHeader {
  uint32_t magic;      // GP_WPN_MAGIC
  uint32_t version;    // 6
  uint32_t count;      // number of requests
  uint32_t max_points; // spline points capacity per request
} GP_WpnHeader;

typedef struct GP_WpnRequest {
  uint32_t request_id;
  uint32_t weapon_slot; // 1 or 2
  float analog;         // [0..1] or trigger magnitude
  char source[32];
  char weapon_type[32];

  // Weapon kinematics + sim parameters (MUST be provided from weapon stats/config).
  // kinematics_mode: 0=beam, 1=fire (projectile), 2=fall (released)
  uint32_t kinematics_mode;
  uint32_t inherit_ship_velocity; // 0/1
  float add_velocity;             // vel_u along ship forward axis
  uint32_t sim_points;            // requested spline points (<= max_points)
  float sim_t_end;                // total sim time (seconds in prototype units)
  float sim_beam_len;             // max ray length for beam
  float sim_drop_off;             // initial position offset along radial-down

  // Ship snapshot (optional). If not provided, leave zeros.
  float ship_pos[3];
  float ship_vel[3];
  float ship_fwd[3];

  // Optional weapon frame override (keeps ship_fwd intact):
  // - bit0: weapon_origin/weapon_dir are valid and should be used.
  uint32_t weapon_flags;
  float weapon_origin[3];
  float weapon_dir[3];

  // World snapshot (optional).
  // Flags:
  // - bit0: terrain heightmap pointer is valid
  // - bit1: node positions pointer is valid
  uint32_t world_flags;
  float planet_surface_r; // base radius before heightmap displacement
  float gravity_g;        // ballistic gravity magnitude (toward center)

  // Terrain heightmap (equirectangular lon/lat, grayscale in [0..1]).
  // The surface radius at direction dir is:
  //   r_surface = planet_surface_r + (h(dir) - terrain_height_bias) * terrain_height_scale
  uintptr_t terrain_hm_ptr;   // points to float32[hm_h][hm_w] (in-process pointer)
  uint32_t terrain_hm_w;
  uint32_t terrain_hm_h;
  uint32_t terrain_hm_stride; // elements per row (usually == w)
  float terrain_height_scale;
  float terrain_height_bias;

  // Optional node positions for ray hits (future use).
  uintptr_t nodes_pos_ptr;       // points to float32[nodes_count][3]
  uint32_t nodes_count;
  uint32_t nodes_pos_stride;     // floats per node (usually 3)
  uintptr_t nodes_radius_ptr;    // points to float32[nodes_count]
  uint32_t nodes_radius_stride;  // floats per node (usually 1)
} GP_WpnRequest;

typedef struct GP_WpnResult {
  uint32_t ok;            // 0/1
  uint32_t impact_valid;  // 0/1
  uint32_t spline_n;      // number of points written
  uint32_t victim_id;     // 0 if none/unknown
  float impact_point[3];
  // Fixed capacity: [max_points] points; actual used is spline_n.
  float spline_points[16][3];
} GP_WpnResult;

typedef struct GP_WpnBatch {
  GP_WpnHeader hdr;
  GP_WpnRequest req[8];
  GP_WpnResult out[8];
} GP_WpnBatch;
#pragma pack(pop)

GP_EXPORT size_t gp_required_bytes(uint32_t n, uint32_t d, uint32_t spring_cap);
GP_EXPORT size_t gp_required_bytes_ex(uint32_t n, uint32_t d, uint32_t spring_cap, uint32_t history_cap);
GP_EXPORT int gp_init(void* state_mem, uint32_t n, uint32_t d, uint32_t spring_cap);
GP_EXPORT int gp_init_ex(void* state_mem, uint32_t n, uint32_t d, uint32_t spring_cap, uint32_t history_cap);
GP_EXPORT int gp_set_n_active(void* state_mem, uint32_t n_active);
GP_EXPORT int gp_set_params(void* state_mem, double k_spring, double k_coulomb, double G, double damping, double softening);
GP_EXPORT int gp_set_south(void* state_mem, uint32_t south_enabled, double south_strength, int32_t south_axis);
GP_EXPORT int gp_set_springs(void* state_mem, const GP_Spring* springs, uint32_t spring_count);
GP_EXPORT void gp_step(void* state_mem, double dt, uint32_t steps);

// Process a batch of weapon fire requests in-place.
// For now the DLL prints requests and writes placeholder impact/spline outputs.
GP_EXPORT int gp_weapon_process_batch(GP_WpnBatch* batch);

GP_EXPORT int gp_set_lorentz(void* state_mem, double lorentz_k, double Bx, double By, double Bz);
GP_EXPORT int gp_set_temp_params(void* state_mem, double temp_ambient, double temp_conv, double temp_noise, double temp_heat_gain);
GP_EXPORT int gp_seed(void* state_mem, uint32_t seed);

GP_EXPORT int gp_set_lifecycle_params(
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
);

GP_EXPORT int gp_set_bond_params(
  void* state_mem,
  uint32_t bonds_enabled,
  uint32_t bond_max_per_node,
  double bond_link_angle,
  double bond_shear_ratio,
  double bond_k,
  uint32_t ionic_bonds,
  uint32_t ionic_valence
);

// Accessors for rendering/debugging: read-only view into the current spring list.
// These are intended to be used after a reader-driven swap to avoid races.
GP_EXPORT uintptr_t gp_get_springs_ptr(void* state_mem);
GP_EXPORT uint32_t gp_get_spring_count(void* state_mem);
GP_EXPORT uint32_t gp_get_spring_cap(void* state_mem);

// History accessors. Age is relative: 0=newest frame, 1=previous, ...
GP_EXPORT uint32_t gp_get_history_cap(void* state_mem);
GP_EXPORT uint32_t gp_get_history_len(void* state_mem);
GP_EXPORT uint32_t gp_get_history_head(void* state_mem);
GP_EXPORT uint32_t gp_get_history_seq(void* state_mem);
GP_EXPORT uint32_t gp_history_age_to_index(void* state_mem, uint32_t age);
GP_EXPORT uintptr_t gp_get_history_pos_ptr(void* state_mem);
GP_EXPORT uintptr_t gp_get_history_vel_ptr(void* state_mem);
GP_EXPORT uintptr_t gp_get_history_dt_ptr(void* state_mem);

// Collision helpers (first pass; intended to be extended).
// - Point overlaps: O(n_active^2), uses linear motion between two history frames.
GP_EXPORT uint32_t gp_detect_point_overlaps(
  void* state_mem,
  uint32_t age_old,
  uint32_t age_new,
  double threshold,
  GP_BallisticContact* out,
  uint32_t out_cap
);

// Geodesic variant (recommended):
// - threshold is an angular radius in radians.
// - Uses normalized interpolation on the sphere and a dot-product compare.
GP_EXPORT uint32_t gp_detect_point_overlaps_geodesic(
  void* state_mem,
  uint32_t age_old,
  uint32_t age_new,
  double angle_threshold,
  uint32_t samples,
  GP_BallisticContact* out,
  uint32_t out_cap
);

// - Spring proximity: O(spring_count^2 * samples), samples linear interpolation in time.
GP_EXPORT uint32_t gp_detect_spring_proximity(
  void* state_mem,
  uint32_t age_old,
  uint32_t age_new,
  double threshold,
  uint32_t samples,
  GP_SegmentContact* out,
  uint32_t out_cap
);

// Request the writer to swap front/back at a step boundary.
GP_EXPORT void gp_request_swap(void* state_mem);

// Return the swap sequence counter (increments when a swap is completed).
GP_EXPORT uint32_t gp_get_swap_seq(void* state_mem);

#ifdef __cplusplus
}
#endif
