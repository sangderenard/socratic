#pragma once
#include <cstdint>

extern "C" {

typedef struct RopeSim RopeSim;

// Create a simulator capable of holding up to max_ropes each with up to max_segments segments.
RopeSim* rope_sim_create(int max_ropes, int max_segments_per_rope);
void rope_sim_destroy(RopeSim* s);

// Add a rope with endpoints (ax,ay)-(bx,by). Returns rope index or -1 on error.
int rope_sim_add_rope(RopeSim* s, float ax, float ay, float bx, float by, int segments, float slack_fraction);

// Update endpoints for an existing rope (rope_idx returned from rope_sim_add_rope)
int rope_sim_set_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by);

// Move endpoints but preserve previous positions so the integrator sees the endpoint velocity.
int rope_sim_move_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by);

// Step the simulation. constraint_iters controls how many constraint passes are applied.
int rope_sim_step(RopeSim* s, float dt, float gravity, int constraint_iters, float damping);

// Query functions: get vertex count for rope, and copy XY vertices into out_xy (length must be >= 2*count)
int rope_sim_get_vertex_count(RopeSim* s, int rope_idx);
int rope_sim_get_vertices(RopeSim* s, int rope_idx, float* out_xy, int max_count);

} // extern C
