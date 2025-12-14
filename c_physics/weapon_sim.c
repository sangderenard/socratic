#include "geodesic_physics.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef GP_PI
#define GP_PI 3.14159265358979323846f
#endif

static float _fclamp(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static float _len3(const float v[3]) {
  return (float)sqrt((double)v[0] * (double)v[0] + (double)v[1] * (double)v[1] + (double)v[2] * (double)v[2]);
}

static float _dot3(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void _add3(float out[3], const float a[3], const float b[3]) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

static void _sub3(float out[3], const float a[3], const float b[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static void _mad3(float out[3], const float a[3], float s, const float b[3]) {
  // out = a + s*b
  out[0] = a[0] + s * b[0];
  out[1] = a[1] + s * b[1];
  out[2] = a[2] + s * b[2];
}

static void _norm3(float v[3]) {
  float n = _len3(v);
  if (n > 1e-8f) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
}

static void _safe_cstr(char* dst, size_t cap, const char* src) {
  if (!dst || cap == 0) {
    return;
  }
  if (!src) {
    dst[0] = '\0';
    return;
  }
  // Ensure NUL termination.
  strncpy(dst, src, cap - 1);
  dst[cap - 1] = '\0';
}

static int _hm_valid(const GP_WpnRequest* r) {
  if (!r) return 0;
  if ((r->world_flags & 1u) == 0u) return 0;
  if (r->terrain_hm_ptr == 0u) return 0;
  if (r->terrain_hm_w < 2u || r->terrain_hm_h < 2u) return 0;
  if (r->terrain_hm_stride < r->terrain_hm_w) return 0;
  return 1;
}

static float _terrain_surface_r_at_dir(const GP_WpnRequest* r, const float dir_unit[3]);

static int _nodes_valid(const GP_WpnRequest* r) {
  if (!r) return 0;
  if ((r->world_flags & 2u) == 0u) return 0;
  if (r->nodes_pos_ptr == 0u) return 0;
  if (r->nodes_radius_ptr == 0u) return 0;
  if (r->nodes_count == 0u) return 0;
  if (r->nodes_pos_stride < 3u) return 0;
  if (r->nodes_radius_stride < 1u) return 0;
  return 1;
}

static int _ray_sphere_hit(const float ray_o[3], const float ray_d[3], const float c[3], float rad, float t_max, float* out_t) {
  if (!out_t) return 0;
  if (!(rad > 0.0f)) return 0;
  // Solve |(o + t d) - c|^2 = r^2
  float oc[3] = { ray_o[0] - c[0], ray_o[1] - c[1], ray_o[2] - c[2] };
  float a = _dot3(ray_d, ray_d);
  float b = _dot3(oc, ray_d);
  float cc = _dot3(oc, oc) - rad * rad;
  float disc = b * b - a * cc;
  if (disc < 0.0f || !(a > 1e-12f)) return 0;
  float s = sqrtf(disc);
  float inva = 1.0f / a;
  float t0 = (-b - s) * inva;
  float t1 = (-b + s) * inva;
  float t = (t0 >= 0.0f) ? t0 : t1;
  if (t < 0.0f) return 0;
  if (t > t_max) return 0;
  *out_t = t;
  return 1;
}

static int _ray_hit_nodes(const GP_WpnRequest* r, const float ray_o[3], const float ray_d[3], float t_max, float* out_t, uint32_t* out_victim_id) {
  if (!r || !out_t || !out_victim_id) return 0;
  if (!_nodes_valid(r)) return 0;

  const float* pos = (const float*)(uintptr_t)r->nodes_pos_ptr;
  const float* rad = (const float*)(uintptr_t)r->nodes_radius_ptr;
  uint32_t n = r->nodes_count;
  uint32_t ps = r->nodes_pos_stride;
  uint32_t rs = r->nodes_radius_stride;

  float best_t = t_max + 1.0f;
  uint32_t best_id = 0u;
  for (uint32_t i = 0; i < n; ++i) {
    const float* c = pos + (size_t)i * (size_t)ps;
    float rr = rad[(size_t)i * (size_t)rs];
    float t_hit;
    if (_ray_sphere_hit(ray_o, ray_d, c, rr, t_max, &t_hit)) {
      if (t_hit < best_t) {
        best_t = t_hit;
        best_id = i + 1u;  // 1-based; 0 means none
      }
    }
  }
  if (best_id == 0u) return 0;
  *out_t = best_t;
  *out_victim_id = best_id;
  return 1;
}

static int _seg_sphere_hit(const float p0[3], const float p1[3], const float c[3], float rad, float* out_t01) {
  if (!out_t01) return 0;
  if (!(rad > 0.0f)) return 0;
  float d[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
  float m[3] = { p0[0] - c[0], p0[1] - c[1], p0[2] - c[2] };

  float a = _dot3(d, d);
  float b = _dot3(m, d);
  float cc = _dot3(m, m) - rad * rad;
  if (!(a > 1e-12f)) return 0;
  float disc = b * b - a * cc;
  if (disc < 0.0f) return 0;
  float s = sqrtf(disc);
  float inva = 1.0f / a;
  float t0 = (-b - s) * inva;
  float t1 = (-b + s) * inva;
  float t = (t0 >= 0.0f) ? t0 : t1;
  if (t < 0.0f || t > 1.0f) return 0;
  *out_t01 = t;
  return 1;
}

static int _seg_hit_nodes(const GP_WpnRequest* r, const float p0[3], const float p1[3], float* out_t01, uint32_t* out_victim_id) {
  if (!r || !out_t01 || !out_victim_id) return 0;
  if (!_nodes_valid(r)) return 0;

  const float* pos = (const float*)(uintptr_t)r->nodes_pos_ptr;
  const float* rad = (const float*)(uintptr_t)r->nodes_radius_ptr;
  uint32_t n = r->nodes_count;
  uint32_t ps = r->nodes_pos_stride;
  uint32_t rs = r->nodes_radius_stride;

  float best_t = 2.0f;
  uint32_t best_id = 0u;
  for (uint32_t i = 0; i < n; ++i) {
    const float* c = pos + (size_t)i * (size_t)ps;
    float rr = rad[(size_t)i * (size_t)rs];
    float t_hit;
    if (_seg_sphere_hit(p0, p1, c, rr, &t_hit)) {
      if (t_hit < best_t) {
        best_t = t_hit;
        best_id = i + 1u;
      }
    }
  }
  if (best_id == 0u) return 0;
  *out_t01 = best_t;
  *out_victim_id = best_id;
  return 1;
}

static float _terrain_signed_dist(const GP_WpnRequest* r, const float p[3]) {
  if (!r || !_hm_valid(r)) return 1e9f;
  float rr = _len3(p);
  if (!(rr > 1e-6f)) return 1e9f;
  float dir[3] = { p[0] / rr, p[1] / rr, p[2] / rr };
  float rs = _terrain_surface_r_at_dir(r, dir);
  return rr - rs;
}

static int _seg_hit_terrain(const GP_WpnRequest* r, const float p0[3], const float p1[3], float* out_t01) {
  if (!r || !out_t01) return 0;
  if (!_hm_valid(r)) return 0;
  float f0 = _terrain_signed_dist(r, p0);
  float f1 = _terrain_signed_dist(r, p1);
  if (!(f0 > 0.0f && f1 <= 0.0f)) return 0;

  float a = 0.0f;
  float b = 1.0f;
  for (uint32_t k = 0; k < 14u; ++k) {
    float m = 0.5f * (a + b);
    float pm[3] = {
      p0[0] + (p1[0] - p0[0]) * m,
      p0[1] + (p1[1] - p0[1]) * m,
      p0[2] + (p1[2] - p0[2]) * m,
    };
    float fm = _terrain_signed_dist(r, pm);
    if (fm > 0.0f) a = m; else b = m;
  }
  *out_t01 = 0.5f * (a + b);
  return 1;
}

static float _hm_sample_bilinear_f32(const float* hm, uint32_t w, uint32_t h, uint32_t stride, float u, float v) {
  if (!hm || w < 1u || h < 1u || stride < w) return 0.0f;
  // u wraps, v clamps.
  float uu = u - floorf(u);
  float vv = _fclamp(v, 0.0f, 1.0f);

  float x = uu * (float)(w - 1u);
  float y = vv * (float)(h - 1u);
  uint32_t x0 = (uint32_t)floorf(x);
  uint32_t y0 = (uint32_t)floorf(y);
  uint32_t x1 = (x0 + 1u < w) ? (x0 + 1u) : (w - 1u);
  uint32_t y1 = (y0 + 1u < h) ? (y0 + 1u) : (h - 1u);

  float sx = x - (float)x0;
  float sy = y - (float)y0;

  const float h00 = hm[(size_t)y0 * (size_t)stride + (size_t)x0];
  const float h10 = hm[(size_t)y0 * (size_t)stride + (size_t)x1];
  const float h01 = hm[(size_t)y1 * (size_t)stride + (size_t)x0];
  const float h11 = hm[(size_t)y1 * (size_t)stride + (size_t)x1];

  float a = (1.0f - sx) * h00 + sx * h10;
  float b = (1.0f - sx) * h01 + sx * h11;
  return (1.0f - sy) * a + sy * b;
}

static float _terrain_surface_r_at_dir(const GP_WpnRequest* r, const float dir_unit[3]) {
  float base = r ? r->planet_surface_r : 0.0f;
  if (!_hm_valid(r)) return base;

  const float* hm = (const float*)(uintptr_t)r->terrain_hm_ptr;
  float x = dir_unit[0];
  float y = dir_unit[1];
  float z = dir_unit[2];
  // lon = atan2(z, x), lat = asin(y)
  float lon = atan2f(z, x);
  float lat = asinf(_fclamp(y, -1.0f, 1.0f));
  float u = 0.5f + lon * (1.0f / (2.0f * GP_PI));
  float v = 0.5f + lat * (1.0f / GP_PI);
  float h01 = _hm_sample_bilinear_f32(hm, r->terrain_hm_w, r->terrain_hm_h, r->terrain_hm_stride, u, v);
  return base + (h01 - r->terrain_height_bias) * r->terrain_height_scale;
}

static int _ray_march_to_terrain(const GP_WpnRequest* r, const float ray_o[3], const float ray_d[3], float t_max, float* out_t_hit) {
  if (!r || !out_t_hit) return 0;
  if (!_hm_valid(r)) return 0;
  if (!(t_max > 0.0f)) return 0;

  float step = 0.01f * (r->planet_surface_r > 1e-3f ? r->planet_surface_r : 50.0f);
  step = _fclamp(step, 0.25f, 2.0f);

  float t = 0.0f;
  float prev_t = 0.0f;
  float prev_f = 0.0f;
  int have_prev = 0;

  for (uint32_t iter = 0; iter < 8192u && t <= t_max; ++iter) {
    float p[3];
    _mad3(p, ray_o, t, ray_d);
    float rlen = _len3(p);
    if (!(rlen > 1e-6f)) {
      t += step;
      continue;
    }
    float dir[3] = { p[0] / rlen, p[1] / rlen, p[2] / rlen };
    float rs = _terrain_surface_r_at_dir(r, dir);
    float f = rlen - rs;
    if (have_prev && f <= 0.0f && prev_f > 0.0f) {
      // bracket and refine with bisection
      float a = prev_t;
      float b = t;
      for (uint32_t k = 0; k < 18u; ++k) {
        float m = 0.5f * (a + b);
        float pm[3];
        _mad3(pm, ray_o, m, ray_d);
        float ml = _len3(pm);
        if (!(ml > 1e-6f)) break;
        float md[3] = { pm[0] / ml, pm[1] / ml, pm[2] / ml };
        float mrs = _terrain_surface_r_at_dir(r, md);
        float mf = ml - mrs;
        if (mf > 0.0f) a = m; else b = m;
      }
      *out_t_hit = 0.5f * (a + b);
      return 1;
    }
    prev_t = t;
    prev_f = f;
    have_prev = 1;
    t += step;
  }
  return 0;
}

GP_EXPORT int gp_weapon_process_batch(GP_WpnBatch* batch) {
  if (!batch) {
    return 0;
  }
  if (batch->hdr.magic != GP_WPN_MAGIC) {
    // Avoid spamming stderr if the caller is polling frequently (e.g. reticle LOS probes).
    // Still emit a signal once so ABI mismatches are discoverable.
    static uint32_t s_last_bad_magic = 0u;
    static int s_bad_magic_reported = 0;
    const uint32_t got = (uint32_t)batch->hdr.magic;
    if (!s_bad_magic_reported || got != s_last_bad_magic) {
      s_last_bad_magic = got;
      s_bad_magic_reported = 1;
      fprintf(stderr, "[wpn] bad magic: 0x%08X\n", (unsigned)got);
      fflush(stderr);
    }
    return 0;
  }
  if (batch->hdr.version != 6u) {
    // Same spam-avoidance as the magic mismatch case.
    static uint32_t s_last_bad_version = 0u;
    static int s_bad_version_reported = 0;
    const uint32_t got = (uint32_t)batch->hdr.version;
    if (!s_bad_version_reported || got != s_last_bad_version) {
      s_last_bad_version = got;
      s_bad_version_reported = 1;
      fprintf(stderr, "[wpn] bad version: %u\n", (unsigned)got);
      fflush(stderr);
    }
    return 0;
  }

  uint32_t count = batch->hdr.count;
  if (count > 8u) {
    count = 8u;
  }

  // This prototype implementation prints requests and writes placeholder outputs.
  for (uint32_t i = 0; i < count; ++i) {
    GP_WpnRequest* r = &batch->req[i];
    GP_WpnResult* o = &batch->out[i];

    char src[33];
    char wtype[33];
    _safe_cstr(src, sizeof(src), r->source);
    _safe_cstr(wtype, sizeof(wtype), r->weapon_type);

    if ((r->weapon_flags & 2u) != 0u) {
      fprintf(
        stderr,
        "[wpn] req[%u] id=%u slot=%u analog=%.3f source='%s' type='%s' "
        "pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f) "
        "wflags=0x%X worigin=(%.3f,%.3f,%.3f) wdir=(%.3f,%.3f,%.3f) "
        "k_mode=%u inh=%u add_v=%.3f sim_pts=%u t_end=%.3f beam=%.3f drop=%.3f "
        "flags=0x%X hm=%ux%u nodes=%u\n",
        (unsigned)i,
        (unsigned)r->request_id,
        (unsigned)r->weapon_slot,
        (double)r->analog,
        src,
        wtype,
        (double)r->ship_pos[0], (double)r->ship_pos[1], (double)r->ship_pos[2],
        (double)r->ship_vel[0], (double)r->ship_vel[1], (double)r->ship_vel[2],
        (double)r->ship_fwd[0], (double)r->ship_fwd[1], (double)r->ship_fwd[2],
        (unsigned)r->weapon_flags,
        (double)r->weapon_origin[0], (double)r->weapon_origin[1], (double)r->weapon_origin[2],
        (double)r->weapon_dir[0], (double)r->weapon_dir[1], (double)r->weapon_dir[2],
        (unsigned)r->kinematics_mode,
        (unsigned)r->inherit_ship_velocity,
        (double)r->add_velocity,
        (unsigned)r->sim_points,
        (double)r->sim_t_end,
        (double)r->sim_beam_len,
        (double)r->sim_drop_off,
        (unsigned)r->world_flags,
        (unsigned)r->terrain_hm_w,
        (unsigned)r->terrain_hm_h,
        (unsigned)r->nodes_count
      );
    }

    // Initialize outputs; mark invalid until we successfully run the requested mode.
    memset(o, 0, sizeof(*o));
    o->ok = 0u;
    o->impact_valid = 0u;
    o->victim_id = 0u;

    // Requested mode must be explicit (comes from weapon config).
    const uint32_t mode = (uint32_t)r->kinematics_mode;

    // Direction (normalize defensively). Default: ship forward.
    float dir[3] = { r->ship_fwd[0], r->ship_fwd[1], r->ship_fwd[2] };
    if ((r->weapon_flags & 1u) != 0u) {
      dir[0] = r->weapon_dir[0];
      dir[1] = r->weapon_dir[1];
      dir[2] = r->weapon_dir[2];
    }
    _norm3(dir);

    // Weapon origin. Default: ship_pos + nose_off * dir (legacy).
    float origin[3];
    if ((r->weapon_flags & 1u) != 0u) {
      origin[0] = r->weapon_origin[0];
      origin[1] = r->weapon_origin[1];
      origin[2] = r->weapon_origin[2];
    } else {
      const float nose_off = 2.0f;
      origin[0] = r->ship_pos[0] + nose_off * dir[0];
      origin[1] = r->ship_pos[1] + nose_off * dir[1];
      origin[2] = r->ship_pos[2] + nose_off * dir[2];
    }

    // Radial up/down (planet center assumed at origin).
    float up[3] = { origin[0], origin[1], origin[2] };
    _norm3(up);
    float down[3] = { -up[0], -up[1], -up[2] };

    // Beam: from nose forward, 2 points.
    if (mode == 0u) {
      const float beam_len = r->sim_beam_len;
      if (!(beam_len > 0.0f)) {
        continue;
      }
      float p0[3] = { origin[0], origin[1], origin[2] };

      float impact[3] = { p0[0] + beam_len * dir[0], p0[1] + beam_len * dir[1], p0[2] + beam_len * dir[2] };
      uint32_t victim = 0u;
      float best_t = beam_len + 1.0f;

      // Node ray hits.
      {
        float t_node;
        uint32_t v;
        if (_ray_hit_nodes(r, p0, dir, beam_len, &t_node, &v)) {
          best_t = t_node;
          victim = v;
        }
      }

      // Terrain ray hit.
      {
        float t_hit;
        if (_ray_march_to_terrain(r, p0, dir, beam_len, &t_hit)) {
          if (t_hit < best_t) {
            best_t = t_hit;
            victim = 0u;
          }
        }
      }

      if (best_t <= beam_len) {
        _mad3(impact, p0, best_t, dir);
      }

      o->impact_point[0] = impact[0];
      o->impact_point[1] = impact[1];
      o->impact_point[2] = impact[2];
      o->victim_id = victim;
      o->spline_n = 2u;
      o->spline_points[0][0] = p0[0];
      o->spline_points[0][1] = p0[1];
      o->spline_points[0][2] = p0[2];
      o->spline_points[1][0] = o->impact_point[0];
      o->spline_points[1][1] = o->impact_point[1];
      o->spline_points[1][2] = o->impact_point[2];
      o->ok = 1u;
      o->impact_valid = 1u;
      continue;
    }

    // Fall/released: start offset along radial-down, inherit ship vel optionally.
    if (mode == 2u) {
      uint32_t maxp = batch->hdr.max_points;
      if (maxp == 0u) maxp = 16u;
      if (maxp > 16u) maxp = 16u;
      if (maxp < 2u) maxp = 2u;

      if (r->sim_points > 0u && r->sim_points < maxp) {
        maxp = r->sim_points;
      }

      const float drop_off = r->sim_drop_off;
      const float t_end = r->sim_t_end;
      if (!(drop_off > 0.0f) || !(t_end > 0.0f)) {
        continue;
      }

      float g = r->gravity_g;

      float p0[3] = {
        origin[0] + drop_off * down[0],
        origin[1] + drop_off * down[1],
        origin[2] + drop_off * down[2]
      };

      // Initial velocity from config: inherit ship optionally, plus add_velocity along ship forward.
      float v0[3] = { 0.0f, 0.0f, 0.0f };
      if (r->inherit_ship_velocity) {
        v0[0] += r->ship_vel[0];
        v0[1] += r->ship_vel[1];
        v0[2] += r->ship_vel[2];
      }
      v0[0] += r->add_velocity * dir[0];
      v0[1] += r->add_velocity * dir[1];
      v0[2] += r->add_velocity * dir[2];

      float dt = (maxp <= 1u) ? t_end : (t_end / (float)(maxp - 1u));
      float p[3] = { p0[0], p0[1], p0[2] };
      float v[3] = { v0[0], v0[1], v0[2] };

      uint32_t written = 0u;
      uint32_t victim = 0u;
      for (uint32_t k = 0; k < maxp; ++k) {
        o->spline_points[k][0] = p[0];
        o->spline_points[k][1] = p[1];
        o->spline_points[k][2] = p[2];
        written = k + 1u;

        // Semi-implicit Euler with central gravity: a = -g * normalize(p)
        float pr = _len3(p);
        float grav_dir[3] = { 0.0f, 0.0f, 0.0f };
        if (pr > 1e-6f) {
          grav_dir[0] = -p[0] / pr;
          grav_dir[1] = -p[1] / pr;
          grav_dir[2] = -p[2] / pr;
        } else {
          grav_dir[0] = down[0];
          grav_dir[1] = down[1];
          grav_dir[2] = down[2];
        }
        // gravity_g comes from world snapshot; allow 0 for space.
        v[0] += (g * grav_dir[0]) * dt;
        v[1] += (g * grav_dir[1]) * dt;
        v[2] += (g * grav_dir[2]) * dt;

        float p_next[3] = { p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt };

        // Continuous collision along segment: nodes + terrain.
        float best_t01 = 2.0f;
        uint32_t best_victim = 0u;
        {
          float t01;
          uint32_t vid;
          if (_seg_hit_nodes(r, p, p_next, &t01, &vid)) {
            best_t01 = t01;
            best_victim = vid;
          }
        }
        {
          float t01;
          if (_seg_hit_terrain(r, p, p_next, &t01)) {
            if (t01 < best_t01) {
              best_t01 = t01;
              best_victim = 0u;
            }
          }
        }

        if (best_t01 <= 1.0f) {
          // Clamp to the hit point and stop.
          p[0] = p[0] + (p_next[0] - p[0]) * best_t01;
          p[1] = p[1] + (p_next[1] - p[1]) * best_t01;
          p[2] = p[2] + (p_next[2] - p[2]) * best_t01;
          victim = best_victim;
          // Overwrite the last spline point to be the impact location.
          o->spline_points[k][0] = p[0];
          o->spline_points[k][1] = p[1];
          o->spline_points[k][2] = p[2];
          break;
        }

        // No hit: accept the step.
        p[0] = p_next[0];
        p[1] = p_next[1];
        p[2] = p_next[2];
      }
      o->spline_n = written;
      o->impact_point[0] = o->spline_points[written - 1u][0];
      o->impact_point[1] = o->spline_points[written - 1u][1];
      o->impact_point[2] = o->spline_points[written - 1u][2];
      o->victim_id = victim;
      o->ok = 1u;
      o->impact_valid = 1u;
      continue;
    }

    // Fire/projectile: inherits ship vel optionally, plus add_velocity along forward.
    if (mode == 1u) {
      uint32_t maxp = batch->hdr.max_points;
      if (maxp == 0u) maxp = 16u;
      if (maxp > 16u) maxp = 16u;

      if (r->sim_points > 0u && r->sim_points < maxp) {
        maxp = r->sim_points;
      }

      float g = r->gravity_g;
      float muzzle = r->add_velocity;
      const float t_end = r->sim_t_end;
      if (!(t_end > 0.0f)) {
        continue;
      }
      float dt = (maxp <= 1u) ? t_end : (t_end / (float)(maxp - 1u));

      float p[3] = { origin[0], origin[1], origin[2] };
      float v[3] = {
        (r->inherit_ship_velocity ? r->ship_vel[0] : 0.0f) + muzzle * dir[0],
        (r->inherit_ship_velocity ? r->ship_vel[1] : 0.0f) + muzzle * dir[1],
        (r->inherit_ship_velocity ? r->ship_vel[2] : 0.0f) + muzzle * dir[2]
      };

      uint32_t written = 0u;
      uint32_t victim = 0u;
      for (uint32_t k = 0; k < maxp; ++k) {
        o->spline_points[k][0] = p[0];
        o->spline_points[k][1] = p[1];
        o->spline_points[k][2] = p[2];
        written = k + 1u;

        // Semi-implicit Euler with central gravity.
        float pr = _len3(p);
        float grav_dir[3] = { 0.0f, 0.0f, 0.0f };
        if (pr > 1e-6f) {
          grav_dir[0] = -p[0] / pr;
          grav_dir[1] = -p[1] / pr;
          grav_dir[2] = -p[2] / pr;
        } else {
          grav_dir[0] = down[0];
          grav_dir[1] = down[1];
          grav_dir[2] = down[2];
        }
        v[0] += (g * grav_dir[0]) * dt;
        v[1] += (g * grav_dir[1]) * dt;
        v[2] += (g * grav_dir[2]) * dt;

        float p_next[3] = { p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt };

        float best_t01 = 2.0f;
        uint32_t best_victim = 0u;
        {
          float t01;
          uint32_t vid;
          if (_seg_hit_nodes(r, p, p_next, &t01, &vid)) {
            best_t01 = t01;
            best_victim = vid;
          }
        }
        {
          float t01;
          if (_seg_hit_terrain(r, p, p_next, &t01)) {
            if (t01 < best_t01) {
              best_t01 = t01;
              best_victim = 0u;
            }
          }
        }

        if (best_t01 <= 1.0f) {
          p[0] = p[0] + (p_next[0] - p[0]) * best_t01;
          p[1] = p[1] + (p_next[1] - p[1]) * best_t01;
          p[2] = p[2] + (p_next[2] - p[2]) * best_t01;
          victim = best_victim;
          o->spline_points[k][0] = p[0];
          o->spline_points[k][1] = p[1];
          o->spline_points[k][2] = p[2];
          break;
        }

        p[0] = p_next[0];
        p[1] = p_next[1];
        p[2] = p_next[2];
      }

      o->spline_n = written;
      o->impact_point[0] = o->spline_points[written - 1u][0];
      o->impact_point[1] = o->spline_points[written - 1u][1];
      o->impact_point[2] = o->spline_points[written - 1u][2];
      o->victim_id = victim;
      o->ok = 1u;
      o->impact_valid = 1u;
      continue;
    }
  }

  fflush(stderr);
  return 1;
}
