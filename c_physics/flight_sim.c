#include "geodesic_physics.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float _dot3(const float a[3], const float b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static float _norm3(const float v[3]) {
  return sqrtf(_dot3(v, v));
}

static void _scale3(float out[3], const float v[3], float s) {
  out[0] = v[0]*s;
  out[1] = v[1]*s;
  out[2] = v[2]*s;
}

static void _add3(float io[3], const float v[3]) {
  io[0] += v[0];
  io[1] += v[1];
  io[2] += v[2];
}

static void _sub3(float out[3], const float a[3], const float b[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static void _normalize3(float io[3]) {
  float n = _norm3(io);
  if (n > 1e-9f) {
    float inv = 1.0f / n;
    io[0] *= inv;
    io[1] *= inv;
    io[2] *= inv;
  } else {
    io[0] = 0.0f;
    io[1] = 0.0f;
    io[2] = 0.0f;
  }
}

// ---------------- Quaternion helpers (q = [w,x,y,z]) ----------------

static float _clampf(float x, float a, float b) {
  return fmaxf(a, fminf(b, x));
}

static void _quat_normalize(float q[4]) {
  float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  if (n > 1e-12f) {
    float inv = 1.0f / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
  } else {
    q[0] = 1.0f;
    q[1] = q[2] = q[3] = 0.0f;
  }
}

static float _quat_dot(const float a[4], const float b[4]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

static void _quat_slerp(const float qa[4], const float qb_in[4], float t, float out[4]) {
  // Robust slerp that takes the shortest path.
  float qb[4] = { qb_in[0], qb_in[1], qb_in[2], qb_in[3] };
  float dot = _quat_dot(qa, qb);
  if (dot < 0.0f) {
    dot = -dot;
    qb[0] = -qb[0];
    qb[1] = -qb[1];
    qb[2] = -qb[2];
    qb[3] = -qb[3];
  }
  dot = _clampf(dot, -1.0f, 1.0f);

  if (dot > 0.9995f) {
    // Nearly identical: lerp.
    out[0] = qa[0] + t * (qb[0] - qa[0]);
    out[1] = qa[1] + t * (qb[1] - qa[1]);
    out[2] = qa[2] + t * (qb[2] - qa[2]);
    out[3] = qa[3] + t * (qb[3] - qa[3]);
    _quat_normalize(out);
    return;
  }

  float theta0 = acosf(dot);
  float sin_theta0 = sinf(theta0);
  float theta = theta0 * _clampf(t, 0.0f, 1.0f);
  float sin_theta = sinf(theta);
  float s0 = cosf(theta) - dot * (sin_theta / sin_theta0);
  float s1 = sin_theta / sin_theta0;

  out[0] = s0 * qa[0] + s1 * qb[0];
  out[1] = s0 * qa[1] + s1 * qb[1];
  out[2] = s0 * qa[2] + s1 * qb[2];
  out[3] = s0 * qa[3] + s1 * qb[3];
  _quat_normalize(out);
}

static void _quat_rotate_vec(const float q[4], const float v[3], float out[3]) {
  // v' = v + 2*cross(q_xyz, cross(q_xyz,v) + q_w*v)
  const float qw = q[0];
  const float qx = q[1];
  const float qy = q[2];
  const float qz = q[3];
  float uv[3] = {
    qy * v[2] - qz * v[1],
    qz * v[0] - qx * v[2],
    qx * v[1] - qy * v[0],
  };
  float uuv[3] = {
    qy * uv[2] - qz * uv[1],
    qz * uv[0] - qx * uv[2],
    qx * uv[1] - qy * uv[0],
  };
  out[0] = v[0] + 2.0f * (qw * uv[0] + uuv[0]);
  out[1] = v[1] + 2.0f * (qw * uv[1] + uuv[1]);
  out[2] = v[2] + 2.0f * (qw * uv[2] + uuv[2]);
}

static void _quat_conj(const float q[4], float out[4]) {
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
}

static void _quat_mul(const float a[4], const float b[4], float out[4]) {
  const float aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const float bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[0] = aw*bw - ax*bx - ay*by - az*bz;
  out[1] = aw*bx + ax*bw + ay*bz - az*by;
  out[2] = aw*by - ax*bz + ay*bw + az*bx;
  out[3] = aw*bz + ax*by - ay*bx + az*bw;
}

static void _cross3(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1]*b[2] - a[2]*b[1];
  out[1] = a[2]*b[0] - a[0]*b[2];
  out[2] = a[0]*b[1] - a[1]*b[0];
}

static void _quat_to_axis_angle_shortest(const float q_in[4], float axis_out[3], float* angle_out) {
  float q[4] = { q_in[0], q_in[1], q_in[2], q_in[3] };
  _quat_normalize(q);
  // Ensure shortest-path representation.
  if (q[0] < 0.0f) {
    q[0] = -q[0];
    q[1] = -q[1];
    q[2] = -q[2];
    q[3] = -q[3];
  }
  float w = _clampf(q[0], -1.0f, 1.0f);
  float ang = 2.0f * acosf(w);
  float s = sqrtf(fmaxf(0.0f, 1.0f - w*w));
  if (s > 1e-6f && ang > 1e-7f) {
    axis_out[0] = q[1] / s;
    axis_out[1] = q[2] / s;
    axis_out[2] = q[3] / s;
  } else {
    axis_out[0] = 0.0f;
    axis_out[1] = 0.0f;
    axis_out[2] = 0.0f;
    ang = 0.0f;
  }
  if (angle_out) {
    *angle_out = ang;
  }
}

static int _solve_linear_system(float* A, float* b, float* x, int n) {
  // Gauss-Jordan elimination with partial pivoting.
  // A is n x n row-major; b is n.
  if (!A || !b || !x || n <= 0 || n > 8) {
    return 0;
  }

  float aug[8 * 9];
  // Build augmented matrix [A | b].
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < n; c++) {
      aug[r * (n + 1) + c] = A[r * n + c];
    }
    aug[r * (n + 1) + n] = b[r];
  }

  for (int col = 0; col < n; col++) {
    // Pivot.
    int piv = col;
    float best = fabsf(aug[piv * (n + 1) + col]);
    for (int r = col + 1; r < n; r++) {
      float v = fabsf(aug[r * (n + 1) + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 1e-9f) || !isfinite(best)) {
      return 0;
    }
    if (piv != col) {
      for (int c = col; c < n + 1; c++) {
        float tmp = aug[col * (n + 1) + c];
        aug[col * (n + 1) + c] = aug[piv * (n + 1) + c];
        aug[piv * (n + 1) + c] = tmp;
      }
    }

    float diag = aug[col * (n + 1) + col];
    if (!(fabsf(diag) > 1e-9f) || !isfinite(diag)) {
      return 0;
    }
    float inv = 1.0f / diag;
    for (int c = col; c < n + 1; c++) {
      aug[col * (n + 1) + c] *= inv;
    }

    for (int r = 0; r < n; r++) {
      if (r == col) continue;
      float f = aug[r * (n + 1) + col];
      if (fabsf(f) <= 1e-12f) continue;
      for (int c = col; c < n + 1; c++) {
        aug[r * (n + 1) + c] -= f * aug[col * (n + 1) + c];
      }
    }
  }

  for (int i = 0; i < n; i++) {
    x[i] = aug[i * (n + 1) + n];
  }
  return 1;
}

static void _mat3_from_quat(const float q[4], float R[9]) {
  // R maps body -> world. Column-major not required; we store row-major.
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  const float xx = x*x, yy = y*y, zz = z*z;
  const float xy = x*y, xz = x*z, yz = y*z;
  const float wx = w*x, wy = w*y, wz = w*z;

  R[0] = 1.0f - 2.0f*(yy + zz);
  R[1] = 2.0f*(xy - wz);
  R[2] = 2.0f*(xz + wy);

  R[3] = 2.0f*(xy + wz);
  R[4] = 1.0f - 2.0f*(xx + zz);
  R[5] = 2.0f*(yz - wx);

  R[6] = 2.0f*(xz - wy);
  R[7] = 2.0f*(yz + wx);
  R[8] = 1.0f - 2.0f*(xx + yy);
}

static void _mat3_mul_vec(const float R[9], const float v[3], float out[3]) {
  out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
  out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
  out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

static void _mat3_mul_vec_T(const float R[9], const float v[3], float out[3]) {
  // out = R^T * v
  out[0] = R[0]*v[0] + R[3]*v[1] + R[6]*v[2];
  out[1] = R[1]*v[0] + R[4]*v[1] + R[7]*v[2];
  out[2] = R[2]*v[0] + R[5]*v[1] + R[8]*v[2];
}

// ---------------- Terrain helpers (matches Python mapping) ----------------

static float _hm_sample_bilinear(const float* hm, uint32_t w, uint32_t h, uint32_t stride, float u, float v) {
  if (!hm || w == 0 || h == 0 || stride == 0) {
    return 0.0f;
  }
  if (w == 1 || h == 1) {
    return hm[0];
  }
  float uu = u - floorf(u);
  float vv = _clampf(v, 0.0f, 1.0f);

  float x = uu * (float)(w - 1);
  float y = vv * (float)(h - 1);
  int x0 = (int)floorf(x);
  int y0 = (int)floorf(y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  if (x1 >= (int)w) x1 = (int)w - 1;
  if (y1 >= (int)h) y1 = (int)h - 1;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;

  float sx = x - (float)x0;
  float sy = y - (float)y0;
  const float h00 = hm[(uint32_t)y0 * stride + (uint32_t)x0];
  const float h10 = hm[(uint32_t)y0 * stride + (uint32_t)x1];
  const float h01 = hm[(uint32_t)y1 * stride + (uint32_t)x0];
  const float h11 = hm[(uint32_t)y1 * stride + (uint32_t)x1];
  float h0 = (1.0f - sx) * h00 + sx * h10;
  float h1 = (1.0f - sx) * h01 + sx * h11;
  return (1.0f - sy) * h0 + sy * h1;
}

static float _terrain_surface_r(const GP_FlightConfig* cfg, const float pos[3]) {
  const float base = cfg ? cfg->planet_surface_r : 0.0f;
  if (!cfg || cfg->terrain_hm_ptr == (uintptr_t)0 || cfg->terrain_height_scale <= 0.0f) {
    return base;
  }
  float dir[3] = { pos[0], pos[1], pos[2] };
  _normalize3(dir);
  if (_norm3(dir) <= 1e-9f) {
    return base;
  }
  float lat = asinf(_clampf(dir[1], -1.0f, 1.0f));
  float lon = atan2f(dir[2], dir[0]);
  float u = lon / (2.0f * (float)M_PI) + 0.5f;
  float v = lat / (float)M_PI + 0.5f;

  const float* hm = (const float*)(cfg->terrain_hm_ptr);
  float h01 = _hm_sample_bilinear(hm, cfg->terrain_hm_w, cfg->terrain_hm_h, cfg->terrain_hm_stride, u, v);
  float disp = (h01 - cfg->terrain_height_bias) * cfg->terrain_height_scale;
  return base + disp;
}

// ---------------- Flight buffer API ----------------

GP_EXPORT size_t gp_flight_required_bytes(void) {
  return sizeof(GP_FlightBuffer);
}

GP_EXPORT int gp_flight_init(void* mem) {
  if (!mem) return 0;
  GP_FlightBuffer* b = (GP_FlightBuffer*)mem;
  memset(b, 0, sizeof(*b));
  b->hdr.magic = GP_FLIGHT_MAGIC;
  b->hdr.version = 4;
  b->hdr.front_idx = 0;
  b->hdr.sim_idx = 1;
  b->hdr.swap_requested = 0;
  b->hdr.swap_seq = 0;
  b->hdr.controls_seq = 0;
  // Identity quats by default.
  b->state[0].q[0] = 1.0f;
  b->state[1].q[0] = 1.0f;

  // Default per-arm telemetry/state.
  for (uint32_t si = 0; si < 2u; si++) {
    for (uint32_t i = 0; i < GP_FLIGHT_ARM_CAP; i++) {
      b->state[si].arm_temp[i] = 0.0f;
      b->state[si].arm_eff[i] = 1.0f;
    }
  }

  // Default rigid-body params.
  b->cfg.mass = 1.0f;
  b->cfg.inertia_diag[0] = 1.0f;
  b->cfg.inertia_diag[1] = 1.0f;
  b->cfg.inertia_diag[2] = 1.0f;
  b->arm_count = 0;
  return 1;
}

GP_EXPORT void gp_flight_request_swap(void* mem) {
  if (!mem) return;
  GP_FlightBuffer* b = (GP_FlightBuffer*)mem;
  b->hdr.swap_requested = 1;
}

GP_EXPORT uint32_t gp_flight_get_swap_seq(void* mem) {
  if (!mem) return 0;
  GP_FlightBuffer* b = (GP_FlightBuffer*)mem;
  return (uint32_t)b->hdr.swap_seq;
}

static void _flight_one_step(GP_FlightBuffer* b, float dt) {
  // Read controls with a simple seq check.
  GP_FlightControls ctl;
  uint32_t s0 = b->hdr.controls_seq;
  memcpy(&ctl, &b->ctl, sizeof(ctl));
  uint32_t s1 = b->hdr.controls_seq;
  if (s0 != s1) {
    // Best-effort retry once.
    memcpy(&ctl, &b->ctl, sizeof(ctl));
  }

  const uint32_t si = (uint32_t)(b->hdr.sim_idx & 1u);
  GP_FlightState* st = &b->state[si];

  // Per-arm telemetry is written per tick. Clear eff by default.
  for (uint32_t i = 0; i < GP_FLIGHT_ARM_CAP; i++) {
    st->arm_eff[i] = 0.0f;
  }

  float pos[3] = { st->pos[0], st->pos[1], st->pos[2] };
  float vel[3] = { st->vel[0], st->vel[1], st->vel[2] };
  float q_cur[4] = { st->q[0], st->q[1], st->q[2], st->q[3] };
  float omega_b[3] = { st->omega_b[0], st->omega_b[1], st->omega_b[2] };
  _quat_normalize(q_cur);

  // Desired orientation and error (for auto mode).
  float q_des[4] = { q_cur[0], q_cur[1], q_cur[2], q_cur[3] };
  const int desired_valid = ((ctl.flags & 1u) != 0u);
  if (desired_valid) {
    q_des[0] = ctl.desired_q[0];
    q_des[1] = ctl.desired_q[1];
    q_des[2] = ctl.desired_q[2];
    q_des[3] = ctl.desired_q[3];
    _quat_normalize(q_des);
  }

  // Arms configured?
  uint32_t n_arms = b->arm_count;
  if (n_arms > GP_FLIGHT_ARM_CAP) n_arms = GP_FLIGHT_ARM_CAP;

  // Optional fallback: if no arms exist, preserve the legacy behavior in auto mode
  // by clamping directly toward desired_q.
  float q_forces[4] = { q_cur[0], q_cur[1], q_cur[2], q_cur[3] };
  float q_new[4] = { q_cur[0], q_cur[1], q_cur[2], q_cur[3] };
  float requested_ang = 0.0f;
  float applied_ang = 0.0f;
  float clamp_ratio = 1.0f;

  if (ctl.mode == 1u && desired_valid && n_arms == 0u) {
    float dot = _quat_dot(q_cur, q_des);
    if (dot < 0.0f) dot = -dot;
    dot = _clampf(dot, -1.0f, 1.0f);
    requested_ang = 2.0f * acosf(dot);
    float max_ang = (ctl.max_ang_rate > 0.0f) ? (ctl.max_ang_rate * dt) : requested_ang;
    float t = 1.0f;
    if (requested_ang > 1e-7f && max_ang > 0.0f && requested_ang > max_ang) {
      t = max_ang / requested_ang;
    }
    _quat_slerp(q_cur, q_des, t, q_new);
    // Angular velocity estimate from applied delta.
    if (requested_ang > 1e-7f && dt > 1e-9f) {
      float q_conj[4], dq[4];
      _quat_conj(q_cur, q_conj);
      _quat_mul(q_conj, q_new, dq);
      float axis[3];
      float aa = 0.0f;
      _quat_to_axis_angle_shortest(dq, axis, &aa);
      omega_b[0] = axis[0] * (aa / dt);
      omega_b[1] = axis[1] * (aa / dt);
      omega_b[2] = axis[2] * (aa / dt);
    }
    q_forces[0] = q_new[0];
    q_forces[1] = q_new[1];
    q_forces[2] = q_new[2];
    q_forces[3] = q_new[3];
    applied_ang = requested_ang * t;
    clamp_ratio = (requested_ang > 1e-7f) ? _clampf(applied_ang / requested_ang, 0.0f, 1.0f) : 1.0f;
  }

  // Body->world rotation matrix and basis.
  float R[9];
  _mat3_from_quat(q_forces, R);
  float right_w[3], up_w[3], fwd_w[3];
  const float ex[3] = {1.0f, 0.0f, 0.0f};
  const float ey[3] = {0.0f, 1.0f, 0.0f};
  const float ez[3] = {0.0f, 0.0f, 1.0f};
  _mat3_mul_vec(R, ex, right_w);
  _mat3_mul_vec(R, ey, up_w);
  _mat3_mul_vec(R, ez, fwd_w);

  // Radial up + terrain/atmosphere.
  float up_rad[3] = { pos[0], pos[1], pos[2] };
  _normalize3(up_rad);
  if (_norm3(up_rad) <= 1e-9f) {
    up_rad[0] = 0.0f; up_rad[1] = 1.0f; up_rad[2] = 0.0f;
  }
  float r_surface = _terrain_surface_r(&b->cfg, pos);
  float r = _norm3(pos);
  float alt = fmaxf(0.0f, r - r_surface);
  // Atmosphere density ratio in [0..1].
  // atmosphere_alt_max semantics:
  // - >0: density falls off to 0 at altitude==alt_max
  // -  0: vacuum (rho=0)
  // - <0: uniform atmosphere (rho=1)
  float rho = 0.0f;
  if (b->cfg.atmosphere_alt_max < -1e-6f) {
    rho = 1.0f;
  } else if (b->cfg.atmosphere_alt_max > 1e-6f) {
    float x = _clampf(alt / b->cfg.atmosphere_alt_max, 0.0f, 1.0f);
    float t0 = (1.0f - x);
    rho = _clampf(t0 * t0, 0.0f, 1.0f);
  }

  // Rigid-body net force/torque.
  const float mass = (b->cfg.mass > 1e-6f) ? b->cfg.mass : 1.0f;
  const float throttle = _clampf(ctl.throttle, -1.0f, 1.0f);
  const float strafe = _clampf(ctl.strafe, -1.0f, 1.0f);

  float Fw[3] = {0.0f, 0.0f, 0.0f};
  float tau_b[3] = {0.0f, 0.0f, 0.0f};

  // Control-system report.
  GP_FlightControlSystem ctrl;
  memset(&ctrl, 0, sizeof(ctrl));
  ctrl.mode = (uint32_t)(ctl.mode ? 1u : 0u);
  ctrl.desired_q[0] = q_des[0];
  ctrl.desired_q[1] = q_des[1];
  ctrl.desired_q[2] = q_des[2];
  ctrl.desired_q[3] = q_des[3];

  float channel_cmd[8];
  for (int i = 0; i < 8; i++) {
    channel_cmd[i] = _clampf(ctl.channels[i], -1.0f, 1.0f);
  }

  // Auto mode: compute channel_cmd to chase desired_q using arms.
  if (ctl.mode == 1u && desired_valid && n_arms > 0u) {
    // Orientation error in body frame: q_err = conj(q_cur) * q_des.
    float q_conj[4], q_err[4];
    _quat_conj(q_cur, q_conj);
    _quat_mul(q_conj, q_des, q_err);
    float err_axis[3];
    float err_ang = 0.0f;
    _quat_to_axis_angle_shortest(q_err, err_axis, &err_ang);

    // Clamp requested angle by max_ang_rate for a stable controller.
    float err_ang_cmd = err_ang;
    if (ctl.max_ang_rate > 1e-6f) {
      float max_ang = ctl.max_ang_rate * dt;
      if (err_ang_cmd > max_ang) {
        err_ang_cmd = max_ang;
      }
    }

    ctrl.err_axis_b[0] = err_axis[0];
    ctrl.err_axis_b[1] = err_axis[1];
    ctrl.err_axis_b[2] = err_axis[2];
    ctrl.err_angle = err_ang;
    requested_ang = err_ang;
    applied_ang = err_ang_cmd;
    clamp_ratio = (err_ang > 1e-7f) ? _clampf(err_ang_cmd / err_ang, 0.0f, 1.0f) : 1.0f;

    float err_vec[3] = { err_axis[0] * err_ang_cmd, err_axis[1] * err_ang_cmd, err_axis[2] * err_ang_cmd };
    const float kp = (ctl.auto_kp > 0.0f) ? ctl.auto_kp : 0.0f;
    const float kd = (ctl.auto_kd > 0.0f) ? ctl.auto_kd : 0.0f;
    float tau_des[3] = {
      kp * err_vec[0] - kd * omega_b[0],
      kp * err_vec[1] - kd * omega_b[1],
      kp * err_vec[2] - kd * omega_b[2],
    };
    ctrl.tau_des_b[0] = tau_des[0];
    ctrl.tau_des_b[1] = tau_des[1];
    ctrl.tau_des_b[2] = tau_des[2];

    // Compute torque sensitivity per free channel (2..7) and drag/bias torque.
    const int free_idx[6] = {2, 3, 4, 5, 6, 7};
    float Mcol[6][3];
    for (int i = 0; i < 6; i++) {
      Mcol[i][0] = Mcol[i][1] = Mcol[i][2] = 0.0f;
    }
    float tau_fixed[3] = {0.0f, 0.0f, 0.0f};
    float tau_bias[3] = {0.0f, 0.0f, 0.0f};

    float v_b[3];
    _mat3_mul_vec_T(R, vel, v_b);

    for (uint32_t i = 0; i < n_arms; i++) {
      const GP_FlightArm* arm = &b->arms[i];
      if (!arm || arm->type == 0u) continue;
      const uint32_t ci = arm->input_idx;

      float r_b[3] = { arm->pos_b[0], arm->pos_b[1], arm->pos_b[2] };

      if (arm->type == 1u) {
        // Thruster torque per unit input: cross(r, dir)*max_force.
        float dir_b[3] = { arm->dir_b[0], arm->dir_b[1], arm->dir_b[2] };
        _normalize3(dir_b);
        float tau_u[3];
        float tx[3];
        _cross3(r_b, dir_b, tx);
        _scale3(tau_u, tx, arm->max_force);

        float u = 0.0f;
        if (ci < 8u) {
          u = _clampf(ctl.channels[ci], -1.0f, 1.0f);
        }

        if (ci == 0u || ci == 1u || ci >= 8u) {
          tau_fixed[0] += tau_u[0] * u;
          tau_fixed[1] += tau_u[1] * u;
          tau_fixed[2] += tau_u[2] * u;
        } else {
          for (int k = 0; k < 6; k++) {
            if ((uint32_t)free_idx[k] == ci) {
              Mcol[k][0] += tau_u[0];
              Mcol[k][1] += tau_u[1];
              Mcol[k][2] += tau_u[2];
              break;
            }
          }
        }
      } else if (arm->type == 2u) {
        // Aero surface: lift is linear in input; drag is bias.
        float wxr[3];
        _cross3(omega_b, r_b, wxr);
        float v_loc[3] = { v_b[0] + wxr[0], v_b[1] + wxr[1], v_b[2] + wxr[2] };
        float sp = _norm3(v_loc);
        if (sp > 1e-6f && rho > 0.0f) {
          float vhat[3] = { v_loc[0]/sp, v_loc[1]/sp, v_loc[2]/sp };
          float axis[3] = { arm->axis_b[0], arm->axis_b[1], arm->axis_b[2] };
          _normalize3(axis);
          float proj[3];
          _scale3(proj, vhat, _dot3(axis, vhat));
          float lift_dir[3];
          _sub3(lift_dir, axis, proj);
          _normalize3(lift_dir);
          float qdyn = rho * sp * sp;

          // Drag bias.
          float drag_mag = arm->k_drag * qdyn;
          float Fd[3] = { -vhat[0] * drag_mag, -vhat[1] * drag_mag, -vhat[2] * drag_mag };
          float td[3];
          _cross3(r_b, Fd, td);
          tau_bias[0] += td[0];
          tau_bias[1] += td[1];
          tau_bias[2] += td[2];

          // Lift torque per unit input.
          float lift_coef = arm->k_lift * qdyn;
          float tau_u[3];
          float tx[3];
          _cross3(r_b, lift_dir, tx);
          _scale3(tau_u, tx, lift_coef);

          float u = 0.0f;
          if (ci < 8u) {
            u = _clampf(ctl.channels[ci], -1.0f, 1.0f);
          }

          if (ci == 0u || ci == 1u || ci >= 8u) {
            tau_fixed[0] += tau_u[0] * u;
            tau_fixed[1] += tau_u[1] * u;
            tau_fixed[2] += tau_u[2] * u;
          } else {
            for (int k = 0; k < 6; k++) {
              if ((uint32_t)free_idx[k] == ci) {
                Mcol[k][0] += tau_u[0];
                Mcol[k][1] += tau_u[1];
                Mcol[k][2] += tau_u[2];
                break;
              }
            }
          }
        }
      } else if (arm->type == 3u) {
        float wxr[3];
        _cross3(omega_b, r_b, wxr);
        float v_loc[3] = { v_b[0] + wxr[0], v_b[1] + wxr[1], v_b[2] + wxr[2] };
        float sp = _norm3(v_loc);
        if (sp > 1e-6f && rho > 0.0f) {
          float vhat[3] = { v_loc[0]/sp, v_loc[1]/sp, v_loc[2]/sp };
          float qdyn = rho * sp * sp;
          float drag_mag = arm->k_drag * qdyn;
          float Fd[3] = { -vhat[0] * drag_mag, -vhat[1] * drag_mag, -vhat[2] * drag_mag };
          float td[3];
          _cross3(r_b, Fd, td);
          tau_bias[0] += td[0];
          tau_bias[1] += td[1];
          tau_bias[2] += td[2];
        }
      }
    }

    float tau_tgt[3] = { tau_des[0] - tau_fixed[0] - tau_bias[0], tau_des[1] - tau_fixed[1] - tau_bias[1], tau_des[2] - tau_fixed[2] - tau_bias[2] };

    // Build normal equations for the 6 free channels.
    float A[6*6];
    float bb[6];
    float x[6];
    for (int i = 0; i < 6*6; i++) A[i] = 0.0f;
    for (int i = 0; i < 6; i++) { bb[i] = 0.0f; x[i] = 0.0f; }

    // Regularization.
    const float lambda = 1e-3f;
    float sens_sum = 0.0f;
    for (int i = 0; i < 6; i++) {
      sens_sum += fabsf(Mcol[i][0]) + fabsf(Mcol[i][1]) + fabsf(Mcol[i][2]);
    }
    int solved = 0;
    if (sens_sum > 1e-8f) {
      for (int i = 0; i < 6; i++) {
        // b = M^T * tau
        bb[i] = Mcol[i][0]*tau_tgt[0] + Mcol[i][1]*tau_tgt[1] + Mcol[i][2]*tau_tgt[2];
        for (int j = 0; j < 6; j++) {
          A[i*6 + j] = Mcol[i][0]*Mcol[j][0] + Mcol[i][1]*Mcol[j][1] + Mcol[i][2]*Mcol[j][2];
        }
        A[i*6 + i] += lambda;
      }
      solved = _solve_linear_system(A, bb, x, 6);
    }

    if (solved) {
      ctrl.flags |= 1u;
      for (int i = 0; i < 6; i++) {
        int ci = free_idx[i];
        channel_cmd[ci] = _clampf(x[i], -1.0f, 1.0f);
      }
    }
  }

  // Record commanded channels.
  for (int i = 0; i < 8; i++) {
    ctrl.channel_cmd[i] = channel_cmd[i];
  }

  // Always apply the legacy scalar model for baseline translation forces.
  // Arms (when configured) add additional forces + torques.
  if (fabsf(throttle) > 1e-7f && ctl.thrust_accel != 0.0f) {
    Fw[0] += fwd_w[0] * (mass * ctl.thrust_accel * throttle);
    Fw[1] += fwd_w[1] * (mass * ctl.thrust_accel * throttle);
    Fw[2] += fwd_w[2] * (mass * ctl.thrust_accel * throttle);
  }
  if (fabsf(strafe) > 1e-7f && ctl.strafe_accel != 0.0f) {
    Fw[0] += right_w[0] * (mass * ctl.strafe_accel * strafe);
    Fw[1] += right_w[1] * (mass * ctl.strafe_accel * strafe);
    Fw[2] += right_w[2] * (mass * ctl.strafe_accel * strafe);
  }
  float speed = _norm3(vel);
  if (speed > 1e-6f) {
    float vhat[3] = { vel[0]/speed, vel[1]/speed, vel[2]/speed };
    // AoA proxy: angle between craft forward axis and velocity direction.
    // This prevents the legacy scalar lift from being purely "fast == lift".
    float aoa_cos = _clampf(_dot3(fwd_w, vhat), -1.0f, 1.0f);
    float aoa = acosf(aoa_cos);
    float lift_gate = sinf(aoa);
    float proj[3];
    _scale3(proj, vhat, _dot3(up_w, vhat));
    float lift_dir[3];
    _sub3(lift_dir, up_w, proj);
    _normalize3(lift_dir);
    if (ctl.lift_k != 0.0f && rho > 0.0f) {
      float k = mass * ctl.lift_k * rho * speed * speed * lift_gate;
      Fw[0] += lift_dir[0] * k;
      Fw[1] += lift_dir[1] * k;
      Fw[2] += lift_dir[2] * k;
    }
    if (ctl.drag_k != 0.0f && rho > 0.0f) {
      float k = mass * ctl.drag_k * rho * speed * speed;
      Fw[0] += -vhat[0] * k;
      Fw[1] += -vhat[1] * k;
      Fw[2] += -vhat[2] * k;
    }
  }

  // If arms are configured, accumulate forces/torques from them.
  if (n_arms > 0u) {
    // Precompute vel in body frame.
    float v_b[3];
    _mat3_mul_vec_T(R, vel, v_b);

    for (uint32_t i = 0; i < n_arms; i++) {
      const GP_FlightArm* arm = &b->arms[i];
      if (!arm || arm->type == 0u) continue;

      float input = 0.0f;
      if (arm->type == 1u && (arm->flags & 1u) != 0u) {
        // Special: thruster uses throttle as its input.
        input = throttle;
      } else if (arm->type == 1u && (arm->flags & 2u) != 0u) {
        // Special: thruster uses strafe as its input.
        input = strafe;
      } else if (arm->input_idx < 8u) {
        input = _clampf(channel_cmd[arm->input_idx], -1.0f, 1.0f);
      }

      float r_b[3] = { arm->pos_b[0], arm->pos_b[1], arm->pos_b[2] };
      float F_b[3] = {0.0f, 0.0f, 0.0f};

      if (arm->type == 1u) {
        // Thruster: along dir_b.
        float dir_b[3] = { arm->dir_b[0], arm->dir_b[1], arm->dir_b[2] };
        _normalize3(dir_b);

        // Optional efficiency scaling based on rho, local airspeed, and temperature.
        float eff = 1.0f;

        // Local point velocity (body frame): v + omega x r.
        float wxr[3];
        _cross3(omega_b, r_b, wxr);
        float v_loc[3] = { v_b[0] + wxr[0], v_b[1] + wxr[1], v_b[2] + wxr[2] };
        float sp = _norm3(v_loc);

        // rho scaling: rho^p
        if (arm->eff_rho_pow > 1e-6f) {
          eff *= powf(_clampf(rho, 0.0f, 1.0f), arm->eff_rho_pow);
        }

        // speed scaling: 1/(1 + (sp/ref)^p)
        if (arm->eff_speed_ref > 1e-6f && arm->eff_speed_pow > 1e-6f) {
          float x = sp / arm->eff_speed_ref;
          eff *= 1.0f / (1.0f + powf(fmaxf(0.0f, x), arm->eff_speed_pow));
        }

        // temperature state + overheat scaling.
        float temp = st->arm_temp[i];
        if (arm->temp_heat_rate > 0.0f || arm->temp_cool_rate > 0.0f) {
          temp += dt * (arm->temp_heat_rate * fabsf(input) - arm->temp_cool_rate * temp);
          if (!isfinite(temp)) temp = 0.0f;
          temp = fmaxf(0.0f, temp);
          st->arm_temp[i] = temp;
        }
        float temp_eff = 1.0f;
        if (arm->temp_overheat_end > arm->temp_overheat_start && temp > arm->temp_overheat_start) {
          float t = (temp - arm->temp_overheat_start) / (arm->temp_overheat_end - arm->temp_overheat_start);
          temp_eff = _clampf(1.0f - t, 0.0f, 1.0f);
        }
        eff = _clampf(eff * temp_eff, 0.0f, 1.0f);
        st->arm_eff[i] = eff;

        float mag = arm->max_force * input;
        // If this thruster is driven by throttle/strafe, interpret max_force as a scale
        // over the legacy translation force magnitude.
        if ((arm->flags & 1u) != 0u) {
          mag = arm->max_force * (mass * ctl.thrust_accel * input);
        } else if ((arm->flags & 2u) != 0u) {
          mag = arm->max_force * (mass * ctl.strafe_accel * input);
        }
        mag *= eff;
        F_b[0] = dir_b[0] * mag;
        F_b[1] = dir_b[1] * mag;
        F_b[2] = dir_b[2] * mag;
      } else if (arm->type == 2u) {
        // Aero surface: lift along axis_b projected perpendicular to local airflow.
        // Local point velocity: v + omega x r.
        float wxr[3];
        _cross3(omega_b, r_b, wxr);
        float v_loc[3] = { v_b[0] + wxr[0], v_b[1] + wxr[1], v_b[2] + wxr[2] };
        float sp = _norm3(v_loc);
        if (sp > 1e-6f && rho > 0.0f) {
          float vhat[3] = { v_loc[0]/sp, v_loc[1]/sp, v_loc[2]/sp };
          float axis[3] = { arm->axis_b[0], arm->axis_b[1], arm->axis_b[2] };
          _normalize3(axis);
          // Project axis perpendicular to v.
          float proj[3];
          _scale3(proj, vhat, _dot3(axis, vhat));
          float lift_dir[3];
          _sub3(lift_dir, axis, proj);
          _normalize3(lift_dir);
          float qdyn = rho * sp * sp;
          float lift_mag = arm->k_lift * qdyn * input;
          float drag_mag = arm->k_drag * qdyn;
          F_b[0] += lift_dir[0] * lift_mag;
          F_b[1] += lift_dir[1] * lift_mag;
          F_b[2] += lift_dir[2] * lift_mag;
          F_b[0] += -vhat[0] * drag_mag;
          F_b[1] += -vhat[1] * drag_mag;
          F_b[2] += -vhat[2] * drag_mag;
        }
      } else if (arm->type == 3u) {
        // Drag center: pure drag opposing airflow at that point.
        float wxr[3];
        _cross3(omega_b, r_b, wxr);
        float v_loc[3] = { v_b[0] + wxr[0], v_b[1] + wxr[1], v_b[2] + wxr[2] };
        float sp = _norm3(v_loc);
        if (sp > 1e-6f && rho > 0.0f) {
          float vhat[3] = { v_loc[0]/sp, v_loc[1]/sp, v_loc[2]/sp };
          float qdyn = rho * sp * sp;
          float drag_mag = arm->k_drag * qdyn;
          F_b[0] += -vhat[0] * drag_mag;
          F_b[1] += -vhat[1] * drag_mag;
          F_b[2] += -vhat[2] * drag_mag;
        }
      }

      // Torque about COM in body frame.
      float tx[3];
      _cross3(r_b, F_b, tx);
      tau_b[0] += tx[0];
      tau_b[1] += tx[1];
      tau_b[2] += tx[2];

      // Convert force to world and accumulate.
      float Fw_i[3];
      _mat3_mul_vec(R, F_b, Fw_i);
      Fw[0] += Fw_i[0];
      Fw[1] += Fw_i[1];
      Fw[2] += Fw_i[2];
    }
  }

  // Gravity (world).
  if (ctl.gravity_g != 0.0f) {
    Fw[0] += (-up_rad[0]) * (mass * ctl.gravity_g);
    Fw[1] += (-up_rad[1]) * (mass * ctl.gravity_g);
    Fw[2] += (-up_rad[2]) * (mass * ctl.gravity_g);
  }

  // Linear acceleration.
  vel[0] += (Fw[0] / mass) * dt;
  vel[1] += (Fw[1] / mass) * dt;
  vel[2] += (Fw[2] / mass) * dt;

  // Angular dynamics in body frame.
  // When no arms exist, we preserve the legacy auto-mode clamp (above) by not
  // re-integrating quaternion from omega.
  if (n_arms > 0u) {
    float Ix = (b->cfg.inertia_diag[0] > 1e-6f) ? b->cfg.inertia_diag[0] : 1.0f;
    float Iy = (b->cfg.inertia_diag[1] > 1e-6f) ? b->cfg.inertia_diag[1] : 1.0f;
    float Iz = (b->cfg.inertia_diag[2] > 1e-6f) ? b->cfg.inertia_diag[2] : 1.0f;
    // Euler rigid body: I*w_dot = tau - w x (I*w)
    float Iw[3] = { Ix*omega_b[0], Iy*omega_b[1], Iz*omega_b[2] };
    float wxIw[3];
    _cross3(omega_b, Iw, wxIw);
    float wdot[3] = {
      (tau_b[0] - wxIw[0]) / Ix,
      (tau_b[1] - wxIw[1]) / Iy,
      (tau_b[2] - wxIw[2]) / Iz,
    };
    omega_b[0] += wdot[0] * dt;
    omega_b[1] += wdot[1] * dt;
    omega_b[2] += wdot[2] * dt;

    // Optional max rate clamp.
    if (ctl.max_ang_rate > 1e-6f) {
      float wn = sqrtf(omega_b[0]*omega_b[0] + omega_b[1]*omega_b[1] + omega_b[2]*omega_b[2]);
      if (wn > ctl.max_ang_rate) {
        float s = ctl.max_ang_rate / wn;
        omega_b[0] *= s;
        omega_b[1] *= s;
        omega_b[2] *= s;
      }
    }

    // Integrate quaternion from omega_b (body rates): qdot = 0.5 * q ⊗ [0, omega_b]
    float wq[4] = {0.0f, omega_b[0], omega_b[1], omega_b[2]};
    float qdot[4];
    _quat_mul(q_cur, wq, qdot);
    q_new[0] = q_cur[0] + 0.5f * qdot[0] * dt;
    q_new[1] = q_cur[1] + 0.5f * qdot[1] * dt;
    q_new[2] = q_cur[2] + 0.5f * qdot[2] * dt;
    q_new[3] = q_cur[3] + 0.5f * qdot[3] * dt;
    _quat_normalize(q_new);
  }

  // Atmosphere-limited max speed.
  if (ctl.max_speed > 1e-6f) {
    float ms_eff = 0.0f;
    if (rho > 1e-4f) {
      ms_eff = ctl.max_speed / sqrtf(fmaxf(1e-6f, rho));
    }
    if (ms_eff > 1e-6f) {
      float sp = _norm3(vel);
      if (sp > ms_eff) {
        float s = ms_eff / sp;
        vel[0] *= s;
        vel[1] *= s;
        vel[2] *= s;
      }
    }
  }

  pos[0] += vel[0] * dt;
  pos[1] += vel[1] * dt;
  pos[2] += vel[2] * dt;

  // Dynamic radius band based on terrain displacement.
  float clearance_min = fmaxf(0.0f, b->cfg.flight_r_min - b->cfg.planet_surface_r);
  float clearance_max = fmaxf(0.0f, b->cfg.flight_r_max - b->cfg.planet_surface_r);
  float r_surface2 = _terrain_surface_r(&b->cfg, pos);
  float rmin = fmaxf(0.0f, r_surface2 + clearance_min);
  float rmax = fmaxf(rmin, r_surface2 + clearance_max);
  float r_before = _norm3(pos);
  float r_clamped = fmaxf(rmin, fminf(rmax, r_before));
  if (r_before > 1e-9f && fabsf(r_clamped - r_before) > 0.0f) {
    float s = r_clamped / r_before;
    pos[0] *= s;
    pos[1] *= s;
    pos[2] *= s;
    float up2[3] = { pos[0], pos[1], pos[2] };
    _normalize3(up2);
    float dv = _dot3(vel, up2);
    vel[0] -= up2[0] * dv;
    vel[1] -= up2[1] * dv;
    vel[2] -= up2[2] * dv;
  }

  // Write back.
  st->pos[0] = pos[0];
  st->pos[1] = pos[1];
  st->pos[2] = pos[2];
  st->vel[0] = vel[0];
  st->vel[1] = vel[1];
  st->vel[2] = vel[2];
  st->q[0] = q_new[0];
  st->q[1] = q_new[1];
  st->q[2] = q_new[2];
  st->q[3] = q_new[3];

  st->omega_b[0] = omega_b[0];
  st->omega_b[1] = omega_b[1];
  st->omega_b[2] = omega_b[2];

  st->rho = rho;
  st->r_surface = r_surface2;
  st->altitude = fmaxf(0.0f, _norm3(pos) - r_surface2);

  // Control-system telemetry.
  ctrl.tau_est_b[0] = tau_b[0];
  ctrl.tau_est_b[1] = tau_b[1];
  ctrl.tau_est_b[2] = tau_b[2];
  st->ctrl = ctrl;

  // Optional debug logging (rate-limited) to verify control mode/solver behavior.
  // Enable by setting ctl.flags bit1 from the caller.
  if ((ctl.flags & 2u) != 0u) {
    static float dbg_accum = 0.0f;
    dbg_accum += dt;
    if (dbg_accum >= 0.25f) {
      dbg_accum = 0.0f;
      const float sp = _norm3(vel);
      printf(
        "[flight] mode=%u desired=%d arms=%u sol=%u kp=%.3f kd=%.3f th_acc=%.3f g=%.3f ms=%.3f thr=%.2f str=%.2f sp=%.2f rho=%.2f req=%.4f appl=%.4f clamp=%.3f err=%.4f cmd(p,y,r)=(%.3f,%.3f,%.3f) w=(%.3f,%.3f,%.3f) tau_des=(%.3f,%.3f,%.3f) tau_est=(%.3f,%.3f,%.3f)\n",
        (unsigned)(ctl.mode ? 1u : 0u),
        desired_valid,
        (unsigned)n_arms,
        (unsigned)((ctrl.flags & 1u) != 0u),
        (double)ctl.auto_kp,
        (double)ctl.auto_kd,
        (double)ctl.thrust_accel,
        (double)ctl.gravity_g,
        (double)ctl.max_speed,
        (double)throttle,
        (double)strafe,
        (double)sp,
        (double)rho,
        (double)requested_ang,
        (double)applied_ang,
        (double)clamp_ratio,
        (double)ctrl.err_angle,
        (double)channel_cmd[2],
        (double)channel_cmd[3],
        (double)channel_cmd[4],
        (double)omega_b[0],
        (double)omega_b[1],
        (double)omega_b[2],
        (double)ctrl.tau_des_b[0],
        (double)ctrl.tau_des_b[1],
        (double)ctrl.tau_des_b[2],
        (double)tau_b[0],
        (double)tau_b[1],
        (double)tau_b[2]
      );
      fflush(stdout);
    }
  }

  st->last_requested_ang = requested_ang;
  st->last_applied_ang = applied_ang;
  st->last_ang_clamp_ratio = clamp_ratio;
}

GP_EXPORT void gp_flight_buf_step(void* mem, float dt, uint32_t steps) {
  if (!mem) return;
  GP_FlightBuffer* b = (GP_FlightBuffer*)mem;
  if (b->hdr.magic != GP_FLIGHT_MAGIC || b->hdr.version != 4u) {
    return;
  }
  if (!(dt > 0.0f) || !isfinite(dt)) {
    return;
  }
  uint32_t n = (steps == 0u) ? 1u : steps;
  for (uint32_t i = 0; i < n; i++) {
    _flight_one_step(b, dt);
  }
  if (b->hdr.swap_requested) {
    uint32_t old_front = (uint32_t)(b->hdr.front_idx & 1u);
    uint32_t old_sim = (uint32_t)(b->hdr.sim_idx & 1u);
    uint32_t new_sim = old_front;
    // Copy latest sim state into the buffer that will become the new sim buffer.
    memcpy(&b->state[new_sim], &b->state[old_sim], sizeof(GP_FlightState));
    b->hdr.front_idx = old_sim;
    b->hdr.sim_idx = new_sim;
    b->hdr.swap_requested = 0;
    b->hdr.swap_seq += 1u;
  }
}

GP_EXPORT int gp_flight_step(const GP_FlightIn* in, GP_FlightOut* out) {
  if (!in || !out) {
    return 0;
  }

  const float dt = in->dt;
  if (!(dt > 0.0f) || !isfinite(dt)) {
    return 0;
  }

  float pos[3] = { in->pos[0], in->pos[1], in->pos[2] };
  float vel[3] = { in->vel[0], in->vel[1], in->vel[2] };

  const float throttle = fmaxf(-1.0f, fminf(1.0f, in->throttle));
  const float strafe = fmaxf(-1.0f, fminf(1.0f, in->strafe));
  const float rho = fmaxf(0.0f, fminf(1.0f, in->rho));

  float a[3] = {0.0f, 0.0f, 0.0f};

  if (fabsf(throttle) > 1e-7f && in->thrust_accel != 0.0f) {
    float tmp[3];
    _scale3(tmp, in->fwd_b, in->thrust_accel * throttle);
    _add3(a, tmp);
  }
  if (fabsf(strafe) > 1e-7f && in->strafe_accel != 0.0f) {
    float tmp[3];
    _scale3(tmp, in->right_b, in->strafe_accel * strafe);
    _add3(a, tmp);
  }

  float speed = _norm3(vel);
  if (speed > 1e-6f) {
    float vhat[3] = { vel[0]/speed, vel[1]/speed, vel[2]/speed };

    // AoA proxy: angle between craft forward axis and velocity direction.
    float aoa_cos = _clampf(_dot3(in->fwd_b, vhat), -1.0f, 1.0f);
    float aoa = acosf(aoa_cos);
    float lift_gate = sinf(aoa);

    // Lift direction: craft up projected perpendicular to velocity.
    float proj[3];
    _scale3(proj, vhat, _dot3(in->up_b, vhat));
    float lift_dir[3];
    _sub3(lift_dir, in->up_b, proj);
    _normalize3(lift_dir);

    if (in->lift_k != 0.0f && rho > 0.0f) {
      float k = in->lift_k * rho * speed * speed * lift_gate;
      float tmp[3];
      _scale3(tmp, lift_dir, k);
      _add3(a, tmp);
    }

    if (in->drag_k != 0.0f && rho > 0.0f) {
      float k = in->drag_k * rho * speed * speed;
      float tmp[3];
      _scale3(tmp, vhat, -k);
      _add3(a, tmp);
    }
  }

  if (in->gravity_g != 0.0f) {
    float tmp[3];
    _scale3(tmp, in->up_rad, -in->gravity_g);
    _add3(a, tmp);
  }

  // Semi-implicit Euler.
  vel[0] += a[0] * dt;
  vel[1] += a[1] * dt;
  vel[2] += a[2] * dt;

  // Atmosphere-limited max speed (same policy as Python).
  float ms = in->max_speed;
  if (ms > 1e-6f) {
    float ms_eff = 0.0f;
    if (rho > 1e-4f) {
      ms_eff = ms / sqrtf(fmaxf(1e-6f, rho));
    }
    if (ms_eff > 1e-6f) {
      float sp = _norm3(vel);
      if (sp > ms_eff) {
        float s = ms_eff / sp;
        vel[0] *= s;
        vel[1] *= s;
        vel[2] *= s;
      }
    }
  }

  pos[0] += vel[0] * dt;
  pos[1] += vel[1] * dt;
  pos[2] += vel[2] * dt;

  out->pos[0] = pos[0];
  out->pos[1] = pos[1];
  out->pos[2] = pos[2];
  out->vel[0] = vel[0];
  out->vel[1] = vel[1];
  out->vel[2] = vel[2];
  return 1;
}
