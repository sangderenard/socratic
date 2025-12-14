"""Small, headless demo of the C-backed double-buffer physics state.

This is intentionally minimal: it exercises the contiguous ctypes buffer and the
"flip on read" swap protocol without touching the OpenGL animator.
"""

from __future__ import annotations

import math
import numpy as np

from c_physics.geodesic_ctypes import GeodesicPhysicsState


def main():
    n, d = 64, 3
    # Give the C backend room to form/break bonds dynamically.
    s = GeodesicPhysicsState(n, d, spring_cap=2048)
    s.set_params(k_spring=0.0, k_coulomb=0.3, G=0.0, damping=0.02, softening=1e-4)
    # Enable temperature effects: some thermal noise + convection + damping heating.
    s.set_lorentz(lorentz_k=0.3, Bx=0.0, By=0.0, Bz=1.0)
    s.set_temp(ambient=0.5, convection=0.5, noise=0.5, heat_gain=1.0)
    s.seed(1)

    # Enable bond formation: close neighbors form bonds; overstretched bonds break.
    s.set_bonds(
        enabled=True,
        max_per_node=3,
        link_angle=0.55,   # radians
        shear_ratio=1.9,
        k=4.0,
        ionic=True,
        ionic_valence=2,
    )

    rng = np.random.default_rng(0)
    p0 = rng.standard_normal((n, d))
    p0 /= np.linalg.norm(p0, axis=1, keepdims=True) + 1e-9
    v0 = np.zeros((n, d))

    # Initialize both buffers so the writer can advance the back buffer.
    s.pos[0][:] = p0
    s.vel[0][:] = v0
    s.pos[1][:] = p0
    s.vel[1][:] = v0
    s.mass[:] = 1.0
    s.charge[:] = rng.choice([-1.0, 1.0], size=(n,))
    s.temps[:] = 0.5

    for t in range(300):
        # "Read" requests a swap; the writer performs it at the step boundary.
        s.request_swap()
        s.step(0.01, steps=1)
        p, v = s.get_front_views()
        if t % 60 == 0:
            sp = s.springs_view()
            # should remain (very close to) unit length
            radii = np.linalg.norm(p, axis=1)
            print(
                f"t={t:4d} front={s.front_idx} r_min={radii.min():.6f} r_max={radii.max():.6f} "
                f"v_rms={math.sqrt((v*v).mean()):.6f} temp_mean={float(s.temps.mean()):.3f} "
                f"springs={int(sp.shape[0])}"
            )


if __name__ == "__main__":
    main()
