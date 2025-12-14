from __future__ import annotations

import os
import sys
import ctypes
from ctypes import c_uint32, c_int32, c_double, c_size_t

import numpy as np


_MAGIC = 0x48505047  # 'GPPH'


_SPRING_DTYPE = np.dtype(
    [
        ("i", np.uint32),
        ("j", np.uint32),
        ("rest_angle", np.float64),
        ("k", np.float64),
    ],
    align=False,
)


class GP_Header(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("n", c_uint32),
        ("n_active", c_uint32),
        ("n_active_sim", c_uint32),
        ("d", c_uint32),
        ("spring_cap", c_uint32),
        ("spring_count", c_uint32),
        ("spring_count_sim", c_uint32),
        ("history_cap", c_uint32),
        ("history_len", c_uint32),
        ("history_head", c_uint32),
        ("history_seq", c_uint32),
        ("front_idx", c_uint32),
        ("sim_idx", c_uint32),
        ("swap_requested", c_uint32),
        ("swap_seq", c_uint32),
        ("pair_count", c_uint32),
        ("bonds_enabled", c_uint32),
        ("bond_max_per_node", c_uint32),
        ("ionic_bonds", c_uint32),
        ("ionic_valence", c_uint32),
        ("bond_link_angle", c_double),
        ("bond_shear_ratio", c_double),
        ("bond_k", c_double),
        ("k_spring", c_double),
        ("k_coulomb", c_double),
        ("G", c_double),
        ("damping", c_double),
        ("softening", c_double),
        ("south_enabled", c_uint32),
        ("south_axis", c_int32),
        ("_pad_south", c_uint32),
        ("south_strength", c_double),
        ("lorentz_k", c_double),
        ("Bx", c_double),
        ("By", c_double),
        ("Bz", c_double),
        ("temp_ambient", c_double),
        ("temp_conv", c_double),
        ("temp_noise", c_double),
        ("temp_heat_gain", c_double),
        ("max_speed", c_double),
        ("radius_scale", c_double),
        ("mass_ref_cuberoot", c_double),
        ("collide_gain", c_double),
        ("merge_speed_frac", c_double),
        ("merge_size_frac", c_double),
        ("shatter_speed_frac", c_double),
        ("shatter_size_frac", c_double),
        ("shatter_k_max", c_uint32),
        ("collision_pair_limit", c_uint32),
        ("mass_vapor_thresh", c_double),
        ("condense_temp_thresh", c_double),
        ("condense_chunk_mass", c_double),
        ("vapor_mass", c_double),
        ("vapor_charge", c_double),
        ("accel_shatter_thresh", c_double),
        ("accel_shatter_fraction", c_double),
        ("accel_shatter_k", c_uint32),
        ("accel_shatter_kick", c_double),
        ("rng_state", c_uint32),
        ("_pad1", c_uint32),
    ]


class GP_Spring(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("i", c_uint32),
        ("j", c_uint32),
        ("rest_angle", c_double),
        ("k", c_double),
    ]


class GP_BallisticContact(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("i", c_uint32),
        ("j", c_uint32),
        ("t", c_double),
        ("dist", c_double),
        ("rel_speed", c_double),
    ]


class GP_SegmentContact(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("sa", c_uint32),
        ("sb", c_uint32),
        ("t", c_double),
        ("dist", c_double),
    ]


def _align8(x: int) -> int:
    return (x + 7) & ~7


def _default_dll_name() -> str:
    return "geodesic_physics.dll" if os.name == "nt" else "libgeodesic_physics.so"


def load_lib(search_dir: str | None = None) -> ctypes.CDLL:
    here = os.path.dirname(__file__)
    dll_name = _default_dll_name()
    candidates = []
    if search_dir:
        candidates.append(os.path.join(search_dir, dll_name))
    # Prefer build outputs (avoids issues overwriting a loaded DLL on Windows).
    candidates.append(os.path.join(here, "build", "Release", dll_name))
    candidates.append(os.path.join(here, "build", "Debug", dll_name))
    candidates.append(os.path.join(here, dll_name))
    candidates.append(os.path.join(os.getcwd(), dll_name))

    last_err: Exception | None = None
    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                break
            except Exception as e:  # pragma: no cover
                last_err = e
    else:
        raise FileNotFoundError(
            f"Could not find {dll_name}. Build it in {here} or pass search_dir.\n"
            f"Tried: {candidates}\n"
            f"Last error: {last_err}"
        )

    lib.gp_required_bytes.argtypes = [c_uint32, c_uint32, c_uint32]
    lib.gp_required_bytes.restype = c_size_t

    # v6+ extended init with history ring-buffer preallocation
    try:
        _required_bytes_ex = lib.gp_required_bytes_ex
        _init_ex = lib.gp_init_ex
    except AttributeError:
        _required_bytes_ex = None
        _init_ex = None
    if _required_bytes_ex is not None:
        _required_bytes_ex.argtypes = [c_uint32, c_uint32, c_uint32, c_uint32]
        _required_bytes_ex.restype = c_size_t
    if _init_ex is not None:
        _init_ex.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_uint32, c_uint32]
        _init_ex.restype = ctypes.c_int

    lib.gp_init.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_uint32]
    lib.gp_init.restype = ctypes.c_int

    # Header ABI v5+ requires gp_set_n_active. If the symbol is missing, the user
    # is likely running an older DLL; fail fast with an actionable message.
    try:
        _set_n_active = lib.gp_set_n_active
    except AttributeError as e:
        raise RuntimeError(
            "geodesic_physics.dll is out of date (missing gp_set_n_active / header v5). "
            "Rebuild the DLL in c_physics (e.g. run c_physics\\build_geodesic_physics.ps1)."
        ) from e
    _set_n_active.argtypes = [ctypes.c_void_p, c_uint32]
    _set_n_active.restype = ctypes.c_int

    lib.gp_set_params.argtypes = [ctypes.c_void_p, c_double, c_double, c_double, c_double, c_double]
    lib.gp_set_params.restype = ctypes.c_int

    if hasattr(lib, "gp_set_south"):
        lib.gp_set_south.argtypes = [ctypes.c_void_p, c_uint32, c_double, c_int32]
        lib.gp_set_south.restype = ctypes.c_int

    lib.gp_set_springs.argtypes = [ctypes.c_void_p, ctypes.POINTER(GP_Spring), c_uint32]
    lib.gp_set_springs.restype = ctypes.c_int

    lib.gp_step.argtypes = [ctypes.c_void_p, c_double, c_uint32]
    lib.gp_step.restype = None

    lib.gp_request_swap.argtypes = [ctypes.c_void_p]
    lib.gp_request_swap.restype = None

    lib.gp_get_swap_seq.argtypes = [ctypes.c_void_p]
    lib.gp_get_swap_seq.restype = c_uint32

    lib.gp_set_lorentz.argtypes = [ctypes.c_void_p, c_double, c_double, c_double, c_double]
    lib.gp_set_lorentz.restype = ctypes.c_int

    lib.gp_set_temp_params.argtypes = [ctypes.c_void_p, c_double, c_double, c_double, c_double]
    lib.gp_set_temp_params.restype = ctypes.c_int

    if hasattr(lib, "gp_set_lifecycle_params"):
        lib.gp_set_lifecycle_params.argtypes = [
            ctypes.c_void_p,
            c_double,  # max_speed
            c_double,  # radius_scale
            c_double,  # mass_ref_cuberoot
            c_double,  # collide_gain
            c_double,  # merge_speed_frac
            c_double,  # merge_size_frac
            c_double,  # shatter_speed_frac
            c_double,  # shatter_size_frac
            c_uint32,  # shatter_k_max
            c_uint32,  # collision_pair_limit
            c_double,  # mass_vapor_thresh
            c_double,  # condense_temp_thresh
            c_double,  # condense_chunk_mass
            c_double,  # accel_shatter_thresh
            c_double,  # accel_shatter_fraction
            c_uint32,  # accel_shatter_k
            c_double,  # accel_shatter_kick
        ]
        lib.gp_set_lifecycle_params.restype = ctypes.c_int

    lib.gp_seed.argtypes = [ctypes.c_void_p, c_uint32]
    lib.gp_seed.restype = ctypes.c_int

    lib.gp_set_bond_params.argtypes = [
        ctypes.c_void_p,
        c_uint32,
        c_uint32,
        c_double,
        c_double,
        c_double,
        c_uint32,
        c_uint32,
    ]
    lib.gp_set_bond_params.restype = ctypes.c_int

    lib.gp_get_springs_ptr.argtypes = [ctypes.c_void_p]
    lib.gp_get_springs_ptr.restype = c_size_t

    lib.gp_get_spring_count.argtypes = [ctypes.c_void_p]
    lib.gp_get_spring_count.restype = c_uint32

    lib.gp_get_spring_cap.argtypes = [ctypes.c_void_p]
    lib.gp_get_spring_cap.restype = c_uint32

    # History exports (v6+)
    if hasattr(lib, "gp_get_history_cap"):
        lib.gp_get_history_cap.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_cap.restype = c_uint32

        lib.gp_get_history_len.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_len.restype = c_uint32

        lib.gp_get_history_head.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_head.restype = c_uint32

        lib.gp_get_history_seq.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_seq.restype = c_uint32

        lib.gp_history_age_to_index.argtypes = [ctypes.c_void_p, c_uint32]
        lib.gp_history_age_to_index.restype = c_uint32

        lib.gp_get_history_pos_ptr.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_pos_ptr.restype = c_size_t

        lib.gp_get_history_vel_ptr.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_vel_ptr.restype = c_size_t

        lib.gp_get_history_dt_ptr.argtypes = [ctypes.c_void_p]
        lib.gp_get_history_dt_ptr.restype = c_size_t

    # Collision exports (v6+)
    if hasattr(lib, "gp_detect_point_overlaps"):
        lib.gp_detect_point_overlaps.argtypes = [
            ctypes.c_void_p,
            c_uint32,
            c_uint32,
            c_double,
            ctypes.POINTER(GP_BallisticContact),
            c_uint32,
        ]
        lib.gp_detect_point_overlaps.restype = c_uint32

    if hasattr(lib, "gp_detect_point_overlaps_geodesic"):
        lib.gp_detect_point_overlaps_geodesic.argtypes = [
            ctypes.c_void_p,
            c_uint32,
            c_uint32,
            c_double,
            c_uint32,
            ctypes.POINTER(GP_BallisticContact),
            c_uint32,
        ]
        lib.gp_detect_point_overlaps_geodesic.restype = c_uint32

    if hasattr(lib, "gp_detect_spring_proximity"):
        lib.gp_detect_spring_proximity.argtypes = [
            ctypes.c_void_p,
            c_uint32,
            c_uint32,
            c_double,
            c_uint32,
            ctypes.POINTER(GP_SegmentContact),
            c_uint32,
        ]
        lib.gp_detect_spring_proximity.restype = c_uint32

    return lib


class GeodesicPhysicsState:
    def __init__(
        self,
        n: int,
        d: int,
        spring_cap: int = 0,
        lib: ctypes.CDLL | None = None,
        n_cap: int | None = None,
        history_cap: int = 0,
    ):
        if n <= 0 or d < 2:
            raise ValueError("n must be >0 and d must be >=2")
        self.n_active0 = int(n)
        self.d = int(d)
        self.spring_cap = int(spring_cap)
        self.n_cap = int(n_cap) if (n_cap is not None) else int(n)
        self.history_cap = int(history_cap)
        if self.history_cap < 0:
            self.history_cap = 0
        if self.n_cap < self.n_active0:
            self.n_cap = self.n_active0

        self.lib = lib or load_lib()
        n_u32 = c_uint32(self.n_cap)
        d_u32 = c_uint32(self.d)
        cap_u32 = c_uint32(self.spring_cap)

        # Prefer v6+ extended sizing/init to preallocate history.
        if hasattr(self.lib, "gp_required_bytes_ex") and hasattr(self.lib, "gp_init_ex"):
            self.nbytes = int(self.lib.gp_required_bytes_ex(n_u32, d_u32, cap_u32, c_uint32(self.history_cap)))
        else:
            self.nbytes = int(self.lib.gp_required_bytes(n_u32, d_u32, cap_u32))
        self.buf = ctypes.create_string_buffer(self.nbytes)

        if hasattr(self.lib, "gp_init_ex"):
            ok = self.lib.gp_init_ex(ctypes.byref(self.buf), n_u32, d_u32, cap_u32, c_uint32(self.history_cap))
        else:
            ok = self.lib.gp_init(ctypes.byref(self.buf), n_u32, d_u32, cap_u32)
        if not ok:
            raise RuntimeError("gp_init failed")

        # Set active node count (<= capacity).
        ok = self.lib.gp_set_n_active(ctypes.byref(self.buf), c_uint32(self.n_active0))
        if not ok:
            raise RuntimeError("gp_set_n_active failed")

        self._header_p = ctypes.cast(ctypes.byref(self.buf), ctypes.POINTER(GP_Header))
        if self._header_p.contents.magic != _MAGIC:
            raise RuntimeError("bad header magic")
        if int(self._header_p.contents.version) != 9:
            raise RuntimeError(
                f"Unsupported C state ABI version {int(self._header_p.contents.version)} (expected 9). "
                "Rebuild geodesic_physics.dll to match this Python wrapper (c_physics/build_geodesic_physics.ps1)."
            )

        # Create stable numpy views for both ping-pong buffers.
        off = _align8(ctypes.sizeof(GP_Header))
        nd = self.n_cap * self.d
        base_ptr = ctypes.addressof(self.buf) + off

        def view_double(offset_doubles: int, shape):
            ptr = ctypes.cast(base_ptr + offset_doubles * ctypes.sizeof(c_double), ctypes.POINTER(c_double))
            arr = np.ctypeslib.as_array(ptr, shape=(int(np.prod(shape)),))
            return arr.reshape(shape)

        self.pos = [
            view_double(0, (self.n_cap, self.d)),
            view_double(nd, (self.n_cap, self.d)),
        ]
        self.vel = [
            view_double(2 * nd, (self.n_cap, self.d)),
            view_double(3 * nd, (self.n_cap, self.d)),
        ]
        self.mass = view_double(4 * nd, (self.n_cap,))
        self.charge = view_double(4 * nd + self.n_cap, (self.n_cap,))
        self.temps = view_double(4 * nd + 2 * self.n_cap, (self.n_cap,))

        # Optional history views (v6+); None if not present or not allocated.
        self.history_pos: np.ndarray | None = None
        self.history_vel: np.ndarray | None = None
        self.history_dt: np.ndarray | None = None
        if hasattr(self.lib, "gp_get_history_cap") and int(self.header.history_cap) > 0:
            hcap = int(self.header.history_cap)
            nd_cap = self.n_cap * self.d

            pos_ptr = int(self.lib.gp_get_history_pos_ptr(ctypes.byref(self.buf)))
            vel_ptr = int(self.lib.gp_get_history_vel_ptr(ctypes.byref(self.buf)))
            dt_ptr = int(self.lib.gp_get_history_dt_ptr(ctypes.byref(self.buf)))
            if pos_ptr and vel_ptr and dt_ptr:
                pos_arr = np.ctypeslib.as_array(ctypes.cast(pos_ptr, ctypes.POINTER(c_double)), shape=(hcap * nd_cap,))
                vel_arr = np.ctypeslib.as_array(ctypes.cast(vel_ptr, ctypes.POINTER(c_double)), shape=(hcap * nd_cap,))
                dt_arr = np.ctypeslib.as_array(ctypes.cast(dt_ptr, ctypes.POINTER(c_double)), shape=(hcap,))
                self.history_pos = pos_arr.reshape((hcap, self.n_cap, self.d))
                self.history_vel = vel_arr.reshape((hcap, self.n_cap, self.d))
                self.history_dt = dt_arr

        # NOTE: The C state also contains scratch + springs after this; we don't need
        # Python views for those to stream positions directly into OpenGL.

    @property
    def header(self) -> GP_Header:
        # Always read live values (front_idx, params) from shared memory.
        return self._header_p.contents

    @property
    def front_idx(self) -> int:
        return int(self.header.front_idx & 1)

    @property
    def n_active(self) -> int:
        # Published count (safe to pair with front_idx).
        return int(self.header.n_active)

    @property
    def n_active_sim(self) -> int:
        # Writer/simulation count (may differ until next swap).
        return int(getattr(self.header, "n_active_sim", self.header.n_active))

    @property
    def swap_seq(self) -> int:
        return int(self.header.swap_seq)

    def request_swap(self) -> None:
        """Ask the writer to swap front/back at a step boundary."""
        self.lib.gp_request_swap(ctypes.byref(self.buf))

    def history_index(self, age: int) -> int:
        if not hasattr(self.lib, "gp_history_age_to_index"):
            raise RuntimeError("history API not available in loaded DLL")
        idx = int(self.lib.gp_history_age_to_index(ctypes.byref(self.buf), c_uint32(int(age))))
        if idx == 0xFFFFFFFF:
            raise IndexError(f"history age {age} out of range")
        return idx

    def detect_point_overlaps(
        self,
        *,
        threshold: float,
        age_old: int = 1,
        age_new: int = 0,
        out_cap: int = 4096,
    ) -> np.ndarray:
        if not hasattr(self.lib, "gp_detect_point_overlaps"):
            raise RuntimeError("collision API not available in loaded DLL")
        out_cap_i = int(out_cap)
        if out_cap_i <= 0:
            return np.zeros((0,), dtype=np.dtype([(f, np.float64) for f in ("i", "j", "t", "dist", "rel_speed")]))
        out_buf = (GP_BallisticContact * out_cap_i)()
        n = int(
            self.lib.gp_detect_point_overlaps(
                ctypes.byref(self.buf),
                c_uint32(int(age_old)),
                c_uint32(int(age_new)),
                c_double(float(threshold)),
                out_buf,
                c_uint32(out_cap_i),
            )
        )
        n = min(n, out_cap_i)
        if n <= 0:
            return np.zeros((0,), dtype=np.dtype([("i", np.uint32), ("j", np.uint32), ("t", np.float64), ("dist", np.float64), ("rel_speed", np.float64)]))
        arr = np.ctypeslib.as_array(out_buf, shape=(n,))
        return arr.copy()

    def detect_point_overlaps_geodesic(
        self,
        *,
        angle_threshold: float,
        samples: int = 5,
        age_old: int = 1,
        age_new: int = 0,
        out_cap: int = 4096,
    ) -> np.ndarray:
        """Geodesic overlap check using on-sphere interpolation.

        `angle_threshold` is radians; returned `dist` is also radians.
        """
        if not hasattr(self.lib, "gp_detect_point_overlaps_geodesic"):
            raise RuntimeError("geodesic collision API not available in loaded DLL")
        out_cap_i = int(out_cap)
        if out_cap_i <= 0:
            return np.zeros((0,), dtype=np.dtype([("i", np.uint32), ("j", np.uint32), ("t", np.float64), ("dist", np.float64), ("rel_speed", np.float64)]))
        out_buf = (GP_BallisticContact * out_cap_i)()
        n = int(
            self.lib.gp_detect_point_overlaps_geodesic(
                ctypes.byref(self.buf),
                c_uint32(int(age_old)),
                c_uint32(int(age_new)),
                c_double(float(angle_threshold)),
                c_uint32(int(samples)),
                out_buf,
                c_uint32(out_cap_i),
            )
        )
        n = min(n, out_cap_i)
        if n <= 0:
            return np.zeros((0,), dtype=np.dtype([("i", np.uint32), ("j", np.uint32), ("t", np.float64), ("dist", np.float64), ("rel_speed", np.float64)]))
        arr = np.ctypeslib.as_array(out_buf, shape=(n,))
        return arr.copy()

    def detect_spring_proximity(
        self,
        *,
        threshold: float,
        samples: int = 3,
        age_old: int = 1,
        age_new: int = 0,
        out_cap: int = 4096,
    ) -> np.ndarray:
        if not hasattr(self.lib, "gp_detect_spring_proximity"):
            raise RuntimeError("collision API not available in loaded DLL")
        out_cap_i = int(out_cap)
        if out_cap_i <= 0:
            return np.zeros((0,), dtype=np.dtype([("sa", np.uint32), ("sb", np.uint32), ("t", np.float64), ("dist", np.float64)]))
        out_buf = (GP_SegmentContact * out_cap_i)()
        n = int(
            self.lib.gp_detect_spring_proximity(
                ctypes.byref(self.buf),
                c_uint32(int(age_old)),
                c_uint32(int(age_new)),
                c_double(float(threshold)),
                c_uint32(int(samples)),
                out_buf,
                c_uint32(out_cap_i),
            )
        )
        n = min(n, out_cap_i)
        if n <= 0:
            return np.zeros((0,), dtype=np.dtype([("sa", np.uint32), ("sb", np.uint32), ("t", np.float64), ("dist", np.float64)]))
        arr = np.ctypeslib.as_array(out_buf, shape=(n,))
        return arr.copy()

    def set_n_active(self, n_active: int) -> None:
        n_active_i = int(n_active)
        if n_active_i < 0:
            n_active_i = 0
        if n_active_i > self.n_cap:
            n_active_i = self.n_cap
        ok = self.lib.gp_set_n_active(ctypes.byref(self.buf), c_uint32(n_active_i))
        if not ok:
            raise RuntimeError("gp_set_n_active failed")

    def set_lifecycle(
        self,
        *,
        max_speed: float,
        radius_scale: float,
        mass_ref_cuberoot: float,
        collide_gain: float,
        merge_speed_frac: float,
        merge_size_frac: float,
        shatter_speed_frac: float,
        shatter_size_frac: float,
        shatter_k_max: int,
        collision_pair_limit: int,
        mass_vapor_thresh: float,
        condense_temp_thresh: float,
        condense_chunk_mass: float,
        accel_shatter_thresh: float,
        accel_shatter_fraction: float,
        accel_shatter_k: int,
        accel_shatter_kick: float,
    ) -> None:
        if not hasattr(self.lib, "gp_set_lifecycle_params"):
            raise RuntimeError("lifecycle API not available in loaded DLL")
        ok = self.lib.gp_set_lifecycle_params(
            ctypes.byref(self.buf),
            c_double(float(max_speed)),
            c_double(float(radius_scale)),
            c_double(float(mass_ref_cuberoot)),
            c_double(float(collide_gain)),
            c_double(float(merge_speed_frac)),
            c_double(float(merge_size_frac)),
            c_double(float(shatter_speed_frac)),
            c_double(float(shatter_size_frac)),
            c_uint32(int(shatter_k_max)),
            c_uint32(int(collision_pair_limit)),
            c_double(float(mass_vapor_thresh)),
            c_double(float(condense_temp_thresh)),
            c_double(float(condense_chunk_mass)),
            c_double(float(accel_shatter_thresh)),
            c_double(float(accel_shatter_fraction)),
            c_uint32(int(accel_shatter_k)),
            c_double(float(accel_shatter_kick)),
        )
        if not ok:
            raise RuntimeError("gp_set_lifecycle_params failed")
    def front_pos_ptr(self, *, n_active: int | None = None) -> tuple[int, int]:
        """Return (address, nbytes) for the current front position buffer.

        This is the most direct path to OpenGL: you can pass the pointer into
        `glBufferData`/`glBufferSubData` without converting through torch.
        """
        idx = self.front_idx
        rows = self.n_active if (n_active is None) else int(n_active)
        if rows < 0:
            rows = 0
        if rows > self.n_cap:
            rows = self.n_cap
        base = self.pos[idx][:rows]
        addr = int(base.__array_interface__["data"][0])
        return addr, int(base.nbytes)

    def front_vel_ptr(self, *, n_active: int | None = None) -> tuple[int, int]:
        idx = self.front_idx
        rows = self.n_active if (n_active is None) else int(n_active)
        if rows < 0:
            rows = 0
        if rows > self.n_cap:
            rows = self.n_cap
        base = self.vel[idx][:rows]
        addr = int(base.__array_interface__["data"][0])
        return addr, int(base.nbytes)

    def set_params(self, *, k_spring=1.0, k_coulomb=0.0, G=0.0, damping=0.0, softening=1e-6):
        ok = self.lib.gp_set_params(
            ctypes.byref(self.buf),
            float(k_spring),
            float(k_coulomb),
            float(G),
            float(damping),
            float(softening),
        )
        if not ok:
            raise RuntimeError("gp_set_params failed")

    def set_south(self, *, enabled: bool = False, strength: float = 0.0, axis: int = -1) -> None:
        if not hasattr(self.lib, "gp_set_south"):
            raise RuntimeError("C DLL is missing gp_set_south (rebuild c_physics DLL)")
        ok = self.lib.gp_set_south(
            ctypes.byref(self.buf),
            c_uint32(1 if enabled else 0),
            c_double(float(strength)),
            c_int32(int(axis)),
        )
        if not ok:
            raise RuntimeError("gp_set_south failed")

    def set_bonds(
        self,
        *,
        enabled: bool = True,
        max_per_node: int = 0,
        link_angle: float = 0.0,
        shear_ratio: float = 2.0,
        k: float = 1.0,
        ionic: bool = False,
        ionic_valence: int = 1,
    ) -> None:
        ok = self.lib.gp_set_bond_params(
            ctypes.byref(self.buf),
            c_uint32(1 if enabled else 0),
            c_uint32(int(max_per_node)),
            c_double(float(link_angle)),
            c_double(float(shear_ratio)),
            c_double(float(k)),
            c_uint32(1 if ionic else 0),
            c_uint32(int(ionic_valence)),
        )
        if not ok:
            raise RuntimeError("gp_set_bond_params failed")

    def set_springs(self, springs: np.ndarray):
        """springs: float/np array shape (E,4): i, j, rest_angle, k"""
        if springs is None:
            arr = np.zeros((0, 4), dtype=np.float64)
        else:
            arr = np.asarray(springs, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("springs must be (E,4)")
        if arr.shape[0] > self.spring_cap:
            raise ValueError("too many springs for spring_cap")

        springs_ct = (GP_Spring * arr.shape[0])()
        for idx in range(arr.shape[0]):
            springs_ct[idx].i = int(arr[idx, 0])
            springs_ct[idx].j = int(arr[idx, 1])
            springs_ct[idx].rest_angle = float(arr[idx, 2])
            springs_ct[idx].k = float(arr[idx, 3])
        ok = self.lib.gp_set_springs(ctypes.byref(self.buf), springs_ct, c_uint32(arr.shape[0]))
        if not ok:
            raise RuntimeError("gp_set_springs failed")

    def set_lorentz(self, *, lorentz_k: float = 0.0, Bx: float = 0.0, By: float = 0.0, Bz: float = 1.0):
        ok = self.lib.gp_set_lorentz(ctypes.byref(self.buf), float(lorentz_k), float(Bx), float(By), float(Bz))
        if not ok:
            raise RuntimeError("gp_set_lorentz failed")

    def set_temp(self, *, ambient: float = 0.0, convection: float = 0.0, noise: float = 0.0, heat_gain: float = 1.0):
        ok = self.lib.gp_set_temp_params(ctypes.byref(self.buf), float(ambient), float(convection), float(noise), float(heat_gain))
        if not ok:
            raise RuntimeError("gp_set_temp_params failed")

    def seed(self, seed: int = 0):
        ok = self.lib.gp_seed(ctypes.byref(self.buf), c_uint32(int(seed) & 0xFFFFFFFF))
        if not ok:
            raise RuntimeError("gp_seed failed")

    def step(self, dt: float, steps: int = 1):
        self.lib.gp_step(ctypes.byref(self.buf), float(dt), c_uint32(int(steps)))

    def get_front_views(self):
        idx = self.front_idx
        return self.pos[idx], self.vel[idx]

    def springs_view(self) -> np.ndarray:
        """Structured numpy view of the current spring list.

        Intended usage: call after a reader-driven swap (request_swap + wait for
        swap_seq to advance) to avoid races with the writer thread.

        Returns an array of dtype GP_Spring with length spring_count.
        """
        ptr = int(self.lib.gp_get_springs_ptr(ctypes.byref(self.buf)))
        if ptr == 0:
            return np.zeros((0,), dtype=_SPRING_DTYPE)
        cap = int(self.lib.gp_get_spring_cap(ctypes.byref(self.buf)))
        count = int(self.lib.gp_get_spring_count(ctypes.byref(self.buf)))
        if cap <= 0 or count <= 0:
            return np.zeros((0,), dtype=_SPRING_DTYPE)
        if count > cap:
            count = cap
        nbytes = cap * ctypes.sizeof(GP_Spring)
        raw = (ctypes.c_ubyte * nbytes).from_address(ptr)
        view = np.frombuffer(raw, dtype=_SPRING_DTYPE, count=cap)
        return view[:count]
