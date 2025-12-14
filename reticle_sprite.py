from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pygame


class ReticleStage(str, Enum):
    IDLE = "idle"  # grey circle
    HOVER = "hover"  # green circle
    LOCKED = "locked"  # yellow circle + yellow triangle
    READY = "ready"  # red rectangle


@dataclass(frozen=True)
class ReticleAssets:
    surfaces: dict[ReticleStage, pygame.Surface]
    rgba_u8: dict[ReticleStage, np.ndarray]
    tensors: dict[ReticleStage, Any]
    size_px: int


def _surf_to_rgba_u8(surf: pygame.Surface) -> np.ndarray:
    w, h = surf.get_size()
    raw = pygame.image.tostring(surf, "RGBA", True)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return arr


def _maybe_torch_tensor(arr: np.ndarray):
    try:
        import torch  # type: ignore

        return torch.from_numpy(arr)
    except Exception:
        return arr


def _mk_surface(size_px: int) -> pygame.Surface:
    return pygame.Surface((int(size_px), int(size_px)), flags=pygame.SRCALPHA)


def _draw_circle(surface: pygame.Surface, *, color_rgba: tuple[int, int, int, int], radius: int, width: int) -> None:
    cx = surface.get_width() // 2
    cy = surface.get_height() // 2
    pygame.draw.circle(surface, color_rgba, (cx, cy), int(radius), int(width))


def _draw_triangle(surface: pygame.Surface, *, color_rgba: tuple[int, int, int, int]) -> None:
    w, h = surface.get_size()
    cx = w // 2
    top = int(0.18 * h)
    left = int(0.26 * w)
    right = int(0.74 * w)
    base = int(0.44 * h)
    pts = [(cx, top), (right, base), (left, base)]
    pygame.draw.polygon(surface, color_rgba, pts, 0)


def _draw_rect(surface: pygame.Surface, *, color_rgba: tuple[int, int, int, int]) -> None:
    w, h = surface.get_size()
    pad = int(0.18 * w)
    rect = pygame.Rect(pad, pad, w - 2 * pad, h - 2 * pad)
    pygame.draw.rect(surface, color_rgba, rect, width=int(max(1, 0.12 * w)))


def ensure_reticle_pngs(folder: str = "assets/reticle", *, size_px: int = 64) -> dict[ReticleStage, str]:
    os.makedirs(folder, exist_ok=True)

    paths = {
        ReticleStage.IDLE: os.path.join(folder, "reticle_idle.png"),
        ReticleStage.HOVER: os.path.join(folder, "reticle_hover.png"),
        ReticleStage.LOCKED: os.path.join(folder, "reticle_locked.png"),
        ReticleStage.READY: os.path.join(folder, "reticle_ready.png"),
    }

    # Generate missing sprites via pygame drawing (no extra deps).
    for stage, path in paths.items():
        if os.path.exists(path):
            continue

        surf = _mk_surface(size_px)
        surf.fill((0, 0, 0, 0))
        radius = int(0.34 * size_px)
        stroke = int(max(2, 0.08 * size_px))

        if stage == ReticleStage.IDLE:
            _draw_circle(surf, color_rgba=(160, 160, 160, 220), radius=radius, width=stroke)
        elif stage == ReticleStage.HOVER:
            _draw_circle(surf, color_rgba=(0, 210, 0, 230), radius=radius, width=stroke)
        elif stage == ReticleStage.LOCKED:
            _draw_circle(surf, color_rgba=(235, 235, 0, 235), radius=radius, width=stroke)
            _draw_triangle(surf, color_rgba=(235, 235, 0, 210))
        elif stage == ReticleStage.READY:
            _draw_rect(surf, color_rgba=(230, 30, 30, 235))
        else:
            _draw_circle(surf, color_rgba=(255, 255, 255, 220), radius=radius, width=stroke)

        try:
            pygame.image.save(surf, path)
        except Exception:
            # If saving fails, we'll still be able to use the in-memory surfaces
            # once loaded via load_reticle_assets (it will fall back to regenerate).
            pass

    return paths


def load_reticle_assets(folder: str = "assets/reticle", *, size_px: int = 64) -> ReticleAssets:
    paths = ensure_reticle_pngs(folder, size_px=size_px)

    surfaces: dict[ReticleStage, pygame.Surface] = {}
    rgba_u8: dict[ReticleStage, np.ndarray] = {}
    tensors: dict[ReticleStage, Any] = {}

    for stage, path in paths.items():
        surf = None
        try:
            surf = pygame.image.load(path).convert_alpha()
        except Exception:
            # Regenerate in-memory if load failed.
            tmp_paths = ensure_reticle_pngs(folder, size_px=size_px)
            try:
                surf = pygame.image.load(tmp_paths[stage]).convert_alpha()
            except Exception:
                surf = _mk_surface(size_px)
                surf.fill((0, 0, 0, 0))

        # Ensure expected size.
        if surf.get_width() != int(size_px) or surf.get_height() != int(size_px):
            surf = pygame.transform.smoothscale(surf, (int(size_px), int(size_px)))

        surfaces[stage] = surf
        arr = _surf_to_rgba_u8(surf)
        rgba_u8[stage] = arr
        tensors[stage] = _maybe_torch_tensor(arr)

    return ReticleAssets(surfaces=surfaces, rgba_u8=rgba_u8, tensors=tensors, size_px=int(size_px))


class ReticleAnimator:
    def __init__(self, *, lock_delay_s: float = 0.80, ready_delay_s: float = 1.60):
        self.lock_delay_s = float(lock_delay_s)
        self.ready_delay_s = float(ready_delay_s)
        self._on_since: float | None = None

    def update(self, *, on_target: bool, now_s: float) -> ReticleStage:
        if not on_target:
            self._on_since = None
            return ReticleStage.IDLE

        if self._on_since is None:
            self._on_since = float(now_s)
            return ReticleStage.HOVER

        dt = float(now_s) - float(self._on_since)
        if dt >= self.ready_delay_s:
            return ReticleStage.READY
        if dt >= self.lock_delay_s:
            return ReticleStage.LOCKED
        return ReticleStage.HOVER
