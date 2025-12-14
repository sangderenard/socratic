"""
OpenGL + pygame physics animator for socratic graphs.

Usage (expects pre-physics export from socratic_spring_export.py):
    python gl_animator.py prephysics.json

Controls:
    Space : pause/resume
    R     : reset to initial positions
    Q/ESC : quit
    P     : print simple stats

Physics:
    - Velocity-Verlet integrator (second-order)
    - Springs on graph edges using initial edge distances as rest lengths
    - Global repulsion between all node pairs (softened Coulomb-like)
    - Optional velocity damping
    - Simulation runs in native embedding dimension; rendering projects to 3D

Rendering:
    - Simple point/line rendering via PyOpenGL
    - Positions are 3D (embeddings are PCA-projected if higher dimensional)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import pygame
import pygame.font
from pygame.locals import DOUBLEBUF, OPENGL

try:
    import torch
except Exception:  # torch optional
    torch = None

try:
    from OpenGL.GL import (
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glBlendFunc,
        glDisable,
        glPushMatrix,
        glPopMatrix,
        glPointSize,
        glVertex3f,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LINES,
        GL_QUADS,
        GL_TRIANGLES,
        GL_POINTS,
        GL_PROJECTION,
        GL_MODELVIEW,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        GL_BLEND,
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_UNPACK_ALIGNMENT,
            glGetDoublev,
            glGetIntegerv,
            GL_MODELVIEW_MATRIX,
            GL_PROJECTION_MATRIX,
            GL_VIEWPORT,
    )
    from OpenGL.GLU import gluPerspective, gluLookAt
except Exception as exc:  # pragma: no cover - import/runtime guard
    print("PyOpenGL is required to run this animator.", file=sys.stderr)
    raise

def load_prephysics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    return nodes, edges


def compute_projection_basis(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, P) where P projects D-dim vectors to 3D; fixed over time."""
    _, D = vecs.shape
    mean = vecs.mean(axis=0, keepdims=True).astype(np.float32)
    if D == 3:
        P = np.eye(3, dtype=np.float32)
    elif D < 3:
        P = np.zeros((D, 3), dtype=np.float32)
        for i in range(D):
            P[i, i] = 1.0
    else:
        Xc = vecs - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:3].T.astype(np.float32)  # (D,3)
    return mean, P


def project_positions(vecs: np.ndarray, P: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Project high-D positions to 3D with an optional precomputed mean for consistent centering."""
    mean_use = mean if mean is not None else vecs.mean(axis=0, keepdims=True)
    return (vecs - mean_use) @ P
 

def build_sim_state(nodes: List[Dict], edges: List[Dict]):
    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    embeds = []
    masses = []
    colors = []
    fixed_mask = []
    for n in nodes:
        emb = np.asarray(n.get("embedding", []), dtype=float)
        embeds.append(emb)
        if "mass" in n:
            masses.append(float(n.get("mass", 1.0)))
        else:
            kind = n.get("kind", "concept")
            if kind == "sentence":
                masses.append(3.0)
            else:
                masses.append(1.0)
        fixed_mask.append(bool(n.get("fixed", False)))
        kind = n.get("kind", "concept")
        # simple kind-based colors
        if kind == "sentence":
            colors.append((1.0, 0.4, 0.2))
        elif kind == "modifier":
            colors.append((0.4, 0.8, 0.4))
        elif kind == "verb":
            colors.append((0.9, 0.9, 0.2))
        else:
            colors.append((0.3, 0.6, 1.0))

    embeds = np.vstack(embeds).astype(np.float32)
    if not np.isfinite(embeds).all():
        print("[warn] Non-finite embedding values detected; sanitizing to zeros")
        np.nan_to_num(embeds, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean, proj = compute_projection_basis(embeds)

    velocities = np.zeros_like(embeds)
    masses = np.asarray(masses, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    fixed_mask_arr = np.asarray(fixed_mask, dtype=bool)

    springs = []
    for e in edges:
        # opt-out: if an edge marks physics=False/0, skip adding a spring
        if e.get("physics") is False or e.get("force") is False:
            continue
        src = e.get("source")
        tgt = e.get("target")
        if src not in id_to_idx or tgt not in id_to_idx:
            continue
        i = id_to_idx[src]
        j = id_to_idx[tgt]
        # Rest length is the embedding-space distance so initial positions have zero spring energy
        rest = float(np.linalg.norm(embeds[i] - embeds[j]))
        k_edge = float(e.get("k", 1.0))
        springs.append((i, j, rest, k_edge))

    springs_arr = np.asarray(springs, dtype=np.float32) if springs else np.zeros((0, 4), dtype=np.float32)

    return node_ids, embeds, velocities, masses, colors, springs_arr, mean, proj, fixed_mask_arr


# ---------------------------------------------------------------------------
# Physics (Velocity-Verlet)
# ---------------------------------------------------------------------------

def compute_forces(
    pos: np.ndarray,
    springs: np.ndarray,
    neighbors: List[List[int]],
    pair_n1: np.ndarray,
    pair_n2: np.ndarray,
    pair_hub: np.ndarray,
    k_spring: float,
    k_rep: float,
    k_angle: float,
    temp: float,
    softening: float,
    F: np.ndarray,
    scratch: dict,
    extra_force_fn=None,
    temp_per_node: np.ndarray | None = None,
    force_scale_per_node: np.ndarray | None = None,
    k_edge_scale: np.ndarray | None = None,
):
    """
    Vectorized force computation.

    springs: (E,4) array with cols (i, j, rest, k_edge)
    scratch: preallocated buffers used in-place to avoid reallocs
    """
    F.fill(0.0)

    # --- springs (vectorized) ---
    if springs.size:
        i_idx = springs[:, 0].astype(np.int64)
        j_idx = springs[:, 1].astype(np.int64)
        rest = springs[:, 2]
        k_edge = springs[:, 3] if springs.shape[1] > 3 else 1.0
        if k_edge_scale is not None and k_edge_scale.size:
            k_edge = k_edge * k_edge_scale

        diff = scratch["diff_s"]
        np.subtract(pos[j_idx], pos[i_idx], out=diff)

        dist = scratch["dist_s"]
        np.einsum("ij,ij->i", diff, diff, out=dist)
        np.add(dist, 1e-12, out=dist)  # avoid div0
        np.sqrt(dist, out=dist)
        np.nan_to_num(dist, copy=False, nan=1e-6, posinf=1e6, neginf=1e6)
        dist[dist < 1e-6] = 1e-6  # clamp tiny/negative

        rest = np.nan_to_num(rest, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        stretch = scratch["stretch"]
        # Hooke: force proportional to (rest - dist); positive when compressed
        np.subtract(rest, dist, out=stretch)

        force_mag = scratch["force_mag"]
        force_mag.fill(0.0)
        np.divide(k_spring * k_edge * stretch, dist, out=force_mag, where=dist > 0)

        F_s = scratch["F_s"]
        # F_s = force_mag[:,None] * diff
        np.multiply(force_mag[:, None], diff, out=F_s)

        # Scatter-add to node forces
        np.add.at(F, i_idx, -F_s)
        np.add.at(F, j_idx, F_s)

    # --- angular spreading (vectorized over neighbor pairs per hub) ---
    if k_angle != 0.0 and pair_n1.size:
        v1 = pos[pair_n1] - pos[pair_hub]
        v2 = pos[pair_n2] - pos[pair_hub]
        norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9
        norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
        u1 = v1 / norm1
        u2 = v2 / norm2
        dot = np.sum(u1 * u2, axis=1, keepdims=True)
        Fi = -k_angle * dot * (u2 - dot * u1) / norm1
        Fj = -k_angle * dot * (u1 - dot * u2) / norm2
        np.add.at(F, pair_n1, Fi)
        np.add.at(F, pair_n2, Fj)
        np.add.at(F, pair_hub, -(Fi + Fj))

    # --- repulsion (all-pairs, vectorized) ---
    N = pos.shape[0]
    if k_rep != 0.0:
        diff_full = scratch["diff_full"]
        np.subtract(pos[:, None, :], pos[None, :, :], out=diff_full)

        dist2 = scratch["dist2_full"]
        np.einsum("ijk,ijk->ij", diff_full, diff_full, out=dist2)
        dist2 += softening

        inv_dist = scratch["inv_dist_full"]
        np.sqrt(dist2, out=inv_dist)
        np.reciprocal(inv_dist, out=inv_dist, where=inv_dist > 0)

        rep_mag = scratch["rep_mag_full"]
        np.multiply(k_rep, inv_dist * inv_dist, out=rep_mag)

        # force = rep_mag * diff * inv_dist
        force_full = scratch["force_full"]
        np.multiply(diff_full, rep_mag[..., None] * inv_dist[..., None], out=force_full)

        # zero diagonal to avoid self-force
        diag_idx = np.arange(N)
        force_full[diag_idx, diag_idx, :] = 0.0

        # sum over j
        F += force_full.sum(axis=1)

    # --- thermal noise ---
    if temp_per_node is not None:
        noise = scratch["temp_noise"]
        noise[:] = np.random.standard_normal(size=pos.shape).astype(np.float32)
        noise *= temp_per_node[:, None] * math.sqrt(scratch.get("dt", 1.0))
        F += noise
    elif temp > 0.0:
        noise = scratch["temp_noise"]
        noise[:] = np.random.standard_normal(size=pos.shape).astype(np.float32)
        # scale by sqrt(dt) to make noise time-step stable
        noise *= temp * math.sqrt(scratch.get("dt", 1.0))
        F += noise

    # --- custom extra forces (optional) ---
    if extra_force_fn is not None:
        extra = extra_force_fn(pos, scratch)
        if extra is not None:
            F += extra

    if force_scale_per_node is not None:
        F *= force_scale_per_node[:, None]

    return F


def step_verlet(pos, vel, mass, springs, neighbors, pair_n1, pair_n2, pair_hub, dt, k_spring, k_rep, temp, softening, damping, scratch, extra_force_fn=None, temp_per_node=None, force_scale_per_node=None, k_edge_scale=None):
    scratch["dt"] = dt
    inv_mass = scratch["inv_mass"]
    np.divide(1.0, mass, out=inv_mass)
    inv_mass = inv_mass[:, None]

    F = scratch["F"]
    compute_forces(pos, springs, neighbors, pair_n1, pair_n2, pair_hub, k_spring, k_rep, scratch.get("k_angle", 0.0), temp, softening, F, scratch, extra_force_fn, temp_per_node=temp_per_node, force_scale_per_node=force_scale_per_node, k_edge_scale=k_edge_scale)
    acc = scratch["acc"]
    np.multiply(F, inv_mass, out=acc)

    vel += 0.5 * dt * acc
    pos += dt * vel

    compute_forces(pos, springs, neighbors, pair_n1, pair_n2, pair_hub, k_spring, k_rep, scratch.get("k_angle", 0.0), temp, softening, F, scratch, extra_force_fn, temp_per_node=temp_per_node, force_scale_per_node=force_scale_per_node, k_edge_scale=k_edge_scale)
    np.multiply(F, inv_mass, out=acc)
    vel += 0.5 * dt * acc

    if isinstance(damping, np.ndarray):
        vel *= (1.0 - damping)[:, None]
    elif damping > 0:
        vel *= (1.0 - damping)
    return pos, vel


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def init_gl(width: int, height: int, fov_deg: float = 45.0, z_near: float = 0.1, z_far: float = 1000.0):
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.08, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = (float(width) / float(height)) if float(height) != 0.0 else 1.0
    gluPerspective(float(fov_deg), aspect, float(z_near), float(z_far))
    glMatrixMode(GL_MODELVIEW)


def draw_hud(font, fps_render: float, fps_phys: float, text_lines=None, text_lines_right=None):
    """Draw FPS + optional text lines without clearing the frame."""
    # HUD must always render on top of the 3D scene.
    # glDrawPixels is depth-tested if GL_DEPTH_TEST is enabled, which can cause
    # HUD text to be clipped by whatever was written to the depth buffer.
    from OpenGL.GL import glIsEnabled, glDepthMask, glGetBooleanv, GL_DEPTH_TEST, GL_DEPTH_WRITEMASK
    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    depth_mask_prev = bool(glGetBooleanv(GL_DEPTH_WRITEMASK))
    if depth_was_enabled:
        glDisable(GL_DEPTH_TEST)
    glDepthMask(False)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    def render_cell_text(text: str):
        # Render monospace text with an individual black background cell
        # behind every character (including spaces), for readability.
        s = str(text).expandtabs(4)
        # Use a stable cell size; Consolas is monospace, but this also works
        # reasonably for other fonts.
        try:
            cell_w, cell_h = font.size("M")
        except Exception:
            cell_w, cell_h = (10, font.get_height())
        cell_w = max(1, int(cell_w))
        cell_h = max(1, int(cell_h))
        if len(s) == 0:
            surf = pygame.Surface((cell_w, cell_h), flags=pygame.SRCALPHA)
            surf.fill((0, 0, 0, 255))
            return surf
        surf = pygame.Surface((cell_w * len(s), cell_h), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for i, ch in enumerate(s):
            x = int(i * cell_w)
            pygame.draw.rect(surf, (0, 0, 0, 255), pygame.Rect(x, 0, cell_w, cell_h))
            if ch != " ":
                glyph = font.render(ch, True, (255, 255, 255))
                try:
                    glyph = glyph.convert_alpha()
                except Exception:
                    pass
                gx, gy = glyph.get_size()
                off_x = max(0, (cell_w - gx) // 2)
                off_y = max(0, (cell_h - gy) // 2)
                surf.blit(glyph, (x + off_x, off_y))
        return surf

    def draw_text(x0, y0, text):
        surf = render_cell_text(text)
        text_data = pygame.image.tostring(surf, "RGBA", True)
        from OpenGL.GL import glRasterPos2f, glDrawPixels, glPixelStorei, glBlendFunc as _glBlendFunc_lbl, glDisable as _glDisable_lbl
        glEnable(GL_BLEND)
        _glBlendFunc_lbl(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glRasterPos2f(x0, y0)
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        _glDisable_lbl(GL_BLEND)

    # fps overlay (top-right)
    draw_text(0.78, 0.96, f"render fps: {fps_render:.1f}")
    draw_text(0.78, 0.94, f"physics fps: {fps_phys:.1f}")

    # live text lines (left side)
    if text_lines:
        y = 0.88
        for line in text_lines[:18]:
            draw_text(0.02, y, str(line)[:200])
            y -= 0.03

    # optional live text lines (right side)
    if text_lines_right:
        y = 0.88
        for line in text_lines_right[:18]:
            draw_text(0.56, y, str(line)[:200])
            y -= 0.03

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    # Restore depth state.
    glDepthMask(bool(depth_mask_prev))
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)


def draw_scene(pos_nd, pos3d, colors, springs, center, eye, font, fps_render, fps_phys, text_lines=None, node_labels=None, proj_mats=None, highlight_idx=None, edge_colors=None, thrust_lines=None, sentence_triangles=None, point_sizes=None, draw_nodes: bool = True, pre_draw_fn=None, draw_hud_now: bool = True):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # ensure blend functions are available (defensive against partial GL imports)
    from OpenGL.GL import glBlendFunc as _glBlendFunc, glDisable as _glDisable
    glBlendFunc_local = _glBlendFunc
    glDisable_local = _glDisable
    cx, cy, cz = center
    ex, ey, ez = eye
    gluLookAt(ex, ey, ez, cx, cy, cz, 0, 1, 0)

    if pre_draw_fn is not None:
        pre_draw_fn()
    mv_mat = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj_mat = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    if proj_mats is None:
        proj_mats = (mv_mat, proj_mat, viewport)

    # fast edges (no depth quantization or segmentation for speed)
    if springs.size:
        i_idx = springs[:, 0].astype(np.int64)
        j_idx = springs[:, 1].astype(np.int64)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for k in range(springs.shape[0]):
            ii = int(i_idx[k])
            jj = int(j_idx[k])
            if ii >= len(pos3d) or jj >= len(pos3d):
                continue
            base_col = edge_colors[k] if edge_colors is not None and k < len(edge_colors) else (0.8, 0.8, 0.85)
            glColor3f(*base_col)
            glVertex3f(pos3d[ii][0], pos3d[ii][1], pos3d[ii][2])
            glVertex3f(pos3d[jj][0], pos3d[jj][1], pos3d[jj][2])
        glEnd()

    # thrust vectors rendered as tails from sentence nodes
    if thrust_lines:
        glLineWidth(1.5)
        glColor3f(0.6, 0.9, 1.0)
        glBegin(GL_LINES)
        for start, end in thrust_lines:
            glVertex3f(start[0], start[1], start[2])
            glVertex3f(end[0], end[1], end[2])
        glEnd()

    # translucent triangles fanning from each sentence node to consecutive tokens
    if sentence_triangles:
        glEnable(GL_BLEND)
        glBlendFunc_local(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.7, 0.9, 1.0, 0.08)
        glBegin(GL_TRIANGLES)
        for a_idx, b_idx, c_idx in sentence_triangles:
            if a_idx >= len(pos3d) or b_idx >= len(pos3d) or c_idx >= len(pos3d):
                continue
            ax, ay, az = pos3d[a_idx]
            bx, by, bz = pos3d[b_idx]
            cx, cy, cz = pos3d[c_idx]
            glVertex3f(ax, ay, az)
            glVertex3f(bx, by, bz)
            glVertex3f(cx, cy, cz)
        glEnd()
        glDisable_local(GL_BLEND)

    # nodes (allow per-node point sizes when provided)
    if draw_nodes:
        # depth shading for nodes
        z_vals = pos3d[:, 2]
        z_min = float(z_vals.min()) if len(z_vals) else 0.0
        z_max = float(z_vals.max()) if len(z_vals) else 1.0
        if abs(z_max - z_min) < 1e-6:
            z_max = z_min + 1.0
        depth_layers = 6

        def node_shade(z, base):
            if not math.isfinite(z):
                factor = 0.6
                return (base[0] * factor, base[1] * factor, base[2] * factor)
            denom = (z_max - z_min)
            if not math.isfinite(denom) or abs(denom) < 1e-12:
                factor = 0.6
                return (base[0] * factor, base[1] * factor, base[2] * factor)
            t = (z - z_min) / denom
            if not math.isfinite(t):
                t = 0.0
            layer = min(depth_layers - 1, max(0, int(t * depth_layers)))
            factor = 0.35 + 0.65 * (1.0 - layer / (depth_layers - 1))
            return (base[0] * factor, base[1] * factor, base[2] * factor)

        if point_sizes is None:
            glPointSize(6.0)
            glBegin(GL_POINTS)
            for (x, y, z), (r, g, b) in zip(pos3d, colors):
                cr, cg, cb = node_shade(z, (r, g, b))
                glColor3f(cr, cg, cb)
                glVertex3f(x, y, z)
            glEnd()
        else:
            # point sizes are per-node; clamp for safety
            sizes = np.asarray(point_sizes, dtype=np.float32)
            if sizes.shape[0] != len(pos3d):
                glPointSize(6.0)
                glBegin(GL_POINTS)
                for (x, y, z), (r, g, b) in zip(pos3d, colors):
                    cr, cg, cb = node_shade(z, (r, g, b))
                    glColor3f(cr, cg, cb)
                    glVertex3f(x, y, z)
                glEnd()
            else:
                # Keep a safety clamp, but allow large nodes to be visibly large.
                sizes = np.clip(sizes, 1.0, 256.0)
                for (x, y, z), (r, g, b), s_px in zip(pos3d, colors, sizes):
                    glPointSize(float(s_px))
                    glBegin(GL_POINTS)
                    cr, cg, cb = node_shade(z, (r, g, b))
                    glColor3f(cr, cg, cb)
                    glVertex3f(x, y, z)
                    glEnd()

    # highlight overlay
    if highlight_idx:
        glPointSize(10.0)
        glBegin(GL_POINTS)
        for idx in highlight_idx:
            if idx < 0 or idx >= len(pos3d):
                continue
            glColor3f(1.0, 0.9, 0.3)
            glVertex3f(*pos3d[idx])
        glEnd()

    # node labels (screen-space overlay using real projection)
    if node_labels and proj_mats:
        mv_mat, proj_mat, viewport = proj_mats
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        from OpenGL.GLU import gluProject
        for lbl in node_labels:
            idx = lbl.get("idx")
            text = lbl.get("text", "")
            if idx is None or not text:
                continue
            wx, wy, wz = gluProject(pos_nd[idx][0], pos_nd[idx][1], pos_nd[idx][2], mv_mat, proj_mat, viewport)
            if wz < 0.0 or wz > 1.0:
                continue
            # convert window coords to 0..1
            sx = max(0.01, min(0.99, wx / viewport[2]))
            sy = max(0.01, min(0.99, wy / viewport[3] + 0.012))
            surf = font.render(text, True, (220, 220, 220))
            text_data = pygame.image.tostring(surf, "RGBA", True)
            from OpenGL.GL import glRasterPos2f, glDrawPixels, glPixelStorei, glBlendFunc as _glBlendFunc_lbl, glDisable as _glDisable_lbl
            glEnable(GL_BLEND)
            _glBlendFunc_lbl(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glRasterPos2f(sx, sy)
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            _glDisable_lbl(GL_BLEND)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    if draw_hud_now:
        draw_hud(font=font, fps_render=fps_render, fps_phys=fps_phys, text_lines=text_lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    path: str | None,
    dt: float,
    k_spring: float,
    k_rep: float,
    k_angle: float,
    temp: float,
    damping: float,
    softening: float,
    steps_per_frame: int,
    max_speed: float | None = 5.0,
    use_torch: bool = False,
    torch_device: str = "cuda",
    nodes_override: list[dict] | None = None,
    edges_override: list[dict] | None = None,
    state_hook=None,
    hook_interval_ms: int = 500,
    text_provider=None,
    label_provider=None,
    temp_fn=None,
    extra_force_fn=None,
    extra_force_fn_t=None,
    damping_per_node: np.ndarray | None = None,
    damping_provider=None,
    damping_provider_t=None,
    temp_per_node: np.ndarray | None = None,
    temp_provider=None,
    temp_provider_t=None,
    force_scale_per_node: np.ndarray | None = None,
    force_scale_provider=None,
    force_scale_provider_t=None,
    spring_k_scale: np.ndarray | None = None,
    spring_k_scale_provider=None,
    spring_k_scale_provider_t=None,
    highlight_provider=None,
    train_k_live: bool = False,
    train_skip_src: np.ndarray | None = None,
    train_skip_dst: np.ndarray | None = None,
    train_skip_w: np.ndarray | None = None,
    train_k_lr: float = 0.05,
    train_k_every: int = 60,
    train_k_unroll: int = 1,
    train_k_field: bool = False,
    train_k_hidden: int = 64,
    constrain_unit_sphere: bool = False,
    constrain_radius: float = 1.0,
    train_k_grad_clip: float = 1.0,
    physics_lockstep: bool = False,
    train_k_live_lockstep: bool = False,
    train_k_reg: float = 0.0,
    k_smooth: float = 0.0,
    k_norm_mean_floor: float = 1e-3,
    k_norm_sum1: bool = False,
    train_k_neg_samples: int = 0,
    train_k_neg_margin: float = 0.5,
    train_k_neg_weight: float = 1.0,
    train_rest: bool = False,
    rest_min: float = 1e-4,
    rest_max: float | None = None,
    show_k_colors: bool = False,
    train_k_thrust: bool = False,
    train_k_damp: bool = False,
    train_k_grad_acc: int = 1,
    train_k_node_depth: int = 1,
    train_k_edge_depth: int = 1,
    train_k_separate_heads: bool = False,
):
    if nodes_override is not None and edges_override is not None:
        nodes, edges = nodes_override, edges_override
    else:
        if path is None:
            raise SystemExit("path is required when no overrides are provided")
        nodes, edges = load_prephysics(path)
    if not nodes:
        raise SystemExit("No nodes found for simulation")

    node_ids, pos0, vel0, masses, colors, springs, mean, proj, fixed_mask = build_sim_state(nodes, edges)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    sentence_triangles = []
    # build per-sentence token chains using id pattern tok:<sent>:<idx>
    tokens_by_sent: dict[int, list[int]] = {}
    for nid in node_ids:
        if nid.startswith("tok:"):
            parts = nid.split(":")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                si = int(parts[1])
                ti = int(parts[2])
                tokens_by_sent.setdefault(si, []).append((ti, id_to_idx[nid]))
    for nid in node_ids:
        if nid.startswith("sent:"):
            parts = nid.split(":")
            if len(parts) >= 2 and parts[1].isdigit():
                si = int(parts[1])
                sent_idx = id_to_idx[nid]
                toks = tokens_by_sent.get(si, [])
                if len(toks) < 2:
                    continue
                toks_sorted = [idx for _, idx in sorted(toks, key=lambda x: x[0])]
                for a, b in zip(toks_sorted[:-1], toks_sorted[1:]):
                    sentence_triangles.append((sent_idx, a, b))
    sentence_mask = np.array([n.get("kind") == "sentence" for n in nodes], dtype=bool)
    pos = pos0.copy()
    vel = vel0.copy()
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    pos_fixed = pos0.copy()
    physics_step_counter = 0
    physics_last_t = 0
    physics_fps = 0.0
    hook_last_t = 0

    damping_vec = None
    if damping_per_node is not None:
        damping_vec = np.asarray(damping_per_node, dtype=np.float32)
    damping_vec_t = None
    if use_torch and torch is not None and damping_vec is not None:
        damping_vec_t = torch.tensor(damping_vec, dtype=torch.float32, device=torch_device)

    temp_vec = None
    if temp_per_node is not None:
        temp_vec = np.asarray(temp_per_node, dtype=np.float32)
    temp_vec_t = None
    if use_torch and torch is not None and temp_vec is not None:
        temp_vec_t = torch.tensor(temp_vec, dtype=torch.float32, device=torch_device)

    force_scale_vec = None
    if force_scale_per_node is not None:
        force_scale_vec = np.asarray(force_scale_per_node, dtype=np.float32)
    force_scale_vec_t = None
    if use_torch and torch is not None and force_scale_vec is not None:
        force_scale_vec_t = torch.tensor(force_scale_vec, dtype=torch.float32, device=torch_device)

    spring_scale_vec = None
    if spring_k_scale is not None:
        spring_scale_vec = np.asarray(spring_k_scale, dtype=np.float32)
    spring_scale_vec_t = None
    if use_torch and torch is not None and spring_scale_vec is not None:
        spring_scale_vec_t = torch.tensor(spring_scale_vec, dtype=torch.float32, device=torch_device)

    # adjacency for angular spreading (neighbor-neighbor repulsion per hub)
    neighbors: List[List[int]] = [[] for _ in range(pos.shape[0])]
    for i, j in springs[:, :2].astype(np.int64):
        neighbors[i].append(j)
        neighbors[j].append(i)
    pair_n1: List[int] = []
    pair_n2: List[int] = []
    pair_hub: List[int] = []
    for c, neigh in enumerate(neighbors):
        deg = len(neigh)
        if deg < 2:
            continue
        for a in range(deg):
            for b in range(a + 1, deg):
                pair_n1.append(neigh[a])
                pair_n2.append(neigh[b])
                pair_hub.append(c)
    pair_n1_arr = np.asarray(pair_n1, dtype=np.int64)
    pair_n2_arr = np.asarray(pair_n2, dtype=np.int64)
    pair_hub_arr = np.asarray(pair_hub, dtype=np.int64)

    use_torch = use_torch and (torch is not None)
    if use_torch:
        device = torch_device if torch.cuda.is_available() or torch_device == "cpu" else "cpu"
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
        vel_t = torch.tensor(vel, dtype=torch.float32, device=device)
        mass_t = torch.tensor(masses, dtype=torch.float32, device=device)
        embed_feat_t = pos_t  # will be refreshed from live positions for k nets
        springs_t = torch.tensor(springs, dtype=torch.float32, device=device)
        i_idx_t = springs_t[:, 0].long()
        j_idx_t = springs_t[:, 1].long()
        rest_t = springs_t[:, 2]
        k_edge_t = springs_t[:, 3] if springs_t.shape[1] > 3 else None
        fixed_mask_t = torch.tensor(fixed_mask, dtype=torch.bool, device=device)
        pos_fixed_t = torch.tensor(pos_fixed, dtype=torch.float32, device=device)
        sentence_mask_t = torch.tensor(sentence_mask, dtype=torch.bool, device=device)

        rest_param = torch.nn.Parameter(rest_t.clone()) if train_rest else None

        N_nodes = pos_t.shape[0]
        E_edges = springs_t.shape[0]

        def _record_vis(rep_t, angle_t, spring_t, rest_t_vis):
            if not show_k_colors:
                return
            if rep_t is None or angle_t is None or spring_t is None or rest_t_vis is None:
                return
            def as_vec(val, length):
                if isinstance(val, torch.Tensor):
                    if val.ndim == 0 or val.numel() == 1:
                        return torch.full((length,), float(val.item()), device=val.device)
                    if val.numel() == length:
                        return val.view(-1)
                    return val.flatten()[:length]
                return torch.full((length,), float(val), device=pos_t.device)

            rep_v = as_vec(rep_t, N_nodes)
            angle_v = as_vec(angle_t, N_nodes)
            spring_v = as_vec(spring_t, E_edges) if E_edges else torch.tensor([], device=pos_t.device)
            rest_v = rest_t_vis if isinstance(rest_t_vis, torch.Tensor) else torch.tensor(rest_t_vis, device=pos_t.device)
            if rest_v.ndim == 0 or rest_v.numel() == 1:
                rest_v = torch.full((E_edges,), float(rest_v.item()), device=pos_t.device)
            else:
                rest_v = rest_v.view(-1)[:E_edges]
            rep_np = rep_v.detach().cpu().numpy().astype(np.float32)
            angle_np = angle_v.detach().cpu().numpy().astype(np.float32)
            spring_np = spring_v.detach().cpu().numpy().astype(np.float32)
            rest_np = rest_v.detach().cpu().numpy().astype(np.float32)
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                state["vis_node_rep"] = rep_np
                state["vis_node_angle"] = angle_np
                state["vis_edge_spring"] = spring_np
                state["vis_edge_rest"] = rest_np

        # trainable k parameters (softplus to ensure positive)
        k_spring_param = torch.nn.Parameter(torch.tensor(k_spring, dtype=torch.float32, device=device)) if not train_k_field else None
        k_rep_param = torch.nn.Parameter(torch.tensor(k_rep, dtype=torch.float32, device=device)) if not train_k_field else None
        k_angle_param = torch.nn.Parameter(torch.tensor(k_angle, dtype=torch.float32, device=device)) if not train_k_field else None

        node_k_net = None
        edge_k_net = None

        def make_mlp(in_dim, hid, out_dim, depth):
            depth = max(1, depth)
            layers = []
            last = in_dim
            for _ in range(depth):
                layers.append(torch.nn.Linear(last, hid))
                layers.append(torch.nn.SiLU())
                last = hid
            layers.append(torch.nn.Linear(last, out_dim))
            return torch.nn.Sequential(*layers)

        class NodeKHeads(torch.nn.Module):
            def __init__(self, in_dim, hid, depth, use_damp, use_thrust, thrust_dim):
                super().__init__()
                self.trunk = make_mlp(in_dim, hid, hid, depth)
                self.head_rep = torch.nn.Linear(hid, 1)
                self.head_ang = torch.nn.Linear(hid, 1)
                self.head_damp = torch.nn.Linear(hid, 1) if use_damp else None
                self.head_thrust = torch.nn.Linear(hid, thrust_dim) if use_thrust else None

            def forward(self, x):
                h = self.trunk(x)
                rep = self.head_rep(h)
                ang = self.head_ang(h)
                damp = self.head_damp(h) if self.head_damp is not None else None
                thrust = self.head_thrust(h) if self.head_thrust is not None else None
                return rep, ang, damp, thrust

        if train_k_live and train_k_field:
            D = embed_feat_t.shape[1]
            out_dim = 2 + (1 if train_k_damp else 0) + (D if train_k_thrust else 0)
            if train_k_separate_heads:
                node_k_net = NodeKHeads(D, train_k_hidden * 2, train_k_node_depth, train_k_damp, train_k_thrust, D).to(device)
            else:
                node_k_net = make_mlp(D, train_k_hidden * 2, max(out_dim, 2), train_k_node_depth).to(device)
            edge_k_net = make_mlp(2 * D, train_k_hidden, 1, train_k_edge_depth).to(device)

        params = []
        for p in (k_spring_param, k_rep_param, k_angle_param, rest_param):
            if p is not None:
                params.append(p)
        if node_k_net is not None:
            params += list(node_k_net.parameters())
        if edge_k_net is not None:
            params += list(edge_k_net.parameters())
        k_opt = torch.optim.Adam(params, lr=train_k_lr) if train_k_live else None

        def _pg_stats():
            if k_opt is None:
                return ""
            parts = []
            with torch.no_grad():
                for gi, pg in enumerate(k_opt.param_groups):
                    g_parts = []
                    for pi, p in enumerate(pg.get("params", [])):
                        if p is None or p.data.numel() == 0:
                            continue
                        data = p.data.detach().float().view(-1)
                        mean = float(data.mean())
                        std = float(data.std(unbiased=False))
                        pmin = float(data.min())
                        pmax = float(data.max())
                        g_parts.append(f"p{pi}:m={mean:.3f} s={std:.3f} lo={pmin:.3f} hi={pmax:.3f}")
                    if g_parts:
                        parts.append(f"pg{gi}[{len(g_parts)}]:" + "; ".join(g_parts))
            return " | ".join(parts)

        # smoothing buffers for ks (torch physics path)
        k_spring_filt = None
        k_rep_filt = None
        k_angle_filt = None
        if k_smooth > 0.0:
            if train_k_field and edge_k_net is not None:
                with torch.no_grad():
                    edge_feat = torch.cat([embed_feat_t[i_idx_t], embed_feat_t[j_idx_t]], dim=1)
                    k_init = torch.nn.functional.softplus(edge_k_net(edge_feat).squeeze(-1)) + 1e-4
                    k_init = torch.clamp(k_init, min=1e-4, max=200.0)
                k_spring_filt = k_init.detach()
            else:
                k_base = torch.nn.functional.softplus(k_spring_param).detach()
                k_spring_filt = torch.clamp(k_base, min=1e-4, max=200.0)
            if train_k_field and node_k_net is not None:
                with torch.no_grad():
                    node_out = node_k_net(embed_feat_t)
                    k_rep_filt = torch.clamp(torch.nn.functional.softplus(node_out[:, 0]), min=1e-4, max=5.0).detach()
                    k_angle_filt = torch.clamp(torch.nn.functional.softplus(node_out[:, 1]), min=1e-4, max=50.0).detach()
            else:
                k_rep_filt = torch.clamp(torch.nn.functional.softplus(k_rep_param).detach(), min=1e-4, max=5.0)
                k_angle_filt = torch.clamp(torch.nn.functional.softplus(k_angle_param).detach(), min=1e-4, max=50.0)

        state = {"pos": pos.copy(), "pos_t": pos_t, "vel_t": vel_t, "sentence_triangles": sentence_triangles}
        lock = None if physics_lockstep else threading.Lock()
        stop_evt = threading.Event() if not physics_lockstep else None

        pair_n1_t = torch.tensor(pair_n1_arr, dtype=torch.long, device=device) if pair_n1_arr.size else torch.tensor([], dtype=torch.long, device=device)
        pair_n2_t = torch.tensor(pair_n2_arr, dtype=torch.long, device=device) if pair_n2_arr.size else torch.tensor([], dtype=torch.long, device=device)
        pair_hub_t = torch.tensor(pair_hub_arr, dtype=torch.long, device=device) if pair_hub_arr.size else torch.tensor([], dtype=torch.long, device=device)
        constrain_radius_t = torch.tensor(constrain_radius, dtype=torch.float32, device=device) if use_torch else None

        def torch_step(pos_t, vel_t):
            nonlocal k_spring_filt, k_rep_filt, k_angle_filt
            damp_now = None
            thrust_now = None
            with torch.no_grad():
                embed_feat_t = pos_t.detach()
                # predict per-node/edge ks (detached to keep physics stable)
                if train_k_field and node_k_net is not None and edge_k_net is not None:
                    if train_k_separate_heads:
                        rep_raw, ang_raw, damp_raw, thrust_raw = node_k_net(embed_feat_t)
                        rep_raw = rep_raw.squeeze(-1)
                        ang_raw = ang_raw.squeeze(-1)
                        k_rep_now = torch.clamp(torch.nn.functional.softplus(rep_raw), min=1e-4, max=5.0)
                        k_angle_now = torch.clamp(torch.nn.functional.softplus(ang_raw), min=1e-4, max=50.0)
                        if train_k_damp and damp_raw is not None:
                            damp_now = torch.clamp(torch.sigmoid(damp_raw.squeeze(-1)), min=0.0, max=0.8)
                        if train_k_thrust and thrust_raw is not None:
                            thrust_now = torch.tanh(thrust_raw) * 0.5
                    else:
                        node_out = node_k_net(embed_feat_t)
                        rep_raw = node_out[:, 0]
                        ang_raw = node_out[:, 1]
                        k_rep_now = torch.clamp(torch.nn.functional.softplus(rep_raw), min=1e-4, max=5.0)
                        k_angle_now = torch.clamp(torch.nn.functional.softplus(ang_raw), min=1e-4, max=50.0)
                        idx_cursor = 2
                        if train_k_damp:
                            damp_raw = node_out[:, idx_cursor]
                            damp_now = torch.clamp(torch.sigmoid(damp_raw), min=0.0, max=0.8)
                            idx_cursor += 1
                        if train_k_thrust:
                            thrust_raw = node_out[:, idx_cursor : idx_cursor + pos_t.shape[1]]
                            thrust_now = torch.tanh(thrust_raw) * 0.5
                    edge_feat = torch.cat([embed_feat_t[i_idx_t], embed_feat_t[j_idx_t]], dim=1)
                    k_spring_now = torch.nn.functional.softplus(edge_k_net(edge_feat).squeeze(-1)) + 1e-4
                    k_spring_now = torch.clamp(k_spring_now, min=1e-4, max=200.0)
                else:
                    k_spring_now = torch.clamp(torch.nn.functional.softplus(k_spring_param).detach(), min=1e-4, max=200.0)
                    k_rep_now = torch.clamp(torch.nn.functional.softplus(k_rep_param).detach(), min=1e-4, max=5.0)
                    k_angle_now = torch.clamp(torch.nn.functional.softplus(k_angle_param).detach(), min=1e-4, max=50.0)

                if k_smooth > 0.0:
                    k_spring_now = k_smooth * k_spring_filt + (1 - k_smooth) * k_spring_now
                    k_rep_now = k_smooth * k_rep_filt + (1 - k_smooth) * k_rep_now
                    k_angle_now = k_smooth * k_angle_filt + (1 - k_smooth) * k_angle_now
                    k_spring_filt = k_spring_now.detach()
                    k_rep_filt = k_rep_now.detach()
                    k_angle_filt = k_angle_now.detach()
                # mean-floor normalization to avoid collapse
                mean_s = k_spring_now.mean()
                if torch.isfinite(mean_s) and mean_s < k_norm_mean_floor:
                    k_spring_now = k_spring_now + (k_norm_mean_floor - mean_s)
                mean_r = k_rep_now.mean()
                if torch.isfinite(mean_r) and mean_r < k_norm_mean_floor:
                    k_rep_now = k_rep_now + (k_norm_mean_floor - mean_r)
                mean_a = k_angle_now.mean()
                if torch.isfinite(mean_a) and mean_a < k_norm_mean_floor:
                    k_angle_now = k_angle_now + (k_norm_mean_floor - mean_a)
                if k_norm_sum1:
                    k_spring_now = torch.clamp(k_spring_now, min=1e-6)
                    k_rep_now = torch.clamp(k_rep_now, min=1e-6)
                    k_angle_now = torch.clamp(k_angle_now, min=1e-6)
                    s_sum = k_spring_now.sum()
                    r_sum = k_rep_now.sum()
                    a_sum = k_angle_now.sum()
                    if torch.isfinite(s_sum) and s_sum > 0:
                        k_spring_now = k_spring_now / s_sum
                    if torch.isfinite(r_sum) and r_sum > 0:
                        k_rep_now = k_rep_now / r_sum
                    if torch.isfinite(a_sum) and a_sum > 0:
                        k_angle_now = k_angle_now / a_sum
                k_spring_now = torch.clamp(k_spring_now, min=1e-4, max=200.0)
                k_rep_now = torch.clamp(k_rep_now, min=1e-4, max=5.0)
                k_angle_now = torch.clamp(k_angle_now, min=1e-4, max=50.0)

                diff = pos_t[j_idx_t] - pos_t[i_idx_t]
                dist2 = (diff * diff).sum(dim=1) + 1e-12
                dist = dist2.sqrt()
                if rest_param is not None:
                    rest_eff = rest_param.detach()
                    rest_eff = torch.clamp(rest_eff, min=rest_min, max=rest_max) if rest_max is not None else torch.clamp(rest_eff, min=rest_min)
                else:
                    rest_eff = rest_t
                if show_k_colors:
                    _record_vis(k_rep_now, k_angle_now, k_spring_now, rest_eff)
                    if train_k_thrust and thrust_now is not None:
                        thr_np = thrust_now.detach().cpu().numpy().astype(np.float32)
                        if lock is not None:
                            with lock:
                                state["vis_thrust"] = thr_np
                        else:
                            state["vis_thrust"] = thr_np
                stretch = rest_eff - dist  # positive when compressed
                edge_k = k_edge_t if k_edge_t is not None else 1.0
                if train_k_field and edge_k_net is not None:
                    edge_feat = torch.cat([embed_feat_t[i_idx_t], embed_feat_t[j_idx_t]], dim=1)
                    k_spring_now = torch.clamp(torch.nn.functional.softplus(edge_k_net(edge_feat).squeeze(-1)) + 1e-4, max=200.0)
                spring_scale_current = spring_scale_vec_t
                if spring_k_scale_provider_t is not None:
                    ss_dyn = spring_k_scale_provider_t()
                    if ss_dyn is not None:
                        spring_scale_current = ss_dyn
                if spring_scale_current is not None:
                    edge_k = edge_k * spring_scale_current
                force_mag = (k_spring_now * edge_k) * stretch / dist
                F_edge = diff * (force_mag / dist)[:, None]
                F = torch.zeros_like(pos_t)
                F.index_add_(0, i_idx_t, -F_edge)
                F.index_add_(0, j_idx_t, F_edge)

                if pair_n1_t.numel() > 0:
                    k_angle_hub = k_angle_now[pair_hub_t] if k_angle_now.ndim > 0 else k_angle_now
                    if k_angle_hub.mean().item() != 0.0:
                        v1 = pos_t[pair_n1_t] - pos_t[pair_hub_t]
                        v2 = pos_t[pair_n2_t] - pos_t[pair_hub_t]
                        norm1 = v1.norm(dim=1, keepdim=True) + 1e-9
                        norm2 = v2.norm(dim=1, keepdim=True) + 1e-9
                        u1 = v1 / norm1
                        u2 = v2 / norm2
                        dot = (u1 * u2).sum(dim=1, keepdim=True)
                        Fi = -k_angle_hub.unsqueeze(-1) * dot * (u2 - dot * u1) / norm1
                        Fj = -k_angle_hub.unsqueeze(-1) * dot * (u1 - dot * u2) / norm2
                        F.index_add_(0, pair_n1_t, Fi)
                        F.index_add_(0, pair_n2_t, Fj)
                        F.index_add_(0, pair_hub_t, -(Fi + Fj))

                if (k_rep_now.mean() if k_rep_now.ndim > 0 else k_rep_now).item() != 0.0:
                    diff_full = pos_t[:, None, :] - pos_t[None, :, :]
                    dist2_full = (diff_full * diff_full).sum(dim=2) + softening
                    inv_dist = dist2_full.sqrt().reciprocal()
                    if k_rep_now.ndim > 0:
                        rep_pair = torch.sqrt(k_rep_now[:, None] * k_rep_now[None, :])
                    else:
                        rep_pair = k_rep_now
                    rep_mag = rep_pair * inv_dist * inv_dist
                    force_full = diff_full * rep_mag[:, :, None] * inv_dist[:, :, None]
                    force_full[torch.arange(pos_t.shape[0]), torch.arange(pos_t.shape[0])] = 0.0
                    F = F + force_full.sum(dim=1)

                if thrust_now is not None:
                    thrust_force = thrust_now.clone()
                    if sentence_mask_t is not None and sentence_mask_t.any():
                        thrust_force = thrust_force * sentence_mask_t.unsqueeze(-1)
                    F = F + thrust_force

                temp_now = temp_fn(time.time()) if temp_fn is not None else temp
                temp_local = temp_now
                if temp_provider_t is not None:
                    tp = temp_provider_t()
                    if tp is not None:
                        temp_local = tp
                elif temp_vec_t is not None:
                    temp_local = temp_vec_t
                if isinstance(temp_local, torch.Tensor):
                    F = F + temp_local.unsqueeze(-1) * torch.randn_like(pos_t) * math.sqrt(dt)
                elif temp_local > 0.0:
                    F = F + temp_local * torch.randn_like(pos_t) * math.sqrt(dt)

                if extra_force_fn_t is not None:
                    extra = extra_force_fn_t(pos_t)
                    if extra is not None:
                        F = F + extra

                force_scale_current = force_scale_vec_t
                if force_scale_provider_t is not None:
                    fs_dyn = force_scale_provider_t()
                    if fs_dyn is not None:
                        force_scale_current = fs_dyn
                if force_scale_current is not None:
                    F = F * force_scale_current.unsqueeze(-1)

                acc = F / mass_t[:, None]
                vel_t = vel_t + 0.5 * dt * acc
                pos_t = pos_t + dt * vel_t

                # second force
                diff = pos_t[j_idx_t] - pos_t[i_idx_t]
                dist2 = (diff * diff).sum(dim=1) + 1e-12
                dist = dist2.sqrt()
                stretch = rest_t - dist
                edge_k = k_edge_t if k_edge_t is not None else 1.0
                spring_scale_current = spring_scale_vec_t
                if spring_k_scale_provider_t is not None:
                    ss_dyn = spring_k_scale_provider_t()
                    if ss_dyn is not None:
                        spring_scale_current = ss_dyn
                if spring_scale_current is not None:
                    edge_k = edge_k * spring_scale_current
                force_mag = (k_spring_now * edge_k) * stretch / dist
                F_edge = diff * (force_mag / dist)[:, None]
                F2 = torch.zeros_like(pos_t)
                F2.index_add_(0, i_idx_t, -F_edge)
                F2.index_add_(0, j_idx_t, F_edge)
                if pair_n1_t.numel() > 0:
                    k_angle_hub = k_angle_now[pair_hub_t] if k_angle_now.ndim > 0 else k_angle_now
                    if k_angle_hub.mean().item() != 0.0:
                        v1 = pos_t[pair_n1_t] - pos_t[pair_hub_t]
                        v2 = pos_t[pair_n2_t] - pos_t[pair_hub_t]
                        norm1 = v1.norm(dim=1, keepdim=True) + 1e-9
                        norm2 = v2.norm(dim=1, keepdim=True) + 1e-9
                        u1 = v1 / norm1
                        u2 = v2 / norm2
                        dot = (u1 * u2).sum(dim=1, keepdim=True)
                        Fi = -k_angle_hub.unsqueeze(-1) * dot * (u2 - dot * u1) / norm1
                        Fj = -k_angle_hub.unsqueeze(-1) * dot * (u1 - dot * u2) / norm2
                        F2.index_add_(0, pair_n1_t, Fi)
                        F2.index_add_(0, pair_n2_t, Fj)
                        F2.index_add_(0, pair_hub_t, -(Fi + Fj))
                if (k_rep_now.mean() if k_rep_now.ndim > 0 else k_rep_now).item() != 0.0:
                    diff_full = pos_t[:, None, :] - pos_t[None, :, :]
                    dist2_full = (diff_full * diff_full).sum(dim=2) + softening
                    inv_dist = dist2_full.sqrt().reciprocal()
                    if k_rep_now.ndim > 0:
                        rep_pair = torch.sqrt(k_rep_now[:, None] * k_rep_now[None, :])
                    else:
                        rep_pair = k_rep_now
                    rep_mag = rep_pair * inv_dist * inv_dist
                    force_full = diff_full * rep_mag[:, :, None] * inv_dist[:, :, None]
                    force_full[torch.arange(pos_t.shape[0]), torch.arange(pos_t.shape[0])] = 0.0
                    F2 = F2 + force_full.sum(dim=1)
                temp_now = temp_fn(time.time()) if temp_fn is not None else temp
                temp_local = temp_now
                if temp_provider_t is not None:
                    tp = temp_provider_t()
                    if tp is not None:
                        temp_local = tp
                elif temp_vec_t is not None:
                    temp_local = temp_vec_t
                if isinstance(temp_local, torch.Tensor):
                    F2 = F2 + temp_local.unsqueeze(-1) * torch.randn_like(pos_t) * math.sqrt(dt)
                elif temp_local > 0.0:
                    F2 = F2 + temp_local * torch.randn_like(pos_t) * math.sqrt(dt)
                if extra_force_fn_t is not None:
                    extra = extra_force_fn_t(pos_t)
                    if extra is not None:
                        F2 = F2 + extra
                force_scale_current = force_scale_vec_t
                if force_scale_provider_t is not None:
                    fs_dyn = force_scale_provider_t()
                    if fs_dyn is not None:
                        force_scale_current = fs_dyn
                if force_scale_current is not None:
                    F2 = F2 * force_scale_current.unsqueeze(-1)

                acc2 = F2 / mass_t[:, None]
                vel_t = vel_t + 0.5 * dt * acc2
                damp_current = damping
                if damp_now is not None:
                    damp_current = damp_now
                elif damping_provider_t is not None:
                    damp_dyn = damping_provider_t()
                    if damp_dyn is not None:
                        damp_current = damp_dyn
                elif damping_vec_t is not None:
                    damp_current = damping_vec_t
                if isinstance(damp_current, torch.Tensor):
                    if damp_current.ndim == 1:
                        vel_t = vel_t * (1.0 - damp_current).unsqueeze(-1)
                    else:
                        vel_t = vel_t * (1.0 - damp_current)
                elif damp_current > 0:
                    vel_t = vel_t * (1.0 - damp_current)

                if constrain_unit_sphere:
                    mask = ~fixed_mask_t if fixed_mask_t is not None else torch.ones_like(pos_t[:, 0], dtype=torch.bool, device=pos_t.device)
                    if mask.any():
                        p = pos_t[mask]
                        nrm = p.norm(dim=1, keepdim=True).clamp(min=1e-9)
                        n = p / nrm
                        pos_t = pos_t.clone()
                        vel_t = vel_t.clone()
                        pos_t[mask] = n * constrain_radius_t
                        v = vel_t[mask]
                        v_rad = (v * n).sum(dim=1, keepdim=True) * n
                        vel_t[mask] = v - v_rad

                pos_t = torch.nan_to_num(pos_t, nan=0.0, posinf=0.0, neginf=0.0)
                vel_t = torch.nan_to_num(vel_t, nan=0.0, posinf=0.0, neginf=0.0)
                if fixed_mask_t.any():
                    vel_t[fixed_mask_t] = 0.0
                    pos_t[fixed_mask_t] = pos_fixed_t[fixed_mask_t]
                return pos_t, vel_t

        def physics_worker():
            nonlocal pos_t, vel_t, physics_step_counter
            while not stop_evt.is_set():
                for _ in range(steps_per_frame):
                    pos_t, vel_t = torch_step(pos_t, vel_t)
                with lock:
                    physics_step_counter += steps_per_frame
                with lock:
                    state["pos_t"] = pos_t
                    state["vel_t"] = vel_t
                    state["pos"] = pos_t.detach().cpu().numpy().astype(np.float32)

        if not physics_lockstep:
            worker = threading.Thread(target=physics_worker, daemon=True)
            worker.start()

    # Sanity check: with rest lengths set from embeddings and zero repulsion/temp, net forces should be ~0
    if springs.size:
        N, D = pos.shape
        E = springs.shape[0]
        check_scratch = {
            "diff_s": np.zeros((E, D), dtype=np.float32),
            "dist_s": np.zeros((E,), dtype=np.float32),
            "stretch": np.zeros((E,), dtype=np.float32),
            "force_mag": np.zeros((E,), dtype=np.float32),
            "F_s": np.zeros((E, D), dtype=np.float32),
            "F": np.zeros((N, D), dtype=np.float32),
            "acc": np.zeros((N, D), dtype=np.float32),
            "inv_mass": np.zeros((N,), dtype=np.float32),
            "diff_full": np.zeros((N, N, D), dtype=np.float32),
            "dist2_full": np.zeros((N, N), dtype=np.float32),
            "inv_dist_full": np.zeros((N, N), dtype=np.float32),
            "rep_mag_full": np.zeros((N, N), dtype=np.float32),
            "force_full": np.zeros((N, N, D), dtype=np.float32),
            "temp_noise": np.zeros((N, D), dtype=np.float32),
        }
        F0 = check_scratch["F"]
        compute_forces(pos, springs, neighbors=[], pair_n1=np.zeros((0,), dtype=np.int64), pair_n2=np.zeros((0,), dtype=np.int64), pair_hub=np.zeros((0,), dtype=np.int64), k_spring=1.0, k_rep=0.0, k_angle=0.0, temp=0.0, softening=1e-6, F=F0, scratch=check_scratch, extra_force_fn=extra_force_fn)
        max_force = float(np.abs(F0).max())
        if max_force > 1e-4:
            print(f"[warn] Initial net force not near zero (max |F|={max_force:.3e}); check embeddings/rest lengths")

    # scratch buffers for vectorized physics
    N, D = pos.shape
    E = springs.shape[0]
    scratch = {
        "diff_s": np.zeros((E, D), dtype=np.float32) if E else np.zeros((0, D), dtype=np.float32),
        "dist_s": np.zeros((E,), dtype=np.float32) if E else np.zeros((0,), dtype=np.float32),
        "stretch": np.zeros((E,), dtype=np.float32) if E else np.zeros((0,), dtype=np.float32),
        "force_mag": np.zeros((E,), dtype=np.float32) if E else np.zeros((0,), dtype=np.float32),
        "F_s": np.zeros((E, D), dtype=np.float32) if E else np.zeros((0, D), dtype=np.float32),
        "F": np.zeros((N, D), dtype=np.float32),
        "acc": np.zeros((N, D), dtype=np.float32),
        "inv_mass": np.zeros((N,), dtype=np.float32),
        "diff_full": np.zeros((N, N, D), dtype=np.float32),
        "dist2_full": np.zeros((N, N), dtype=np.float32),
        "inv_dist_full": np.zeros((N, N), dtype=np.float32),
        "rep_mag_full": np.zeros((N, N), dtype=np.float32),
        "force_full": np.zeros((N, N, D), dtype=np.float32),
        "temp_noise": np.zeros((N, D), dtype=np.float32),
        "k_angle": k_angle,
    }

    pygame.init()
    pygame.font.init()
    font_small = pygame.font.SysFont("consolas", 12)
    width, height = 960, 720
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Socratic Graph Physics (OpenGL)")
    clock = pygame.time.Clock()
    physics_last_t = pygame.time.get_ticks()
    hook_last_t = physics_last_t

    init_gl(width, height)
    paused = False

    # view state
    view_center = None
    view_radius = None
    cam_min = 2.2
    cam_scale = 2.2  # fixed distance used after normalization
    smooth = 0.18
    rotate_enabled = False  # default no rotation
    normalize_enabled = True  # default normalization on
    t_start = pygame.time.get_ticks()

        # tensors for live k training (skip-gram weights)
    skip_src_t = None
    skip_dst_t = None
    skip_w_t = None
    if use_torch and train_k_live and train_skip_src is not None and train_skip_src.size:
        dev = pos_t.device
        skip_src_t = torch.tensor(train_skip_src, dtype=torch.long, device=dev)
        skip_dst_t = torch.tensor(train_skip_dst, dtype=torch.long, device=dev)
        skip_w_t = torch.tensor(train_skip_w, dtype=torch.float32, device=dev)
    train_frame_counter = 0
    grad_acc_counter_lockstep = 0

    def do_live_train(pos_live):
        nonlocal train_frame_counter, k_spring_filt, k_rep_filt, k_angle_filt
        if k_opt is None or skip_src_t is None or pos_live is None:
            return
        train_frame_counter += 1
        if train_frame_counter < train_k_every:
            return
        train_frame_counter = 0
        pos_train = pos_live.clone()
        vel_train = torch.zeros_like(pos_train)
        embed_feat_t = pos_live.detach()
        if grad_acc_counter_lockstep % max(1, train_k_grad_acc) == 0:
            k_opt.zero_grad()
        if train_k_field and node_k_net is not None and edge_k_net is not None:
            if train_k_separate_heads:
                rep_raw, ang_raw, _, _ = node_k_net(embed_feat_t)
                k_rep_now = torch.clamp(torch.nn.functional.softplus(rep_raw.squeeze(-1)), min=1e-4, max=5.0).clone()
                k_angle_now = torch.clamp(torch.nn.functional.softplus(ang_raw.squeeze(-1)), min=1e-4, max=50.0).clone()
            else:
                node_out = node_k_net(embed_feat_t)
                k_rep_now = torch.clamp(torch.nn.functional.softplus(node_out[:, 0]), min=1e-4, max=5.0).clone()
                k_angle_now = torch.clamp(torch.nn.functional.softplus(node_out[:, 1]), min=1e-4, max=50.0).clone()
            edge_feat_live = torch.cat([embed_feat_t[i_idx_t], embed_feat_t[j_idx_t]], dim=1)
            k_spring_edge = torch.nn.functional.softplus(edge_k_net(edge_feat_live).squeeze(-1)) + 1e-4
            k_spring_edge = torch.clamp(k_spring_edge, min=1e-4, max=200.0).clone()
        else:
            k_spring_edge = torch.clamp(torch.nn.functional.softplus(k_spring_param), min=1e-4, max=200.0)
            k_rep_now = torch.clamp(torch.nn.functional.softplus(k_rep_param), min=1e-4, max=5.0)
            k_angle_now = torch.clamp(torch.nn.functional.softplus(k_angle_param), min=1e-4, max=50.0)

        if k_smooth > 0.0:
            k_spring_edge = k_smooth * k_spring_filt + (1 - k_smooth) * k_spring_edge
            k_rep_now = k_smooth * k_rep_filt + (1 - k_smooth) * k_rep_now
            k_angle_now = k_smooth * k_angle_filt + (1 - k_smooth) * k_angle_now
            k_spring_filt = k_spring_edge.detach()
            k_rep_filt = k_rep_now.detach()
            k_angle_filt = k_angle_now.detach()
        mean_s = k_spring_edge.mean()
        if torch.isfinite(mean_s) and mean_s < k_norm_mean_floor:
            k_spring_edge = k_spring_edge + (k_norm_mean_floor - mean_s)
        mean_r = k_rep_now.mean()
        if torch.isfinite(mean_r) and mean_r < k_norm_mean_floor:
            k_rep_now = k_rep_now + (k_norm_mean_floor - mean_r)
        mean_a = k_angle_now.mean()
        if torch.isfinite(mean_a) and mean_a < k_norm_mean_floor:
            k_angle_now = k_angle_now + (k_norm_mean_floor - mean_a)
        if k_norm_sum1:
            k_spring_edge = torch.clamp(k_spring_edge, min=1e-6)
            k_rep_now = torch.clamp(k_rep_now, min=1e-6)
            k_angle_now = torch.clamp(k_angle_now, min=1e-6)
            s_sum = k_spring_edge.sum()
            r_sum = k_rep_now.sum()
            a_sum = k_angle_now.sum()
            if torch.isfinite(s_sum) and s_sum > 0:
                k_spring_edge = k_spring_edge / s_sum
            if torch.isfinite(r_sum) and r_sum > 0:
                k_rep_now = k_rep_now / r_sum
            if torch.isfinite(a_sum) and a_sum > 0:
                k_angle_now = k_angle_now / a_sum
        k_spring_edge = torch.clamp(k_spring_edge, min=1e-4, max=200.0)
        k_rep_now = torch.clamp(k_rep_now, min=1e-4, max=5.0)
        k_angle_now = torch.clamp(k_angle_now, min=1e-4, max=50.0)

        p_pos = pos_train
        p_vel = vel_train
        for _ in range(max(1, train_k_unroll)):
            diff = p_pos[j_idx_t] - p_pos[i_idx_t]
            dist = torch.linalg.norm(diff, dim=1) + 1e-9
            if rest_param is not None:
                rest_eff = rest_param
                rest_eff = torch.clamp(rest_eff, min=rest_min, max=rest_max) if rest_max is not None else torch.clamp(rest_eff, min=rest_min)
            else:
                rest_eff = rest_t
            stretch = rest_eff - dist
            edge_k_local = k_edge_t if k_edge_t is not None else 1.0
            k_edge_eff = k_spring_edge
            force_mag = (k_edge_eff * edge_k_local) * stretch / dist
            F_edge = diff * (force_mag / dist)[:, None]
            Ftrain = torch.zeros_like(p_pos)
            Ftrain.index_add_(0, i_idx_t, -F_edge)
            Ftrain.index_add_(0, j_idx_t, F_edge)

            if pair_n1_t.numel() > 0:
                k_angle_hub = k_angle_now[pair_hub_t] if k_angle_now.ndim > 0 else k_angle_now
                if k_angle_hub.mean().item() != 0.0:
                    v1 = p_pos[pair_n1_t] - p_pos[pair_hub_t]
                    v2 = p_pos[pair_n2_t] - p_pos[pair_hub_t]
                    norm1 = v1.norm(dim=1, keepdim=True) + 1e-9
                    norm2 = v2.norm(dim=1, keepdim=True) + 1e-9
                    u1 = v1 / norm1
                    u2 = v2 / norm2
                    dot = (u1 * u2).sum(dim=1, keepdim=True)
                    Fi = -k_angle_hub.unsqueeze(-1) * dot * (u2 - dot * u1) / norm1
                    Fj = -k_angle_hub.unsqueeze(-1) * dot * (u1 - dot * u2) / norm2
                    Ftrain.index_add_(0, pair_n1_t, Fi)
                    Ftrain.index_add_(0, pair_n2_t, Fj)
                    Ftrain.index_add_(0, pair_hub_t, -(Fi + Fj))

            if (k_rep_now.mean() if k_rep_now.ndim > 0 else k_rep_now).item() != 0.0:
                diff_full = p_pos[:, None, :] - p_pos[None, :, :]
                dist2_full = (diff_full * diff_full).sum(dim=2) + softening
                inv_dist = dist2_full.sqrt().reciprocal()
                if k_rep_now.ndim > 0:
                    rep_pair = torch.sqrt(k_rep_now[:, None] * k_rep_now[None, :])
                else:
                    rep_pair = k_rep_now
                rep_mag = rep_pair * inv_dist * inv_dist
                force_full = diff_full * rep_mag[:, :, None] * inv_dist[:, :, None]
                force_full[torch.arange(p_pos.shape[0]), torch.arange(p_pos.shape[0])] = 0.0
                Ftrain = Ftrain + force_full.sum(dim=1)

            acc_train = Ftrain
            p_vel = p_vel + dt * acc_train
            p_pos = p_pos + dt * p_vel

            if constrain_unit_sphere:
                mask = ~fixed_mask_t if fixed_mask_t is not None else torch.ones_like(p_pos[:, 0], dtype=torch.bool, device=p_pos.device)
                if mask.any():
                    p = p_pos[mask]
                    nrm = p.norm(dim=1, keepdim=True).clamp(min=1e-9)
                    n = p / nrm
                    p_pos[mask] = n * constrain_radius_t
                    v = p_vel[mask]
                    v_rad = (v * n).sum(dim=1, keepdim=True) * n
                    p_vel[mask] = v - v_rad

        pi = p_pos[skip_src_t]
        pj = p_pos[skip_dst_t]
        dist_skip = torch.linalg.norm(pi - pj, dim=1)
        loss = (skip_w_t * dist_skip).mean()
        if train_k_neg_samples and train_k_neg_samples > 0:
            Ntrain = p_pos.shape[0]
            neg_i = torch.randint(0, Ntrain, (train_k_neg_samples,), device=p_pos.device)
            neg_j = torch.randint(0, Ntrain, (train_k_neg_samples,), device=p_pos.device)
            mask = neg_i != neg_j
            if mask.any():
                neg_i = neg_i[mask]
                neg_j = neg_j[mask]
                if neg_i.numel() > 0:
                    dn = torch.linalg.norm(p_pos[neg_i] - p_pos[neg_j], dim=1)
                    neg_loss = torch.relu(train_k_neg_margin - dn).mean()
                    loss = loss + train_k_neg_weight * neg_loss
        if train_k_reg and train_k_reg > 0:
            reg = (k_spring_edge * k_spring_edge).mean()
            reg = reg + (k_rep_now * k_rep_now).mean()
            reg = reg + (k_angle_now * k_angle_now).mean()
            loss = loss + train_k_reg * reg
        if not torch.isfinite(loss):
            print("[train-k-live] skip nan/inf loss")
            k_opt.zero_grad()
            return
        loss_to_backprop = loss / max(1, train_k_grad_acc)
        loss_to_backprop.backward()
        grad_acc_counter_lockstep += 1
        if grad_acc_counter_lockstep % max(1, train_k_grad_acc) == 0:
            if train_k_grad_clip and train_k_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(k_opt.param_groups[0]["params"], max_norm=train_k_grad_clip)
            k_opt.step()
            for pg in k_opt.param_groups:
                for p in pg["params"]:
                    if p is None:
                        continue
                    p.data = p.data.clamp(min=-8.0, max=6.0)
            if rest_param is not None:
                with torch.no_grad():
                    if rest_max is not None:
                        rest_param.data = rest_param.data.clamp(min=rest_min, max=rest_max)
                    else:
                        rest_param.data = rest_param.data.clamp(min=rest_min)
        print(
            "[train-k-live] loss="
            f"{loss.item():.4f} ks_mean="
            f"{float(torch.nn.functional.softplus(k_spring_param).mean() if k_spring_param is not None else k_spring_edge.mean()):.3f} "
            f"kr_mean={float(torch.nn.functional.softplus(k_rep_param) if k_rep_param is not None else k_rep_now.mean()):.3f} "
            f"ka_mean={float(torch.nn.functional.softplus(k_angle_param) if k_angle_param is not None else k_angle_now.mean()):.3f} "
            f"[{_pg_stats()}]"
        )

    def orbit_eye(center, cam_dist, t):
        # small, steady two-axis orbit for depth perception
        yaw = 0.35 * math.sin(0.35 * t)
        pitch = 0.2 * math.sin(0.47 * t + 0.5)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x, y, z = 0.0, 0.0, cam_dist
        x, z = x * cy + z * sy, -x * sy + z * cy
        y, z = y * cp - z * sp, y * sp + z * cp
        return center[0] + x, center[1] + y, center[2] + z

    # Diagnostic: check initial spring forces (should be ~0 if rest lengths match positions)
    if springs.size:
        Fchk = scratch["F"]
        compute_forces(pos, springs, neighbors=[], pair_n1=np.zeros((0,), dtype=np.int64), pair_n2=np.zeros((0,), dtype=np.int64), pair_hub=np.zeros((0,), dtype=np.int64), k_spring=1.0, k_rep=0.0, k_angle=0.0, temp=0.0, softening=softening, F=Fchk, scratch=scratch, extra_force_fn=extra_force_fn)
        mags = np.linalg.norm(Fchk, axis=1)
        max_mag = float(mags.max())
        if max_mag > 1e-6:
            top_idx = np.argsort(-mags)[:5]
            print(f"[diagnostic] Initial max |F|={max_mag:.3e}; top nodes: " + ", ".join(f"{node_ids[i]}:{mags[i]:.2e}" for i in top_idx))
        i_idx = springs[:, 0].astype(np.int64)
        j_idx = springs[:, 1].astype(np.int64)
        rest = springs[:, 2]
        dist_now = np.linalg.norm(pos[j_idx] - pos[i_idx], axis=1)
        stretch = dist_now - rest
        max_stretch = float(np.abs(stretch).max()) if stretch.size else 0.0
        if max_stretch > 1e-6:
            top_e = np.argsort(-np.abs(stretch))[:5]
            print("[diagnostic] Largest spring stretch:", ", ".join(f"{i_idx[k]}-{j_idx[k]}:{stretch[k]:.2e}" for k in top_e))

    latest_pos = pos

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if use_torch and stop_evt is not None:
                    stop_evt.set()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    if use_torch and stop_evt is not None:
                        stop_evt.set()
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    rotate_enabled = not rotate_enabled
                if event.key == pygame.K_s:
                    normalize_enabled = not normalize_enabled
                if event.key == pygame.K_t:
                    pos[:] = pos0
                    vel[:] = vel0
                    latest_pos = pos
                    if use_torch:
                        pos_t = torch.tensor(pos0, dtype=torch.float32, device=pos_t.device)
                        vel_t = torch.tensor(vel0, dtype=torch.float32, device=pos_t.device)
                        with lock:
                            state["pos_t"] = pos_t
                            state["vel_t"] = vel_t
                            state["pos"] = pos
                if event.key == pygame.K_p:
                    print(f"pos range: {latest_pos.min(axis=0)} to {latest_pos.max(axis=0)}")

        # always refresh latest positions from physics (torch worker keeps running even if paused)
        if use_torch:
            if lock is not None:
                with lock:
                    latest = state.get("pos")
            else:
                latest = state.get("pos")
            if latest is not None:
                latest_pos = latest

        if not paused:
            if use_torch and physics_lockstep:
                # train first (optional), then run torch physics synchronously
                if train_k_live and train_k_live_lockstep:
                    do_live_train(pos_t)
                for _ in range(steps_per_frame):
                    pos_t, vel_t = torch_step(pos_t, vel_t)
                state["pos_t"] = pos_t
                state["vel_t"] = vel_t
                state["pos"] = pos_t.detach().cpu().numpy().astype(np.float32)
                latest_pos = state["pos"]
            elif not use_torch:
                t_sec = (pygame.time.get_ticks() - t_start) / 1000.0
                temp_now = temp_fn(t_sec) if temp_fn is not None else temp
                if temp_provider is not None:
                    t_dyn = temp_provider()
                    if t_dyn is not None:
                        temp_now = t_dyn
                elif temp_vec is not None:
                    temp_now = temp_vec

                damp_current = damping
                if damping_provider is not None:
                    damp_dyn = damping_provider()
                    if damp_dyn is not None:
                        damp_current = damp_dyn
                elif damping_vec is not None:
                    damp_current = damping_vec

                force_scale_current = force_scale_vec
                if force_scale_provider is not None:
                    fs_dyn = force_scale_provider()
                    if fs_dyn is not None:
                        force_scale_current = fs_dyn

                spring_scale_current = spring_scale_vec
                if spring_k_scale_provider is not None:
                    ss_dyn = spring_k_scale_provider()
                    if ss_dyn is not None:
                        spring_scale_current = ss_dyn
                steps = steps_per_frame
                for _ in range(steps_per_frame):
                    step_verlet(
                        pos,
                        vel,
                        masses,
                        springs,
                        neighbors,
                        pair_n1_arr,
                        pair_n2_arr,
                        pair_hub_arr,
                        dt,
                        k_spring,
                        k_rep,
                        temp_now if not isinstance(temp_now, np.ndarray) else 0.0,
                        softening,
                        damp_current,
                        scratch,
                        extra_force_fn=extra_force_fn,
                        temp_per_node=temp_per_node,
                        force_scale_per_node=force_scale_current,
                        k_edge_scale=spring_scale_current,
                    )
                    if constrain_unit_sphere:
                        mask = ~fixed_mask if fixed_mask is not None else np.ones(pos.shape[0], dtype=bool)
                        if mask.any():
                            p = pos[mask]
                            nrm = np.linalg.norm(p, axis=1, keepdims=True)
                            nrm[nrm < 1e-9] = 1e-9
                            n = p / nrm
                            pos[mask] = n * constrain_radius
                            v = vel[mask]
                            v_rad = (v * n).sum(axis=1, keepdims=True) * n
                            vel[mask] = v - v_rad
                    if max_speed:
                        speed = np.linalg.norm(vel, axis=1)
                        over = speed > max_speed
                        if np.any(over):
                            vel[over] *= (max_speed / (speed[over] + 1e-9))[:, None]

                    if fixed_mask.any():
                        vel[fixed_mask] = 0.0
                        pos[fixed_mask] = pos_fixed[fixed_mask]
            now_ms = pygame.time.get_ticks()
            dt_ms = now_ms - physics_last_t
            physics_fps = steps_per_frame * 1000.0 / dt_ms if dt_ms > 0 else physics_fps
            physics_last_t = now_ms
        
        if use_torch and train_k_live and skip_src_t is not None and not (physics_lockstep and train_k_live_lockstep):
            # live micro-training on k params with current positions
            train_frame_counter += 1
            if train_frame_counter >= train_k_every:
                train_frame_counter = 0
                if lock is not None:
                    with lock:
                        pos_live = state.get("pos_t")
                else:
                    pos_live = state.get("pos_t")
                if pos_live is not None:
                    # differentiable short unroll
                    pos_train = pos_live.clone()
                    vel_train = torch.zeros_like(pos_train)
                    embed_feat_t = pos_live.detach()
                    # gradient accumulation: clear only when starting a new cycle
                    if grad_acc_counter_lockstep % max(1, train_k_grad_acc) == 0:
                        k_opt.zero_grad()
                    if train_k_field and node_k_net is not None and edge_k_net is not None:
                        node_out = node_k_net(embed_feat_t)
                        k_rep_now = torch.clamp(torch.nn.functional.softplus(node_out[:, 0]), min=1e-4, max=5.0).clone()
                        k_angle_now = torch.clamp(torch.nn.functional.softplus(node_out[:, 1]), min=1e-4, max=50.0).clone()
                        edge_feat_live = torch.cat([embed_feat_t[i_idx_t], embed_feat_t[j_idx_t]], dim=1)
                        k_spring_edge = torch.nn.functional.softplus(edge_k_net(edge_feat_live).squeeze(-1)) + 1e-4
                        k_spring_edge = torch.clamp(k_spring_edge, min=1e-4, max=200.0).clone()
                    else:
                        k_spring_edge = torch.clamp(torch.nn.functional.softplus(k_spring_param), min=1e-4, max=200.0)
                        k_rep_now = torch.clamp(torch.nn.functional.softplus(k_rep_param), min=1e-4, max=5.0)
                        k_angle_now = torch.clamp(torch.nn.functional.softplus(k_angle_param), min=1e-4, max=50.0)

                    if k_smooth > 0.0:
                        k_spring_edge = k_smooth * k_spring_filt + (1 - k_smooth) * k_spring_edge
                        k_rep_now = k_smooth * k_rep_filt + (1 - k_smooth) * k_rep_now
                        k_angle_now = k_smooth * k_angle_filt + (1 - k_smooth) * k_angle_now
                        k_spring_filt = k_spring_edge.detach()
                        k_rep_filt = k_rep_now.detach()
                        k_angle_filt = k_angle_now.detach()
                    mean_s = k_spring_edge.mean()
                    if torch.isfinite(mean_s) and mean_s < k_norm_mean_floor:
                        k_spring_edge = k_spring_edge + (k_norm_mean_floor - mean_s)
                    mean_r = k_rep_now.mean()
                    if torch.isfinite(mean_r) and mean_r < k_norm_mean_floor:
                        k_rep_now = k_rep_now + (k_norm_mean_floor - mean_r)
                    mean_a = k_angle_now.mean()
                    if torch.isfinite(mean_a) and mean_a < k_norm_mean_floor:
                        k_angle_now = k_angle_now + (k_norm_mean_floor - mean_a)
                    if k_norm_sum1:
                        k_spring_edge = torch.clamp(k_spring_edge, min=1e-6)
                        k_rep_now = torch.clamp(k_rep_now, min=1e-6)
                        k_angle_now = torch.clamp(k_angle_now, min=1e-6)
                        s_sum = k_spring_edge.sum()
                        r_sum = k_rep_now.sum()
                        a_sum = k_angle_now.sum()
                        if torch.isfinite(s_sum) and s_sum > 0:
                            k_spring_edge = k_spring_edge / s_sum
                        if torch.isfinite(r_sum) and r_sum > 0:
                            k_rep_now = k_rep_now / r_sum
                        if torch.isfinite(a_sum) and a_sum > 0:
                            k_angle_now = k_angle_now / a_sum
                    k_spring_edge = torch.clamp(k_spring_edge, min=1e-4, max=200.0)
                    k_rep_now = torch.clamp(k_rep_now, min=1e-4, max=5.0)
                    k_angle_now = torch.clamp(k_angle_now, min=1e-4, max=50.0)

                    p_pos = pos_train
                    p_vel = vel_train
                    for _ in range(max(1, train_k_unroll)):
                        diff = p_pos[j_idx_t] - p_pos[i_idx_t]
                        dist = torch.linalg.norm(diff, dim=1) + 1e-9
                        if rest_param is not None:
                            rest_eff = rest_param
                            rest_eff = torch.clamp(rest_eff, min=rest_min, max=rest_max) if rest_max is not None else torch.clamp(rest_eff, min=rest_min)
                        else:
                            rest_eff = rest_t
                        stretch = rest_eff - dist
                        edge_k_local = k_edge_t if k_edge_t is not None else 1.0
                        k_edge_eff = k_spring_edge if train_k_field and edge_k_net is not None else k_spring_edge
                        force_mag = (k_edge_eff * edge_k_local) * stretch / dist
                        F_edge = diff * (force_mag / dist)[:, None]
                        Ftrain = torch.zeros_like(p_pos)
                        Ftrain.index_add_(0, i_idx_t, -F_edge)
                        Ftrain.index_add_(0, j_idx_t, F_edge)

                        if pair_n1_t.numel() > 0:
                            k_angle_hub = k_angle_now[pair_hub_t] if k_angle_now.ndim > 0 else k_angle_now
                            if k_angle_hub.mean().item() != 0.0:
                                v1 = p_pos[pair_n1_t] - p_pos[pair_hub_t]
                                v2 = p_pos[pair_n2_t] - p_pos[pair_hub_t]
                                norm1 = v1.norm(dim=1, keepdim=True) + 1e-9
                                norm2 = v2.norm(dim=1, keepdim=True) + 1e-9
                                u1 = v1 / norm1
                                u2 = v2 / norm2
                                dot = (u1 * u2).sum(dim=1, keepdim=True)
                                Fi = -k_angle_hub.unsqueeze(-1) * dot * (u2 - dot * u1) / norm1
                                Fj = -k_angle_hub.unsqueeze(-1) * dot * (u1 - dot * u2) / norm2
                                Ftrain.index_add_(0, pair_n1_t, Fi)
                                Ftrain.index_add_(0, pair_n2_t, Fj)
                                Ftrain.index_add_(0, pair_hub_t, -(Fi + Fj))

                        if (k_rep_now.mean() if k_rep_now.ndim > 0 else k_rep_now).item() != 0.0:
                            diff_full = p_pos[:, None, :] - p_pos[None, :, :]
                            dist2_full = (diff_full * diff_full).sum(dim=2) + softening
                            inv_dist = dist2_full.sqrt().reciprocal()
                            if k_rep_now.ndim > 0:
                                rep_pair = torch.sqrt(k_rep_now[:, None] * k_rep_now[None, :])
                            else:
                                rep_pair = k_rep_now
                            rep_mag = rep_pair * inv_dist * inv_dist
                            force_full = diff_full * rep_mag[:, :, None] * inv_dist[:, :, None]
                            force_full[torch.arange(p_pos.shape[0]), torch.arange(p_pos.shape[0])] = 0.0
                            Ftrain = Ftrain + force_full.sum(dim=1)

                        acc_train = Ftrain
                        p_vel = p_vel + dt * acc_train
                        p_pos = p_pos + dt * p_vel

                        if constrain_unit_sphere:
                            mask = ~fixed_mask_t if fixed_mask_t is not None else torch.ones_like(p_pos[:, 0], dtype=torch.bool, device=p_pos.device)
                            if mask.any():
                                p = p_pos[mask]
                                nrm = p.norm(dim=1, keepdim=True).clamp(min=1e-9)
                                n = p / nrm
                                p_pos[mask] = n * constrain_radius_t
                                v = p_vel[mask]
                                v_rad = (v * n).sum(dim=1, keepdim=True) * n
                                p_vel[mask] = v - v_rad

                    pi = p_pos[skip_src_t]
                    pj = p_pos[skip_dst_t]
                    dist_skip = torch.linalg.norm(pi - pj, dim=1)
                    loss = (skip_w_t * dist_skip).mean()
                    if train_k_neg_samples and train_k_neg_samples > 0:
                        Ntrain = p_pos.shape[0]
                        neg_i = torch.randint(0, Ntrain, (train_k_neg_samples,), device=p_pos.device)
                        neg_j = torch.randint(0, Ntrain, (train_k_neg_samples,), device=p_pos.device)
                        mask = neg_i != neg_j
                        if mask.any():
                            neg_i = neg_i[mask]
                            neg_j = neg_j[mask]
                            if neg_i.numel() > 0:
                                dn = torch.linalg.norm(p_pos[neg_i] - p_pos[neg_j], dim=1)
                                neg_loss = torch.relu(train_k_neg_margin - dn).mean()
                                loss = loss + train_k_neg_weight * neg_loss
                    if train_k_reg and train_k_reg > 0:
                        reg = (k_spring_edge * k_spring_edge).mean()
                        reg = reg + (k_rep_now * k_rep_now).mean()
                        reg = reg + (k_angle_now * k_angle_now).mean()
                        loss = loss + train_k_reg * reg
                    if not torch.isfinite(loss):
                        print("[train-k-live] skip nan/inf loss")
                        k_opt.zero_grad()
                    else:
                        loss_to_backprop = loss / max(1, train_k_grad_acc)
                        loss_to_backprop.backward()
                        grad_acc_counter_lockstep += 1
                        stepped = False
                        if grad_acc_counter_lockstep % max(1, train_k_grad_acc) == 0:
                            if train_k_grad_clip and train_k_grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(k_opt.param_groups[0]["params"], max_norm=train_k_grad_clip)
                            k_opt.step()
                            stepped = True
                            # clamp params to keep softplus outputs sane
                            for pg in k_opt.param_groups:
                                for p in pg["params"]:
                                    if p is None or p.grad is None:
                                        continue
                                    p.data = p.data.clamp(min=-8.0, max=6.0)
                            if rest_param is not None:
                                with torch.no_grad():
                                    if rest_max is not None:
                                        rest_param.data = rest_param.data.clamp(min=rest_min, max=rest_max)
                                    else:
                                        rest_param.data = rest_param.data.clamp(min=rest_min)
                            k_opt.zero_grad()
                        print(
                            "[train-k-live] loss="
                            f"{loss.item():.4f} ks_mean="
                            f"{float(torch.nn.functional.softplus(k_spring_param).mean() if k_spring_param is not None else k_spring_edge.mean()):.3f} "
                            f"kr_mean={float(torch.nn.functional.softplus(k_rep_param) if k_rep_param is not None else k_rep_now.mean()):.3f} "
                            f"ka_mean={float(torch.nn.functional.softplus(k_angle_param) if k_angle_param is not None else k_angle_now.mean()):.3f} "
                            f"[{_pg_stats()}]" + (" step" if stepped else " accum")
                        )

        latest_pos = np.nan_to_num(latest_pos, nan=0.0, posinf=0.0, neginf=0.0)
        mean_latest = latest_pos.mean(axis=0, keepdims=True) if latest_pos.size else None
        pos3d_raw = project_positions(latest_pos, proj, mean_latest)
        mins = pos3d_raw.min(axis=0)
        maxs = pos3d_raw.max(axis=0)
        target_center = (mins + maxs) * 0.5
        target_radius = float(np.max(maxs - mins) * 0.5)
        if target_radius < 1e-3:
            target_radius = 1e-3
        # smooth center/radius for stability
        if view_center is None:
            view_center = target_center
            view_radius = target_radius
        else:
            view_center = (1 - smooth) * view_center + smooth * target_center
            view_radius = (1 - smooth) * view_radius + smooth * target_radius

        scale = 1.0
        if normalize_enabled:
            # enforce fit with 25% margin by scaling render coordinates
            margin = 1.25
            scale = 1.0 / (max(view_radius * margin, 1e-3))
            pos3d = (pos3d_raw - view_center) * scale
            render_center = np.zeros(3, dtype=np.float32)
            cam_dist = max(cam_min, cam_scale)  # fixed distance since positions normalized
        else:
            pos3d = pos3d_raw
            render_center = view_center
            cam_dist = max(cam_min, cam_scale * view_radius)

        t_now = (pygame.time.get_ticks() - t_start) / 1000.0
        if rotate_enabled:
            eye = orbit_eye(render_center, cam_dist, t_now)
        else:
            eye = (render_center[0], render_center[1], render_center[2] + cam_dist)
        fps_render = clock.get_fps()
        text_lines = text_provider() if text_provider is not None else None
        node_labels = label_provider() if label_provider is not None else None
        highlight_idx = highlight_provider() if highlight_provider is not None else None
        node_colors_vis = None
        edge_colors_vis = None
        if show_k_colors and use_torch:
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                rep_v = state.get("vis_node_rep")
                ang_v = state.get("vis_node_angle")
                spring_v = state.get("vis_edge_spring")
                rest_v = state.get("vis_edge_rest")
            thrust_v = state.get("vis_thrust")

            def _norm_with_band(arr, floor=0.2):
                if arr is None or len(arr) == 0:
                    return None, None
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                lo = np.percentile(arr, 5)
                hi = np.percentile(arr, 95)
                if not np.isfinite(lo):
                    lo = 0.0
                if not np.isfinite(hi) or hi - lo < 1e-6:
                    hi = lo + 1e-6
                norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
                peak = float(norm.max()) if norm.size else 0.0
                scale_boost = 1.0
                if peak > 0 and peak < 0.7:
                    scale_boost = 0.7 / peak
                adj = floor + (1.0 - floor) * norm * scale_boost
                adj = np.clip(adj, 0.0, 1.0)
                raw_scale = float(np.max(arr)) if arr.size else 0.0
                return adj, raw_scale

            rep_n, rep_scale = _norm_with_band(rep_v)
            ang_n, ang_scale = _norm_with_band(ang_v)

            # Use the unused middle channel to visualize damping per node (normalized with band).
            damp_raw = None
            if damping_provider_t is not None:
                damp_dyn = damping_provider_t()
                if damp_dyn is not None:
                    damp_raw = damp_dyn.detach().cpu().numpy() if hasattr(damp_dyn, "detach") else np.asarray(damp_dyn)
            elif damping_vec_t is not None:
                damp_raw = damping_vec_t.detach().cpu().numpy()
            elif damping_provider is not None:
                damp_dyn = damping_provider()
                if damp_dyn is not None:
                    damp_raw = np.asarray(damp_dyn)
            elif damping_vec is not None:
                damp_raw = damping_vec
            damp_n, damp_scale = _norm_with_band(damp_raw)

            if rep_n is not None and ang_n is not None and len(rep_n) == len(colors):
                mid = damp_n if damp_n is not None and len(damp_n) == len(rep_n) else np.zeros_like(rep_n)
                node_colors_vis = np.stack([rep_n, mid, ang_n], axis=1).astype(np.float32)
            spring_n, spring_scale = _norm_with_band(spring_v)
            rest_n, rest_scale = _norm_with_band(rest_v)
            if spring_n is not None and rest_n is not None and len(spring_n) == springs.shape[0]:
                edge_colors_vis = np.stack([spring_n, np.zeros_like(spring_n), rest_n], axis=1).astype(np.float32)

            # project thrust vectors for sentence nodes (tail rendering)
            thrust_lines = []
            if thrust_v is not None and mean_latest is not None:
                thrust_np = np.asarray(thrust_v)
                thrust_scale = 0.3 * max(target_radius, 1e-3)
                for idx, is_sentence in enumerate(sentence_mask):
                    if not is_sentence or idx >= thrust_np.shape[0] or idx >= latest_pos.shape[0]:
                        continue
                    vec = thrust_np[idx]
                    mag = float(np.linalg.norm(vec))
                    if mag <= 1e-6:
                        continue
                    line_nd = latest_pos[idx] + (vec / mag) * mag * thrust_scale
                    end_raw = (line_nd - mean_latest) @ proj
                    start_raw = pos3d_raw[idx]
                    start = (start_raw - view_center) * scale if normalize_enabled else start_raw
                    end = (end_raw - view_center) * scale if normalize_enabled else end_raw
                    thrust_lines.append((start.astype(np.float32), end.astype(np.float32)))
            state["vis_thrust_lines"] = thrust_lines if thrust_lines else None

            scale_lines = []
            if rep_scale is not None:
                scale_lines.append(f"Node R rep max {rep_scale:.3g}")
            if damp_scale is not None:
                scale_lines.append(f"Node G damp max {damp_scale:.3g}")
            if ang_scale is not None:
                scale_lines.append(f"Node B angle max {ang_scale:.3g}")
            if spring_scale is not None:
                scale_lines.append(f"Edge R spring max {spring_scale:.3g}")
            if rest_scale is not None:
                scale_lines.append(f"Edge B rest max {rest_scale:.3g}")
            if scale_lines:
                if text_lines is None:
                    text_lines = []
                text_lines = scale_lines + list(text_lines)

        render_colors = node_colors_vis if node_colors_vis is not None else colors
        draw_scene(
            pos3d,
            pos3d,
            render_colors,
            springs,
            render_center,
            eye,
            font_small,
            fps_render,
            physics_fps,
            text_lines=text_lines,
            node_labels=node_labels,
            proj_mats=None,
            highlight_idx=highlight_idx,
            edge_colors=edge_colors_vis,
            thrust_lines=state.get("vis_thrust_lines"),
            sentence_triangles=state.get("sentence_triangles"),
        )

        # optional state hook for external consumers (e.g., text generation)
        if state_hook is not None:
            now_ms_hook = pygame.time.get_ticks()
            if now_ms_hook - hook_last_t >= hook_interval_ms:
                try:
                    state_hook(latest_pos.copy(), node_ids)
                except Exception as exc:  # pragma: no cover - hook robustness
                    print(f"[warn] state_hook error: {exc}")
                hook_last_t = now_ms_hook
        pygame.display.flip()
        clock.tick(60)


def main():
    parser = argparse.ArgumentParser(description="OpenGL/pygame physics animator for socratic graphs")
    parser.add_argument("prephysics_json", help="Path to pre-physics JSON (from socratic_spring_export.py)")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step per sub-step")
    parser.add_argument("--k-spring", type=float, default=1.0, help="Spring stiffness")
    parser.add_argument("--k-rep", type=float, default=0.0, help="Repulsion coefficient (set 0 to rely on thermal noise)")
    parser.add_argument("--k-angle", type=float, default=0.0, help="Angular spreading stiffness along edges (tangent force)")
    parser.add_argument("--temp", type=float, default=0.0, help="Thermal noise strength (stddev of Gaussian force per axis)")
    parser.add_argument("--damping", type=float, default=0.01, help="Velocity damping per sub-step (0-1)")
    parser.add_argument("--softening", type=float, default=1e-2, help="Softening term to avoid singularities")
    parser.add_argument("--steps-per-frame", type=int, default=2, help="Integrator sub-steps per frame")
    parser.add_argument("--max-speed", type=float, default=5.0, help="Clamp particle speeds (set 0 to disable)")
    parser.add_argument("--use-torch", action="store_true", help="Run physics on torch (GPU if available)")
    parser.add_argument("--torch-device", type=str, default="cuda", help="Torch device to use (cuda|cpu)")
    parser.add_argument("--k-norm-mean-floor", type=float, default=1e-3, help="If mean k drops below this, shift up to avoid collapse")
    parser.add_argument("--k-norm-sum1", action="store_true", help="Normalize each k set to sum to 1 after smoothing")
    parser.add_argument("--train-k-neg-samples", type=int, default=0, help="Number of negative pairs to sample per live train step (0 to disable)")
    parser.add_argument("--train-k-neg-margin", type=float, default=0.5, help="Hinge margin for negative pairs")
    parser.add_argument("--train-k-neg-weight", type=float, default=1.0, help="Loss weight for negative pairs")
    parser.add_argument("--train-k-thrust", action="store_true", help="Predict per-sentence thrust vectors from k field net (torch only)")
    parser.add_argument("--train-k-damp", action="store_true", help="Predict per-node damping from k field net (torch only)")
    parser.add_argument("--train-rest", action="store_true", help="Learn edge rest lengths during live training (torch only)")
    parser.add_argument("--rest-min", type=float, default=1e-4, help="Minimum rest length when learning rest (clamp)")
    parser.add_argument("--rest-max", type=float, default=None, help="Maximum rest length when learning rest (clamp; blank for none)")
    parser.add_argument("--train-k-reg", type=float, default=0.0, help="L2 regularization weight for learned ks")
    parser.add_argument("--k-smooth", type=float, default=0.0, help="EMA smoothing factor (0-1) for k values")
    parser.add_argument("--show-k-colors", action="store_true", help="Visualize k rep/angle per node and k spring/rest per edge via colors")
    parser.add_argument("--train-k-grad-acc", type=int, default=1, help="Accumulate gradients over this many live-train iterations before optimizer step")
    parser.add_argument("--train-k-node-depth", type=int, default=1, help="Hidden depth (>=1) for node k network trunk")
    parser.add_argument("--train-k-edge-depth", type=int, default=1, help="Hidden depth (>=1) for edge k network")
    parser.add_argument("--train-k-separate-heads", action="store_true", help="Use separate heads for rep/angle/damp/thrust (shared trunk)")

    args = parser.parse_args()
    run(
        path=args.prephysics_json,
        dt=args.dt,
        k_spring=args.k_spring,
        k_rep=args.k_rep,
        k_angle=args.k_angle,
        temp=args.temp,
        damping=args.damping,
        softening=args.softening,
        steps_per_frame=args.steps_per_frame,
        max_speed=None if args.max_speed == 0 else args.max_speed,
        use_torch=args.use_torch,
        torch_device=args.torch_device,
        train_k_reg=args.train_k_reg,
        k_smooth=args.k_smooth,
        k_norm_mean_floor=args.k_norm_mean_floor,
        k_norm_sum1=args.k_norm_sum1,
        train_k_neg_samples=args.train_k_neg_samples,
        train_k_neg_margin=args.train_k_neg_margin,
        train_k_neg_weight=args.train_k_neg_weight,
        train_rest=args.train_rest,
        rest_min=args.rest_min,
        rest_max=args.rest_max,
        show_k_colors=args.show_k_colors,
    )


if __name__ == "__main__":
    main()
