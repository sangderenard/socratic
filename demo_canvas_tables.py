from __future__ import annotations

"""
Small pygame demo that places a `GP_CanvasContext` at the bottom of the hitbox demo
and spawns two simple modules (tables) with left/right LED contact nodes connected
by a rope simulated and rendered by the canvas C backend.

Run:
    python demo_canvas_tables.py

Requires: pygame, and the native `c_menu` library built in `c_menu/build/Release`.
"""

import sys
import time
import pathlib
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pygame

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "c_menu"))
import table_api as ta

# Locate native c_menu DLL
DLL_PATH = ROOT / "c_menu" / "build" / "Release" / "c_menu.dll"

@dataclass
class CanvasModuleDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("w", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("left_contacts", ctypes.c_int32),
        ("right_contacts", ctypes.c_int32),
        ("label", ctypes.c_char * 64),
    ]

@dataclass
class CanvasEdgeDesc(ctypes.Structure):
    _fields_ = [
        ("a_module", ctypes.c_int32),
        ("a_contact_idx", ctypes.c_int32),
        ("b_module", ctypes.c_int32),
        ("b_contact_idx", ctypes.c_int32),
    ]


def load_canvas_lib():
    if not DLL_PATH.exists():
        raise OSError(f"c_menu DLL not found at {DLL_PATH}. Build c_menu first.")
    lib = ctypes.CDLL(str(DLL_PATH))
    # prototypes
    lib.gp_canvas_create.restype = ctypes.c_void_p
    lib.gp_canvas_create.argtypes = (ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_destroy.restype = None
    lib.gp_canvas_destroy.argtypes = (ctypes.c_void_p,)
    lib.gp_canvas_add_module.restype = ctypes.c_int
    lib.gp_canvas_add_module.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasModuleDesc))
    lib.gp_canvas_move_module.restype = ctypes.c_int
    lib.gp_canvas_move_module.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_add_edge.restype = ctypes.c_int
    lib.gp_canvas_add_edge.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasEdgeDesc))
    lib.gp_canvas_set_cable_style.restype = ctypes.c_int
    lib.gp_canvas_set_cable_style.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_set_edge_hues.restype = ctypes.c_int
    lib.gp_canvas_set_edge_hues.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float)
    lib.gp_canvas_raster_rgba.restype = ctypes.c_int
    lib.gp_canvas_raster_rgba.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int)
    lib.gp_canvas_on_click.restype = ctypes.c_int
    lib.gp_canvas_on_click.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    # new mouse/drag and table helpers
    # prefer the real on_mouse_down symbol if exported; otherwise fall back to on_click
    try:
        lib.gp_canvas_on_mouse_down.restype = ctypes.c_int
        lib.gp_canvas_on_mouse_down.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    except AttributeError:
        # fallback: alias to gp_canvas_on_click so older builds still work
        lib.gp_canvas_on_mouse_down = lib.gp_canvas_on_click
        lib.gp_canvas_on_mouse_down.restype = ctypes.c_int
        lib.gp_canvas_on_mouse_down.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_on_mouse_move.restype = ctypes.c_int
    lib.gp_canvas_on_mouse_move.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_on_mouse_up.restype = ctypes.c_int
    lib.gp_canvas_on_mouse_up.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    lib.gp_canvas_step.restype = ctypes.c_int
    lib.gp_canvas_step.argtypes = (ctypes.c_void_p, ctypes.c_float)
    lib.gp_canvas_attach_table.restype = ctypes.c_int
    lib.gp_canvas_attach_table.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_create_table.restype = ctypes.c_int
    lib.gp_canvas_create_table.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.gp_canvas_destroy_table.restype = ctypes.c_int
    lib.gp_canvas_destroy_table.argtypes = (ctypes.c_void_p, ctypes.c_int)
    return lib


def make_module(x, y, w, h, left, right, label: str) -> CanvasModuleDesc:
    m = CanvasModuleDesc()
    m.x = int(x)
    m.y = int(y)
    m.w = int(w)
    m.h = int(h)
    m.left_contacts = int(left)
    m.right_contacts = int(right)
    b = label.encode("utf-8")[:63]
    padded = b + b"\0" * (64 - len(b))
    m.label = padded
    return m


def main() -> int:
    pygame.init()
    try:
        style = ta.default_style(width_px=1000)
        ctx = ta.TableContext(style=style)
        # build simple columns and rows using table_api helpers
        c1 = ta.GP_TableColumn()
        c1.kind = ta.GP_TableCellKind.LEDS
        c1.width_px = 140
        c1.align = 0
        c2 = ta.GP_TableColumn()
        c2.kind = ta.GP_TableCellKind.AXIS
        c2.width_px = 240
        c2.align = 0
        cols = [c1, c2]
        r = ta.make_row(kind=ta.GP_TableRowKind.NOTE, label="Demo row", depth=0, expanded=True, cells=(ta.make_cell_leds(flags=0b111111111), ta.make_cell_axis(value=0.0)))
        rows = [r]
        ctx.set_columns(cols)
        ctx.set_rows(rows)
    except OSError as e:
        print("Failed to load native table library (c_menu). Ensure it is built and on disk.")
        print(e)
        return 1

    # table render dims
    g = ctx.geom()
    table_w = int(g.width_px)
    table_h = int(g.height_px)

    canvas_h = 220
    info_h = 24
    screen = pygame.display.set_mode((table_w, table_h + canvas_h + info_h))
    pygame.display.set_caption("Canvas + Tables demo")

    # load canvas lib
    try:
        lib = load_canvas_lib()
    except Exception as e:
        print("Failed to load canvas API:", e)
        return 1

    # create canvas instance sized to table width and canvas_h
    cctx = lib.gp_canvas_create(table_w, canvas_h)
    if not cctx:
        print("gp_canvas_create failed")
        return 1

    # set a pleasant cable style
    lib.gp_canvas_set_cable_style(cctx, 5, 2)
    # set some hues to demonstrate colored core (copy of a small travelling gradient)
    hues = (ctypes.c_float * 3)(0.0, 0.33, 0.66)
    lib.gp_canvas_set_edge_hues(cctx, hues, 3, ctypes.c_float(0.9))

    # Start with an empty canvas (no pre-spawned modules or edges).
    # The demo previously spawned two example modules and connected them; that
    # fake content has been removed so the canvas starts empty and the host
    # can programmatically add modules when desired.

    # buffer for canvas RGBA (table_w * canvas_h * 4)
    buf_len = table_w * canvas_h * 4
    buf = (ctypes.c_ubyte * buf_len)()

    # Ensure canvas has created its internal rope_sim by forcing one raster.
    ok = lib.gp_canvas_raster_rgba(cctx, buf, buf_len)

    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont(None, 18)

    # buffer for canvas RGBA (table_w * canvas_h * 4)
    buf_len = table_w * canvas_h * 4
    buf = (ctypes.c_ubyte * buf_len)()

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            # handle mouse down/up/move and forward into canvas when inside canvas area
            elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                mx, my = ev.pos
                if my >= table_h and my < table_h + canvas_h:
                    cx = int(mx)
                    cy = int(my - table_h)
                    if ev.type == pygame.MOUSEBUTTONDOWN:
                        lib.gp_canvas_on_mouse_down(cctx, cx, cy)
                    elif ev.type == pygame.MOUSEBUTTONUP:
                        lib.gp_canvas_on_mouse_up(cctx, cx, cy)
                    elif ev.type == pygame.MOUSEMOTION:
                        lib.gp_canvas_on_mouse_move(cctx, cx, cy)

        # render table at top using table_api
        w, h, rgba, geom, hits = ctx.render_with_state(ta.GP_TableRenderState())
        surf_table = pygame.image.frombuffer(bytearray(rgba), (w, h), "RGBA").convert_alpha()

        # step canvas sim to enable live solving (dt seconds)
        lib.gp_canvas_step(cctx, ctypes.c_float(1.0/60.0))
        # render canvas into buffer and create surface
        ok = lib.gp_canvas_raster_rgba(cctx, buf, buf_len)
        if ok:
            # construct bytes from buf
            pybuf = bytes(bytearray(buf))
            surf_canvas = pygame.image.frombuffer(pybuf, (table_w, canvas_h), "RGBA").convert_alpha()
        else:
            surf_canvas = None

        screen.fill((12, 12, 16))
        screen.blit(surf_table, (0, 0))
        if surf_canvas:
            screen.blit(surf_canvas, (0, table_h))

        # info line
        info_rect = pygame.Rect(0, table_h + canvas_h, table_w, info_h)
        pygame.draw.rect(screen, (30, 30, 36), info_rect)
        txt = font.render("Canvas demo: two modules connected by a colored spline", True, (220, 220, 220))
        screen.blit(txt, (8, table_h + canvas_h + 4))

        pygame.display.flip()
        clock.tick(60)

    # cleanup
    lib.gp_canvas_destroy(cctx)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
