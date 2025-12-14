from __future__ import annotations

import pygame


def render_cell_text(font: pygame.font.Font, text: str) -> pygame.Surface:
    """Render monospace-ish HUD text with per-character black background cells."""
    s = str(text).expandtabs(4)
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
