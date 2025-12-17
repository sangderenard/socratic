#include "table_abi.h"
#include "menu_waveform_abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <cstdio>
#include <memory>

namespace {

struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

static void memset_rect(uint8_t* img, int w, int h, int pitch, int x0, int y0, int rw, int rh, Color c);

static inline Color to_color(const uint8_t rgba[4]) {
    return Color{rgba[0], rgba[1], rgba[2], rgba[3]};
}
static void draw_calib_strip(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int cw,
    int ch,
    bool inv_on,
    int mode,
    Color dim,
    Color active,
    Color inv_col) {
    if (!img || cw <= 0 || ch <= 0) return;
    const int slots = 5; // INV, TRIM, CAP, DED, RST
    int gap = 2;
    int slot_w = std::max(4, (cw - gap * (slots - 1)) / slots);
    int slot_h = std::max(4, ch);
    int y = y0 + (ch - slot_h) / 2;

    for (int i = 0; i < slots; ++i) {
        int x = x0 + i * (slot_w + gap);
        Color c = dim;
        if (i == 0 && inv_on) {
            c = inv_col;
        } else if (i == mode && mode > 0) {
            c = active;
        }
        memset_rect(img, w, h, pitch, x, y, slot_w, slot_h, c);
        // Thin outline for readability.
        memset_rect(img, w, h, pitch, x, y, slot_w, 1, Color{255, 255, 255, 32});
        memset_rect(img, w, h, pitch, x, y + slot_h - 1, slot_w, 1, Color{0, 0, 0, 64});
    }
}

struct Style {
    int w = 640;
    int row_h = 20;
    int indent = 14;
    int expand_w = 12;
    int name_w = 160;

    Color bg{8, 8, 12, 255};
    Color bg_sel{18, 18, 26, 255};
    Color hdr{20, 20, 28, 255};
    Color text{240, 240, 245, 255};
    Color text_hdr{255, 255, 255, 255};
    Color led_on{255, 210, 90, 255};
    Color led_off{70, 70, 80, 255};
    Color led_edge{255, 255, 255, 255};
    Color axis_bg{18, 18, 22, 255};
    Color axis_tick{200, 200, 200, 255};
    Color axis_val{255, 255, 140, 255};
    Color timer{110, 180, 255, 255};
    Color wave_bg{6, 6, 8, 255};
    Color wave_fg{255, 255, 255, 255};

    uint32_t led_mask[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
    uint32_t led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
};

static Style load_style(const GP_TableStyle* s) {
    Style out;
    if (!s) return out;
    out.w = std::max(64, int(s->width_px));
    out.row_h = std::max(8, int(s->row_h_px));
    out.indent = std::max(0, int(s->indent_px));
    out.expand_w = std::max(8, int(s->expand_w_px));
    out.name_w = std::max(20, int(s->name_w_px));
    out.bg = to_color(s->bg_rgba);
    out.bg_sel = to_color(s->bg_sel_rgba);
    out.hdr = to_color(s->hdr_rgba);
    out.text = to_color(s->text_rgba);
    out.text_hdr = to_color(s->text_hdr_rgba);
    out.led_on = to_color(s->led_on_rgba);
    out.led_off = to_color(s->led_off_rgba);
    out.led_edge = to_color(s->led_edge_rgba);
    out.axis_bg = to_color(s->axis_bg_rgba);
    out.axis_tick = to_color(s->axis_tick_rgba);
    out.axis_val = to_color(s->axis_val_rgba);
    out.timer = to_color(s->timer_rgba);
    out.wave_bg = to_color(s->wave_bg_rgba);
    out.wave_fg = to_color(s->wave_fg_rgba);
    for (int i = 0; i < 9; ++i) out.led_mask[i] = s->led_mask[i];
    out.led_edge_mask = s->led_edge_mask;
    return out;
}

static GP_TableStyle make_default_style() {
    GP_TableStyle s{};
    s.width_px = 640;
    s.row_h_px = 20;
    s.indent_px = 14;
    s.expand_w_px = 12;
    s.name_w_px = 160;
    auto set = [](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
    set(s.bg_rgba, 8, 8, 12, 255);
    set(s.bg_sel_rgba, 18, 18, 26, 255);
    set(s.hdr_rgba, 20, 20, 28, 255);
    set(s.text_rgba, 240, 240, 245, 255);
    set(s.text_hdr_rgba, 255, 255, 255, 255);
    set(s.led_on_rgba, 255, 210, 90, 255);
    set(s.led_off_rgba, 70, 70, 80, 255);
    set(s.led_edge_rgba, 255, 255, 255, 255);
    set(s.axis_bg_rgba, 18, 18, 22, 255);
    set(s.axis_tick_rgba, 200, 200, 200, 255);
    set(s.axis_val_rgba, 255, 255, 140, 255);
    set(s.timer_rgba, 110, 180, 255, 255);
    set(s.wave_bg_rgba, 6, 6, 8, 255);
    set(s.wave_fg_rgba, 255, 255, 255, 255);
    uint32_t default_bits[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
    for (int i = 0; i < 9; ++i) s.led_mask[i] = default_bits[i];
    s.led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
    return s;
}

static void memset_rect(uint8_t* img, int w, int h, int pitch, int x0, int y0, int rw, int rh, Color c) {
    if (!img) return;
    int x1 = std::min(w, x0 + rw);
    int y1 = std::min(h, y0 + rh);
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    if (x0 >= x1 || y0 >= y1) return;
    for (int y = y0; y < y1; ++y) {
        uint8_t* row = img + y * pitch + x0 * 4;
        for (int x = x0; x < x1; ++x) {
            row[0] = c.r;
            row[1] = c.g;
            row[2] = c.b;
            row[3] = c.a;
            row += 4;
        }
    }
}

static void draw_circle(uint8_t* img, int w, int h, int pitch, int cx, int cy, int r, Color c) {
    if (!img) return;
    const int r2 = r * r;
    for (int dy = -r; dy <= r; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= w) continue;
            if (dx * dx + dy * dy > r2) continue;
            uint8_t* p = img + y * pitch + x * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = c.a;
        }
    }
}

static void draw_line(uint8_t* img, int w, int h, int pitch, int x0, int y0, int x1, int y1, int thick, Color c) {
    if (!img) return;
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int x = x0;
    int y = y0;
    while (true) {
        memset_rect(img, w, h, pitch, x - thick / 2, y - thick / 2, thick, thick, c);
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

static void draw_axis_bar_ext(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int bar_w,
    int bar_h,
    float v_cur,
    float v_min,
    float v_max,
    float cap_min,
    float cap_max,
    float trim,
    float deadzone,
    bool seen_min,
    bool seen_max,
    Color bg,
    Color tick,
    Color val) {
    if (!img || bar_w <= 0 || bar_h <= 0) return;

    auto clamp_unit = [](float v) { return std::max(-1.0f, std::min(1.0f, v)); };
    auto x_at = [&](float vv) {
        float v01 = 0.5f * (clamp_unit(vv) + 1.0f);
        return x0 + int(std::lround(v01 * float(std::max(1, bar_w - 1))));
    };

    // Background
    memset_rect(img, w, h, pitch, x0, y0, bar_w, bar_h, bg);

    // End ticks: red if never reached, white otherwise.
    Color col_end_ok{255, 255, 255, 255};
    Color col_end_bad{255, 80, 80, 255};
    draw_line(img, w, h, pitch, x_at(-1.0f), y0 - 2, x_at(-1.0f), y0 + bar_h + 2, 1, seen_min ? col_end_ok : col_end_bad);
    draw_line(img, w, h, pitch, x_at(+1.0f), y0 - 2, x_at(+1.0f), y0 + bar_h + 2, 1, seen_max ? col_end_ok : col_end_bad);

    // Center tick
    draw_line(img, w, h, pitch, x_at(0.0f), y0 - 2, x_at(0.0f), y0 + bar_h + 2, 1, col_end_ok);

    // Deadzone band around trim point
    float cap_lo = cap_min;
    float cap_hi = cap_max;
    if (!(cap_hi > cap_lo + 1e-6f)) {
        cap_lo = -1.0f;
        cap_hi = 1.0f;
    }
    float mid = 0.5f * (cap_lo + cap_hi);
    float span = 0.5f * (cap_hi - cap_lo);
    if (span < 1e-6f) span = 1e-6f;

    // Trim tick (orange) and deadzone bounds (blue) if provided.
    Color col_trim{255, 140, 80, 255};
    Color col_dz{64, 140, 255, 255};
    float trim_raw = mid + trim * span;
    draw_line(img, w, h, pitch, x_at(trim_raw), y0 - 2, x_at(trim_raw), y0 + bar_h + 2, 1, col_trim);

    if (deadzone > 0.0f) {
        float dz_raw = deadzone * span;
        int x0_dz = x_at(trim_raw - dz_raw);
        int x1_dz = x_at(trim_raw + dz_raw);
        int xa = std::min(x0_dz, x1_dz);
        int xb = std::max(x0_dz, x1_dz);
        memset_rect(img, w, h, pitch, xa, y0, std::max(1, xb - xa + 1), bar_h, Color{32, 32, 32, 255});
        draw_line(img, w, h, pitch, x0_dz, y0 - 2, x0_dz, y0 + bar_h + 2, 1, col_dz);
        draw_line(img, w, h, pitch, x1_dz, y0 - 2, x1_dz, y0 + bar_h + 2, 1, col_dz);
    }

    // Observed min/max ticks (green)
    Color col_seen{64, 255, 64, 255};
    draw_line(img, w, h, pitch, x_at(v_min), y0 - 1, x_at(v_min), y0 + bar_h + 1, 1, col_seen);
    draw_line(img, w, h, pitch, x_at(v_max), y0 - 1, x_at(v_max), y0 + bar_h + 1, 1, col_seen);

    // Current value (yellow)
    draw_line(img, w, h, pitch, x_at(v_cur), y0 - 3, x_at(v_cur), y0 + bar_h + 3, 2, val);
}

static void draw_timers(uint8_t* img, int w, int h, int pitch, int x0, int y0, int w_px, int h_px, float hold_s, float last_s, Color c) {
    if (!img || w_px <= 0 || h_px <= 0) return;
    const float clamp_hold = std::min(1.0f, std::max(0.0f, hold_s));
    const float clamp_last = std::min(1.0f, std::max(0.0f, last_s));
    int hold_w = int(std::lround(clamp_hold * float(w_px)));
    int last_w = int(std::lround(clamp_last * float(w_px)));
    int mid = y0 + h_px / 2;
    memset_rect(img, w, h, pitch, x0, mid - h_px / 2, hold_w, h_px / 2, c);
    memset_rect(img, w, h, pitch, x0, mid, last_w, h_px / 2, c);
}

static void draw_waveform(uint8_t* img, int w, int h, int pitch, int x0, int y0, int ww, int wh, const GP_MenuWaveform* wf, Color bg, Color fg) {
    if (!img || ww <= 0 || wh <= 0) return;
    memset_rect(img, w, h, pitch, x0, y0, ww, wh, bg);
    if (!wf) return;
    const int need = ww * wh * 4;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(need));
    if (!gp_menu_waveform_raster_rgba(const_cast<GP_MenuWaveform*>(wf), tmp.data(), int32_t(tmp.size()))) {
        return;
    }
    for (int yy = 0; yy < wh; ++yy) {
        for (int xx = 0; xx < ww; ++xx) {
            int tx = x0 + xx;
            int ty = y0 + yy;
            if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
            const uint8_t* src = &tmp[(yy * ww + xx) * 4];
            uint8_t* dst = img + ty * pitch + tx * 4;
            dst[0] = uint8_t((int(src[0]) * int(fg.r)) / 255);
            dst[1] = uint8_t((int(src[1]) * int(fg.g)) / 255);
            dst[2] = uint8_t((int(src[2]) * int(fg.b)) / 255);
            dst[3] = src[3];
        }
    }
}

static void draw_expand_box(uint8_t* img, int w, int h, int pitch, int x0, int y0, int size, bool expanded, Color c) {
    int s = std::max(6, size);
    int y_mid = y0 + s / 2;
    int x_mid = x0 + s / 2;
    // box outline
    memset_rect(img, w, h, pitch, x0, y0, s, 1, c);
    memset_rect(img, w, h, pitch, x0, y0 + s - 1, s, 1, c);
    memset_rect(img, w, h, pitch, x0, y0, 1, s, c);
    memset_rect(img, w, h, pitch, x0 + s - 1, y0, 1, s, c);
    // plus/minus
    draw_line(img, w, h, pitch, x0 + 2, y_mid, x0 + s - 3, y_mid, 1, c);
    if (!expanded) {
        draw_line(img, w, h, pitch, x_mid, y0 + 2, x_mid, y0 + s - 3, 1, c);
    }
}

static void draw_led_strip(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int cy,
    int cw,
    int count,
    uint32_t on_mask,
    uint32_t edge_mask,
    uint32_t active_mask,
    int radius,
    Color on,
    Color off,
    Color edge,
    Color active_only) {
    if (!img || cw <= 0 || count <= 0) return;
    int led_spacing = std::max(radius * 2 + 2, cw / std::max(1, count + 1));
    int cx0 = x0 + led_spacing;
    for (int li = 0; li < count; ++li) {
        int cx = cx0 + li * led_spacing;
        uint32_t bit = 1u << li;
        bool on_b = (on_mask & bit) != 0;
        bool edge_b = (edge_mask & bit) != 0;
        bool active_but_off = !on_b && (active_mask & bit);
        Color lc = on_b ? on : (active_but_off ? active_only : off);
        if (edge_b) lc = edge;
        draw_circle(img, w, h, pitch, cx, cy, radius, lc);
    }
}

static void draw_scrollbar(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int cw,
    int ch,
    float value01,
    int total_rows,
    int visible_rows,
    bool arrow_up_pressed,
    bool arrow_dn_pressed,
    Color track,
    Color thumb,
    Color arrow_up,
    Color arrow_dn,
    Color stripe_even,
    Color stripe_odd) {
    if (!img || cw <= 4 || ch <= 4) return;
    int bar_x = x0;
    int bar_w = std::max(6, cw);
    int bar_h = ch;
    int arrow_h = std::max(8, bar_h / 6);
    // Track
    memset_rect(img, w, h, pitch, bar_x, y0, bar_w, bar_h, track);
    // Arrows
    memset_rect(img, w, h, pitch, bar_x, y0, bar_w, arrow_h, arrow_up_pressed ? arrow_dn : arrow_up);
    memset_rect(img, w, h, pitch, bar_x, y0 + bar_h - arrow_h, bar_w, arrow_h, arrow_dn_pressed ? arrow_up : arrow_dn);

    int track_y0 = y0 + arrow_h;
    int track_h = bar_h - 2 * arrow_h;
    if (track_h <= 0) return;

    // Alternating stripes to suggest rows.
    int rows = std::max(1, total_rows);
    float stripe_h = float(track_h) / float(rows);
    for (int r = 0; r < rows; ++r) {
        int sy0 = track_y0 + int(std::floor(stripe_h * r));
        int sy1 = track_y0 + int(std::floor(stripe_h * (r + 1)));
        int sh = std::max(1, sy1 - sy0);
        Color sc = (r % 2 == 0) ? stripe_even : stripe_odd;
        memset_rect(img, w, h, pitch, bar_x, sy0, bar_w, sh, sc);
    }

    // Thumb size ~ visible/total.
    float vis_frac = std::clamp(float(visible_rows) / float(std::max(visible_rows, total_rows)), 0.05f, 1.0f);
    int thumb_h = std::max(6, int(vis_frac * track_h));
    float clamped_v = std::clamp(value01, 0.0f, 1.0f);
    int thumb_y = track_y0 + int((track_h - thumb_h) * clamped_v);
    memset_rect(img, w, h, pitch, bar_x, thumb_y, bar_w, thumb_h, thumb);
    // Outline
    memset_rect(img, w, h, pitch, bar_x, thumb_y, bar_w, 1, Color{255, 255, 255, 48});
    memset_rect(img, w, h, pitch, bar_x, thumb_y + thumb_h - 1, bar_w, 1, Color{0, 0, 0, 96});
}

static void compute_columns(const GP_TableColumn* cols, int col_count, int table_w, int name_w, int* out_x0, int* out_w) {
    int x = name_w;
    for (int i = 0; i < col_count && i < 8; ++i) {
        int cw = std::max(1, int(cols[i].width_px));
        out_x0[i] = x;
        out_w[i] = cw;
        x += cw;
    }
    for (int i = col_count; i < 8; ++i) {
        out_x0[i] = x;
        out_w[i] = 0;
    }
}

} // namespace

extern "C" {

int32_t gp_table_calc_size(const GP_TableStyle* style, int32_t row_count, int32_t col_count, GP_TableGeom* out_geom) {
    if (!out_geom || row_count < 0 || col_count < 0) return 0;
    Style st = load_style(style);
    out_geom->width_px = st.w;
    out_geom->height_px = std::max(1, int(row_count) * st.row_h);
    compute_columns(nullptr, 0, st.w, st.name_w, out_geom->col_x0, out_geom->col_w);
    return 1;
}

int32_t gp_table_raster_rgba_with_hits(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!rows || !cols || !out_rgba || row_count <= 0 || col_count <= 0) return 0;
    Style st = load_style(style);
    int w = st.w;
    int h = std::max(1, int(row_count) * st.row_h);
    const int need = w * h * 4;
    if (out_len_bytes < need) return 0;

    const bool want_hits = hitboxes_out != nullptr && hitboxes_cap > 0;
    int hit_count = 0;
    auto push_hit = [&](int x0, int y0, int x1, int y1, int row_idx, int col_idx, GP_TableCellKind ck, GP_TableHitPart part, int aux0, int aux1, uint32_t flags) {
        if (!want_hits) return;
        if (hit_count >= hitboxes_cap) return;
        GP_TableHitBox hb{};
        hb.x0 = x0;
        hb.y0 = y0;
        hb.x1 = x1;
        hb.y1 = y1;
        hb.row_idx = row_idx;
        hb.col_idx = col_idx;
        hb.cell_kind = static_cast<int32_t>(ck);
        hb.part = static_cast<int32_t>(part);
        hb.aux0 = aux0;
        hb.aux1 = aux1;
        hb.flags = flags;
        hitboxes_out[hit_count++] = hb;
    };

    // Clear
    memset(out_rgba, 0, static_cast<std::size_t>(need));
    memset_rect(out_rgba, w, h, w * 4, 0, 0, w, h, st.bg);

    int col_x0[8] = {0};
    int col_w[8] = {0};
    compute_columns(cols, col_count, w, st.name_w, col_x0, col_w);
    if (out_geom) {
        out_geom->width_px = w;
        out_geom->height_px = h;
        for (int i = 0; i < 8; ++i) {
            out_geom->col_x0[i] = col_x0[i];
            out_geom->col_w[i] = col_w[i];
        }
    }

    const int led_count = 9;
    for (int i = 0; i < row_count; ++i) {
        const GP_TableRow& r = rows[i];
        int y0 = i * st.row_h;
        Color bg = (r.kind == GP_TABLE_ROW_HEADER) ? st.hdr : (r.selected ? st.bg_sel : st.bg);
        memset_rect(out_rgba, w, h, w * 4, 0, y0, w, st.row_h, bg);

        // Expand box only on header/device rows to avoid clutter on leaf rows.
        int indent_px = st.indent * std::max(0, r.depth);
        if (r.kind == GP_TABLE_ROW_HEADER || r.kind == GP_TABLE_ROW_DEVICE) {
            int exp_x = 4 + indent_px;
            int exp_y = y0 + (st.row_h - st.expand_w) / 2;
            draw_expand_box(out_rgba, w, h, w * 4, exp_x, exp_y, st.expand_w, r.expanded != 0, st.text);
            push_hit(exp_x, exp_y, exp_x + st.expand_w, exp_y + st.expand_w, i, -1, GP_TABLE_CELL_TEXT, GP_TABLE_HIT_EXPAND, 0, 0, 0);
        }
        // Label gutter still reserves name_w; callers overlay text as needed.

        // Cells
        int cy = y0 + st.row_h / 2;
        for (int c = 0; c < r.cell_count && c < col_count && c < 8; ++c) {
            const GP_TableCell& cell = r.cells[c];
            int x0 = col_x0[c];
            int cw = col_w[c];
            if (cw <= 0) continue;

            // Whole-cell hitbox (moved to emit AFTER per-part hitboxes)
            // (previously emitted here which caused coarse hits to shadow per-LED hits)

            switch (cell.kind) {
                case GP_TABLE_CELL_LEDS: {
                    uint32_t on_mask = cell.flags;
                    uint32_t edge_mask = st.led_edge_mask ? (cell.flags & st.led_edge_mask) : 0u;
                    int eff_w = std::max(1, cw - 4);
                    int led_spacing = std::max(4, eff_w / std::max(1, led_count + 1));
                    int cx0 = x0 + 2 + led_spacing;
                    for (int li = 0; li < led_count; ++li) {
                        int cx = cx0 + li * led_spacing;
                        push_hit(cx - 6, cy - 6, cx + 6, cy + 6, i, c, GP_TABLE_CELL_LEDS, GP_TABLE_HIT_LED, li, 0, 0);
                    }
                    draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy, eff_w, led_count, on_mask, edge_mask, on_mask, 4, st.led_on, st.led_off, st.led_edge, st.led_off);
                    break;
                }
                case GP_TABLE_CELL_LEDS_ARG: {
                    int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                    if (count == 0) count = (cell.flags & 0xFF);
                    if (count == 0) count = 12;
                    uint32_t linked_mask = cell.flags;
                    uint32_t required_mask = static_cast<uint32_t>(cell.reserved0);
                    if (required_mask == 0 && count > 0) required_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
                    int eff_w = std::max(1, cw - 4);
                    int led_spacing = std::max(4, eff_w / std::max(1, count + 1));
                    int cx0 = x0 + 2 + led_spacing;
                    for (int li = 0; li < count; ++li) {
                        int cx = cx0 + li * led_spacing;
                        uint32_t bit = 1u << li;
                        uint32_t flags = (required_mask & bit) ? 1u : 0u;
                        push_hit(cx - 6, cy - 6, cx + 6, cy + 6, i, c, GP_TABLE_CELL_LEDS_ARG, GP_TABLE_HIT_LED_ARG, li, 0, flags);
                    }
                    draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy, eff_w, count, linked_mask, 0u, required_mask, 4, st.led_on, st.led_off, st.led_edge, st.axis_tick);
                    break;
                }
                case GP_TABLE_CELL_LEDS_TABLE: {
                    struct PackedStrip {
                        uint8_t count;
                        uint8_t reserved[3];
                        uint32_t on_mask;
                        uint32_t edge_mask;
                        uint32_t active_mask;
                    };
                    struct PackedTable {
                        uint8_t n;
                        uint8_t pad[3];
                        PackedStrip strips[3];
                    } pt{};
                    std::memcpy(&pt, cell.text, std::min<std::size_t>(sizeof(pt), sizeof(cell.text)));
                    int n = std::max(0, std::min<int>(pt.n, 3));
                    if (n <= 0) break;
                    int eff_w = std::max(1, cw - 4);
                    int slot_h = std::max(6, st.row_h / std::max(1, n + 1));
                    for (int si = 0; si < n; ++si) {
                        const PackedStrip& ps = pt.strips[si];
                        int cy_slot = y0 + (st.row_h * (si + 1)) / (n + 1);
                        int led_spacing = std::max(3 * 2 + 2, eff_w / std::max(1, int(ps.count) + 1));
                        int cx0 = x0 + 2 + led_spacing;
                        for (int li = 0; li < ps.count; ++li) {
                            int cx = cx0 + li * led_spacing;
                            push_hit(cx - 5, cy_slot - 5, cx + 5, cy_slot + 5, i, c, GP_TABLE_CELL_LEDS_TABLE, GP_TABLE_HIT_LED_TABLE, si, li, 0);
                        }
                        draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy_slot, eff_w, std::max<int>(0, ps.count), ps.on_mask, ps.edge_mask, ps.active_mask, 3, st.led_on, st.led_off, st.led_edge, st.axis_tick);
                    }
                    break;
                }
                case GP_TABLE_CELL_SCROLL: {
                    float v = cell.value;            // 0..1 scroll fraction
                    int total = std::max(1, int(cell.hold_s));    // total rows/items
                    int visible = std::max(1, int(cell.last_s));  // visible rows/items
                    bool up_press = (cell.flags & 0x1u) != 0;
                    bool dn_press = (cell.flags & 0x2u) != 0;
                    int bar_w = std::max(1, cw - 2);
                    int arrow_h = std::max(8, (st.row_h - 2) / 6);
                    draw_scrollbar(out_rgba, w, h, w * 4, x0 + 1, y0 + 1, bar_w, st.row_h - 2, v, total, visible, up_press, dn_press, st.axis_bg, st.axis_val, st.text, st.text_hdr, st.bg_sel, st.bg);
                    // Hitboxes: arrows + thumb
                    push_hit(x0 + 1, y0 + 1, x0 + 1 + bar_w, y0 + 1 + arrow_h, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_UP, 0, 0, 0);
                    push_hit(x0 + 1, y0 + st.row_h - 1 - arrow_h, x0 + 1 + bar_w, y0 + st.row_h - 1, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_DOWN, 0, 0, 0);
                    int track_y0 = y0 + 1 + arrow_h;
                    int track_h = (st.row_h - 2) - 2 * arrow_h;
                    if (track_h > 0) {
                        float vis_frac = std::clamp(float(visible) / float(std::max(visible, total)), 0.05f, 1.0f);
                        int thumb_h = std::max(6, int(vis_frac * track_h));
                        float clamped_v = std::clamp(v, 0.0f, 1.0f);
                        int thumb_y = track_y0 + int((track_h - thumb_h) * clamped_v);
                        push_hit(x0 + 1, thumb_y, x0 + 1 + bar_w, thumb_y + thumb_h, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_THUMB, 0, 0, 0);
                    }
                    break;
                }
                case GP_TABLE_CELL_AXIS: {
                    int bar_h = std::max(6, st.row_h / 3);
                    int bar_y = y0 + (st.row_h - bar_h) / 2;

                    // Optional extras: hold_s/min, last_s/max, calib params encoded as text.
                    float v_min = cell.hold_s;
                    float v_max = cell.last_s;
                    float cap_min = -1.0f;
                    float cap_max = 1.0f;
                    float trim = 0.0f;
                    float deadzone = 0.10f;
                    bool seen_min = (cell.flags & (1u << 0)) != 0;
                    bool seen_max = (cell.flags & (1u << 1)) != 0;
                    if ((cell.flags & 0x3u) == 0) {
                        // Default to "seen" so legacy callers render white end ticks.
                        seen_min = true;
                        seen_max = true;
                    }

                    if (cell.text[0] != '\0') {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
                        if (std::sscanf(cell.text, "%f %f %f %f", &t0, &t1, &t2, &t3) == 4) {
                            cap_min = t0;
                            cap_max = t1;
                            trim = t2;
                            deadzone = std::max(0.0f, t3);
                        }
                    }

                    int bar_w = std::max(1, cw - 4);
                    draw_axis_bar_ext(
                        out_rgba,
                        w,
                        h,
                        w * 4,
                        x0 + 2,
                        bar_y,
                        bar_w,
                        bar_h,
                        cell.value,
                        v_min,
                        v_max,
                        cap_min,
                        cap_max,
                        trim,
                        deadzone,
                        seen_min,
                        seen_max,
                        st.axis_bg,
                        st.axis_tick,
                        st.axis_val);
                    push_hit(x0 + 2, bar_y, x0 + 2 + bar_w, bar_y + bar_h, i, c, GP_TABLE_CELL_AXIS, GP_TABLE_HIT_AXIS, 0, 0, 0);
                    break;
                }
                case GP_TABLE_CELL_CALIB: {
                    bool inv_on = (cell.flags & 0x1u) != 0;
                    int mode = int((cell.flags >> 1) & 0x3u); // 0:none,1:trim,2:cap,3:ded
                    int strip_h = std::max(6, st.row_h / 3);
                    int strip_y = y0 + (st.row_h - strip_h) / 2;
                    int eff_w = std::max(1, cw - 4);
                    draw_calib_strip(out_rgba, w, h, w * 4, x0 + 2, strip_y, eff_w, strip_h, inv_on, mode, st.axis_tick, st.timer, st.led_on);
                    // Slots: 0 INV, 1 TRIM, 2 CAP, 3 DED, 4 RST
                    int slots = 5;
                    int gap = 2;
                    int slot_w = std::max(4, (eff_w - gap * (slots - 1)) / slots);
                    int slot_y = strip_y;
                    int slot_h = strip_h;
                    for (int si = 0; si < slots; ++si) {
                        int sx = x0 + 2 + si * (slot_w + gap);
                        push_hit(sx, slot_y, sx + slot_w, slot_y + slot_h, i, c, GP_TABLE_CELL_CALIB, GP_TABLE_HIT_CALIB_SLOT, si, 0, 0);
                    }
                    break;
                }
                case GP_TABLE_CELL_TIMERS: {
                    int t_h = std::max(2, st.row_h / 4);
                    int t_y = y0 + (st.row_h - t_h) / 2;
                    draw_timers(out_rgba, w, h, w * 4, x0 + 2, t_y, std::max(1, cw - 4), t_h, cell.hold_s, cell.last_s, st.timer);
                    break;
                }
                case GP_TABLE_CELL_WAVE: {
                    int wh = std::min(st.row_h - 4, cw);
                    int wy = y0 + (st.row_h - wh) / 2;
                    draw_waveform(out_rgba, w, h, w * 4, x0 + 2, wy, std::max(1, cw - 4), wh, cell.wave, st.wave_bg, st.wave_fg);
                    break;
                }
                case GP_TABLE_CELL_TEXT:
                default:
                    // Text is not rendered in this minimal raster; leave blank.
                    break;
            }
            // Emit whole-cell hitbox after all per-part hitboxes so small parts (LEDs)
            // are found before the coarse whole-cell region by hit iteration order.
            push_hit(x0, y0, x0 + cw, y0 + st.row_h, i, c, static_cast<GP_TableCellKind>(cell.kind), GP_TABLE_HIT_CELL, 0, 0, 0);
        }
    }

    if (hitboxes_written) *hitboxes_written = hit_count;

    return 1;
}

int32_t gp_table_raster_rgba(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom) {
    return gp_table_raster_rgba_with_hits(rows, row_count, cols, col_count, style, out_rgba, out_len_bytes, out_geom, nullptr, 0, nullptr);
}

// Stateful context ----------------------------------------------------------

struct GP_TableContext {
    GP_TableStyle style_raw{};
    Style st{};
    std::vector<GP_TableRow> rows;
    std::vector<GP_TableColumn> cols;
    GP_TableGeom geom{};
};

static void recompute_geom(GP_TableContext* ctx) {
    if (!ctx) return;
    ctx->geom.width_px = ctx->st.w;
    ctx->geom.height_px = std::max<int32_t>(1, static_cast<int32_t>(ctx->rows.size()) * ctx->st.row_h);
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, ctx->geom.col_x0, ctx->geom.col_w);
}

GP_TableContext* gp_table_create(const GP_TableStyle* style) {
    std::unique_ptr<GP_TableContext> ctx(new GP_TableContext());
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols.resize(0);
    ctx->rows.resize(0);
    recompute_geom(ctx.get());
    return ctx.release();
}

void gp_table_destroy(GP_TableContext* ctx) {
    delete ctx;
}

int32_t gp_table_set_style(GP_TableContext* ctx, const GP_TableStyle* style) {
    if (!ctx) return 0;
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_set_columns(GP_TableContext* ctx, const GP_TableColumn* cols, int32_t col_count) {
    if (!ctx) return 0;
    if (col_count < 0 || col_count > 8) return 0;
    if (col_count > 0 && !cols) return 0;
    if (col_count == 0) {
        ctx->cols.clear();
    } else {
        ctx->cols.assign(cols, cols + col_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_set_rows(GP_TableContext* ctx, const GP_TableRow* rows, int32_t row_count) {
    if (!ctx) return 0;
    if (row_count < 0) return 0;
    if (row_count > 0 && !rows) return 0;
    if (row_count == 0) {
        ctx->rows.clear();
    } else {
        ctx->rows.assign(rows, rows + row_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_get_geom(const GP_TableContext* ctx, GP_TableGeom* out_geom) {
    if (!ctx || !out_geom) return 0;
    *out_geom = ctx->geom;
    return 1;
}

int32_t gp_table_render_rgba_with_hits(
    GP_TableContext* ctx,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!ctx) return 0;
    return gp_table_raster_rgba_with_hits(
        ctx->rows.data(), static_cast<int32_t>(ctx->rows.size()),
        ctx->cols.data(), static_cast<int32_t>(ctx->cols.size()),
        &ctx->style_raw,
        out_rgba,
        out_len_bytes,
        out_geom,
        hitboxes_out,
        hitboxes_cap,
        hitboxes_written);
}

} // extern "C"

#ifdef __cplusplus
// C++ container helpers for table_abi (no new files). These mirror the C structs but provide
// builder-style APIs. They do not change raster behavior.
namespace gp {

struct RenderedTable {
    int width_px = 0;
    int height_px = 0;
    GP_TableGeom geom{};
    std::vector<uint8_t> rgba; // width*height*4
    bool ok() const { return width_px > 0 && height_px > 0 && rgba.size() == static_cast<std::size_t>(width_px * height_px * 4); }
};

struct TableStyle {
    GP_TableStyle style{};
    static TableStyle defaults(int width_px = 640, int row_h_px = 20) {
        TableStyle s;
        s.style = GP_TableStyle{};
        s.style.width_px = width_px;
        s.style.row_h_px = row_h_px;
        s.style.indent_px = 14;
        s.style.expand_w_px = 12;
        s.style.name_w_px = 160;
        auto set = [&](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
        set(s.style.bg_rgba, 8, 8, 12, 255);
        set(s.style.bg_sel_rgba, 18, 18, 26, 255);
        set(s.style.hdr_rgba, 20, 20, 28, 255);
        set(s.style.text_rgba, 240, 240, 245, 255);
        set(s.style.text_hdr_rgba, 255, 255, 255, 255);
        set(s.style.led_on_rgba, 255, 210, 90, 255);
        set(s.style.led_off_rgba, 70, 70, 80, 255);
        set(s.style.led_edge_rgba, 255, 255, 255, 255);
        set(s.style.axis_bg_rgba, 18, 18, 22, 255);
        set(s.style.axis_tick_rgba, 200, 200, 200, 255);
        set(s.style.axis_val_rgba, 255, 255, 140, 255);
        set(s.style.timer_rgba, 110, 180, 255, 255);
        set(s.style.wave_bg_rgba, 6, 6, 8, 255);
        set(s.style.wave_fg_rgba, 255, 255, 255, 255);
        uint32_t default_bits[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
        for (int i = 0; i < 9; ++i) s.style.led_mask[i] = default_bits[i];
        s.style.led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
        return s;
    }
};

struct TableCell {
    GP_TableCell cell{};
    TableCell() { cell.kind = GP_TABLE_CELL_TEXT; }

    static TableCell text(std::string_view t, int align = 0) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        auto len = std::min<std::size_t>(t.size(), sizeof(c.cell.text) - 1);
        std::memcpy(c.cell.text, t.data(), len);
        c.cell.text[len] = '\0';
        c.cell.flags = static_cast<uint32_t>(align);
        return c;
    }

    static TableCell leds(uint32_t flags) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS;
        c.cell.flags = flags;
        return c;
    }

    static TableCell axis(float value, bool seen_min, bool seen_max) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_AXIS;
        c.cell.value = value;
        c.cell.flags = (seen_min ? 1u : 0u) | (seen_max ? 2u : 0u);
        return c;
    }

    static TableCell axis_calib(float value, float v_min, float v_max, float cap_min, float cap_max, float trim, float deadzone, bool seen_min, bool seen_max) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_AXIS;
        c.cell.value = value;
        c.cell.hold_s = v_min;
        c.cell.last_s = v_max;
        c.cell.flags = (seen_min ? 1u : 0u) | (seen_max ? 2u : 0u);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "%0.3f %0.3f %0.3f %0.3f", double(cap_min), double(cap_max), double(trim), double(deadzone));
        return c;
    }

    static TableCell timers(float hold_s, float last_s) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TIMERS;
        c.cell.hold_s = hold_s;
        c.cell.last_s = last_s;
        return c;
    }

    static TableCell wave(const GP_MenuWaveform* wf) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_WAVE;
        c.cell.wave = wf;
        return c;
    }

    // Signal-list helpers (right-pane signals: name, args LEDs, history, op, channel).
    // These are encoded into TEXT/WAVE cells so the existing raster can carry them; UI overlays
    // can decode the packed flags if they want richer markers (arg counts, expansion state).
    static TableCell signal_args(int req, int linked, int max_leds = 12, bool expandable = false) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        const int r = std::max(0, req);
        const int l = std::max(0, linked);
        const int m = std::max(0, max_leds);
        // Pack small integers so overlays can render LED dots without extra sidecar data.
        c.cell.flags = (static_cast<uint32_t>(r) & 0xFFu) |
                       ((static_cast<uint32_t>(l) & 0xFFu) << 8) |
                       ((static_cast<uint32_t>(m) & 0xFFu) << 16) |
                       (expandable ? (1u << 24) : 0u);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "args %d/%d/%d%s", l, r, m, expandable ? "*" : "");
        return c;
    }

    static TableCell signal_hist(const GP_MenuWaveform* wf) {
        // History strip is carried as a waveform handle; caller feeds samples externally.
        return wave(wf);
    }

    static TableCell signal_op(std::string_view op_lbl) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        auto len = std::min<std::size_t>(op_lbl.size(), sizeof(c.cell.text) - 1);
        std::memcpy(c.cell.text, op_lbl.data(), len);
        c.cell.text[len] = '\0';
        return c;
    }

    static TableCell signal_channel(int ch) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        std::string tmp = (ch < 0) ? std::string("-") : std::to_string(ch);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "ch:%s", tmp.c_str());
        return c;
    }

    static TableCell calib_strip(bool invert_on, int mode /*0:none,1:trim,2:cap,3:ded*/) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_CALIB;
        uint32_t flags = 0;
        if (invert_on) flags |= 1u;
        flags |= (static_cast<uint32_t>(mode) & 0x3u) << 1;
        c.cell.flags = flags;
        return c;
    }

    static TableCell leds_arg(int count, uint32_t linked_mask, uint32_t required_mask) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS_ARG;
        c.cell.value = static_cast<float>(count);
        c.cell.flags = linked_mask;
        c.cell.reserved0 = static_cast<int32_t>(required_mask);
        return c;
    }

    struct PackedStrip {
        uint8_t count = 0;
        uint8_t reserved[3]{};
        uint32_t on_mask = 0;
        uint32_t edge_mask = 0;
        uint32_t active_mask = 0;
    };

    struct PackedTable {
        uint8_t n = 0;
        uint8_t pad[3]{};
        PackedStrip strips[3]{};
    };

    static TableCell leds_table(const std::vector<PackedStrip>& strips) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS_TABLE;
        PackedTable pt{};
        pt.n = static_cast<uint8_t>(std::min<std::size_t>(strips.size(), 3));
        for (std::size_t i = 0; i < strips.size() && i < 3; ++i) pt.strips[i] = strips[i];
        std::memcpy(c.cell.text, &pt, std::min<std::size_t>(sizeof(pt), sizeof(c.cell.text)));
        return c;
    }

    static TableCell scrollbar(float value01, int total_rows, int visible_rows, bool arrow_up_pressed, bool arrow_dn_pressed) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_SCROLL;
        c.cell.value = value01;
        c.cell.hold_s = static_cast<float>(total_rows);
        c.cell.last_s = static_cast<float>(visible_rows);
        uint32_t flags = 0;
        if (arrow_up_pressed) flags |= 1u;
        if (arrow_dn_pressed) flags |= 1u << 1;
        c.cell.flags = flags;
        return c;
    }
};

struct TableRow {
    GP_TableRow row{};
    std::vector<TableCell> cells;

    TableRow(GP_TableRowKind kind, std::string_view label, int depth = 0, bool expanded = false, bool selected = false) {
        row.kind = static_cast<int32_t>(kind);
        row.depth = depth;
        row.expanded = expanded ? 1 : 0;
        row.selected = selected ? 1 : 0;
        auto len = std::min<std::size_t>(label.size(), sizeof(row.label) - 1);
        std::memcpy(row.label, label.data(), len);
        row.label[len] = '\0';
    }

    TableRow& add_cell(const TableCell& c) {
        if (cells.size() < 8) cells.push_back(c);
        return *this;
    }

    GP_TableRow finalize() const {
        GP_TableRow out = row;
        const int n = static_cast<int>(std::min<std::size_t>(cells.size(), 8));
        for (int i = 0; i < n; ++i) out.cells[i] = cells[static_cast<std::size_t>(i)].cell;
        out.cell_count = n;
        return out;
    }
};

// Convenience builder for right-pane signal rows (name + args + history + op + channel).
struct SignalRow {
    TableRow row;

    explicit SignalRow(std::string_view label, bool selected = false)
        : row(GP_TABLE_ROW_NOTE, label, /*depth=*/0, /*expanded=*/false, selected) {}

    SignalRow& args(int req, int linked, int max_leds = 12, bool expandable = false) {
        row.add_cell(TableCell::signal_args(req, linked, max_leds, expandable));
        return *this;
    }

    SignalRow& history(const GP_MenuWaveform* wf) {
        row.add_cell(TableCell::signal_hist(wf));
        return *this;
    }

    SignalRow& op(std::string_view op_lbl) {
        row.add_cell(TableCell::signal_op(op_lbl));
        return *this;
    }

    SignalRow& channel(int ch) {
        row.add_cell(TableCell::signal_channel(ch));
        return *this;
    }

    GP_TableRow finalize() const { return row.finalize(); }
};

class TableTexture {
public:
    TableTexture() : style_(TableStyle::defaults()) {}

    TableTexture& set_style(const GP_TableStyle& s) {
        style_.style = s;
        return *this;
    }

    TableTexture& add_column(GP_TableCellKind kind, int width_px, int align = 0) {
        GP_TableColumn c{};
        c.kind = static_cast<int32_t>(kind);
        c.width_px = width_px;
        c.align = align;
        cols_.push_back(c);
        return *this;
    }

    TableRow& add_row(GP_TableRowKind kind, std::string_view label, int depth = 0, bool expanded = false, bool selected = false) {
        rows_.emplace_back(kind, label, depth, expanded, selected);
        return rows_.back();
    }

    TableTexture& add_signal_row(const SignalRow& s) {
        rows_.push_back(s.row);
        return *this;
    }

    RenderedTable render() const {
        RenderedTable out;
        if (rows_.empty() || cols_.empty()) return out;
        std::vector<GP_TableRow> packed;
        packed.reserve(rows_.size());
        for (const auto& r : rows_) packed.push_back(r.finalize());

        GP_TableGeom geom{};
        const int row_count = static_cast<int>(packed.size());
        const int col_count = static_cast<int>(cols_.size());
        GP_TableStyle st = style_.style;
        if (!gp_table_calc_size(&st, row_count, col_count, &geom)) return out;
        const int need = geom.width_px * geom.height_px * 4;
        if (need <= 0) return out;
        out.rgba.resize(static_cast<std::size_t>(need));
        if (!gp_table_raster_rgba(packed.data(), row_count, cols_.data(), col_count, &st, out.rgba.data(), static_cast<int32_t>(out.rgba.size()), &geom)) {
            out.rgba.clear();
            return out;
        }
        out.width_px = geom.width_px;
        out.height_px = geom.height_px;
        out.geom = geom;
        return out;
    }

private:
    TableStyle style_;
    std::vector<GP_TableColumn> cols_;
    std::vector<TableRow> rows_;
};

} // namespace gp
#endif // __cplusplus
