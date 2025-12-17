#include "workbench_rows_abi.h"
#include "menu_waveform_abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

static inline Color to_color(const uint8_t rgba[4]) {
    return Color{rgba[0], rgba[1], rgba[2], rgba[3]};
}

struct Style {
    int w = 640;
    int row_h = 20;
    int name_w = 160;
    int led_spacing = 14;
    int led_r = 4;
    int axis_w = 180;
    int wave_w = 70;
    int wave_h = 14;

    Color bg{8, 8, 12, 255};
    Color bg_sel{18, 18, 26, 255};
    Color hdr{20, 20, 28, 255};
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

static Style load_style(const GP_WbStyle* s) {
    Style out;
    if (!s) return out;
    out.w = std::max(64, int(s->width_px));
    out.row_h = std::max(8, int(s->row_h_px));
    out.name_w = std::max(20, int(s->name_w_px));
    out.led_spacing = std::max(6, int(s->led_spacing_px));
    out.led_r = std::max(2, int(s->led_radius_px));
    out.axis_w = std::max(40, int(s->axis_w_px));
    out.wave_w = std::max(0, int(s->wave_w_px));
    out.wave_h = std::max(0, int(std::min(s->wave_h_px, s->row_h_px)));

    out.bg = to_color(s->bg_rgba);
    out.bg_sel = to_color(s->bg_sel_rgba);
    out.hdr = to_color(s->hdr_rgba);
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

static void draw_waveform(uint8_t* img, int w, int h, int pitch, int x0, int y0, int ww, int wh, const GP_MenuWaveform* wf, Color bg, Color fg) {
    if (!img || ww <= 0 || wh <= 0) return;
    // Fill background.
    memset_rect(img, w, h, pitch, x0, y0, ww, wh, bg);
    if (!wf) return;
    const int need = ww * wh * 4;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(need));
    if (!gp_menu_waveform_raster_rgba(const_cast<GP_MenuWaveform*>(wf), tmp.data(), int32_t(tmp.size()))) {
        return;
    }
    // Blit with alpha (premult not needed; source is opaque) and clamp.
    for (int yy = 0; yy < wh; ++yy) {
        for (int xx = 0; xx < ww; ++xx) {
            int tx = x0 + xx;
            int ty = y0 + yy;
            if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
            const uint8_t* src = &tmp[(yy * ww + xx) * 4];
            uint8_t* dst = img + ty * pitch + tx * 4;
            // Simple copy; tmp already has alpha.
            dst[0] = uint8_t((int(src[0]) * int(fg.r)) / 255);
            dst[1] = uint8_t((int(src[1]) * int(fg.g)) / 255);
            dst[2] = uint8_t((int(src[2]) * int(fg.b)) / 255);
            dst[3] = src[3];
        }
    }
}

static void draw_axis_bar(uint8_t* img, int w, int h, int pitch, int x0, int y0, int bar_w, int bar_h, float v, Color bg, Color tick, Color val) {
    if (!img || bar_w <= 0 || bar_h <= 0) return;
    memset_rect(img, w, h, pitch, x0, y0, bar_w, bar_h, bg);
    const int mid_y = y0 + bar_h / 2;
    // ticks at -1,0,+1
    for (int i = 0; i < 3; ++i) {
        float t = (i == 0) ? -1.0f : (i == 1 ? 0.0f : 1.0f);
        float t01 = 0.5f * (t + 1.0f);
        int xx = x0 + int(std::lround(t01 * float(bar_w - 1)));
        draw_line(img, w, h, pitch, xx, y0 - 1, xx, y0 + bar_h + 1, 1, tick);
    }
    float v_clamped = std::max(-1.0f, std::min(1.0f, v));
    float v01 = 0.5f * (v_clamped + 1.0f);
    int xv = x0 + int(std::lround(v01 * float(bar_w - 1)));
    draw_line(img, w, h, pitch, xv, y0 - 2, xv, y0 + bar_h + 2, 2, val);
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

} // namespace

extern "C" {

int32_t gp_wb_rows_calc_size(const GP_WbStyle* style, int32_t row_count, int32_t* out_w, int32_t* out_h) {
    if (!out_w || !out_h || row_count < 0) return 0;
    Style st = load_style(style);
    *out_w = st.w;
    *out_h = std::max(1, int(row_count) * st.row_h);
    return 1;
}

int32_t gp_wb_rows_raster_rgba(const GP_WbRow* rows, int32_t row_count, const GP_WbStyle* style, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!rows || !out_rgba || row_count <= 0) return 0;
    Style st = load_style(style);
    int w = st.w;
    int h = std::max(1, int(row_count) * st.row_h);
    const int need = w * h * 4;
    if (out_len_bytes < need) return 0;

    // Clear to bg.
    memset(out_rgba, 0, static_cast<std::size_t>(need));
    memset_rect(out_rgba, w, h, w * 4, 0, 0, w, h, st.bg);

    const int led_count = 9;
    for (int i = 0; i < row_count; ++i) {
        const GP_WbRow& r = rows[i];
        int y0 = i * st.row_h;
        Color bg = (r.kind == GP_WB_ROW_DEVICE_HEADER) ? st.hdr : (r.selected ? st.bg_sel : st.bg);
        memset_rect(out_rgba, w, h, w * 4, 0, y0, w, st.row_h, bg);

        int cx_name0 = 4;
        int led_x0 = cx_name0 + st.name_w + 8;
        int axis_x0 = led_x0 + led_count * st.led_spacing + 12;
        int axis_w = st.axis_w;
        int wave_x0 = axis_x0 + axis_w + 10;
        int wave_w = st.wave_w;
        int wave_h = std::min(st.wave_h, st.row_h - 4);
        if (wave_h < 0) wave_h = 0;

        // LEDs
        int cy = y0 + st.row_h / 2;
        for (int li = 0; li < led_count; ++li) {
            int cx = led_x0 + li * st.led_spacing;
            uint32_t mask = st.led_mask[li];
            bool on = mask != 0 && (r.flags & mask);
            bool edge = on && (st.led_edge_mask != 0) && ((r.flags & st.led_edge_mask) != 0);
            Color lc = edge ? st.led_edge : (on ? st.led_on : st.led_off);
            draw_circle(out_rgba, w, h, w * 4, cx, cy, st.led_r, lc);
        }

        // Axis bar for axes; buttons skip axis draw.
        if (r.kind == GP_WB_ROW_AXIS) {
            int bar_h = std::max(6, st.row_h / 3);
            int bar_y = y0 + (st.row_h - bar_h) / 2;
            draw_axis_bar(out_rgba, w, h, w * 4, axis_x0, bar_y, axis_w, bar_h, r.value, st.axis_bg, st.axis_tick, st.axis_val);
        }

        // Timers stripe (for buttons): two stacked mini bars using hold_s and last_s.
        if (r.kind == GP_WB_ROW_BUTTON) {
            int t_w = axis_w;
            int t_h = std::max(2, st.row_h / 4);
            int t_y = y0 + (st.row_h - t_h) / 2;
            float hold = std::max(0.0f, std::min(1.0f, r.hold_s));
            float last = std::max(0.0f, std::min(1.0f, r.last_s));
            draw_timers(out_rgba, w, h, w * 4, axis_x0, t_y, t_w, t_h, hold, last, st.timer);
        }

        // Waveform (optional).
        if (r.wave && wave_w > 0 && wave_h > 0) {
            int wy = y0 + (st.row_h - wave_h) / 2;
            draw_waveform(out_rgba, w, h, w * 4, wave_x0, wy, wave_w, wave_h, r.wave, st.wave_bg, st.wave_fg);
        }
    }

    return 1;
}

} // extern "C"

#ifdef __cplusplus

// C++ helpers to build workbench rows as containers and render to RGBA without manual C struct wiring.
namespace gp {

struct RenderedWbRows {
    int width_px = 0;
    int height_px = 0;
    std::vector<uint8_t> rgba; // width*height*4
    bool ok() const {
        return width_px > 0 && height_px > 0 && rgba.size() == static_cast<std::size_t>(width_px * height_px * 4);
    }
};

struct WbStyle {
    GP_WbStyle style{};

    static WbStyle defaults(int width_px = 640, int row_h_px = 20) {
        WbStyle s;
        s.style = GP_WbStyle{};
        s.style.width_px = width_px;
        s.style.row_h_px = row_h_px;
        s.style.name_w_px = 160;
        s.style.led_spacing_px = 14;
        s.style.led_radius_px = 4;
        s.style.axis_w_px = 180;
        s.style.wave_w_px = 70;
        s.style.wave_h_px = 14;
        auto set = [&](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = a;
        };
        set(s.style.bg_rgba, 8, 8, 12, 255);
        set(s.style.bg_sel_rgba, 18, 18, 26, 255);
        set(s.style.hdr_rgba, 20, 20, 28, 255);
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

struct WbRow {
    GP_WbRow row{};

    explicit WbRow(GP_WbRowKind kind, std::string_view label = "", bool selected = false) {
        row.kind = static_cast<int32_t>(kind);
        row.selected = selected ? 1 : 0;
        auto len = std::min<std::size_t>(label.size(), sizeof(row.label) - 1);
        std::memcpy(row.label, label.data(), len);
        row.label[len] = '\0';
    }

    static WbRow device_header(std::string_view label, bool selected = false) {
        return WbRow(GP_WB_ROW_DEVICE_HEADER, label, selected);
    }

    static WbRow device_note(std::string_view label, bool selected = false) {
        return WbRow(GP_WB_ROW_DEVICE_NOTE, label, selected);
    }

    static WbRow axis(std::string_view label, float value, uint32_t flags, bool selected = false, const GP_MenuWaveform* wave = nullptr) {
        WbRow r(GP_WB_ROW_AXIS, label, selected);
        r.row.value = value;
        r.row.flags = flags;
        r.row.wave = wave;
        return r;
    }

    static WbRow button(std::string_view label, uint32_t flags, float hold_s, float last_s, bool selected = false, const GP_MenuWaveform* wave = nullptr) {
        WbRow r(GP_WB_ROW_BUTTON, label, selected);
        r.row.flags = flags;
        r.row.hold_s = hold_s;
        r.row.last_s = last_s;
        r.row.wave = wave;
        return r;
    }

    // Hat matrix helper: packs direction LEDs into flags (bits 0-7) and emits two axis rows (x/y) plus header.
    // This is a convenience; raster still draws using axis/button primitives.
    static void add_hat(std::string_view label, int dir_idx, float x_val, float y_val, std::vector<GP_WbRow>& out_rows, bool selected = false) {
        uint32_t dir_flags = 0;
        if (dir_idx >= 0 && dir_idx < 8) dir_flags |= (1u << static_cast<uint32_t>(dir_idx));
        // Header row carries direction LEDs.
        WbRow hdr(GP_WB_ROW_DEVICE_HEADER, label, selected);
        hdr.row.flags = dir_flags;
        out_rows.push_back(hdr.row);
        // X axis row
        WbRow rx(GP_WB_ROW_AXIS, std::string(label) + ".x", selected);
        rx.row.value = x_val;
        rx.row.flags = dir_flags;
        out_rows.push_back(rx.row);
        // Y axis row
        WbRow ry(GP_WB_ROW_AXIS, std::string(label) + ".y", selected);
        ry.row.value = y_val;
        ry.row.flags = dir_flags;
        out_rows.push_back(ry.row);
    }
};

struct AxisRow {
    GP_WbRow row{};
    AxisRow(std::string_view label, float value, uint32_t flags = 0, bool selected = false, const GP_MenuWaveform* wave = nullptr) {
        row.kind = static_cast<int32_t>(GP_WB_ROW_AXIS);
        row.value = value;
        row.flags = flags;
        row.selected = selected ? 1 : 0;
        row.wave = wave;
        auto len = std::min<std::size_t>(label.size(), sizeof(row.label) - 1);
        std::memcpy(row.label, label.data(), len);
        row.label[len] = '\0';
    }
};

struct ButtonRow {
    GP_WbRow row{};
    ButtonRow(std::string_view label, uint32_t flags, float hold_s, float last_s, bool selected = false, const GP_MenuWaveform* wave = nullptr) {
        row.kind = static_cast<int32_t>(GP_WB_ROW_BUTTON);
        row.flags = flags;
        row.hold_s = hold_s;
        row.last_s = last_s;
        row.selected = selected ? 1 : 0;
        row.wave = wave;
        auto len = std::min<std::size_t>(label.size(), sizeof(row.label) - 1);
        std::memcpy(row.label, label.data(), len);
        row.label[len] = '\0';
    }
};

struct HatRows {
    GP_WbRow header{};
    GP_WbRow axis_x{};
    GP_WbRow axis_y{};

    HatRows(std::string_view label, int dir_idx, float x_val, float y_val, bool selected = false) {
        uint32_t dir_flags = 0;
        if (dir_idx >= 0 && dir_idx < 8) dir_flags |= (1u << static_cast<uint32_t>(dir_idx));

        header.kind = static_cast<int32_t>(GP_WB_ROW_DEVICE_HEADER);
        header.flags = dir_flags;
        header.selected = selected ? 1 : 0;
        {
            auto len = std::min<std::size_t>(label.size(), sizeof(header.label) - 1);
            std::memcpy(header.label, label.data(), len);
            header.label[len] = '\0';
        }

        axis_x.kind = static_cast<int32_t>(GP_WB_ROW_AXIS);
        axis_x.value = x_val;
        axis_x.flags = dir_flags;
        axis_x.selected = selected ? 1 : 0;
        {
            std::string lx = std::string(label) + ".x";
            auto len = std::min<std::size_t>(lx.size(), sizeof(axis_x.label) - 1);
            std::memcpy(axis_x.label, lx.data(), len);
            axis_x.label[len] = '\0';
        }

        axis_y.kind = static_cast<int32_t>(GP_WB_ROW_AXIS);
        axis_y.value = y_val;
        axis_y.flags = dir_flags;
        axis_y.selected = selected ? 1 : 0;
        {
            std::string ly = std::string(label) + ".y";
            auto len = std::min<std::size_t>(ly.size(), sizeof(axis_y.label) - 1);
            std::memcpy(axis_y.label, ly.data(), len);
            axis_y.label[len] = '\0';
        }
    }
};

class WbTexture {
public:
    WbTexture() : style_(WbStyle::defaults()) {}

    WbTexture& set_style(const GP_WbStyle& s) {
        style_.style = s;
        return *this;
    }

    WbTexture& add_row(const WbRow& r) {
        rows_.push_back(r.row);
        return *this;
    }

    WbTexture& add_axis(const AxisRow& a) {
        rows_.push_back(a.row);
        return *this;
    }

    WbTexture& add_button(const ButtonRow& b) {
        rows_.push_back(b.row);
        return *this;
    }

    // Convenience: append hat rows (header + x axis + y axis) with direction flags.
    WbTexture& add_hat(std::string_view label, int dir_idx, float x_val, float y_val, bool selected = false) {
        WbRow::add_hat(label, dir_idx, x_val, y_val, rows_, selected);
        return *this;
    }

    WbTexture& add_hat(const HatRows& h) {
        rows_.push_back(h.header);
        rows_.push_back(h.axis_x);
        rows_.push_back(h.axis_y);
        return *this;
    }

    RenderedWbRows render() const {
        RenderedWbRows out;
        if (rows_.empty()) return out;
        GP_WbStyle st = style_.style;
        int w = 0, h = 0;
        if (!gp_wb_rows_calc_size(&st, static_cast<int32_t>(rows_.size()), &w, &h)) return out;
        const int need = w * h * 4;
        if (need <= 0) return out;
        out.rgba.resize(static_cast<std::size_t>(need));
        if (!gp_wb_rows_raster_rgba(rows_.data(), static_cast<int32_t>(rows_.size()), &st, out.rgba.data(), static_cast<int32_t>(out.rgba.size()))) {
            out.rgba.clear();
            return out;
        }
        out.width_px = w;
        out.height_px = h;
        return out;
    }

private:
    WbStyle style_;
    std::vector<GP_WbRow> rows_;
};

} // namespace gp

#endif // __cplusplus
