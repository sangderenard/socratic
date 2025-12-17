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
#include <fstream>
#include <iterator>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <filesystem>
#include "text_render_helper.h"
#include "table_node_groups.h"
#include "rope_sim.h"

namespace {

struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

// Global template library directory (can be set by gp_table_set_library_dir).
static std::string g_template_library_dir;

static std::string make_template_filename(const char* dir, const char* name) {
    std::string d;
    if (dir && dir[0]) d = std::string(dir);
    else d = g_template_library_dir;
    if (d.empty()) d = ".";
    std::filesystem::path p(d);
    std::string fname = name ? std::string(name) : std::string("untitled");
    p /= (fname + std::string(".gptbl"));
    return p.string();
}

extern "C" int32_t gp_table_set_library_dir(const char* dir) {
    if (!dir || dir[0] == '\0') { g_template_library_dir.clear(); return 1; }
    try { g_template_library_dir = std::string(dir); return 1; } catch (...) { return 0; }
}

extern "C" int32_t gp_table_get_library_dir(char* out_buf, int32_t out_len) {
    if (!out_buf && out_len != 0) return 0;
    const std::string &d = g_template_library_dir;
    if (!out_buf) return static_cast<int32_t>(d.size());
    int32_t to_write = std::min<int32_t>(static_cast<int32_t>(d.size()), out_len);
    if (to_write > 0) memcpy(out_buf, d.data(), static_cast<size_t>(to_write));
    return to_write;
}

int32_t gp_table_save_template(GP_TableContext* ctx, const char* dir, const char* name) {
    if (!ctx || !name) return 0;
    int32_t need = gp_table_serialize(ctx, nullptr, 0);
    if (need <= 0) return 0;
    std::vector<char> buf(static_cast<size_t>(need));
    int32_t wrote = gp_table_serialize(ctx, buf.data(), need);
    if (wrote != need) return 0;
    std::string path = make_template_filename(dir, name);
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) return 0;
        ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        ofs.close();
    } catch (...) {
        return 0;
    }
    return 1;
}

int32_t gp_table_load_template(GP_TableContext* ctx, const char* dir, const char* name) {
    if (!ctx || !name) return 0;
    std::string path = make_template_filename(dir, name);
    try {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return 0;
        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
        if (buf.empty()) return 0;
        return gp_table_deserialize(ctx, buf.data(), static_cast<int32_t>(buf.size()));
    } catch (...) {
        return 0;
    }
}

int32_t gp_table_list_templates(const char* dir, char* out_buf, int32_t out_len) {
    std::string d;
    if (dir && dir[0]) d = std::string(dir);
    else d = g_template_library_dir;
    if (d.empty()) d = ".";
    std::string listing;
    try {
        for (auto &entry : std::filesystem::directory_iterator(d)) {
            if (!entry.is_regular_file()) continue;
            auto p = entry.path();
            if (p.extension() == ".gptbl") {
                listing += p.stem().string();
                listing += '\n';
            }
        }
    } catch (...) {
        return 0;
    }
    if (!out_buf) return static_cast<int32_t>(listing.size());
    int32_t to_write = std::min<int32_t>(static_cast<int32_t>(listing.size()), out_len);
    if (to_write > 0) memcpy(out_buf, listing.data(), static_cast<size_t>(to_write));
    return to_write;
}

// Simple default-table spec used by the C++ table implementation.
// Keeps a single place to tune the "charcoal" default appearance.
struct DefaultTableSpec {
    int width_px = 640;
    int row_h_px = 20;
    uint8_t bg_rgba[4] = {40, 40, 50, 255}; // charcoal
};
static const DefaultTableSpec kDefaultTableSpec;

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

    // Calibration slot label texts and colors (text color still used; background will be transparent)
    static const char* calib_labels_lower[5] = { "inv", "trim", "cap", "ded", "rst" };
    static const char* calib_labels_upper[5] = { "INV", "TRIM", "CAP", "DED", "RST" };
    static const uint8_t calib_label_color[5][4] = {
        {160,160,160,255}, // grey (inv)
        {255,255,255,255}, // white (trim)
        {64,220,64,255},   // green (cap)
        {220,64,64,255},   // red (ded)
        {255,255,255,255}  // white (rst)
    };

    // Pre-render labels to determine required slot height
    std::array<TextBitmap,5> bms;
    int max_bh = 0;
    for (int i = 0; i < slots; ++i) {
        const char* txt = (i == 2 || i == 3) ? calib_labels_upper[i] : calib_labels_lower[i];
        std::array<unsigned char,4> col = { calib_label_color[i][0], calib_label_color[i][1], calib_label_color[i][2], calib_label_color[i][3] };
        bms[i] = render_text_to_rgba(std::string(txt), 1.0f, col);
        if (!bms[i].pixels.empty()) max_bh = std::max<int>(max_bh, bms[i].height);
    }

    const int vpad = 6; // vertical padding inside slot
    slot_h = std::max(slot_h, max_bh + vpad);
    int y = y0 + (ch - slot_h) / 2;

    // Draw only outlines for simple text buttons (no colored fills)
    for (int i = 0; i < slots; ++i) {
        int x = x0 + i * (slot_w + gap);
        // thin outline for button
        memset_rect(img, w, h, pitch, x, y, slot_w, 1, Color{255,255,255,32});
        memset_rect(img, w, h, pitch, x, y + slot_h - 1, slot_w, 1, Color{0,0,0,64});
        memset_rect(img, w, h, pitch, x, y, 1, slot_h, Color{255,255,255,32});
        memset_rect(img, w, h, pitch, x + slot_w - 1, y, 1, slot_h, Color{0,0,0,64});
    }

    // Render labels centered in each slot. Bold simulated by overdrawing with 1px offset.
    for (int i = 0; i < slots; ++i) {
        int sx = x0 + i * (slot_w + gap);
        int sy = y;
        const TextBitmap &bm = bms[i];
        if (bm.pixels.empty()) continue;
        int bx = sx + (slot_w - bm.width) / 2;
        int by = sy + (slot_h - bm.height) / 2;
        for (int yy = 0; yy < bm.height; ++yy) {
            int dst_y = by + yy;
            if (dst_y < 0 || dst_y >= h) continue;
            for (int xx = 0; xx < bm.width; ++xx) {
                int dst_x = bx + xx;
                if (dst_x < 0 || dst_x >= w) continue;
                unsigned char* dst = img + dst_y * pitch + dst_x * 4;
                const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                float sa = src[3] / 255.0f;
                if (sa >= 0.999f) {
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                } else if (sa > 0.001f) {
                    for (int cch = 0; cch < 3; ++cch) {
                        float s = src[cch] / 255.0f;
                        float d = dst[cch] / 255.0f;
                        float out = s * sa + d * (1.0f - sa);
                        dst[cch] = static_cast<uint8_t>(std::lround(out * 255.0f));
                    }
                    float da = dst[3] / 255.0f;
                    float outa = sa + da * (1.0f - sa);
                    dst[3] = static_cast<uint8_t>(std::lround(outa * 255.0f));
                }
            }
        }
        // bold: overdraw with 1px offset
        for (int yy = 0; yy < bm.height; ++yy) {
            int dst_y = by + yy + 1;
            if (dst_y < 0 || dst_y >= h) continue;
            for (int xx = 0; xx < bm.width; ++xx) {
                int dst_x = bx + xx + 1;
                if (dst_x < 0 || dst_x >= w) continue;
                unsigned char* dst = img + dst_y * pitch + dst_x * 4;
                const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                float sa = src[3] / 255.0f;
                if (sa >= 0.999f) {
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                } else if (sa > 0.001f) {
                    for (int cch = 0; cch < 3; ++cch) {
                        float s = src[cch] / 255.0f;
                        float d = dst[cch] / 255.0f;
                        float out = s * sa + d * (1.0f - sa);
                        dst[cch] = static_cast<uint8_t>(std::lround(out * 255.0f));
                    }
                    float da = dst[3] / 255.0f;
                    float outa = sa + da * (1.0f - sa);
                    dst[3] = static_cast<uint8_t>(std::lround(outa * 255.0f));
                }
            }
        }
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
    // Cable / edge render parameters
    int cable_segments = 24;         // samples along cable
    int cable_jacket_px = 5;        // outer jacket radius in pixels
    int cable_jacket_border = 2;    // inner border thickness (core = jacket - border)
    float cable_core_alpha = 0.92f; // core translucency (0..1)
    float cable_sag = 0.10f;        // sag factor relative to distance (0..1)
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
    s.width_px = kDefaultTableSpec.width_px;
    s.row_h_px = kDefaultTableSpec.row_h_px;
    s.indent_px = 14;
    s.expand_w_px = 12;
    s.name_w_px = 160;
    auto set = [](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
    set(s.bg_rgba, kDefaultTableSpec.bg_rgba[0], kDefaultTableSpec.bg_rgba[1], kDefaultTableSpec.bg_rgba[2], kDefaultTableSpec.bg_rgba[3]);
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
    const GP_TableRenderState* render_state,
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
    // If caller provided an output geometry (e.g. canvas wants to render into
    // a specific module rect), honor its width/height as the target buffer
    // size. This allows embedding tables into arbitrary-sized module boxes
    // and ensures hitbox coordinates align with the provided buffer.
    if (out_geom && out_geom->width_px > 0 && out_geom->height_px > 0) {
        w = out_geom->width_px;
        h = out_geom->height_px;
    }
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
    // If caller overrode the output geometry size, compute an effective
    // per-row height so rows are laid out to match the target buffer
    // height. This ensures hitbox coordinates line up with the rendered
    // pixels when the renderer is asked to draw into a specific buffer
    // (e.g. embedding a table into a larger module rect).
    int row_h_eff = std::max(1, st.row_h);
    if (row_count > 0) row_h_eff = std::max(1, h / row_count);
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
        int y0 = i * row_h_eff;
        Color bg = (r.kind == GP_TABLE_ROW_HEADER) ? st.hdr : (r.selected ? st.bg_sel : st.bg);
        memset_rect(out_rgba, w, h, w * 4, 0, y0, w, row_h_eff, bg);

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
        int cy = y0 + row_h_eff / 2;
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
                    int slot_h = std::max(6, row_h_eff / std::max(1, n + 1));
                    for (int si = 0; si < n; ++si) {
                        const PackedStrip& ps = pt.strips[si];
                        int cy_slot = y0 + (row_h_eff * (si + 1)) / (n + 1);
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
                    int arrow_h = std::max(8, (row_h_eff - 2) / 6);
                    draw_scrollbar(out_rgba, w, h, w * 4, x0 + 1, y0 + 1, bar_w, row_h_eff - 2, v, total, visible, up_press, dn_press, st.axis_bg, st.axis_val, st.text, st.text_hdr, st.bg_sel, st.bg);
                    // Hitboxes: arrows + thumb
                    push_hit(x0 + 1, y0 + 1, x0 + 1 + bar_w, y0 + 1 + arrow_h, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_UP, 0, 0, 0);
                    push_hit(x0 + 1, y0 + st.row_h - 1 - arrow_h, x0 + 1 + bar_w, y0 + st.row_h - 1, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_DOWN, 0, 0, 0);
                    int track_y0 = y0 + 1 + arrow_h;
                    int track_h = (row_h_eff - 2) - 2 * arrow_h;
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
                    int bar_h = std::max(6, row_h_eff / 3);
                    int bar_y = y0 + (row_h_eff - bar_h) / 2;

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
                    int strip_h = std::max(6, row_h_eff / 3);
                    int strip_y = y0 + (row_h_eff - strip_h) / 2;
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
                    int t_h = std::max(2, row_h_eff / 4);
                    int t_y = y0 + (row_h_eff - t_h) / 2;
                    draw_timers(out_rgba, w, h, w * 4, x0 + 2, t_y, std::max(1, cw - 4), t_h, cell.hold_s, cell.last_s, st.timer);
                    break;
                }
                case GP_TABLE_CELL_WAVE: {
                    int wh = std::min(row_h_eff - 4, cw);
                    int wy = y0 + (row_h_eff - wh) / 2;
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

    // Apply transient render-state overlays (highlights) if requested.
    if (render_state && out_rgba) {
        // Load style for layout math
        Style st_local = load_style(style);
        int w_local = st_local.w;
        int h_local = std::max(1, int(row_count) * st_local.row_h);
        int row_h_local = std::max(1, st_local.row_h);
        if (row_count > 0) row_h_local = std::max(1, h_local / row_count);
        int pitch_local = w_local * 4;

        // Determine highlight color
        Color hcol{255, 210, 90, 200};
        bool have_custom = false;
        if (render_state->highlight_color[0] || render_state->highlight_color[1] || render_state->highlight_color[2] || render_state->highlight_color[3]) {
            hcol = Color{render_state->highlight_color[0], render_state->highlight_color[1], render_state->highlight_color[2], render_state->highlight_color[3]};
            have_custom = true;
        }

        // helper to draw an outline rect
        auto draw_outline = [&](int rx0, int ry0, int rw, int rh, Color col){
            if (rw <= 0 || rh <= 0) return;
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0, rw, 1, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0 + rh - 1, rw, 1, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0, 1, rh, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0 + rw - 1, ry0, 1, rh, col);
        };

        int col_x0[8] = {0};
        int col_w[8] = {0};
        compute_columns(cols, col_count, st_local.w, st_local.name_w, col_x0, col_w);

        if (render_state->highlight_row >= 0) {
            int r = render_state->highlight_row;
            if (r >= 0 && r < row_count) {
                int y0 = r * row_h_local;
                // whole-cell highlight if col not provided
                if (render_state->highlight_col < 0) {
                    draw_outline(0, y0, w_local, row_h_local, hcol);
                } else {
                    int c = render_state->highlight_col;
                    if (c >= 0 && c < col_count) {
                        int x0 = col_x0[c];
                        int cw = col_w[c];
                        // Decide based on highlighted part
                        int part = render_state->highlight_part;
                        if (part == GP_TABLE_HIT_LED || part == GP_TABLE_HIT_LED_ARG || part == GP_TABLE_HIT_LED_TABLE) {
                            // compute LED positions similar to raster
                            const GP_TableRow &row = rows[r];
                            const GP_TableCell &cell = row.cells[c];
                            int led_count = 9;
                            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                                if (count == 0) count = (cell.flags & 0xFF);
                                if (count == 0) count = 12;
                                led_count = count;
                            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                                // best-effort: use 8 as fallback
                                led_count = 8;
                            }
                            int eff_w = std::max(1, cw - 4);
                            int radius = 4;
                            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
                            int cx0 = x0 + 2 + led_spacing;
                            int li = render_state->highlight_aux0;
                            if (li >= 0 && li < led_count) {
                                int cx = cx0 + li * led_spacing;
                                int cy = y0 + row_h_local / 2;
                                // slightly larger ring for highlight
                                draw_circle(out_rgba, w_local, h_local, pitch_local, cx, cy, radius + 2, hcol);
                            }
                        } else if (part == GP_TABLE_HIT_CALIB_SLOT) {
                            // compute calib slot rects similar to draw_calib_strip assumptions
                            int slots = 5;
                            int gap = 2;
                            int eff_w = std::max(1, cw - 4);
                            int slot_w = std::max(4, (eff_w - gap * (slots - 1)) / slots);
                            int sy = y0 + (row_h_local - slot_w) / 2; // approximate
                            int si = render_state->highlight_aux0;
                            if (si >= 0 && si < slots) {
                                int sx = x0 + 2 + si * (slot_w + gap);
                                draw_outline(sx, y0, slot_w, row_h_local, hcol);
                            }
                        } else if (part == GP_TABLE_HIT_SCROLL_THUMB) {
                            draw_outline(x0, y0, cw, row_h_local, hcol);
                        } else {
                            // default: whole-cell outline
                            draw_outline(x0, y0, cw, st_local.row_h, hcol);
                        }
                    }
                }
            }
        } else if (render_state->mouse_x >= 0 && render_state->mouse_y >= 0) {
            // mouse-based highlight: find hitbox under mouse if hitboxes were emitted
            if (hitboxes_out && hitboxes_cap > 0) {
                int mx = render_state->mouse_x;
                int my = render_state->mouse_y;
                for (int hi = 0; hi < hit_count; ++hi) {
                    const GP_TableHitBox &hb = hitboxes_out[hi];
                    if (mx >= hb.x0 && mx < hb.x1 && my >= hb.y0 && my < hb.y1) {
                        draw_outline(hb.x0, hb.y0, hb.x1 - hb.x0, hb.y1 - hb.y0, hcol);
                        break;
                    }
                }
            }
        }
    }

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
    return gp_table_raster_rgba_with_hits(rows, row_count, cols, col_count, style, /*render_state=*/nullptr, out_rgba, out_len_bytes, out_geom, nullptr, 0, nullptr);
}

// Stateful context ----------------------------------------------------------

struct GP_TableContext {
    GP_TableStyle style_raw{};
    Style st{};
    std::vector<GP_TableRow> rows;
    std::vector<GP_TableColumn> cols;
    GP_TableGeom geom{};
    // Scroll state: row offset (first visible row index) and fraction (0..1)
    int32_t scroll_row_offset = 0;
    float scroll_frac = 0.0f;
    float scroll_frac_x = 0.0f;
    // selected LED keys: packed (row<<32) | (col<<16) | led_index
    std::unordered_set<uint64_t> selected_leds;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    // Relaxation state (per-edge values/velocities are maintained in parallel to edges)
    int32_t relax_mode = GP_TABLE_RELAX_OFF;
    float relax_stiffness = 10.0f;
    float relax_damping = 2.0f;
    float relax_threshold = 1e-3f;
    int32_t relax_max_iters = 200;
    double relax_last_time = 0.0; // seconds since epoch
    std::vector<float> relax_value; // 0..1 value per edge (1.0 = relaxed)
    std::vector<float> relax_vel;   // velocity per edge
    // rope simulator instance and per-edge rope indices
    RopeSim* rope_sim = nullptr;
    int rope_sim_owned = 0; // 1 if this context owns and should destroy the sim
    std::vector<int> rope_sim_idx; // rope index per edge (same order as ctx->edges)
    // Prospective live-edge state (used when one node selected and mode enabled)
    int32_t prospective_mode = 0; // 0=off,1=on
    bool prospective_initialized = false;
    float prospective_x = 0.0f;
    float prospective_y = 0.0f;
    float prospective_vx = 0.0f;
    float prospective_vy = 0.0f;
    // queue of recent mouse targets (front = oldest to be cleared next)
    std::vector<std::pair<float,float>> prospective_targets;
    int32_t prospective_max_history = 8;
    float prospective_slack = 4.0f;
    float prospective_rope_length = 0.0f;
    int prospective_rope_idx = -1; // temporary rope index for live prospective cable
    // editable flag (editor potential). Default: editable (1).
    int32_t editable = 1;
    // per-key type hints and io sets
    std::unordered_map<uint64_t,int32_t> key_type_hint; // key -> type_id
    std::unordered_set<uint64_t> key_is_input; // keys marked input
    std::unordered_set<uint64_t> key_is_output; // keys marked output
    // reading direction for sides (0=input,1=output). 0=LTR,1=TTB,2=RTL,3=BTB
    int32_t side_reading_dir[2] = {0, 0};
    // LED grid preference (cols, rows, aspect)
    int32_t led_pref_cols = 0;
    int32_t led_pref_rows = 0;
    float led_pref_aspect = 0.0f;
    // Optional step callback for node/table shims
    GP_TableStepFn step_callback = nullptr;
    void* step_user = nullptr;
};

// Attach/detach an external RopeSim instance to the table context.
// If `sim` is non-null the table will use that simulator for all rope
// allocations and updates. `take_ownership` indicates whether the table
// should destroy the provided simulator when the table is destroyed
// (1 = table destroys the sim, 0 = caller retains ownership). Passing
// `sim == NULL` detaches any external simulator; the table may create
// its own simulator later on demand. Returns 1 on success.
int32_t gp_table_attach_rope_sim(GP_TableContext* ctx, RopeSim* sim, int32_t take_ownership) {
    if (!ctx) return 0;
    // If we currently own a sim, destroy it first
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
    }
    ctx->rope_sim = sim;
    ctx->rope_sim_owned = (sim != nullptr) ? (take_ownership ? 1 : 0) : 0;
    // reset indices so edges will create ropes in the new sim when next rendered
    ctx->rope_sim_idx.clear();
    return 1;
}

int32_t gp_table_set_step_callback(GP_TableContext* ctx, GP_TableStepFn cb, void* user) {
    if (!ctx) return 0;
    ctx->step_callback = cb;
    ctx->step_user = user;
    return 1;
}

int32_t gp_table_clear_step_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->step_callback = nullptr;
    ctx->step_user = nullptr;
    return 1;
}

int32_t gp_table_step(GP_TableContext* ctx, const float* inputs, int32_t in_count, float* outputs, int32_t out_count, double dt) {
    if (!ctx) return 0;
    if (!ctx->step_callback) return 0;
    // Defensive: allow null arrays as zero-length
    if ((in_count > 0 && !inputs) || (out_count > 0 && !outputs)) return 0;
    try {
        ctx->step_callback(ctx->step_user, inputs, in_count, outputs, out_count, dt);
        return 1;
    } catch (...) {
        return 0;
    }
}

static void recompute_geom(GP_TableContext* ctx) {
    if (!ctx) return;
    ctx->geom.width_px = ctx->st.w;
    ctx->geom.height_px = std::max<int32_t>(1, static_cast<int32_t>(ctx->rows.size()) * ctx->st.row_h);
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, ctx->geom.col_x0, ctx->geom.col_w);
}

static void draw_circle_outline(uint8_t* img, int w, int h, int pitch, int cx, int cy, int r, int thickness, Color c) {
    if (!img || r <= 0) return;
    const int r2 = r * r;
    const int outer = r + thickness;
    const int outer2 = outer * outer;
    for (int dy = -outer; dy <= outer; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -outer; dx <= outer; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= w) continue;
            int d2 = dx * dx + dy * dy;
            if (d2 >= r2 && d2 <= outer2) {
                uint8_t* p = img + y * pitch + x * 4;
                p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
            }
        }
    }
}

// Blend src color (with alpha 0..255) over destination pixel in-place
static inline void blend_pixel(uint8_t* dst, uint8_t sr, uint8_t sg, uint8_t sb, uint8_t sa) {
    if (!dst) return;
    float a = sa / 255.0f;
    if (a <= 0.0f) return;
    float inv = 1.0f - a;
    float dr = dst[0] / 255.0f;
    float dg = dst[1] / 255.0f;
    float db = dst[2] / 255.0f;
    float da = dst[3] / 255.0f;
    float srf = sr / 255.0f;
    float sgf = sg / 255.0f;
    float sbf = sb / 255.0f;
    float outa = a + da * inv;
    if (outa <= 0.0f) {
        dst[0] = dst[1] = dst[2] = dst[3] = 0;
        return;
    }
    float out_r = (srf * a + dr * da * inv) / outa;
    float out_g = (sgf * a + dg * da * inv) / outa;
    float out_b = (sbf * a + db * da * inv) / outa;
    dst[0] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_r)) * 255.0f));
    dst[1] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_g)) * 255.0f));
    dst[2] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_b)) * 255.0f));
    dst[3] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, outa)) * 255.0f));
}

// Draw a filled circle with blending (soft core) used for cable sampling points
static void draw_blob_blend(uint8_t* img, int w, int h, int pitch, int cx, int cy, int radius, Color c) {
    if (!img || radius <= 0) return;
    int r = radius;
    int r2 = r * r;
    int y0 = std::max(0, cy - r);
    int y1 = std::min(h - 1, cy + r);
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        int dx_limit = static_cast<int>(std::floor(std::sqrt((double)r2 - double(dy * dy))));
        int x0 = std::max(0, cx - dx_limit);
        int x1 = std::min(w - 1, cx + dx_limit);
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            int d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            // simple linear falloff alpha across radius
            float t = 1.0f - (std::sqrt((float)d2) / (float)r);
            uint8_t sa = static_cast<uint8_t>(std::lround(c.a * t));
            uint8_t sr = c.r;
            uint8_t sg = c.g;
            uint8_t sb = c.b;
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, sr, sg, sb, sa);
        }
    }
}

// Draw a droopy blended cable between two points.
static void draw_cable_blend(uint8_t* img, int w, int h, int pitch, int ax, int ay, int bx, int by, int jacket_px, int jacket_border, Color core_col, int segments, float sag_factor, float relax_v) {
    if (!img) return;
    if (segments < 4) segments = 4;
    float dx = float(bx - ax);
    float dy = float(by - ay);
    float dist = std::sqrt(dx*dx + dy*dy);
    float sag = dist * sag_factor;
    // core color modulated by relax value but keep a minimum visibility so cables never fully disappear
    Color core = core_col;
    float rv = std::min(1.0f, std::max(0.0f, relax_v));
    const float min_vis = 0.18f;
    float vis = std::max(min_vis, rv);
    core.a = static_cast<uint8_t>(std::lround(core.a * vis));
    // jacket color (almost black)
    Color jacket{10,10,10,220};
    Color jacket_edge{40,40,40,160};
    // choose samples so blob spacing is <= ~0.6 * jacket_px to avoid visible gaps
    int min_seg_for_spacing = 1;
    if (jacket_px > 0) min_seg_for_spacing = static_cast<int>(std::ceil(dist / (std::max(1.0f, float(jacket_px) * 0.6f))));
    int use_segments = std::max(segments, std::max(4, min_seg_for_spacing));
    // sample points and draw overlapping blobs for a continuous tube
    for (int si = 0; si <= use_segments; ++si) {
        float t = float(si) / float(use_segments);
        float px = float(ax) + dx * t;
        float py = float(ay) + dy * t + sag * std::sin(3.14159265f * t);
        int ipx = static_cast<int>(std::lround(px));
        int ipy = static_cast<int>(std::lround(py));
        // outer jacket blob (solid-ish)
        draw_blob_blend(img, w, h, pitch, ipx, ipy, jacket_px, jacket);
        // thin jacket edge to give depth
        draw_blob_blend(img, w, h, pitch, ipx, ipy, std::max(1, jacket_px - 1), jacket_edge);
        // core
        int core_r = std::max(1, jacket_px - jacket_border);
        Color corec = core;
        // slightly boosted alpha near center
        corec.a = static_cast<uint8_t>(std::lround(corec.a * 1.0f));
        draw_blob_blend(img, w, h, pitch, ipx, ipy, core_r, corec);
    }
    // end plugs: black outer circles
    draw_circle(img, w, h, pitch, ax, ay, jacket_px, Color{0,0,0,255});
    draw_circle(img, w, h, pitch, bx, by, jacket_px, Color{0,0,0,255});
    // inner core caps
    Color corecap = core_col;
    corecap.a = static_cast<uint8_t>(std::lround(corecap.a * 1.0f));
    draw_circle(img, w, h, pitch, ax, ay, std::max(1, jacket_px - jacket_border), corecap);
    draw_circle(img, w, h, pitch, bx, by, std::max(1, jacket_px - jacket_border), corecap);
}

// Draw a smooth blended rope/tube along given interleaved vertices using Catmull-Rom
// verts: float array [x0,y0, x1,y1, ...], count = number of vertices
static void draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, Color core_col, int samples_per_segment) {
    if (!img || !verts || count < 2) return;
    if (samples_per_segment < 2) samples_per_segment = 2;

    auto get = [&](int idx) {
        // clamp
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };

    auto catmull = [&](int i, float t) {
        // control points p0..p3 for segment between i and i+1
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        // Catmull-Rom with tension 0.5
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    // Build dense samples along the spline then rasterize as thick blended segments
    std::vector<std::pair<float,float>> samples;
    samples.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        // estimate chord length for this segment (p1-p2)
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // sample spacing: ~0.6*jacket_px to ensure overlap
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        for (int s = 0; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            samples.emplace_back(p.first, p.second);
        }
    }

    if (samples.empty()) return;

    // helper: draw a blended thick segment by per-pixel distance to the segment
    auto draw_segment_blend = [&](float x1, float y1, float x2, float y2, int radius, Color col) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        float rplus = float(radius) + 1.0f; // allow soft edge
        int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
        int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
        int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
        int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, w - 1);
        maxy = std::min(maxy, h - 1);
        for (int py = miny; py <= maxy; ++py) {
            for (int px = minx; px <= maxx; ++px) {
                float cx = px + 0.5f;
                float cy = py + 0.5f;
                float t = 0.0f;
                if (len2 > 1e-6f) {
                    t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                    if (t < 0.0f) t = 0.0f;
                    else if (t > 1.0f) t = 1.0f;
                }
                float closestX = x1 + dx * t;
                float closestY = y1 + dy * t;
                float ddx = cx - closestX;
                float ddy = cy - closestY;
                float dist = std::sqrt(ddx*ddx + ddy*ddy);
                if (dist <= rplus) {
                    float coverage = 1.0f - (dist / rplus);
                    uint8_t a = static_cast<uint8_t>(std::lround(float(col.a) * coverage));
                    if (a == 0) continue;
                    Color cc = col;
                    cc.a = a;
                    uint8_t* dstp = img + py * pitch + px * 4;
                    blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
                }
            }
        }
    };

    // draw black backing rims then an inner core cap so rope drawn afterwards overwrites interior
    auto start = samples.front();
    auto end = samples.back();
    int rim_r = jacket_px + 2; // slightly larger rim
    draw_blob_blend(img, w, h, pitch, static_cast<int>(std::lround(start.first)), static_cast<int>(std::lround(start.second)), rim_r, Color{0,0,0,255});
    draw_blob_blend(img, w, h, pitch, static_cast<int>(std::lround(end.first)), static_cast<int>(std::lround(end.second)), rim_r, Color{0,0,0,255});
    // inner core cap (rope color) to sit inside rim
    int core_r_cap = std::max(1, jacket_px - jacket_border);
    Color corecap = core_col;
    corecap.a = static_cast<uint8_t>(std::lround(corecap.a));
    draw_blob_blend(img, w, h, pitch, static_cast<int>(std::lround(start.first)), static_cast<int>(std::lround(start.second)), core_r_cap, corecap);
    draw_blob_blend(img, w, h, pitch, static_cast<int>(std::lround(end.first)), static_cast<int>(std::lround(end.second)), core_r_cap, corecap);

    // draw jacket (slightly slimmer to reduce chunkiness)
    int eff_jacket = std::max(1, jacket_px - 1);
    Color jacket_col = Color{200,200,200, static_cast<uint8_t>(std::lround(180.0f))};
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        draw_segment_blend(a.first, a.second, b.first, b.second, eff_jacket, jacket_col);
    }

    // draw core (thinner, translucent core_col)
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        draw_segment_blend(a.first, a.second, b.first, b.second, core_r, core_col);
    }
}

// Convert HSV (h in 0..1, s 0..1, v 0..1) to Color (alpha=255)
static inline Color hsv_to_color(float h, float s, float v, uint8_t a=255) {
    h = h - std::floor(h);
    float hh = h * 6.0f;
    int i = static_cast<int>(std::floor(hh));
    float f = hh - float(i);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    float r=0,g=0,b=0;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return Color{ static_cast<uint8_t>(std::lround(r * 255.0f)), static_cast<uint8_t>(std::lround(g * 255.0f)), static_cast<uint8_t>(std::lround(b * 255.0f)), a };
}

// Colored variant: `hues` is an optional array of per-vertex hue values in [0..1]. If null, falls back to neutral drawing.
static void draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity) {
    if (!hues || hue_count <= 0) {
        // fallback
        Color neutral{200,200,200,200};
        draw_rope_curve_blend(img, w, h, pitch, verts, count, jacket_px, jacket_border, neutral, samples_per_segment);
        return;
    }
    if (!img || !verts || count < 2) return;

    // Build samples along spline (same as non-colored variant)
    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    std::vector<std::pair<float,float>> samples;
    samples.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        for (int s = 0; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            samples.emplace_back(p.first, p.second);
        }
    }
    if (samples.empty()) return;

    // build hue samples aligned with `samples` by interpolating provided `hues` across verts
    std::vector<float> hue_samples;
    hue_samples.reserve(samples.size());
    for (size_t si = 0; si < samples.size(); ++si) {
        float u = float(si) / float(std::max<size_t>(1, samples.size() - 1));
        float v = u * float(std::max(1, count - 1));
        int idx = static_cast<int>(std::floor(v));
        float ft = v - float(idx);
        float h0 = hues[std::min(idx, hue_count-1) % hue_count];
        float h1 = hues[std::min(idx+1, hue_count-1) % hue_count];
        float hh = h0 * (1.0f - ft) + h1 * ft;
        hue_samples.push_back(hh);
    }

    // helper to draw segments (reuse draw_segment_blend lambda from earlier by reimplementing minimal inline)
    auto draw_segment_blend_local = [&](float x1, float y1, float x2, float y2, int radius, Color col) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        float rplus = float(radius) + 1.0f;
        int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
        int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
        int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
        int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, w - 1);
        maxy = std::min(maxy, h - 1);
        for (int py = miny; py <= maxy; ++py) {
            for (int px = minx; px <= maxx; ++px) {
                float cx = px + 0.5f;
                float cy = py + 0.5f;
                float t = 0.0f;
                if (len2 > 1e-6f) {
                    t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                    if (t < 0.0f) t = 0.0f;
                    else if (t > 1.0f) t = 1.0f;
                }
                float closestX = x1 + dx * t;
                float closestY = y1 + dy * t;
                float ddx = cx - closestX;
                float ddy = cy - closestY;
                float dist = std::sqrt(ddx*ddx + ddy*ddy);
                if (dist <= rplus) {
                    float coverage = 1.0f - (dist / rplus);
                    uint8_t a = static_cast<uint8_t>(std::lround(float(col.a) * coverage));
                    if (a == 0) continue;
                    Color cc = col;
                    cc.a = a;
                    uint8_t* dstp = img + py * pitch + px * 4;
                    blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
                }
            }
        }
    };

    // Draw neutral jacket first
    int eff_jacket = std::max(1, jacket_px - 1);
    Color jacket_col = Color{200,200,200, static_cast<uint8_t>(std::lround(180.0f))};
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        draw_segment_blend_local(a.first, a.second, b.first, b.second, eff_jacket, jacket_col);
    }

    // Draw colored core using hue_samples
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float hue = hue_samples[i];
        Color hc = hsv_to_color(hue, 1.0f, 1.0f, static_cast<uint8_t>(std::lround(255.0f * std::clamp(hue_intensity, 0.0f, 1.0f))));
        draw_segment_blend_local(samples[i].first, samples[i].second, samples[i+1].first, samples[i+1].second, core_r, hc);
    }
    // note: endpoint rims already drawn by caller if desired
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
    // create rope simulator with reasonable capacities (owned by default)
    int max_ropes = 1024;
    int max_segs = std::max(4, ctx->st.cable_segments);
    ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    ctx->rope_sim_owned = 1;
    ctx->rope_sim_idx.clear();
    return ctx.release();
}

void gp_table_destroy(GP_TableContext* ctx) {
    if (!ctx) return;
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        ctx->rope_sim = nullptr;
    }
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

int32_t gp_table_on_click(GP_TableContext* ctx, int32_t x, int32_t y, GP_TableHitBox* out_hit) {
    if (!ctx) return 0;
    // Render into a temporary buffer to collect hitboxes
    GP_TableGeom geom{};
    gp_table_get_geom(ctx, &geom);
    const int w = geom.width_px;
    const int h = geom.height_px;
    if (w <= 0 || h <= 0) return 0;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    const int hitcap = 4096;
    std::vector<GP_TableHitBox> hits(hitcap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(ctx, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &written);
    if (!ok) return 0;
    // find first hit that contains (x,y)
    for (int i = 0; i < written; ++i) {
        const GP_TableHitBox &hb = hits[i];
        if (x >= hb.x0 && x < hb.x1 && y >= hb.y0 && y < hb.y1) {
            // process default actions
            if (out_hit) *out_hit = hb;
            // expand toggle
            if (hb.part == GP_TABLE_HIT_EXPAND && hb.row_idx >= 0 && hb.row_idx < static_cast<int>(ctx->rows.size())) {
                ctx->rows[hb.row_idx].expanded = ctx->rows[hb.row_idx].expanded ? 0 : 1;
                recompute_geom(ctx);
                return 1;
            }
            // scroll up/down
            if (hb.part == GP_TABLE_HIT_SCROLL_UP) {
                ctx->scroll_row_offset = std::max(0, ctx->scroll_row_offset - 1);
                return 1;
            }
            if (hb.part == GP_TABLE_HIT_SCROLL_DOWN) {
                ctx->scroll_row_offset = std::min<int>(std::max(0, static_cast<int>(ctx->rows.size()) - 1), ctx->scroll_row_offset + 1);
                return 1;
            }
            // LED click: toggle selection (separate from on/off state)
            if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && hb.row_idx >= 0 && hb.col_idx >= 0) {
                int r_idx = hb.row_idx;
                int c_idx = hb.col_idx;
                int led = hb.aux0;
                if (led >= 0) {
                    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(r_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(c_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led));
                    auto it = ctx->selected_leds.find(key);
                    if (it == ctx->selected_leds.end()) ctx->selected_leds.insert(key);
                    else ctx->selected_leds.erase(it);
                            // If exactly one other LED was selected prior to this click, form an edge.
                            if (ctx->selected_leds.size() == 2) {
                                // grab the two keys
                                auto it2 = ctx->selected_leds.begin();
                                uint64_t k0 = *it2; ++it2; uint64_t k1 = *it2;
                                // add edge if allowed by node-group rules (directional: k0 -> k1)
                                int allowed = gp_table_node_group_is_edge_allowed(ctx, k0, k1);
                                if (allowed == 1) {
                                    // use API helper so relax arrays are kept in sync
                                    gp_table_add_edge(ctx, k0, k1);
                                }
                                // clear selections after attempting to form edge
                                ctx->selected_leds.clear();
                            }
                            return 1;
                }
            }
            // Other parts: no default action, but return hit
            return 1;
        }
    }
    return 0;
}

int32_t gp_table_set_scroll_fraction(GP_TableContext* ctx, float frac) {
    if (!ctx) return 0;
    ctx->scroll_frac = std::clamp(frac, 0.0f, 1.0f);
    // compute row offset based on fraction and available rows
    int visible_rows = std::max(1, ctx->geom.height_px / ctx->st.row_h);
    int total = static_cast<int>(ctx->rows.size());
    int max_off = std::max(0, total - visible_rows);
    ctx->scroll_row_offset = static_cast<int>(std::lround(ctx->scroll_frac * float(max_off)));
    return 1;
}

int32_t gp_table_set_scroll_fraction_xy(GP_TableContext* ctx, float frac_x, float frac_y) {
    if (!ctx) return 0;
    ctx->scroll_frac_x = std::clamp(frac_x, 0.0f, 1.0f);
    return gp_table_set_scroll_fraction(ctx, frac_y);
}

// Edge list helpers
int32_t gp_table_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    ctx->edges.emplace_back(a, b);
    // ensure relax arrays stay in sync
    // start unrelaxed so the cable animates into place
    ctx->relax_value.push_back(0.0f);
    ctx->relax_vel.push_back(0.0f);
    // reset prospective state when a real edge is added
    ctx->prospective_initialized = false;
    // create a rope entry in the simulator (if available)
    if (ctx->rope_sim) {
        // compute approximate endpoints in table-local coords
        int ax = 0, ay = 0, bx = 0, by = 0;
        auto compute_center_local = [&](uint64_t key, int &outx, int &outy) {
            outx = -1; outy = -1;
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            if (r_orig >= ctx->rows.size()) return;
            const GP_TableRow &row = ctx->rows[static_cast<size_t>(r_orig)];
            if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
            int col_x0[8] = {0}; int col_w[8] = {0};
            compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);
            const GP_TableCell &cell = row.cells[static_cast<int>(c_idx)];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = static_cast<int>(r_orig) * ctx->st.row_h;
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                outx = cx0 + static_cast<int>(led) * led_spacing;
                outy = y0 + ctx->st.row_h / 2;
            }
        };
        compute_center_local(a, ax, ay);
        compute_center_local(b, bx, by);
        int segs = std::max(4, ctx->st.cable_segments);
        float slack = 0.0f;
        int idx = rope_sim_add_rope(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack);
        ctx->rope_sim_idx.push_back(idx);
    } else {
        ctx->rope_sim_idx.push_back(-1);
    }
    return 1;
}

int32_t gp_table_clear_edges(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->edges.clear();
    ctx->relax_value.clear();
    ctx->relax_vel.clear();
    // reset rope simulator indices and recreate sim to free resources
    ctx->rope_sim_idx.clear();
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        int max_ropes = 1024;
        int max_segs = std::max(4, ctx->st.cable_segments);
        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    } else {
        // if rope_sim is external or null, leave it alone; indices already cleared
    }
    return 1;
}

int32_t gp_table_get_edge_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->edges.size());
}

int32_t gp_table_get_edge(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_a, unsigned long long* out_b) {
    if (!ctx || !out_a || !out_b) return 0;
    if (idx < 0 || idx >= static_cast<int32_t>(ctx->edges.size())) return 0;
    *out_a = ctx->edges[static_cast<size_t>(idx)].first;
    *out_b = ctx->edges[static_cast<size_t>(idx)].second;
    return 1;
}

int32_t gp_table_get_selected_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->selected_leds.size());
}

int32_t gp_table_get_selected_key(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_key) {
    if (!ctx || !out_key) return 0;
    if (idx < 0 || idx >= static_cast<int>(ctx->selected_leds.size())) return 0;
    auto it = ctx->selected_leds.begin();
    std::advance(it, idx);
    *out_key = *it;
    return 1;
}

int32_t gp_table_set_prospective_mode(GP_TableContext* ctx, int32_t enabled) {
    if (!ctx) return 0;
    ctx->prospective_mode = enabled ? 1 : 0;
    if (!ctx->prospective_mode) ctx->prospective_initialized = false;
    return 1;
}

int32_t gp_table_get_prospective_mode(GP_TableContext* ctx, int32_t* out_enabled) {
    if (!ctx || !out_enabled) return 0;
    *out_enabled = ctx->prospective_mode;
    return 1;
}

// IO / type hint APIs
int32_t gp_table_set_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t type_id, int32_t is_input, int32_t is_output) {
    if (!ctx) return 0;
    ctx->key_type_hint[key] = type_id;
    if (is_input) ctx->key_is_input.insert(key); else ctx->key_is_input.erase(key);
    if (is_output) ctx->key_is_output.insert(key); else ctx->key_is_output.erase(key);
    return 1;
}

int32_t gp_table_get_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t* out_type_id, int32_t* out_is_input, int32_t* out_is_output) {
    if (!ctx) return 0;
    auto it = ctx->key_type_hint.find(key);
    if (out_type_id) *out_type_id = (it != ctx->key_type_hint.end()) ? it->second : -1;
    if (out_is_input) *out_is_input = ctx->key_is_input.count(key) ? 1 : 0;
    if (out_is_output) *out_is_output = ctx->key_is_output.count(key) ? 1 : 0;
    return 1;
}

int32_t gp_table_has_io_sections(GP_TableContext* ctx) {
    if (!ctx) return 0;
    return (ctx->key_is_input.size() > 0 && ctx->key_is_output.size() > 0) ? 1 : 0;
}

int32_t gp_table_enumerate_io_keys(GP_TableContext* ctx, int32_t direction, unsigned long long* out_keys, int32_t cap) {
    if (!ctx || !out_keys || cap <= 0) return 0;
    int written = 0;
    if (direction == 0) {
        for (auto k : ctx->key_is_input) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    } else {
        for (auto k : ctx->key_is_output) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    }
    return written;
}

int32_t gp_table_set_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t dir) {
    if (!ctx) return 0;
    if (side < 0 || side > 1) return 0;
    if (dir < 0 || dir > 3) return 0;
    ctx->side_reading_dir[side] = dir;
    return 1;
}

int32_t gp_table_get_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t* out_dir) {
    if (!ctx || !out_dir) return 0;
    if (side < 0 || side > 1) return 0;
    *out_dir = ctx->side_reading_dir[side];
    return 1;
}

int32_t gp_table_set_led_grid_preference(GP_TableContext* ctx, int32_t pref_cols, int32_t pref_rows, float pref_aspect) {
    if (!ctx) return 0;
    ctx->led_pref_cols = std::max(0, pref_cols);
    ctx->led_pref_rows = std::max(0, pref_rows);
    ctx->led_pref_aspect = pref_aspect;
    return 1;
}

int32_t gp_table_get_led_grid_preference(GP_TableContext* ctx, int32_t* out_pref_cols, int32_t* out_pref_rows, float* out_pref_aspect) {
    if (!ctx) return 0;
    if (out_pref_cols) *out_pref_cols = ctx->led_pref_cols;
    if (out_pref_rows) *out_pref_rows = ctx->led_pref_rows;
    if (out_pref_aspect) *out_pref_aspect = ctx->led_pref_aspect;
    return 1;
}

int32_t gp_table_prospective_set_params(GP_TableContext* ctx, int32_t max_history, float slack, float rope_length) {
    if (!ctx) return 0;
    ctx->prospective_max_history = std::max(1, max_history);
    ctx->prospective_slack = slack;
    ctx->prospective_rope_length = rope_length;
    // trim existing queue if needed
    if (static_cast<int>(ctx->prospective_targets.size()) > ctx->prospective_max_history) {
        ctx->prospective_targets.erase(ctx->prospective_targets.begin(), ctx->prospective_targets.begin() + (ctx->prospective_targets.size() - ctx->prospective_max_history));
    }
    return 1;
}

int32_t gp_table_prospective_get_params(GP_TableContext* ctx, int32_t* out_max_history, float* out_slack, float* out_rope_length) {
    if (!ctx) return 0;
    if (out_max_history) *out_max_history = ctx->prospective_max_history;
    if (out_slack) *out_slack = ctx->prospective_slack;
    if (out_rope_length) *out_rope_length = ctx->prospective_rope_length;
    return 1;
}

// Editable flag helpers
int32_t gp_table_set_editable(GP_TableContext* ctx, int32_t editable) {
    if (!ctx) return 0;
    ctx->editable = editable ? 1 : 0;
    return 1;
}

int32_t gp_table_get_editable(GP_TableContext* ctx, int32_t* out_editable) {
    if (!ctx || !out_editable) return 0;
    *out_editable = ctx->editable;
    return 1;
}

// Serialization format (simple binary):
// [8 bytes magic 'GPTBL001'][uint32_t version=1]
// then GP_TableStyle (raw), int32 col_count, cols[], int32 row_count, rows[], int32 edge_count, edges (u64,u64)..., int32 sel_count, sel_keys...
int32_t gp_table_serialize(GP_TableContext* ctx, char* out_buf, int32_t out_len) {
    if (!ctx) return 0;
    const uint32_t version = 1;
    const char magic[8] = {'G','P','T','B','L','0','0','1'};
    int32_t col_count = static_cast<int32_t>(ctx->cols.size());
    int32_t row_count = static_cast<int32_t>(ctx->rows.size());
    int32_t edge_count = static_cast<int32_t>(ctx->edges.size());
    int32_t sel_count = static_cast<int32_t>(ctx->selected_leds.size());
    int32_t need = 0;
    need += 8; // magic
    need += 4; // version
    need += static_cast<int32_t>(sizeof(GP_TableStyle));
    need += 4; // col_count
    need += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    need += 4; // row_count
    need += row_count * static_cast<int32_t>(sizeof(GP_TableRow));
    need += 4; // edge_count
    need += edge_count * static_cast<int32_t>(sizeof(uint64_t) * 2);
    need += 4; // sel_count
    need += sel_count * static_cast<int32_t>(sizeof(uint64_t));

    if (!out_buf) return need;
    if (out_len < need) return 0;

    char* p = out_buf;
    // write magic
    memcpy(p, magic, 8); p += 8;
    // write version
    memcpy(p, &version, 4); p += 4;
    // write style_raw
    memcpy(p, &ctx->style_raw, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // write cols
    memcpy(p, &col_count, 4); p += 4;
    if (col_count > 0) {
        memcpy(p, ctx->cols.data(), static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    }
    // write rows (we zero any waveform pointers)
    memcpy(p, &row_count, 4); p += 4;
    for (int i = 0; i < row_count; ++i) {
        GP_TableRow row = ctx->rows[static_cast<size_t>(i)];
        // clear wave pointers inside cells to avoid serializing pointers
        for (int c = 0; c < row.cell_count && c < 8; ++c) {
            row.cells[c].wave = nullptr;
        }
        memcpy(p, &row, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
    }
    // write edges
    memcpy(p, &edge_count, 4); p += 4;
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a = ctx->edges[static_cast<size_t>(i)].first;
        uint64_t b = ctx->edges[static_cast<size_t>(i)].second;
        memcpy(p, &a, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(p, &b, sizeof(uint64_t)); p += sizeof(uint64_t);
    }
    // write selected keys
    memcpy(p, &sel_count, 4); p += 4;
    for (auto k : ctx->selected_leds) {
        uint64_t key = k;
        memcpy(p, &key, sizeof(uint64_t)); p += sizeof(uint64_t);
    }

    return need;
}

int32_t gp_table_deserialize(GP_TableContext* ctx, const char* in_buf, int32_t in_len) {
    if (!ctx || !in_buf || in_len <= 0) return 0;
    const char expect_magic[8] = {'G','P','T','B','L','0','0','1'};
    if (in_len < 8 + 4 + static_cast<int>(sizeof(GP_TableStyle))) return 0;
    const char* p = in_buf;
    if (memcmp(p, expect_magic, 8) != 0) return 0;
    p += 8;
    uint32_t version = 0;
    memcpy(&version, p, 4); p += 4;
    if (version != 1) return 0;
    // read style
    GP_TableStyle style{};
    memcpy(&style, p, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // columns
    int32_t col_count = 0;
    memcpy(&col_count, p, 4); p += 4;
    if (col_count < 0 || col_count > 8) return 0;
    std::vector<GP_TableColumn> cols;
    if (col_count > 0) {
        cols.resize(static_cast<size_t>(col_count));
        memcpy(cols.data(), p, static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int>(sizeof(GP_TableColumn));
    }
    // rows
    int32_t row_count = 0;
    memcpy(&row_count, p, 4); p += 4;
    if (row_count < 0) return 0;
    std::vector<GP_TableRow> rows;
    if (row_count > 0) {
        rows.resize(static_cast<size_t>(row_count));
        for (int i = 0; i < row_count; ++i) {
            memcpy(&rows[static_cast<size_t>(i)], p, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
            // ensure wave pointers are null for safety
            for (int c = 0; c < rows[static_cast<size_t>(i)].cell_count && c < 8; ++c) rows[static_cast<size_t>(i)].cells[c].wave = nullptr;
        }
    }
    // edges
    int32_t edge_count = 0;
    memcpy(&edge_count, p, 4); p += 4;
    if (edge_count < 0) return 0;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    edges.reserve(static_cast<size_t>(edge_count));
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a=0,b=0;
        memcpy(&a, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(&b, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        edges.emplace_back(a,b);
    }
    // selected keys
    int32_t sel_count = 0;
    memcpy(&sel_count, p, 4); p += 4;
    if (sel_count < 0) return 0;
    std::unordered_set<uint64_t> selset;
    for (int i = 0; i < sel_count; ++i) {
        uint64_t key = 0;
        memcpy(&key, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        selset.insert(key);
    }

    // Commit to ctx: replace style, cols, rows, edges, selected set
    ctx->style_raw = style;
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols = std::move(cols);
    ctx->rows = std::move(rows);
    // clear existing edges via API to keep rope_sim indices consistent
    gp_table_clear_edges(ctx);
    for (auto &e : edges) gp_table_add_edge(ctx, e.first, e.second);
    ctx->selected_leds = std::move(selset);
    recompute_geom(ctx);
    return 1;
}

// Relaxation helper: ensure internal arrays match edges size (called before stepping)
static void ensure_relax_vectors(GP_TableContext* ctx) {
    if (!ctx) return;
    size_t n = ctx->edges.size();
    if (ctx->relax_value.size() < n) {
        ctx->relax_value.resize(n, 1.0f);
        ctx->relax_vel.resize(n, 0.0f);
    } else if (ctx->relax_value.size() > n) {
        ctx->relax_value.resize(n);
        ctx->relax_vel.resize(n);
    }
}

// Relaxation control
int32_t gp_table_relax_set_mode(GP_TableContext* ctx, int32_t mode) {
    if (!ctx) return 0;
    ctx->relax_mode = mode;
    if (mode == GP_TABLE_RELAX_WALL_TIME) {
        // set last_time to now
        using namespace std::chrono;
        ctx->relax_last_time = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
    }
    return 1;
}

int32_t gp_table_relax_get_mode(GP_TableContext* ctx, int32_t* out_mode) {
    if (!ctx || !out_mode) return 0;
    *out_mode = ctx->relax_mode;
    return 1;
}

int32_t gp_table_relax_set_params(GP_TableContext* ctx, float stiffness, float damping, float threshold, int32_t max_iters) {
    if (!ctx) return 0;
    ctx->relax_stiffness = stiffness;
    ctx->relax_damping = damping;
    ctx->relax_threshold = threshold;
    ctx->relax_max_iters = max_iters > 0 ? max_iters : 1;
    return 1;
}

int32_t gp_table_relax_step(GP_TableContext* ctx, float dt) {
    if (!ctx) return 0;
    if (dt <= 0.0f) return 0;
    ensure_relax_vectors(ctx);
    float max_change = 0.0f;
    const float stiffness = ctx->relax_stiffness;
    const float damping = ctx->relax_damping;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        float &v = ctx->relax_value[i];
        float &vel = ctx->relax_vel[i];
        const float target = 1.0f;
        float acc = stiffness * (target - v) - damping * vel;
        vel += acc * dt;
        float dv = vel * dt;
        v += dv;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        max_change = std::max(max_change, std::abs(dv));
    }
    return 1;
}

int32_t gp_table_relax_update(GP_TableContext* ctx) {
    if (!ctx) return 0;
    using namespace std::chrono;
    double now = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
    double last = ctx->relax_last_time;
    if (last <= 0.0) last = now;
    double dt = now - last;
    ctx->relax_last_time = now;
    if (dt <= 0.0) return 0;
    return gp_table_relax_step(ctx, static_cast<float>(dt));
}

int32_t gp_table_relax_run_until_stable(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ensure_relax_vectors(ctx);
    const float threshold = ctx->relax_threshold;
    const int max_iters = ctx->relax_max_iters;
    const float dt = 0.016f; // fixed small step (60Hz)
    for (int it = 0; it < max_iters; ++it) {
        // step and compute max change
        float max_change = 0.0f;
        const float stiffness = ctx->relax_stiffness;
        const float damping = ctx->relax_damping;
        for (size_t i = 0; i < ctx->edges.size(); ++i) {
            float &v = ctx->relax_value[i];
            float &vel = ctx->relax_vel[i];
            const float target = 1.0f;
            float acc = stiffness * (target - v) - damping * vel;
            vel += acc * dt;
            float dv = vel * dt;
            v += dv;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            max_change = std::max(max_change, std::abs(dv));
        }
        if (max_change <= threshold) return 1;
    }
    return 1;
}

int32_t gp_table_get_scroll_fraction(GP_TableContext* ctx, float* out_frac) {
    if (!ctx || !out_frac) return 0;
    *out_frac = ctx->scroll_frac;
    return 1;
}

int32_t gp_table_get_scroll_fraction_xy(GP_TableContext* ctx, float* out_frac_x, float* out_frac_y) {
    if (!ctx) return 0;
    if (out_frac_x) *out_frac_x = ctx->scroll_frac_x;
    if (out_frac_y) *out_frac_y = ctx->scroll_frac;
    return 1;
}

int32_t gp_table_get_row_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->rows.size());
}

int32_t gp_table_get_row(const GP_TableContext* ctx, int32_t idx, GP_TableRow* out_row) {
    if (!ctx || !out_row) return 0;
    if (idx < 0 || idx >= static_cast<int32_t>(ctx->rows.size())) return 0;
    *out_row = ctx->rows[static_cast<size_t>(idx)];
    return 1;
}

int32_t gp_table_set_led_selected(GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, int32_t selected) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    if (selected) ctx->selected_leds.insert(key);
    else ctx->selected_leds.erase(key);
    return 1;
}

int32_t gp_table_get_led_selected(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    return ctx->selected_leds.find(key) != ctx->selected_leds.end() ? 1 : 0;
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
        /*render_state=*/nullptr,
        out_rgba,
        out_len_bytes,
        out_geom,
        hitboxes_out,
        hitboxes_cap,
        hitboxes_written);
}

int32_t gp_table_render_rgba_with_state(
    GP_TableContext* ctx,
    const GP_TableRenderState* render_state,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!ctx) return 0;

    // Build filtered visible rows according to expanded flags and create mapping
    std::vector<GP_TableRow> vis_rows;
    std::vector<int> map_vis_to_orig; // vis index -> original index
    vis_rows.reserve(ctx->rows.size());
    map_vis_to_orig.reserve(ctx->rows.size());
    const int max_depth = 64;
    std::vector<bool> parent_expanded(max_depth, true);
    for (size_t i = 0; i < ctx->rows.size(); ++i) {
        const GP_TableRow &r = ctx->rows[i];
        int d = std::max(0, r.depth);
        if (d >= max_depth) d = max_depth - 1;
        bool visible = true;
        for (int dd = 1; dd <= d; ++dd) {
            if (!parent_expanded[dd]) { visible = false; break; }
        }
        if (visible) {
            vis_rows.push_back(r);
            map_vis_to_orig.push_back(static_cast<int>(i));
        }
        // set expanded flag for children
        parent_expanded[d+1 < max_depth ? d+1 : d] = (r.expanded != 0);
        // clear deeper flags if next row has smaller depth will happen on next iteration
    }

    // Prepare a local render state with translated highlight_row if present
    GP_TableRenderState local_rs{};
    if (render_state) local_rs = *render_state;
    if (render_state && render_state->highlight_row >= 0) {
        // find mapped index
        int orig = render_state->highlight_row;
        int mapped = -1;
        for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == orig) { mapped = static_cast<int>(vi); break; }
        local_rs.highlight_row = mapped;
    }

    // Call the raster on the visible rows
    GP_TableGeom local_geom{};
    int ok = gp_table_raster_rgba_with_hits(
        vis_rows.data(), static_cast<int32_t>(vis_rows.size()),
        ctx->cols.data(), static_cast<int32_t>(ctx->cols.size()),
        &ctx->style_raw,
        render_state ? &local_rs : nullptr,
        out_rgba,
        out_len_bytes,
        &local_geom,
        hitboxes_out,
        hitboxes_cap,
        hitboxes_written);

    if (!ok) return 0;

    // Prepare column geometry for overlay placement
    int col_x0[8] = {0};
    int col_w[8] = {0};
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);

    // Draw selection overlays for any selected LEDs kept in ctx.
    // Selected LEDs are stored as packed keys: (row<<32)|(col<<16)|led
    if (out_rgba && !ctx->selected_leds.empty()) {
        int w_local = local_geom.width_px;
        int h_local = local_geom.height_px;
        int pitch_local = w_local * 4;
        Color sel_col{80, 160, 255, 192};
        // For each selected LED, compute its position in visible rows and draw outline.
        for (uint64_t key : ctx->selected_leds) {
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            // find visible index
            int vis_idx = -1;
            for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == static_cast<int>(r_orig)) { vis_idx = static_cast<int>(vi); break; }
            if (vis_idx < 0) continue;
            const GP_TableRow &row = vis_rows[vis_idx];
            if (c_idx < 0 || c_idx >= static_cast<uint32_t>(row.cell_count)) continue;
            const GP_TableCell &cell = row.cells[c_idx];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = vis_idx * ctx->st.row_h;
            // approximate LED layout similar to raster
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                int cx = cx0 + static_cast<int>(led) * led_spacing;
                int cy = y0 + ctx->st.row_h / 2;
                draw_circle_outline(out_rgba, w_local, h_local, pitch_local, cx, cy, radius + 2, 2, sel_col);
            }
        }
    }


    // Draw edges between LED centers
    // Prospective live-edge: if enabled and exactly one LED selected, we will
    // relax a free end of the cord toward the supplied mouse position and
    // render a live prospective edge. This updates small state in the context
    // so behavior is smooth across frames.
    if (out_rgba) {
        double dt_frame = 1.0 / 60.0; // default frame dt
        // compute wall-time dt once per frame and step relaxation so edges animate between frames
        using namespace std::chrono;
        double now_frame = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
        double last_frame = ctx->relax_last_time;
        if (last_frame <= 0.0) last_frame = now_frame;
        dt_frame = now_frame - last_frame;
        if (dt_frame <= 0.0) dt_frame = 1.0 / 60.0;
        ctx->relax_last_time = now_frame;
        // step relax values by the frame dt (keeps animation in render loop)
        gp_table_relax_step(ctx, static_cast<float>(dt_frame));
        // attempt prospective update/draw if enabled
        if (ctx->prospective_mode && ctx->selected_leds.size() == 1) {
            if (render_state && render_state->mouse_x >= 0 && render_state->mouse_y >= 0) {
                // compute selected node center
                uint64_t sel_key = *ctx->selected_leds.begin();
                int sx = -1, sy = -1;
                {
                    uint32_t r_orig = static_cast<uint32_t>(sel_key >> 32);
                    uint32_t c_idx = static_cast<uint32_t>((sel_key >> 16) & 0xFFFFu);
                    uint32_t led = static_cast<uint32_t>(sel_key & 0xFFFFu);
                    int vis_idx = -1;
                    for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == static_cast<int>(r_orig)) { vis_idx = static_cast<int>(vi); break; }
                    if (vis_idx >= 0) {
                        const GP_TableRow &row = vis_rows[vis_idx];
                        if (static_cast<int>(c_idx) >= 0 && static_cast<int>(c_idx) < row.cell_count) {
                            const GP_TableCell &cell = row.cells[c_idx];
                            int x0 = col_x0[static_cast<int>(c_idx)];
                            int cw = col_w[static_cast<int>(c_idx)];
                            int y0 = vis_idx * ctx->st.row_h;
                            int led_count = 9;
                            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                                if (count == 0) count = (cell.flags & 0xFF);
                                if (count == 0) count = 12;
                                led_count = count;
                            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                                led_count = 8;
                            }
                            int eff_w = std::max(1, cw - 4);
                            int radius = 4;
                            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
                            int cx0 = x0 + 2 + led_spacing;
                            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                                sx = cx0 + static_cast<int>(led) * led_spacing;
                                sy = y0 + ctx->st.row_h / 2;
                            }
                        }
                    }
                }
                if (sx >= 0 && sy >= 0) {
                    // compute local image dims/pitch for drawing (use distinct names)
                    int p_w = local_geom.width_px;
                    int p_h = local_geom.height_px;
                    int p_pitch = p_w * 4;
                        // target is mouse in table-local coords; push into queue
                        float tx = static_cast<float>(render_state->mouse_x);
                        float ty = static_cast<float>(render_state->mouse_y);
                        ctx->prospective_targets.emplace_back(tx, ty);
                        // drop oldest if history exceeds max
                        if (static_cast<int>(ctx->prospective_targets.size()) > ctx->prospective_max_history) {
                            int drop = static_cast<int>(ctx->prospective_targets.size()) - ctx->prospective_max_history;
                            ctx->prospective_targets.erase(ctx->prospective_targets.begin(), ctx->prospective_targets.begin() + drop);
                        }
                        // effective target is newest queued target (latest mouse position)
                        float eff_tx = tx;
                        float eff_ty = ty;
                        if (!ctx->prospective_targets.empty()) {
                            eff_tx = ctx->prospective_targets.back().first;
                            eff_ty = ctx->prospective_targets.back().second;
                        }
                    // initialize prospect pos if needed
                    if (!ctx->prospective_initialized) {
                        ctx->prospective_x = static_cast<float>(sx);
                        ctx->prospective_y = static_cast<float>(sy);
                        ctx->prospective_vx = 0.0f;
                        ctx->prospective_vy = 0.0f;
                        ctx->prospective_initialized = true;
                        // reset time base
                        using namespace std::chrono;
                        ctx->relax_last_time = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
                    }
                    // Immediate snap: follow newest mouse position exactly for responsiveness
                    float &px = ctx->prospective_x;
                    float &py = ctx->prospective_y;
                    float &vx = ctx->prospective_vx;
                    float &vy = ctx->prospective_vy;
                    px = eff_tx;
                    py = eff_ty;
                    // zero motion state so future relax/springs start from rest
                    vx = 0.0f;
                    vy = 0.0f;
                    // update prospective rope endpoints in simulator so last vertex equals mouse immediately
                    int ipx = static_cast<int>(std::lround(px));
                    int ipy = static_cast<int>(std::lround(py));
                    if (!ctx->rope_sim) {
                        int max_ropes = 16;
                        int max_segs = std::max(4, ctx->st.cable_segments);
                        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
                        ctx->rope_sim_idx.clear();
                    }
                    // create a single persistent prospective rope if not present
                    if (ctx->prospective_rope_idx < 0) {
                        int segs = std::max(4, ctx->st.cable_segments);
                        float slack = ctx->prospective_rope_length > 0.0f ? ctx->prospective_rope_length : 0.0f;
                        ctx->prospective_rope_idx = rope_sim_add_rope(ctx->rope_sim, static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(ipx), static_cast<float>(ipy), segs, slack);
                    } else {
                        // move endpoints preserving previous positions so integrator receives velocity impulse
                        rope_sim_move_endpoints(ctx->rope_sim, ctx->prospective_rope_idx, static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(ipx), static_cast<float>(ipy));
                    }
                    // consume the newest queued target when close enough
                    float ddx = eff_tx - px;
                    float ddy = eff_ty - py;
                    float d2 = ddx*ddx + ddy*ddy;
                    if (d2 <= ctx->prospective_slack * ctx->prospective_slack) ctx->prospective_targets.clear();
                    // if no permanent edges will step the sim later, step now so prospective rope animates
                    if (ctx->edges.empty()) {
                        // freer whipping (lower damping) but stronger constraint solve so it settles quickly
                        float gravity = 800.0f;
                        int constraint_iters = 8;
                        float damping = 0.86f;
                        rope_sim_step(ctx->rope_sim, static_cast<float>(dt_frame), gravity, constraint_iters, damping);
                    }
                    // draw prospective rope from sim vertices
                    if (ctx->prospective_rope_idx >= 0) {
                        int vc = rope_sim_get_vertex_count(ctx->rope_sim, ctx->prospective_rope_idx);
                        if (vc >= 2) {
                            std::vector<float> verts(static_cast<size_t>(vc) * 2);
                            int got = rope_sim_get_vertices(ctx->rope_sim, ctx->prospective_rope_idx, verts.data(), static_cast<int>(verts.size()));
                            if (got > 0) {
                                Color pcol{180, 255, 180, 220};
                                // draw smooth curve along simulator vertices
                                int samples_per_segment = std::max(2, ctx->st.cable_segments / std::max(1, got - 1));
                                draw_rope_curve_blend(out_rgba, p_w, p_h, p_pitch, verts.data(), got, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, pcol, samples_per_segment);
                            }
                        }
                    }
                }
            }
        }
    }

    if (out_rgba && !ctx->edges.empty()) {
        int w_local = local_geom.width_px;
        int h_local = local_geom.height_px;
        int pitch_local = w_local * 4;
        Color edge_col{200, 200, 255, 200};

        // ensure rope simulator exists
        if (!ctx->rope_sim) {
            int max_ropes = std::max<int>(1024, static_cast<int>(ctx->edges.size()) + 16);
            int max_segs = std::max(4, ctx->st.cable_segments);
            ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
            ctx->rope_sim_idx.clear();
        }

        // ensure rope_sim_idx matches edge count
        while (ctx->rope_sim_idx.size() < ctx->edges.size()) ctx->rope_sim_idx.push_back(-1);

        // update endpoints in sim (and create ropes if missing)
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            const auto &e = ctx->edges[ei];
            uint64_t ka = e.first;
            uint64_t kb = e.second;
            int ax = -1, ay = -1, bx = -1, by = -1;
            auto compute_center_vis = [&](uint64_t key, int &outx, int &outy) {
                outx = -1; outy = -1;
                uint32_t r_orig = static_cast<uint32_t>(key >> 32);
                uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
                uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
                int vis_idx = -1;
                for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == static_cast<int>(r_orig)) { vis_idx = static_cast<int>(vi); break; }
                if (vis_idx < 0) return;
                const GP_TableRow &row = vis_rows[vis_idx];
                if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
                const GP_TableCell &cell = row.cells[c_idx];
                int x0 = col_x0[static_cast<int>(c_idx)];
                int cw = col_w[static_cast<int>(c_idx)];
                int y0 = vis_idx * ctx->st.row_h;
                int led_count = 9;
                if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                    int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                    if (count == 0) count = (cell.flags & 0xFF);
                    if (count == 0) count = 12;
                    led_count = count;
                } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                    led_count = 8;
                }
                int eff_w = std::max(1, cw - 4);
                int radius = 4;
                int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
                int cx0 = x0 + 2 + led_spacing;
                if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                    outx = cx0 + static_cast<int>(led) * led_spacing;
                    outy = y0 + ctx->st.row_h / 2;
                }
            };
            compute_center_vis(ka, ax, ay);
            compute_center_vis(kb, bx, by);
            if (ax < 0 || ay < 0 || bx < 0 || by < 0) continue;

            int rope_idx = ctx->rope_sim_idx[ei];
            if (rope_idx < 0) {
                int segs = std::max(4, ctx->st.cable_segments);
                float slack = 0.0f;
                int new_idx = rope_sim_add_rope(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack);
                ctx->rope_sim_idx[ei] = new_idx;
            } else {
                rope_sim_move_endpoints(ctx->rope_sim, rope_idx, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by));
            }
        }

        // Step the simulator for this frame: allow more whip (lower damping) but more constraint iterations to settle
        float gravity = 800.0f;
        int constraint_iters = 8;
        float damping = 0.86f;
        // fall back to a reasonable fixed step if frame dt isn't available in this scope
        float sim_dt = 1.0f / 60.0f;
        rope_sim_step(ctx->rope_sim, sim_dt, gravity, constraint_iters, damping);

        // Now render ropes from simulator vertices
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            const auto &e = ctx->edges[ei];
            float ev = 1.0f;
            if (ei < ctx->relax_value.size()) ev = ctx->relax_value[ei];
            Color col = edge_col;
            col.a = static_cast<uint8_t>(std::min<int>(255, static_cast<int>(col.a * ev)));
            int rope_idx = ctx->rope_sim_idx[ei];
            if (rope_idx < 0) continue;
            int vc = rope_sim_get_vertex_count(ctx->rope_sim, rope_idx);
            if (vc < 2) continue;
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = rope_sim_get_vertices(ctx->rope_sim, rope_idx, verts.data(), static_cast<int>(verts.size()));
            if (got <= 0) continue;
            // draw smooth spline curve along simulator vertices
            int samples_per_segment = std::max(2, ctx->st.cable_segments / std::max(1, got - 1));
            draw_rope_curve_blend(out_rgba, w_local, h_local, pitch_local, verts.data(), got, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, col, samples_per_segment);
        }
    }
    // Remap hitboxes' row indices from visible index -> original index
    if (hitboxes_out && hitboxes_cap > 0 && map_vis_to_orig.size() > 0 && hitboxes_written && *hitboxes_written > 0) {
        for (int hi = 0; hi < *hitboxes_written; ++hi) {
            int vis_idx = hitboxes_out[hi].row_idx;
            if (vis_idx >= 0 && vis_idx < static_cast<int>(map_vis_to_orig.size())) {
                hitboxes_out[hi].row_idx = map_vis_to_orig[vis_idx];
            }
        }
    }

    // Populate out_geom in original coordinate system (width/height same)
    if (out_geom) {
        *out_geom = local_geom;
    }

    return 1;
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

// C-linkage wrappers so other translation units can use the exact spline drawer
extern "C" void table_draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca, int samples_per_segment) {
    Color core_col{cr, cg, cb, ca};
    draw_rope_curve_blend(img, w, h, pitch, verts, count, jacket_px, jacket_border, core_col, samples_per_segment);
}

extern "C" void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity) {
    draw_rope_curve_blend_colored(img, w, h, pitch, verts, count, jacket_px, jacket_border, hues, hue_count, samples_per_segment, hue_intensity);
}
