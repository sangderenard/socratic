#include "canvas_abi.h"
#include "table_abi.h"

#include <cstdint>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ModulePlacement {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

GP_TableStyle make_style_for_width(int w_px) {
    GP_TableStyle s{};
    s.width_px = w_px;
    s.row_h_px = 20;
    s.indent_px = 14;
    s.expand_w_px = 12;
    s.name_w_px = 0; // keep LED columns aligned to the module box
    auto set = [](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
        dst[3] = a;
    };
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

GP_TableContext* make_three_led_table() {
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return nullptr;

    GP_TableStyle style = make_style_for_width(160);
    gp_table_set_style(t, &style);

    GP_TableColumn cols[2];
    cols[0].kind = GP_TABLE_CELL_LEDS_ARG;
    cols[0].width_px = 64;
    cols[0].align = 0;
    cols[1].kind = GP_TABLE_CELL_LEDS_ARG;
    cols[1].width_px = 64;
    cols[1].align = 0;
    gp_table_set_columns(t, cols, 2);

    GP_TableRow row{};
    row.kind = GP_TABLE_ROW_DEVICE;
    row.depth = 0;
    row.expanded = 1;
    row.selected = 0;
    row.cell_count = 2;
    const uint32_t on_mask = (1u << 3) - 1u; // three LEDs on each side
    for (int ci = 0; ci < 2; ++ci) {
        row.cells[ci].kind = GP_TABLE_CELL_LEDS_ARG;
        row.cells[ci].value = 3.0f;   // three LEDs
        row.cells[ci].flags = on_mask;
        row.cells[ci].reserved0 = static_cast<int32_t>(on_mask); // required mask
    }
    gp_table_set_rows(t, &row, 1);
    return t;
}

std::optional<std::pair<int, int>> find_led_center(GP_TableContext* t, bool left_column, int led_idx, int w, int h) {
    if (!t) return std::nullopt;
    GP_TableGeom geom{};
    gp_table_get_geom(t, &geom);
    geom.width_px = w;
    geom.height_px = h;

    std::vector<uint8_t> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    const int cap = 256;
    std::vector<GP_TableHitBox> hits(cap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(t, nullptr, rgba.data(), static_cast<int32_t>(rgba.size()), &geom, hits.data(), cap, &written);
    if (!ok || written <= 0) return std::nullopt;
    for (int i = 0; i < written; ++i) {
        const auto& hb = hits[i];
        if (hb.part == GP_TABLE_HIT_LED_ARG && hb.aux0 == led_idx) {
            const bool is_left_col = (hb.col_idx == 0);
            if (is_left_col != left_column) continue;
            int cx = (hb.x0 + hb.x1) / 2;
            int cy = (hb.y0 + hb.y1) / 2;
            return std::make_pair(cx, cy);
        }
    }
    return std::nullopt;
}

int canvas_button_x_center(int index) {
    // Matches gp_canvas_on_click button layout: bx=8, bw=20, spacing=12.
    const int bx = 8;
    const int bw = 20;
    const int spacing = 12;
    return bx + index * (bw + spacing) + bw / 2;
}

int control_bar_y_center() {
    // control_bar_h=28 => by=4, bh=20
    return 4 + 20 / 2;
}

int count_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int w, int h, int x0, int y0, int x1, int y1) {
    if (a.size() != b.size()) return -1;
    int x_start = std::max(0, std::min(x0, x1));
    int x_end = std::min(w, std::max(x0, x1));
    int y_start = std::max(0, std::min(y0, y1));
    int y_end = std::min(h, std::max(y0, y1));
    int count = 0;
    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            size_t idx = static_cast<size_t>(y * w + x) * 4;
            if (idx + 3 < a.size()) {
                if (a[idx + 0] != b[idx + 0] || a[idx + 1] != b[idx + 1] || a[idx + 2] != b[idx + 2] || a[idx + 3] != b[idx + 3]) {
                    ++count;
                }
            }
        }
    }
    return count;
}

std::optional<GP_TableHitBox> table_hit_at_point(GP_TableContext* t, int w, int h, int lx, int ly) {
    if (!t || w <= 0 || h <= 0) return std::nullopt;
    GP_TableGeom geom{};
    gp_table_get_geom(t, &geom);
    geom.width_px = w;
    geom.height_px = h;
    std::vector<uint8_t> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    const int cap = 512;
    std::vector<GP_TableHitBox> hits(cap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(t, nullptr, rgba.data(), static_cast<int32_t>(rgba.size()), &geom, hits.data(), cap, &written);
    if (!ok || written <= 0) return std::nullopt;
    for (int i = 0; i < written; ++i) {
        const auto& hb = hits[i];
        if (lx >= hb.x0 && lx < hb.x1 && ly >= hb.y0 && ly < hb.y1) return hb;
    }
    return std::nullopt;
}

bool is_led_part(int part) {
    return part == GP_TABLE_HIT_LED || part == GP_TABLE_HIT_LED_ARG || part == GP_TABLE_HIT_LED_TABLE;
}

struct LedHit {
    int idx = -1;
    int cx = 0;
    int cy = 0;
    int aux1 = 0;
    int part = 0;
    int col = 0;
};

std::vector<LedHit> list_led_hits(GP_TableContext* t, bool left_column, int w, int h) {
    std::vector<LedHit> hits_out;
    if (!t || w <= 0 || h <= 0) return hits_out;
    GP_TableGeom geom{};
    gp_table_get_geom(t, &geom);
    geom.width_px = w;
    geom.height_px = h;
    std::vector<uint8_t> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    const int cap = 512;
    std::vector<GP_TableHitBox> hits(cap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(t, nullptr, rgba.data(), static_cast<int32_t>(rgba.size()), &geom, hits.data(), cap, &written);
    if (!ok || written <= 0) return hits_out;
    for (int i = 0; i < written; ++i) {
        const auto& hb = hits[i];
        if (!is_led_part(hb.part)) continue;
        bool is_left = (hb.col_idx == 0);
        if (is_left != left_column) continue;
        int cx = (hb.x0 + hb.x1) / 2;
        int cy = (hb.y0 + hb.y1) / 2;
        hits_out.push_back(LedHit{hb.aux0, cx, cy, hb.aux1, hb.part, hb.col_idx});
    }
    return hits_out;
}

struct IOButtons {
    int plus_out_x = 0;
    int plus_in_x = 0;
    int y = 0;
};

IOButtons compute_io_buttons(int canvas_width) {
    const int control_bar_h = 28;
    const int bw = 20;
    const int spacing = 12;
    const int nbw = bw;
    const int num_w = 40; // max(24, nbw*2)
    const int gap = 10;
    const int table_btn_count = 3;
    const int group_width = table_btn_count * (bw + spacing) - spacing; // 84
    const int bx_r = std::max(8, canvas_width - 8 - group_width);
    const int io_base_x = bx_r;
    const int io_shift = (nbw + spacing + 80);
    const int bx_minus_out = io_base_x - io_shift - (nbw + gap + num_w + gap + nbw);
    const int bx_num_out = bx_minus_out + nbw + gap;
    const int bx_plus_out = bx_num_out + num_w + gap;
    const int bx_minus_in = io_base_x - (nbw + gap + num_w + gap + nbw);
    const int bx_num_in = bx_minus_in + nbw + gap;
    const int bx_plus_in = bx_num_in + num_w + gap;
    (void)control_bar_h;
    IOButtons b;
    b.plus_out_x = bx_plus_out + nbw / 2;
    b.plus_in_x = bx_plus_in + nbw / 2;
    b.y = control_bar_y_center();
    return b;
}

} // namespace

int main() {
    const int canvas_w = 800;
    const int canvas_h = 600;
    GP_CanvasContext* canvas = gp_canvas_create(canvas_w, canvas_h);
    if (!canvas) {
        std::cerr << "Failed to create canvas\n";
        return 1;
    }

    // Activate the "new table" canvas tool (index 1).
    gp_canvas_on_click(canvas, canvas_button_x_center(1), control_bar_y_center());

    // Spawn two modules via click-to-create tool.
    gp_canvas_on_click(canvas, 150, 150);
    gp_canvas_on_click(canvas, 450, 150);
    // Toggle tool off to avoid accidental extra modules.
    gp_canvas_on_click(canvas, canvas_button_x_center(1), control_bar_y_center());

    // Known placements.
    std::vector<ModulePlacement> modules = {
        {100, 100, 160, 120},
        {400, 100, 160, 120},
    };
    gp_canvas_move_module(canvas, 0, modules[0].x, modules[0].y);
    gp_canvas_move_module(canvas, 1, modules[1].x, modules[1].y);

    // Replace auto-created tables with deterministic 3-LED tables.
    std::vector<GP_TableContext*> attached_tables;
    for (int mi = 0; mi < 2; ++mi) {
        gp_canvas_destroy_table(canvas, mi);
        GP_TableContext* t = make_three_led_table();
        if (!t) {
            std::cerr << "Failed to create table " << mi << "\n";
            return 1;
        }
        gp_canvas_attach_table(canvas, mi, t, 1);
        attached_tables.push_back(t);
        auto pre_left = list_led_hits(t, true, modules[mi].w, modules[mi].h);
        auto pre_right = list_led_hits(t, false, modules[mi].w, modules[mi].h);
        std::cerr << "Module " << mi << " initial left hits: ";
        for (const auto& h : pre_left) std::cerr << h.idx << "{" << h.aux1 << "," << h.part << "," << h.col << "}@(" << h.cx << "," << h.cy << ") ";
        std::cerr << " right hits: ";
        for (const auto& h : pre_right) std::cerr << h.idx << "{" << h.aux1 << "," << h.part << "," << h.col << "}@(" << h.cx << "," << h.cy << ") ";
        std::cerr << "\n";
    }

    // Bump IO counters to 3 via synthetic control-bar clicks for each module.
    IOButtons io_btns = compute_io_buttons(canvas_w);
    for (int mi = 0; mi < 2; ++mi) {
        // Focus module.
        const auto& m = modules[mi];
        gp_canvas_on_click(canvas, m.x + m.w / 2, m.y + m.h / 2);
        for (int k = 0; k < 3; ++k) {
            gp_canvas_on_click(canvas, io_btns.plus_out_x, io_btns.y);
            gp_canvas_on_click(canvas, io_btns.plus_in_x, io_btns.y);
        }
    }

    std::vector<uint8_t> rgba_before_edges(static_cast<size_t>(canvas_w) * canvas_h * 4);
    if (!gp_canvas_raster_rgba(canvas, rgba_before_edges.data(), static_cast<int32_t>(rgba_before_edges.size()))) {
        std::cerr << "Failed to raster canvas before edge creation\n";
        return 1;
    }

    // Locate LED centers (layout is deterministic) and click through the canvas.
    for (int led = 0; led < 3; ++led) {
        auto out_local = find_led_center(attached_tables[0], false, led, modules[0].w, modules[0].h);
        auto in_local = find_led_center(attached_tables[1], true, led, modules[1].w, modules[1].h);
        if (!out_local || !in_local) {
            std::cerr << "Failed to locate LED " << led << " (out=" << static_cast<bool>(out_local) << ", in=" << static_cast<bool>(in_local) << ")\n";
            auto outs = list_led_hits(attached_tables[0], false, modules[0].w, modules[0].h);
            auto ins = list_led_hits(attached_tables[1], true, modules[1].w, modules[1].h);
            std::cerr << "Module 0 right hits: ";
            for (const auto& h : outs) std::cerr << h.idx << "{" << h.aux1 << "," << h.part << "," << h.col << "}@(" << h.cx << "," << h.cy << ") ";
            std::cerr << "\nModule 1 left hits: ";
            for (const auto& h : ins) std::cerr << h.idx << "{" << h.aux1 << "," << h.part << "," << h.col << "}@(" << h.cx << "," << h.cy << ") ";
            std::cerr << "\n";
            return 1;
        }
        int out_x = modules[0].x + out_local->first;
        int out_y = modules[0].y + out_local->second;
        int in_x = modules[1].x + in_local->first;
        int in_y = modules[1].y + in_local->second;
        auto out_hit = table_hit_at_point(attached_tables[0], modules[0].w, modules[0].h, out_local->first, out_local->second);
        auto in_hit = table_hit_at_point(attached_tables[1], modules[1].w, modules[1].h, in_local->first, in_local->second);
        if (!out_hit || !is_led_part(out_hit->part)) {
            std::cerr << "Module 0 LED " << led << " not hittable at (" << out_local->first << "," << out_local->second << ")\n";
            return 1;
        }
        if (!in_hit || !is_led_part(in_hit->part)) {
            std::cerr << "Module 1 LED " << led << " not hittable at (" << in_local->first << "," << in_local->second << ")\n";
            return 1;
        }
        std::cerr << "Connecting LED " << led << " from (" << out_x << "," << out_y << ") to (" << in_x << "," << in_y << ")\n";

        gp_canvas_on_click(canvas, out_x, out_y);
        gp_canvas_on_click(canvas, in_x, in_y);
    }

    // Persist to a temp file and verify edge creation.
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / "canvas_table_headless_test.txt";
    gp_canvas_save_to_file(canvas, tmp.string().c_str());

    std::ifstream ifs(tmp);
    if (!ifs.good()) {
        std::cerr << "Failed to read canvas dump\n";
        return 1;
    }
    std::vector<std::array<int, 4>> edges;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("EDGE", 0) == 0) {
            std::istringstream ss(line);
            std::string tag;
            std::array<int, 4> e{};
            int type = 0;
            ss >> tag >> e[0] >> e[1] >> e[2] >> e[3] >> type;
            edges.push_back(e);
        }
    }
    if (edges.size() != 3) {
        std::cerr << "Expected 3 edges, found " << edges.size() << "\n";
        return 1;
    }
    bool have_idx[3] = {false, false, false};
    for (const auto& e : edges) {
        if (!((e[0] == 0 && e[2] == 1) || (e[0] == 1 && e[2] == 0))) {
            std::cerr << "Edge modules not connecting 0 and 1\n";
            return 1;
        }
        for (int i = 0; i < 3; ++i) {
            if (e[1] == i && e[3] == i) have_idx[i] = true;
        }
    }
    if (!have_idx[0] || !have_idx[1] || !have_idx[2]) {
        std::cerr << "Did not form edges for contact indices 0,1,2\n";
        return 1;
    }

    // Rasterize and ensure edges appear visually between modules.
    std::vector<uint8_t> rgba(static_cast<size_t>(canvas_w) * canvas_h * 4);
    if (!gp_canvas_raster_rgba(canvas, rgba.data(), static_cast<int32_t>(rgba.size()))) {
        std::cerr << "Failed to raster canvas\n";
        return 1;
    }
    int edge_roi_diff = count_diff(
        rgba_before_edges,
        rgba,
        canvas_w,
        canvas_h,
        modules[0].x + modules[0].w,
        modules[0].y,
        modules[1].x,
        modules[0].y + modules[0].h);
    if (edge_roi_diff <= 0) {
        std::cerr << "Edge pixels not found between modules (diff count " << edge_roi_diff << ")\n";
        return 1;
    }
    std::vector<uint8_t> rgba_after_edges = rgba;

    // Create a prospective edge and verify it animates/responds to mouse moves.
    auto first_out = find_led_center(attached_tables[0], false, 0, modules[0].w, modules[0].h);
    if (!first_out) {
        std::cerr << "Failed to find prospective start LED\n";
        return 1;
    }
    int start_x = modules[0].x + first_out->first;
    int start_y = modules[0].y + first_out->second;
    gp_canvas_on_click(canvas, start_x, start_y);
    // Move mouse to two distinct targets to force rope endpoint updates.
    int tgt1_x = modules[0].x + modules[0].w + 20;
    int tgt1_y = modules[0].y + 220;
    int tgt2_x = modules[0].x + modules[0].w + 90;
    int tgt2_y = modules[0].y + 260;
    gp_canvas_on_mouse_move(canvas, tgt1_x, tgt1_y);
    if (!gp_canvas_raster_rgba(canvas, rgba.data(), static_cast<int32_t>(rgba.size()))) {
        std::cerr << "Failed to raster canvas after prospective move 1\n";
        return 1;
    }
    std::vector<uint8_t> rope_frame_move1 = rgba;
    int rope_at_tgt1 = count_diff(rgba_after_edges, rope_frame_move1, canvas_w, canvas_h, tgt1_x - 20, tgt1_y - 20, tgt1_x + 20, tgt1_y + 20);
    gp_canvas_on_mouse_move(canvas, tgt2_x, tgt2_y);
    if (!gp_canvas_raster_rgba(canvas, rgba.data(), static_cast<int32_t>(rgba.size()))) {
        std::cerr << "Failed to raster canvas after prospective move 2\n";
        return 1;
    }
    std::vector<uint8_t> rope_frame_move2 = rgba;
    int rope_at_tgt2 = count_diff(rgba_after_edges, rope_frame_move2, canvas_w, canvas_h, tgt2_x - 20, tgt2_y - 20, tgt2_x + 20, tgt2_y + 20);
    int rope_residual_at_tgt1 = count_diff(rgba_after_edges, rope_frame_move2, canvas_w, canvas_h, tgt1_x - 20, tgt1_y - 20, tgt1_x + 20, tgt1_y + 20);
    int rope_shift_at_tgt1 = count_diff(rope_frame_move1, rope_frame_move2, canvas_w, canvas_h, tgt1_x - 20, tgt1_y - 20, tgt1_x + 20, tgt1_y + 20);
    if (rope_at_tgt1 <= 0 || rope_at_tgt2 <= 0) {
        std::cerr << "Prospective rope pixels not found at expected targets (t1=" << rope_at_tgt1 << ", t2=" << rope_at_tgt2 << ")\n";
        return 1;
    }
    if (rope_shift_at_tgt1 <= 0 || rope_residual_at_tgt1 * 2 >= rope_at_tgt1) {
        std::cerr << "Prospective rope did not relocate after mouse move (shift=" << rope_shift_at_tgt1 << ", residual=" << rope_residual_at_tgt1 << ", initial=" << rope_at_tgt1 << ")\n";
        return 1;
    }
    if (rope_at_tgt1 == rope_at_tgt2) {
        std::cerr << "Prospective rope did not move between targets (counts equal)\n";
        return 1;
    }

    gp_canvas_destroy(canvas);
    std::cout << "Headless canvas/table wiring test passed\n";
    return 0;
}
