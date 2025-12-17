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

int count_nonzero(const std::vector<uint8_t>& buf, int w, int h, int x0, int y0, int x1, int y1) {
    int x_start = std::max(0, std::min(x0, x1));
    int x_end = std::min(w, std::max(x0, x1));
    int y_start = std::max(0, std::min(y0, y1));
    int y_end = std::min(h, std::max(y0, y1));
    int count = 0;
    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            size_t idx = static_cast<size_t>(y * w + x) * 4;
            if (idx + 3 < buf.size()) {
                if (buf[idx + 0] || buf[idx + 1] || buf[idx + 2] || buf[idx + 3]) ++count;
            }
        }
    }
    return count;
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
    GP_CanvasContext* canvas = gp_canvas_create(800, 600);
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
    for (int mi = 0; mi < 2; ++mi) {
        gp_canvas_destroy_table(canvas, mi);
        GP_TableContext* t = make_three_led_table();
        if (!t) {
            std::cerr << "Failed to create table " << mi << "\n";
            return 1;
        }
        gp_canvas_attach_table(canvas, mi, t, 1);
    }

    // Bump IO counters to 3 via synthetic control-bar clicks for each module.
    IOButtons io_btns = compute_io_buttons(800);
    for (int mi = 0; mi < 2; ++mi) {
        // Focus module.
        const auto& m = modules[mi];
        gp_canvas_on_click(canvas, m.x + m.w / 2, m.y + m.h / 2);
        for (int k = 0; k < 3; ++k) {
            gp_canvas_on_click(canvas, io_btns.plus_out_x, io_btns.y);
            gp_canvas_on_click(canvas, io_btns.plus_in_x, io_btns.y);
        }
    }

    // Locate LED centers (layout is deterministic) and click through the canvas.
    GP_TableContext* table_layout = make_three_led_table();
    if (!table_layout) {
        std::cerr << "Failed to create layout table\n";
        return 1;
    }
    for (int led = 0; led < 3; ++led) {
        auto out_local = find_led_center(table_layout, false, led, modules[0].w, modules[0].h);
        auto in_local = find_led_center(table_layout, true, led, modules[1].w, modules[1].h);
        if (!out_local || !in_local) {
            std::cerr << "Failed to locate LED " << led << "\n";
            gp_table_destroy(table_layout);
            return 1;
        }
        int out_x = modules[0].x + out_local->first;
        int out_y = modules[0].y + out_local->second;
        int in_x = modules[1].x + in_local->first;
        int in_y = modules[1].y + in_local->second;

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
    const int w = 800, h = 600;
    std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
    if (!gp_canvas_raster_rgba(canvas, rgba.data(), static_cast<int32_t>(rgba.size()))) {
        std::cerr << "Failed to raster canvas\n";
        return 1;
    }
    int edge_roi = count_nonzero(
        rgba,
        w,
        h,
        modules[0].x + modules[0].w,
        modules[0].y,
        modules[1].x,
        modules[0].y + modules[0].h);
    if (edge_roi <= 0) {
        std::cerr << "Edge pixels not found between modules\n";
        return 1;
    }

    // Create a prospective edge and verify it animates/responds to mouse moves.
    auto first_out = find_led_center(table_layout, false, 0, modules[0].w, modules[0].h);
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
    int rope_at_tgt1 = count_nonzero(rgba, w, h, tgt1_x - 20, tgt1_y - 20, tgt1_x + 20, tgt1_y + 20);
    gp_canvas_on_mouse_move(canvas, tgt2_x, tgt2_y);
    if (!gp_canvas_raster_rgba(canvas, rgba.data(), static_cast<int32_t>(rgba.size()))) {
        std::cerr << "Failed to raster canvas after prospective move 2\n";
        return 1;
    }
    int rope_at_tgt2 = count_nonzero(rgba, w, h, tgt2_x - 20, tgt2_y - 20, tgt2_x + 20, tgt2_y + 20);
    if (rope_at_tgt1 <= 0 || rope_at_tgt2 <= 0) {
        std::cerr << "Prospective rope pixels not found at expected targets\n";
        return 1;
    }
    if (rope_at_tgt1 == rope_at_tgt2) {
        std::cerr << "Prospective rope did not move between targets\n";
        return 1;
    }

    gp_table_destroy(table_layout);

    gp_canvas_destroy(canvas);
    std::cout << "Headless canvas/table wiring test passed\n";
    return 0;
}
