#pragma once

// Lightweight C++ helpers for building table textures as structured objects.
// These wrap the C ABI in table_abi.h so callers can assemble columns/rows/cells
// using containers instead of raw C structs. Rendering still flows through
// gp_table_raster_rgba; this layer only shapes the data and owns the buffers.

#include "table_abi.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace gp {

struct RenderedTable {
    int width_px = 0;
    int height_px = 0;
    GP_TableGeom geom{};
    std::vector<uint8_t> rgba; // RGBA8 surface, size = width*height*4
    bool ok() const { return width_px > 0 && height_px > 0 && rgba.size() == static_cast<std::size_t>(width_px * height_px * 4); }
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

    static TableCell axis(float value, float seen_min, float seen_max) {
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
        if (cells.size() < 8) {
            cells.push_back(c);
        }
        return *this;
    }

    GP_TableRow finalize() const {
        GP_TableRow out = row;
        const int n = static_cast<int>(std::min<std::size_t>(cells.size(), 8));
        for (int i = 0; i < n; ++i) {
            out.cells[i] = cells[static_cast<std::size_t>(i)].cell;
        }
        out.cell_count = n;
        return out;
    }
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

        // First compute size.
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
