#pragma once

// C++ container helpers for workbench_rows_abi.
// Purpose: let callers assemble rows/styles as objects, then raster via gp_wb_rows_raster_rgba
// without hand-populating C structs. Mirrors legacy workbench data: device headers/notes,
// axes (value + flags), buttons (flags + timers), optional waveforms, selection.

#include "workbench_rows_abi.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace gp {

struct RenderedWbRows {
    int width_px = 0;
    int height_px = 0;
    std::vector<uint8_t> rgba; // size = width*height*4
    bool ok() const { return width_px > 0 && height_px > 0 && rgba.size() == static_cast<std::size_t>(width_px * height_px * 4); }
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
        auto set = [&](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
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
