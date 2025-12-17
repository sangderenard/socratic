#include "menu_waveform_abi.h"
#include "menu_waveform.hpp"

#include <new>

struct GP_MenuWaveform {
    gp_menu::MenuWaveform impl;
    GP_MenuWaveform(int w, int h, int n) : impl(w, h, n) {}
};

extern "C" {

GP_MenuWaveform* gp_menu_waveform_create(int32_t width_px, int32_t height_px, int32_t history_len) {
    try {
        auto* p = new GP_MenuWaveform(int(width_px), int(height_px), int(history_len));
        return p;
    } catch (...) {
        return nullptr;
    }
}

void gp_menu_waveform_destroy(GP_MenuWaveform* wf) {
    delete wf;
}

void gp_menu_waveform_push_sample(GP_MenuWaveform* wf, uint64_t t_ns, float v) {
    if (!wf) return;
    wf->impl.push(std::uint64_t(t_ns), float(v));
}

int32_t gp_menu_waveform_raster_rgba(GP_MenuWaveform* wf, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!wf) return 0;
    return wf->impl.raster_rgba(out_rgba, int(out_len_bytes)) ? 1 : 0;
}

void gp_menu_waveform_clear(GP_MenuWaveform* wf) {
    if (!wf) return;
    wf->impl.clear();
}

} // extern "C"
