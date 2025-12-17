#pragma once

// Minimal C ABI for a pixel waveform "signal window".
//
// This module is intentionally not a renderer.
// It only owns a ring buffer of samples and can rasterize a tiny waveform into an RGBA8 pixel buffer.
//
// Integration idea (future): controller engine calls gp_menu_waveform_push_sample() once per tick.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle.
typedef struct GP_MenuWaveform GP_MenuWaveform;

// Create a waveform ring and raster target.
// width_px/height_px define the pixel window size.
// history_len is the number of samples stored (typically >= width_px).
GP_MenuWaveform* gp_menu_waveform_create(int32_t width_px, int32_t height_px, int32_t history_len);

void gp_menu_waveform_destroy(GP_MenuWaveform* wf);

// Push one raw sample (expected in [-1, +1] but not enforced).
// t_ns is optional and may be 0.
void gp_menu_waveform_push_sample(GP_MenuWaveform* wf, uint64_t t_ns, float v);

// Rasterize current waveform into RGBA8.
// out_rgba must point to at least width_px*height_px*4 bytes.
// Returns 1 on success.
int32_t gp_menu_waveform_raster_rgba(
    GP_MenuWaveform* wf,
    uint8_t* out_rgba,
    int32_t out_len_bytes
);

// Optional: clear history to zeros.
void gp_menu_waveform_clear(GP_MenuWaveform* wf);

#ifdef __cplusplus
}
#endif
