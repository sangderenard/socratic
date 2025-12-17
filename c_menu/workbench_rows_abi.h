#pragma once

// C ABI for rasterizing legacy-style workbench rows (LEDs + axis bar + optional waveform).
//
// The intent is to mirror the legacy Python/GL dense tables but have the pixel work done in C++
// so Python can hand off state arrays and blit the resulting RGBA buffer.
//
// NOTE: This is a minimal first cut; text is not rendered. Callers may overlay text separately
// using existing UiDoc text primitives.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration from menu_waveform_abi.h
struct GP_MenuWaveform;

// Row kind mirrors the legacy semantics.
typedef enum GP_WbRowKind {
    GP_WB_ROW_DEVICE_HEADER = 0,
    GP_WB_ROW_DEVICE_NOTE = 1,
    GP_WB_ROW_AXIS = 2,
    GP_WB_ROW_BUTTON = 3,
} GP_WbRowKind;

// One row worth of state (no dynamic allocation inside the struct).
typedef struct GP_WbRow {
    int32_t kind;          // GP_WbRowKind
    char label[64];        // optional ASCII label (not rendered yet; kept for future)
    float value;           // axis value in [-1, +1]
    uint32_t flags;        // bitmask for LED columns (caller provides bits)
    float hold_s;          // button hold seconds (timer stripe)
    float last_s;          // last hold seconds (timer stripe)
    int32_t selected;      // highlight row if non-zero
    int32_t reserved0;
    const struct GP_MenuWaveform* wave; // optional waveform handle (can be NULL)
    int32_t reserved1;
} GP_WbRow;

// Styling and layout hints. All colors are RGBA8.
typedef struct GP_WbStyle {
    int32_t width_px;       // total row width
    int32_t row_h_px;       // per-row height
    int32_t name_w_px;      // width of the name/label gutter
    int32_t led_spacing_px; // center-to-center LED spacing
    int32_t led_radius_px;  // LED circle radius
    int32_t axis_w_px;      // width of the axis bar
    int32_t wave_w_px;      // width of the optional waveform slot (0 to skip)
    int32_t wave_h_px;      // height of the waveform slot (<= row_h_px)

    uint8_t bg_rgba[4];          // default row background
    uint8_t bg_sel_rgba[4];      // selected row background
    uint8_t hdr_rgba[4];         // device header background
    uint8_t led_on_rgba[4];      // LED on color
    uint8_t led_off_rgba[4];     // LED off color
    uint8_t led_edge_rgba[4];    // LED edge color (if flag matches led_edge_mask)
    uint8_t axis_bg_rgba[4];     // axis bar background
    uint8_t axis_tick_rgba[4];   // axis ticks color
    uint8_t axis_val_rgba[4];    // axis value marker color
    uint8_t timer_rgba[4];       // timer stripe color
    uint8_t wave_bg_rgba[4];     // waveform background fill
    uint8_t wave_fg_rgba[4];     // waveform foreground color

    uint32_t led_mask[9];        // bitmask per LED column (len=9; unused entries may be 0)
    uint32_t led_edge_mask;      // bits considered "edge" for edge coloring
} GP_WbStyle;

// Calculate output surface dimensions for a given row count + style.
// Returns 1 on success, 0 on failure.
int32_t gp_wb_rows_calc_size(const GP_WbStyle* style, int32_t row_count, int32_t* out_w, int32_t* out_h);

// Rasterize rows into an RGBA8 buffer. The buffer must be at least width*height*4 bytes,
// where width/height come from gp_wb_rows_calc_size.
// Returns 1 on success, 0 on failure.
int32_t gp_wb_rows_raster_rgba(
    const GP_WbRow* rows,
    int32_t row_count,
    const GP_WbStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes
);

#ifdef __cplusplus
}
#endif
