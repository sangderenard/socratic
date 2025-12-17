#pragma once

// Generic C ABI for rasterizing dense tables with hierarchical rows, LEDs, axis bars,
// timers, optional waveforms, and simple expand/collapse affordances.
// Text is carried through the API (label + per-cell text) but this minimal raster
// does not render text; callers can overlay text separately using their text system.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GP_MenuWaveform; // from menu_waveform_abi.h

// Cell kinds
typedef enum GP_TableCellKind {
    GP_TABLE_CELL_TEXT = 0,
    GP_TABLE_CELL_LEDS = 1,
    GP_TABLE_CELL_AXIS = 2,
    GP_TABLE_CELL_TIMERS = 3,
    GP_TABLE_CELL_WAVE = 4,
    GP_TABLE_CELL_CALIB = 5, // compact calibration strip (INV/TRIM/CAP/DED/RST)
    GP_TABLE_CELL_LEDS_ARG = 6,    // signal arg LEDs (required vs linked)
    GP_TABLE_CELL_LEDS_TABLE = 7,  // stacked LED strips encoded inside one cell
    GP_TABLE_CELL_SCROLL = 8,      // vertical scrollbar (arrows + track + thumb)
} GP_TableCellKind;

// Hitbox parts for interactive regions.
typedef enum GP_TableHitPart {
    GP_TABLE_HIT_CELL = 0,
    GP_TABLE_HIT_EXPAND = 1,
    GP_TABLE_HIT_LED = 2,
    GP_TABLE_HIT_AXIS = 3,
    GP_TABLE_HIT_CALIB_SLOT = 4,
    GP_TABLE_HIT_SCROLL_UP = 5,
    GP_TABLE_HIT_SCROLL_DOWN = 6,
    GP_TABLE_HIT_SCROLL_THUMB = 7,
    GP_TABLE_HIT_LED_TABLE = 8,
    GP_TABLE_HIT_LED_ARG = 9,
} GP_TableHitPart;

// Hitbox emitted per interactive sub-element (pixel coords in table-local space).
typedef struct GP_TableHitBox {
    int32_t x0, y0, x1, y1; // inclusive-exclusive rectangle
    int32_t row_idx;        // row in input array
    int32_t col_idx;        // column in input array
    int32_t cell_kind;      // GP_TableCellKind
    int32_t part;           // GP_TableHitPart
    int32_t aux0;           // e.g., led index, slot index
    int32_t aux1;           // spare
    uint32_t flags;         // optional
} GP_TableHitBox;

// Row kinds
typedef enum GP_TableRowKind {
    GP_TABLE_ROW_HEADER = 0,
    GP_TABLE_ROW_DEVICE = 1,
    GP_TABLE_ROW_AXIS = 2,
    GP_TABLE_ROW_BUTTON = 3,
    GP_TABLE_ROW_NOTE = 4,
} GP_TableRowKind;

// Column meta
typedef struct GP_TableColumn {
    int32_t kind;      // GP_TableCellKind
    int32_t width_px;  // fixed width; 0 => auto (not implemented, treated as fixed)
    int32_t align;     // 0=left,1=center,2=right (used for text when rendered elsewhere)
} GP_TableColumn;

// One cell
typedef struct GP_TableCell {
    int32_t kind;          // GP_TableCellKind
    char text[96];         // for TEXT; AXIS: optional "cap_min cap_max trim deadzone"
    uint32_t flags;        // for LEDS; AXIS: bit0/bit1 mark seen_min/seen_max
    float value;           // for AXIS (current)
    float hold_s;          // for TIMERS; AXIS: observed min
    float last_s;          // for TIMERS; AXIS: observed max
    const struct GP_MenuWaveform* wave; // for WAVE (nullable)
    int32_t reserved0;
} GP_TableCell;

// One row
typedef struct GP_TableRow {
    int32_t kind;          // GP_TableRowKind
    int32_t depth;         // indentation level (0=root)
    int32_t expanded;      // non-zero => expanded; renderer draws +/- box
    int32_t selected;      // highlight row if non-zero
    char label[96];        // primary label (not rendered in this minimal raster)
    GP_TableCell cells[8]; // fixed cell array
    int32_t cell_count;    // how many cells are valid
    int32_t reserved0;
} GP_TableRow;

// Style
typedef struct GP_TableStyle {
    int32_t width_px;
    int32_t row_h_px;
    int32_t indent_px;
    int32_t expand_w_px;
    int32_t name_w_px;     // label gutter width (used for +/- and potential text overlay)

    uint8_t bg_rgba[4];
    uint8_t bg_sel_rgba[4];
    uint8_t hdr_rgba[4];
    uint8_t text_rgba[4];
    uint8_t text_hdr_rgba[4];
    uint8_t led_on_rgba[4];
    uint8_t led_off_rgba[4];
    uint8_t led_edge_rgba[4];
    uint8_t axis_bg_rgba[4];
    uint8_t axis_tick_rgba[4];
    uint8_t axis_val_rgba[4];
    uint8_t timer_rgba[4];
    uint8_t wave_bg_rgba[4];
    uint8_t wave_fg_rgba[4];

    uint32_t led_mask[9];
    uint32_t led_edge_mask;
} GP_TableStyle;

// Geometry returned to the caller.
typedef struct GP_TableGeom {
    int32_t width_px;
    int32_t height_px;
    int32_t col_x0[8];
    int32_t col_w[8];
} GP_TableGeom;

// Opaque stateful table context. Holds style/columns/rows so callers can update incrementally
// and render repeatedly without repassing everything.
typedef struct GP_TableContext GP_TableContext;

// Calculate output size; returns 1 on success.
int32_t gp_table_calc_size(const GP_TableStyle* style, int32_t row_count, int32_t col_count, GP_TableGeom* out_geom);

// Rasterize rows into RGBA8 buffer; buffer must be >= width*height*4 bytes as reported by calc_size.
// Returns 1 on success.
int32_t gp_table_raster_rgba(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom);

// Rasterize and optionally emit hitboxes; hitboxes_out may be NULL to skip.
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
    int32_t* hitboxes_written);

// Stateful context helpers --------------------------------------------------

// Create/destroy a table context. Style may be NULL to use defaults.
GP_TableContext* gp_table_create(const GP_TableStyle* style);
void gp_table_destroy(GP_TableContext* ctx);

// Update style/columns/rows on an existing context. Any call will recompute
// geometry. Passing NULL for style leaves it unchanged. Returns 1 on success.
int32_t gp_table_set_style(GP_TableContext* ctx, const GP_TableStyle* style);
int32_t gp_table_set_columns(GP_TableContext* ctx, const GP_TableColumn* cols, int32_t col_count);
int32_t gp_table_set_rows(GP_TableContext* ctx, const GP_TableRow* rows, int32_t row_count);

// Retrieve current geometry (width/height/columns). Returns 1 on success.
int32_t gp_table_get_geom(const GP_TableContext* ctx, GP_TableGeom* out_geom);

// Render using stored rows/columns/style. Hitboxes are optional (may be NULL).
int32_t gp_table_render_rgba_with_hits(
    GP_TableContext* ctx,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written);

#ifdef __cplusplus
}
#endif
