#include "canvas_abi.h"
#include "rope_sim.h"
#include "table_abi.h"
#include "text_render_helper.h"

#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>

// local minimal Color and draw helpers (self-contained)
struct Color { uint8_t r=0,g=0,b=0,a=255; };

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
    if (outa <= 0.0f) { dst[0]=dst[1]=dst[2]=dst[3]=0; return; }
    float out_r = (srf * a + dr * da * inv) / outa;
    float out_g = (sgf * a + dg * da * inv) / outa;
    float out_b = (sbf * a + db * da * inv) / outa;
    dst[0] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_r)) * 255.0f));
    dst[1] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_g)) * 255.0f));
    dst[2] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_b)) * 255.0f));
    dst[3] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, outa)) * 255.0f));
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
            row[0] = c.r; row[1] = c.g; row[2] = c.b; row[3] = c.a; row += 4;
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
            if (dx*dx + dy*dy > r2) continue;
            uint8_t* p = img + y * pitch + x * 4;
            p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
        }
    }
}

// draw soft blended filled circle (used for rope blobs)
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
            float t = 1.0f - (std::sqrt((float)d2) / (float)r);
            uint8_t sa = static_cast<uint8_t>(std::lround(c.a * t));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}
// Per-canvas drag state (moved into the canvas object to avoid global map)
struct DragState { int dragging = 0; int module = -1; int offx = 0; int offy = 0; };

// Minimal internal canvas context implementation
struct GP_CanvasContextImpl {
    int width=0, height=0;
    std::vector<GP_CanvasModuleDesc> modules;
    // internal edge info bundles the desc, rope index, and per-edge hues
    struct EdgeInfo {
        GP_CanvasEdgeDesc desc;
        int rope_idx = -1;
        int type_id = 0; // 0 == wildcard / untyped
        std::vector<float> hues;
        float hue_intensity = 0.0f;
    };
    std::vector<EdgeInfo> edges;
    RopeSim* rope_sim = nullptr;
    // optional attached table per module (aligned with `modules` by index)
    std::vector<GP_TableContext*> module_tables;
    std::vector<int> module_table_owned; // 1 if canvas should destroy
    // per-module IO counts (inputs, outputs) exposed in the control bar
    std::vector<int> module_io_in_count;
    std::vector<int> module_io_out_count;
    // cable style/hues
    int jacket_px = 4;
    int jacket_border = 2;
    std::vector<float> hues;
    float hue_intensity = 0.0f;
    // selection state for click-to-connect behavior. `anchor_x/anchor_y` are
    // pixel coordinates (canvas space) for the selected contact when the
    // selection originates from a table hitbox; they remain -1 when unset.
    struct Sel { int module = -1; int contact_idx = -1; int left = -1; int anchor_x = -1; int anchor_y = -1; } selected;
    // per-canvas drag state (moved here to avoid a global map)
    DragState drag;
    // provisional rope index while user is selecting a contact and moving the mouse
    int prospective_rope_idx = -1;
    // tool selection state: separate groups (exclusive within group)
    // canvas tool group: 0 = neutral, 1 = new table, 2 = new module
    int selected_tool_canvas = 0;
    // table tool group: 0 = neutral, 1 = select, 2 = edit (placeholder)
    int selected_tool_table = 0;
    // which module (if any) has keyboard/focus for table editing
    int focused_module = -1;
    // registered host windows (opaque pointers)
    std::vector<void*> windows;
    // mapping from window pointer to stable node id for backing graph
    std::unordered_map<void*, int> window_node_ids;
    // next unique node id for graph nodes
    int next_node_id = 1;
    // per-module node id (aligned with `modules`) or -1 if none
    std::vector<int> module_node_id;
    // graph node/contract representation
    struct NodeContract {
        int node_id = -1;
        int module_idx = -1; // which module this node belongs to (-1 if none)
        std::vector<int> input_types; // supported input type ids
        std::vector<int> output_types; // supported output type ids
    };
    std::vector<NodeContract> nodes;
    // UI control bar height (in canvas-local pixels)
    int control_bar_h = 28;
    // autosave parameters (path may be empty to disable)
    std::string autosave_path;
    double autosave_interval_s = 0.0;
    double autosave_accum_s = 0.0;
    GP_CanvasContextImpl(int w, int h): width(w), height(h) {}
};

// (global drag map removed; each canvas has its own DragState member)


static inline void compute_contact_pos(const GP_CanvasModuleDesc &m, bool left, int idx, int &outx, int &outy) {
    int count = left ? m.left_contacts : m.right_contacts;
    if (count <= 0) count = 1;
    float frac = (idx + 0.5f) / (float)count;
    outy = m.y + static_cast<int>(std::lround(frac * m.h));
    outx = left ? m.x : (m.x + m.w - 1);
}

// compute contact pos given an explicit count (used when table defines IO keys)
static inline void compute_contact_pos_with_count(const GP_CanvasModuleDesc &m, bool left, int idx, int count, int &outx, int &outy) {
    if (count <= 0) count = 1;
    float frac = (idx + 0.5f) / (float)count;
    outy = m.y + static_cast<int>(std::lround(frac * m.h));
    outx = left ? m.x : (m.x + m.w - 1);
}

// Query attached table for input/output IO key counts. If table is null,
// returns zero counts.
static void get_table_io_counts(GP_TableContext* t, int &out_in_count, int &out_out_count) {
    out_in_count = 0; out_out_count = 0;
    if (!t) return;
    const int cap = 4096;
    std::vector<unsigned long long> keys(cap);
    int nin = gp_table_enumerate_io_keys(t, 0, keys.data(), cap);
    if (nin > 0) out_in_count = nin;
    int nout = gp_table_enumerate_io_keys(t, 1, keys.data(), cap);
    if (nout > 0) out_out_count = nout;
}

// Ensure the attached table reflects the canvas' requested IO counts for the module.
// Creates columns/rows and LED cells (GP_TABLE_CELL_LEDS_ARG) to display counts.
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx) {
    // Simplified behavior: only update the focused module's attached table
    // to contain a single LED cell representing the input count from the
    // control-bar. This avoids complex layout logic while producing a
    // stateful LED cell that the existing table rasterizer will render.
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx >= static_cast<int>(ctx->module_tables.size())) return;
    GP_TableContext* t = ctx->module_tables[module_idx];
    if (!t) return;
    // only touch the currently focused module to avoid clobbering other tables
    if (ctx->focused_module != module_idx) return;
    int in_count = 0;
    if (module_idx < static_cast<int>(ctx->module_io_in_count.size())) in_count = ctx->module_io_in_count[module_idx];

    // If there's no attached table but the UI has non-zero IO counts, create
    // a canvas-owned table so the user sees the LED cells immediately.
    if (!t && (in_count > 0 || (module_idx < static_cast<int>(ctx->module_io_out_count.size()) && ctx->module_io_out_count[module_idx] > 0))) {
        GP_TableContext* nt = gp_table_create(nullptr);
        if (nt) {
            ctx->module_tables[module_idx] = nt;
            if (module_idx >= static_cast<int>(ctx->module_table_owned.size())) ctx->module_table_owned.resize(module_idx + 1, 0);
            ctx->module_table_owned[module_idx] = 1;
            // attach to canvas rope sim
            if (!ctx->rope_sim) ctx->rope_sim = rope_sim_create(1024, 64);
            gp_table_attach_rope_sim(nt, ctx->rope_sim, 0);
            gp_table_set_prospective_mode(nt, 1);
            gp_table_prospective_set_params(nt, 8, 4.0f, 0.0f);
            t = nt;
        }
    }
    if (!t) return;

    // create two LED-arg columns (inputs, outputs) and a single header row
    // so the control bar's input/output counts are visible in the table.
    int out_count = 0;
    if (module_idx < static_cast<int>(ctx->module_io_out_count.size())) out_count = ctx->module_io_out_count[module_idx];
    // Create two columns (inputs, outputs). We'll create one row per IO index
    // so each input/output is rendered as a single LED cell (one LED per row
    // cell) rather than packing many LEDs into a single cell.
    int max_count = std::max(0, std::max(in_count, out_count));
    GP_TableColumn cols[2];
    cols[0].kind = GP_TABLE_CELL_LEDS_ARG; cols[0].width_px = 48; cols[0].align = 0;
    cols[1].kind = GP_TABLE_CELL_LEDS_ARG; cols[1].width_px = 48; cols[1].align = 0;
    gp_table_set_columns(t, cols, 2);

    if (max_count <= 0) {
        // no IO: create a single empty header row to keep table geometry stable
        GP_TableRow prow{}; memset(&prow, 0, sizeof(prow));
        prow.kind = GP_TABLE_ROW_HEADER; prow.depth = 0; prow.expanded = 1; prow.selected = 0;
        prow.cell_count = 2;
        // empty text cells
        prow.cells[0].kind = GP_TABLE_CELL_TEXT; prow.cells[1].kind = GP_TABLE_CELL_TEXT;
        gp_table_set_rows(t, &prow, 1);
        return;
    }

    std::vector<GP_TableRow> rows(static_cast<size_t>(max_count));
    for (int ri = 0; ri < max_count; ++ri) {
        GP_TableRow r{}; memset(&r, 0, sizeof(r));
        r.kind = GP_TABLE_ROW_DEVICE; r.depth = 0; r.expanded = 1; r.selected = 0;
        r.cell_count = 2;
        // left/input cell
        if (ri < in_count) {
            r.cells[0].kind = GP_TABLE_CELL_LEDS_ARG;
            r.cells[0].value = 1.0f; // single LED
            r.cells[0].flags = 1u;   // linked/on mask (one LED)
        } else {
            r.cells[0].kind = GP_TABLE_CELL_TEXT;
            r.cells[0].text[0] = '\0';
        }
        // right/output cell
        if (ri < out_count) {
            r.cells[1].kind = GP_TABLE_CELL_LEDS_ARG;
            r.cells[1].value = 1.0f;
            r.cells[1].flags = 1u;
        } else {
            r.cells[1].kind = GP_TABLE_CELL_TEXT;
            r.cells[1].text[0] = '\0';
        }
        rows[static_cast<size_t>(ri)] = r;
    }
    gp_table_set_rows(t, rows.data(), static_cast<int>(rows.size()));
    return;
}
extern "C" GP_CanvasContext* gp_canvas_create(int width, int height) {
    GP_CanvasContextImpl* c = new GP_CanvasContextImpl(width, height);
    // create internal rope sim immediately so canvases use live simulation by default
    c->rope_sim = rope_sim_create(1024, 64);
    return reinterpret_cast<GP_CanvasContext*>(c);
}

extern "C" void gp_canvas_destroy(GP_CanvasContext* ctx) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (!c) return;
    if (c->rope_sim) rope_sim_destroy(c->rope_sim);
    // destroy any owned attached tables
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] && c->module_table_owned[i]) {
            gp_table_destroy(c->module_tables[i]);
        }
    }
    delete c;
}

extern "C" int gp_canvas_add_module(GP_CanvasContext* ctx_, const GP_CanvasModuleDesc* desc) {
    if (!ctx_ || !desc) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    GP_CanvasModuleDesc d = *desc;
    c->modules.push_back(d);
    c->module_tables.push_back(nullptr);
    c->module_table_owned.push_back(0);
    c->module_io_in_count.push_back(0);
    c->module_io_out_count.push_back(0);
    return static_cast<int>(c->modules.size() - 1);
}

extern "C" int gp_canvas_move_module(GP_CanvasContext* ctx_, int module_idx, int x, int y) {
    if (!ctx_) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return -1;
    c->modules[module_idx].x = x;
    c->modules[module_idx].y = y;
    return 1;
}

// forward-declare typed-edge add function so legacy gp_canvas_add_edge can call it
extern "C" int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc, int type_id);


extern "C" int gp_canvas_on_click(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    printf("gp_canvas_on_click: click %d,%d\n", x, y);
    // find contact under point
    const int pick_r = 8;
    // check control bar button regions first — buttons are canvas-local coords
    if (y >= 0 && y < c->control_bar_h) {
        const int canvas_btn_count = 3;
        const int table_btn_count = 3;
        const int spacing = 12;
        int by = 4;
        int bh = std::max(4, c->control_bar_h - 8);
        int bw = bh; // square buttons
        // left canvas group
        int bx = 8;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            if (x >= bx_i && x < bx_i + bw && y >= by && y < by + bh) {
                if (c->selected_tool_canvas == bi) c->selected_tool_canvas = 0; else c->selected_tool_canvas = bi;
                printf("gp_canvas_on_click: canvas tool %d toggled -> selected_tool_canvas=%d\n", bi, c->selected_tool_canvas);
                return 1;
            }
        }
        // right table group
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, c->width - 8 - group_width);
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            if (x >= bx_i && x < bx_i + bw && y >= by && y < by + bh) {
                if (c->selected_tool_table == bi) c->selected_tool_table = 0; else c->selected_tool_table = bi;
                printf("gp_canvas_on_click: table tool %d toggled -> selected_tool_table=%d\n", bi, c->selected_tool_table);
                return 1;
            }
        }
        // IO controls to the left of table buttons: compute positions matching raster
        int io_base_x = bx_r;
        int nbw = bw;
        int num_w = std::max(24, nbw * 2);
        int gap = 10;
        // focused module drives which counts are shown/modified
        int focused = c->focused_module;
        if (focused >= 0 && focused < static_cast<int>(c->module_io_in_count.size())) {
            int in_count = c->module_io_in_count[focused];
            int out_count = c->module_io_out_count[focused];
            // inputs group positions
            int bx_minus_in = io_base_x - (nbw + gap + num_w + gap + nbw);
            int bx_num_in = bx_minus_in + nbw + gap;
            int bx_plus_in = bx_num_in + num_w + gap;
            if (x >= bx_minus_in && x < bx_minus_in + nbw && y >= by && y < by + bh) {
                c->module_io_in_count[focused] = std::max(0, in_count - 1);
                printf("gp_canvas_on_click: dec inputs for module %d -> %d\n", focused, c->module_io_in_count[focused]);
                return 1;
            }
            if (x >= bx_plus_in && x < bx_plus_in + nbw && y >= by && y < by + bh) {
                c->module_io_in_count[focused] = std::min(64, in_count + 1);
                printf("gp_canvas_on_click: inc inputs for module %d -> %d\n", focused, c->module_io_in_count[focused]);
                return 1;
            }
            // outputs group slightly left of inputs group (as drawn)
            int io_shift = (nbw + spacing + 80);
            int bx_minus_out = io_base_x - io_shift - (nbw + gap + num_w + gap + nbw);
            int bx_num_out = bx_minus_out + nbw + gap;
            int bx_plus_out = bx_num_out + num_w + gap;
            if (x >= bx_minus_out && x < bx_minus_out + nbw && y >= by && y < by + bh) {
                c->module_io_out_count[focused] = std::max(0, out_count - 1);
                printf("gp_canvas_on_click: dec outputs for module %d -> %d\n", focused, c->module_io_out_count[focused]);
                return 1;
            }
            if (x >= bx_plus_out && x < bx_plus_out + nbw && y >= by && y < by + bh) {
                c->module_io_out_count[focused] = std::min(64, out_count + 1);
                printf("gp_canvas_on_click: inc outputs for module %d -> %d\n", focused, c->module_io_out_count[focused]);
                return 1;
            }
        }
    }
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        // If module has an attached table, ask the table for hit information
        // for clicks inside the module rect. If the table reports a LED hit,
        // map that hit into the canvas contact selection/rope creation flow
        // so clicks target the actual LED cells rendered by the table.
        if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) {
            if (x >= m.x && x < m.x + m.w && y >= m.y && y < m.y + m.h) {
                // Query the attached table for hitboxes without invoking
                // `gp_table_on_click` (which mutates table selection). This
                // lets the canvas resolve LED hits for edge-mode without
                // competing with the table's own selection logic.
                GP_TableContext* t = c->module_tables[mi];
                int lx = x - m.x;
                int ly = y - m.y;
                int tw = std::max(1, m.w);
                int th = std::max(1, m.h);
                std::vector<uint8_t> tmp(static_cast<size_t>(tw) * static_cast<size_t>(th) * 4);
                GP_TableGeom geom{};
                gp_table_get_geom(t, &geom);
                geom.width_px = tw; geom.height_px = th;
                const int hitcap = 1024;
                std::vector<GP_TableHitBox> hits(hitcap);
                int hits_written = 0;
                int ok = gp_table_render_rgba_with_state(t, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &hits_written);
                if (ok && hits_written > 0) {
                    // find first hit containing local point
                    GP_TableHitBox found{}; bool found_any = false;
                    for (int hi = 0; hi < hits_written; ++hi) {
                        const GP_TableHitBox &hb = hits[hi];
                        if (lx >= hb.x0 && lx < hb.x1 && ly >= hb.y0 && ly < hb.y1) { found = hb; found_any = true; break; }
                    }
                    if (found_any) {
                        // If LED hit, handle canvas-level connection flow. In
                        // edge-drawing mode we avoid calling into the table so
                        // we don't toggle its internal selection state.
                        if (found.part == GP_TABLE_HIT_LED || found.part == GP_TABLE_HIT_LED_ARG || found.part == GP_TABLE_HIT_LED_TABLE) {
                            int ax = m.x + (found.x0 + found.x1) / 2;
                            int ay = m.y + (found.y0 + found.y1) / 2;
                            bool is_left = (ax < m.x + m.w / 2);
                            c->focused_module = mi;
                            // If we're in edge-drawing mode (tool index 2), don't
                            // let the table mutate selection; instead manage
                            // canvas selection and ropes here.
                            if (c->selected.module == -1) {
                                c->selected.module = mi; c->selected.contact_idx = found.aux0; c->selected.left = is_left ? 1 : 0;
                                c->selected.anchor_x = ax; c->selected.anchor_y = ay;
                                if (!c->rope_sim) c->rope_sim = rope_sim_create(1024, 64);
                                int ax0 = ax, ay0 = ay;
                                int bx = ax, by = ay;
                                int segs = 8; float slack = 0.0f;
                                c->prospective_rope_idx = rope_sim_add_rope(c->rope_sim, static_cast<float>(ax0), static_cast<float>(ay0), static_cast<float>(bx), static_cast<float>(by), segs, slack);
                                return 1;
                            } else {
                                if ((c->selected.left == 1 && !is_left) || (c->selected.left == 0 && is_left)) {
                                    GP_CanvasEdgeDesc e;
                                    if (c->selected.left == 1) {
                                        e.a_module = c->selected.module; e.a_contact_idx = c->selected.contact_idx;
                                        e.b_module = mi; e.b_contact_idx = found.aux0;
                                    } else {
                                        e.a_module = mi; e.a_contact_idx = found.aux0;
                                        e.b_module = c->selected.module; e.b_contact_idx = c->selected.contact_idx;
                                    }
                                    int ei = gp_canvas_add_edge(ctx_, &e);
                                    if (c->prospective_rope_idx >= 0) {
                                        c->edges[ei].rope_idx = c->prospective_rope_idx;
                                        c->prospective_rope_idx = -1;
                                    }
                                    c->selected.module = -1; c->selected.contact_idx = -1; c->selected.left = -1; c->selected.anchor_x = -1; c->selected.anchor_y = -1;
                                    return 1;
                                } else {
                                    c->selected.module = mi; c->selected.contact_idx = found.aux0; c->selected.left = is_left ? 1 : 0; c->selected.anchor_x = ax; c->selected.anchor_y = ay;
                                    return 1;
                                }
                            }
                        } else {
                            // Non-LED hit: let table handle the click unless we're
                            // in canvas-level edge-only mode (tool index 2).
                            if (c->selected_tool_canvas != 2) {
                                // forward to table so it can perform its default actions
                                GP_TableHitBox out_hit{};
                                gp_table_on_click(t, lx, ly, &out_hit);
                                return 1;
                            }
                            // if edge-only mode, consume the hit but do not mutate table
                            return 1;
                        }
                    }
                }
            }
        }
        // If module has an attached table, do not run legacy contact-hit logic
        // — table content and hitboxes are authoritative for drawing and hits.
        if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) continue;
        // compute left contact count (prefer table input keys if attached)
        int left_count = m.left_contacts;
        int right_count = m.right_contacts;
        if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) {
            int in_ct=0, out_ct=0; get_table_io_counts(c->module_tables[mi], in_ct, out_ct);
            if (in_ct > 0) left_count = in_ct;
            if (out_ct > 0) right_count = out_ct;
        }
        // left contacts
        for (int ci = 0; ci < left_count; ++ci) {
            int cx, cy; compute_contact_pos_with_count(m, true, ci, left_count, cx, cy);
            int dx = x - cx; int dy = y - cy;
            if (dx*dx + dy*dy <= pick_r*pick_r) {
                // clicked a left contact
                // focus the module
                c->focused_module = mi;
                printf("gp_canvas_on_click: hit left contact module=%d contact=%d\n", mi, ci);
                if (c->selected.module == -1) {
                    printf("gp_canvas_on_click: selecting left %d/%d\n", mi, ci);
                    c->selected.module = mi; c->selected.contact_idx = ci; c->selected.left = 1; c->selected.anchor_x = cx; c->selected.anchor_y = cy;
                    // create a provisional rope that will follow the mouse
                    if (!c->rope_sim) c->rope_sim = rope_sim_create(1024, 64);
                    int ax = cx, ay = cy;
                    int bx = cx, by = cy;
                    int segs = 8;
                    float slack = 0.0f;
                    c->prospective_rope_idx = rope_sim_add_rope(c->rope_sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack);
                    printf("gp_canvas_on_click: created prospective rope %d for left %d/%d\n", c->prospective_rope_idx, mi, ci);
                    return 1;
                } else {
                    // ensure we have opposite sides
                    if (c->selected.left == 0) {
                        // selected was right, now left -> create edge with left as a and right as b
                        GP_CanvasEdgeDesc e;
                        e.a_module = mi; e.a_contact_idx = ci;
                        e.b_module = c->selected.module; e.b_contact_idx = c->selected.contact_idx;
                        int right_mod = c->selected.module;
                        int right_ci = c->selected.contact_idx;
                        int ei = gp_canvas_add_edge(ctx_, &e);
                        // promote provisional rope to the real edge if present
                        if (c->prospective_rope_idx >= 0) {
                            c->edges[ei].rope_idx = c->prospective_rope_idx;
                            printf("gp_canvas_on_click: promoted prospective rope %d to edge %d\n", c->prospective_rope_idx, ei);
                            c->prospective_rope_idx = -1;
                        }
                        c->selected.module = -1; c->selected.contact_idx = -1; c->selected.left = -1; c->selected.anchor_x = -1; c->selected.anchor_y = -1;
                        printf("gp_canvas_on_click: created edge %d (L %d.%d -> R %d.%d)\n", ei, mi, ci, right_mod, right_ci);
                        return 1;
                    } else {
                        // both left; toggle selection to this
                        printf("gp_canvas_on_click: switching selection to left %d/%d\n", mi, ci);
                        c->selected.module = mi; c->selected.contact_idx = ci; c->selected.left = 1; c->selected.anchor_x = cx; c->selected.anchor_y = cy;
                        return 1;
                    }
                }
            }
        }
        // right contacts
        for (int ci = 0; ci < right_count; ++ci) {
            int cx, cy; compute_contact_pos_with_count(m, false, ci, right_count, cx, cy);
            int dx = x - cx; int dy = y - cy;
            if (dx*dx + dy*dy <= pick_r*pick_r) {
                // clicked a right contact
                // focus the module
                c->focused_module = mi;
                printf("gp_canvas_on_click: hit right contact module=%d contact=%d\n", mi, ci);
                if (c->selected.module == -1) {
                    printf("gp_canvas_on_click: selecting right %d/%d\n", mi, ci);
                    c->selected.module = mi; c->selected.contact_idx = ci; c->selected.left = 0; c->selected.anchor_x = cx; c->selected.anchor_y = cy;
                    // create a provisional rope that will follow the mouse
                    if (!c->rope_sim) c->rope_sim = rope_sim_create(1024, 64);
                    int ax = cx, ay = cy;
                    int bx = cx, by = cy;
                    int segs = 8;
                    float slack = 0.0f;
                    c->prospective_rope_idx = rope_sim_add_rope(c->rope_sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack);
                    printf("gp_canvas_on_click: created prospective rope %d for right %d/%d\n", c->prospective_rope_idx, mi, ci);
                    return 1;
                } else {
                    if (c->selected.left == 1) {
                        // selected was left, now right -> create edge with left as a and right as b
                        GP_CanvasEdgeDesc e;
                        e.a_module = c->selected.module; e.a_contact_idx = c->selected.contact_idx;
                        e.b_module = mi; e.b_contact_idx = ci;
                        int left_mod = c->selected.module;
                        int left_ci = c->selected.contact_idx;
                        int ei = gp_canvas_add_edge(ctx_, &e);
                        if (c->prospective_rope_idx >= 0) {
                            c->edges[ei].rope_idx = c->prospective_rope_idx;
                            printf("gp_canvas_on_click: promoted prospective rope %d to edge %d\n", c->prospective_rope_idx, ei);
                            c->prospective_rope_idx = -1;
                        }
                        c->selected.module = -1; c->selected.contact_idx = -1; c->selected.left = -1; c->selected.anchor_x = -1; c->selected.anchor_y = -1;
                        printf("gp_canvas_on_click: created edge %d (L %d.%d -> R %d.%d)\n", ei, left_mod, left_ci, mi, ci);
                        return 1;
                    } else {
                        // both right; toggle selection to this
                        printf("gp_canvas_on_click: switching selection to right %d/%d\n", mi, ci);
                        c->selected.module = mi; c->selected.contact_idx = ci; c->selected.left = 0; c->selected.anchor_x = cx; c->selected.anchor_y = cy;
                        return 1;
                    }
                }
            }
        }
    }
    // click not on any contact: clear selection
    c->selected.module = -1; c->selected.contact_idx = -1; c->selected.left = -1; c->selected.anchor_x = -1; c->selected.anchor_y = -1;
    // discard any provisional rope
    if (c->prospective_rope_idx >= 0) {
        printf("gp_canvas_on_click: discarding prospective rope %d\n", c->prospective_rope_idx);
        c->prospective_rope_idx = -1;
    }

    // If a tool is active and the click is in empty space (not on any module),
    // spawn the tool's module. Tool mapping changed: 0=select, 1=create-table, 2=edge-mode
    bool hit_module = false;
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        if (x >= m.x && x < m.x + m.w && y >= m.y && y < m.y + m.h) { hit_module = true; break; }
    }
    if (!hit_module && c->selected_tool_canvas == 1) {
        GP_CanvasModuleDesc d{};
        int nx = x - 20;
        int ny = y - 16;
        nx = std::max(0, std::min(c->width - 160, nx));
        ny = std::max(0, std::min(c->height - 120, ny));
        d.x = nx; d.y = ny; d.w = 160; d.h = 120; d.left_contacts = 3; d.right_contacts = 3;
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "Table %zu", c->modules.size()); std::memset(d.label,0,sizeof(d.label)); std::memcpy(d.label,lbl,std::min<size_t>(strlen(lbl), sizeof(d.label)-1));
        int new_idx = gp_canvas_add_module(ctx_, &d);
        if (new_idx >= 0) {
            // Tool==1 => create a table at click position
            gp_canvas_create_table(ctx_, new_idx);
            printf("gp_canvas_on_click: spawned new table at %d,%d module=%d\n", nx, ny, new_idx);
            // focus the newly created module
            c->focused_module = new_idx;
        }
        // leave tool selected — user can toggle off with button
        return 1;
    }

    // forward click to any attached table that contains the point
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        if (x >= m.x && x < m.x + m.w && y >= m.y && y < m.y + m.h) {
            // focus this module when clicked
            c->focused_module = mi;
            if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi] && c->selected_tool_canvas == 0) {
                // local coords; only forward clicks into attached tables when
                // canvas is in select/interaction mode (tool 0). In edge-mode
                // we performed non-mutating hit tests earlier and should avoid
                // letting the table change its own selection state here.
                int lx = x - m.x;
                int ly = y - m.y;
                GP_TableHitBox hb{};
                int ok = gp_table_on_click(c->module_tables[mi], lx, ly, &hb);
                if (ok) return 1;
            }
            break;
        }
    }
    return 0;
}

extern "C" int gp_canvas_on_mouse_down(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // if click on a contact, reuse gp_canvas_on_click behavior
    if (gp_canvas_on_click(ctx_, x, y)) return 1;
    // otherwise check for module hit to start dragging
    for (int mi = static_cast<int>(c->modules.size()) - 1; mi >= 0; --mi) {
        const auto &m = c->modules[mi];
        if (x >= m.x && x < m.x + m.w && y >= m.y && y < m.y + m.h) {
            // start drag: record in per-canvas DragState
            c->drag.dragging = 1;
            c->drag.module = mi;
            c->drag.offx = x - m.x;
            c->drag.offy = y - m.y;
            // focus the module being dragged
            c->focused_module = mi;
            printf("gp_canvas_on_mouse_down: start drag canvas=%p module=%d off=%d,%d\n", (void*)c, mi, c->drag.offx, c->drag.offy);
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_on_mouse_move(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    bool handled = false;
    // update provisional rope endpoint to follow mouse
    if (c->prospective_rope_idx >= 0 && c->selected.module >= 0) {
        int fx = 0, fy = 0;
        bool left = (c->selected.left == 1);
        if (c->selected.module >= 0 && c->selected.module < static_cast<int>(c->modules.size())) {
            if (c->selected.anchor_x >= 0 && c->selected.anchor_y >= 0) {
                fx = c->selected.anchor_x; fy = c->selected.anchor_y;
            } else {
                compute_contact_pos(c->modules[c->selected.module], left, c->selected.contact_idx, fx, fy);
            }
            rope_sim_move_endpoints(c->rope_sim, c->prospective_rope_idx, static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(x), static_cast<float>(y));
            handled = true;
        }
    }
    // handle module drag if present (per-canvas drag state)
    DragState ds = c->drag;
    if (ds.dragging && ds.module >= 0) {
        int nx = x - ds.offx;
        int ny = y - ds.offy;
        // clamp within canvas
        nx = std::max(0, std::min(c->width - c->modules[ds.module].w, nx));
        ny = std::max(0, std::min(c->height - c->modules[ds.module].h, ny));
        c->modules[ds.module].x = nx;
        c->modules[ds.module].y = ny;
        printf("gp_canvas_on_mouse_move: canvas=%p module=%d -> %d,%d\n", (void*)c, ds.module, nx, ny);
        handled = true;
    }
    return handled ? 1 : 0;
}

extern "C" int gp_canvas_on_mouse_up(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->drag.dragging) return 0;
    c->drag.dragging = 0;
    printf("gp_canvas_on_mouse_up: canvas=%p module=%d\n", (void*)c, c->drag.module);
    c->drag.module = -1;
    return 1;
}

// create/destroy canvas-owned table helpers
extern "C" int gp_canvas_create_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;
    if (c->module_tables[module_idx]) return 0; // already has a table
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return 0;
    c->module_tables[module_idx] = t;
    c->module_table_owned[module_idx] = 1;
    // ensure module has a backing node in graph
    if (module_idx >= static_cast<int>(c->module_node_id.size())) c->module_node_id.resize(module_idx + 1, -1);
    if (c->module_node_id[module_idx] < 0) {
        int nid = c->next_node_id++;
        c->module_node_id[module_idx] = nid;
        GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = module_idx;
        c->nodes.push_back(std::move(nc));
    }
    // attach table to canvas sim
    if (!c->rope_sim) c->rope_sim = rope_sim_create(1024, 64);
    gp_table_attach_rope_sim(t, c->rope_sim, 0);
    // enable prospective/live mode by default for embedded tables
    gp_table_set_prospective_mode(t, 1);
    gp_table_prospective_set_params(t, 8, 4.0f, 0.0f);
    // If the table has no IO sections and no existing columns, create empty
    // input/output edge sections on the table according to its side-reading
    // direction so canvases render empty input/output strips at the edges.
    // Only do this for truly blank tables to avoid clobbering caller-initialized tables.
    {
        GP_TableGeom gtmp{};
        int col_count_existing = 0;
        if (gp_table_get_geom(t, &gtmp)) {
            for (int ii = 0; ii < 8; ++ii) if (gtmp.col_w[ii] > 0) ++col_count_existing;
        }
        if (!gp_table_has_io_sections(t) && col_count_existing == 0) {
            int in_dir = 0, out_dir = 0;
            gp_table_get_side_reading_direction(t, 0, &in_dir);
            gp_table_get_side_reading_direction(t, 1, &out_dir);
            int dir = in_dir; // prefer input side direction as primary
            if (dir == 0 || dir == 2) {
                // horizontal layout: ensure an input column at left and output at right
                GP_TableColumn cols[2];
                cols[0].kind = GP_TABLE_CELL_LEDS;
                cols[0].width_px = 80;
                cols[0].align = 0;
                cols[1].kind = GP_TABLE_CELL_LEDS;
                cols[1].width_px = 80;
                cols[1].align = 0;
                if (dir == 2) { // RightToLeft -> swap so inputs appear on the right edge
                    GP_TableColumn tmp = cols[0]; cols[0] = cols[1]; cols[1] = tmp;
                }
                gp_table_set_columns(t, cols, 2);
                // leave rows empty for now
                gp_table_set_rows(t, nullptr, 0);
            } else {
                // vertical layout: create a single column and empty top/bottom rows
                GP_TableColumn col;
                col.kind = GP_TABLE_CELL_LEDS;
                col.width_px = 160;
                col.align = 0;
                gp_table_set_columns(t, &col, 1);
                GP_TableRow rows[2];
                memset(rows, 0, sizeof(rows));
                rows[0].kind = GP_TABLE_ROW_HEADER;
                rows[0].depth = 0;
                rows[0].expanded = 1;
                rows[0].selected = 0;
                rows[0].cell_count = 1;
                rows[0].cells[0].kind = GP_TABLE_CELL_LEDS;
                rows[1].kind = GP_TABLE_ROW_NOTE;
                rows[1].depth = 0;
                rows[1].expanded = 1;
                rows[1].selected = 0;
                rows[1].cell_count = 1;
                rows[1].cells[0].kind = GP_TABLE_CELL_LEDS;
                if (dir == 3) {
                    // BottomToTop: place inputs at bottom (swap)
                    GP_TableRow tmp = rows[0]; rows[0] = rows[1]; rows[1] = tmp;
                }
                gp_table_set_rows(t, rows, 2);
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_register_window(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // if already registered, noop
    if (c->window_node_ids.find(window_ptr) != c->window_node_ids.end()) {
        return 1;
    }
    int nid = c->next_node_id++;
    c->window_node_ids[window_ptr] = nid;
    c->windows.push_back(window_ptr);
    return 1;
}

extern "C" int gp_canvas_unregister_window(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    for (auto it = c->windows.begin(); it != c->windows.end(); ++it) {
        if (*it == window_ptr) { c->windows.erase(it); break; }
    }
    auto mit = c->window_node_ids.find(window_ptr);
    if (mit != c->window_node_ids.end()) c->window_node_ids.erase(mit);
    return 1;
}

extern "C" int gp_canvas_get_window_node_id(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto it = c->window_node_ids.find(window_ptr);
    if (it == c->window_node_ids.end()) return -1;
    return it->second;
}

extern "C" int gp_canvas_destroy_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;
    GP_TableContext* t = c->module_tables[module_idx];
    int owned = c->module_table_owned[module_idx];
    c->module_tables[module_idx] = nullptr;
    c->module_table_owned[module_idx] = 0;
    if (t && owned) gp_table_destroy(t);
    // detach node mapping for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_node_id.size())) {
        int nid = c->module_node_id[module_idx];
        c->module_node_id[module_idx] = -1;
        for (auto &n : c->nodes) {
            if (n.node_id == nid) { n.module_idx = -1; break; }
        }
    }
    return 1;
}

extern "C" int gp_canvas_step(GP_CanvasContext* ctx_, float dt) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->rope_sim) return 0;
    rope_sim_step(c->rope_sim, dt, 800.0f, 8, 0.86f);
    // Drive per-module table step callbacks using the UI-specified in/out counts.
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        GP_TableContext* t = nullptr;
        if (mi >= 0 && mi < static_cast<int>(c->module_tables.size())) t = c->module_tables[mi];
        if (!t) continue;
        int in_count = 0, out_count = 0;
        if (mi < static_cast<int>(c->module_io_in_count.size())) in_count = c->module_io_in_count[mi];
        if (mi < static_cast<int>(c->module_io_out_count.size())) out_count = c->module_io_out_count[mi];
        if (in_count <= 0 && out_count <= 0) continue;
        std::vector<float> inputs(static_cast<size_t>(std::max(0, in_count))); 
        std::vector<float> outputs(static_cast<size_t>(std::max(0, out_count)));
        // zero-init
        for (auto &v : inputs) v = 0.0f;
        for (auto &v : outputs) v = 0.0f;
        // invoke the table's step callback (if installed)
        gp_table_step(t, in_count ? inputs.data() : nullptr, in_count, out_count ? outputs.data() : nullptr, out_count, static_cast<double>(dt));
        // (outputs are ignored here; host may later fetch state via other APIs)
    }
    // autosave: accumulate dt and write canvas file when interval reached
    if (!c->autosave_path.empty() && c->autosave_interval_s > 0.0) {
        c->autosave_accum_s += static_cast<double>(dt);
        if (c->autosave_accum_s >= c->autosave_interval_s) {
            // attempt save; ignore failures
            gp_canvas_save_to_file(ctx_, c->autosave_path.c_str());
            c->autosave_accum_s = 0.0;
        }
    }
    return 1;
}

extern "C" void* gp_canvas_get_rope_sim(GP_CanvasContext* ctx_) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    return reinterpret_cast<void*>(c->rope_sim);
}

extern "C" int gp_canvas_attach_table(GP_CanvasContext* ctx_, int module_idx, GP_TableContext* table, int take_ownership) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    // set table pointer and ownership
    c->module_tables[module_idx] = table;
    c->module_table_owned[module_idx] = take_ownership ? 1 : 0;
    // ensure module has a backing node in graph
    if (module_idx >= static_cast<int>(c->module_node_id.size())) c->module_node_id.resize(module_idx + 1, -1);
    if (c->module_node_id[module_idx] < 0) {
        int nid = c->next_node_id++;
        c->module_node_id[module_idx] = nid;
        GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = module_idx;
        c->nodes.push_back(std::move(nc));
    }
    // ensure canvas has a rope_sim and attach to table so table defers simulation
    if (!c->rope_sim) c->rope_sim = rope_sim_create(1024, 64);
    if (table) {
        // attach table's rope sim to canvas sim; do not transfer ownership
        gp_table_attach_rope_sim(table, c->rope_sim, 0);
        // enable prospective/live mode by default for attached tables
        gp_table_set_prospective_mode(table, 1);
        gp_table_prospective_set_params(table, 8, 4.0f, 0.0f);
        // If the table has no IO sections and no existing columns, create empty
        // IO columns/rows so attached blank tables show input/output strips.
        GP_TableGeom gtmp{};
        int col_count_existing = 0;
        if (gp_table_get_geom(table, &gtmp)) {
            for (int ii = 0; ii < 8; ++ii) if (gtmp.col_w[ii] > 0) ++col_count_existing;
        }
        if (!gp_table_has_io_sections(table) && col_count_existing == 0) {
            int in_dir = 0, out_dir = 0;
            gp_table_get_side_reading_direction(table, 0, &in_dir);
            gp_table_get_side_reading_direction(table, 1, &out_dir);
            int dir = in_dir;
            if (dir == 0 || dir == 2) {
                GP_TableColumn cols[2];
                cols[0].kind = GP_TABLE_CELL_LEDS;
                cols[0].width_px = 80;
                cols[0].align = 0;
                cols[1].kind = GP_TABLE_CELL_LEDS;
                cols[1].width_px = 80;
                cols[1].align = 0;
                if (dir == 2) { GP_TableColumn tmp = cols[0]; cols[0] = cols[1]; cols[1] = tmp; }
                gp_table_set_columns(table, cols, 2);
                gp_table_set_rows(table, nullptr, 0);
            } else {
                GP_TableColumn col;
                col.kind = GP_TABLE_CELL_LEDS;
                col.width_px = 160;
                col.align = 0;
                gp_table_set_columns(table, &col, 1);
                GP_TableRow rows[2]; memset(rows, 0, sizeof(rows));
                rows[0].kind = GP_TABLE_ROW_HEADER; rows[0].depth = 0; rows[0].expanded = 1; rows[0].cell_count = 1; rows[0].cells[0].kind = GP_TABLE_CELL_LEDS;
                rows[1].kind = GP_TABLE_ROW_NOTE; rows[1].depth = 0; rows[1].expanded = 1; rows[1].cell_count = 1; rows[1].cells[0].kind = GP_TABLE_CELL_LEDS;
                if (dir == 3) { GP_TableRow tmp = rows[0]; rows[0] = rows[1]; rows[1] = tmp; }
                gp_table_set_rows(table, rows, 2);
            }
        }
        // Probe table IO keys and populate NodeContract input/output type lists
        // so the canvas knows what types this module exposes.
        if (module_idx >= 0) {
            // ensure nodes vector contains the contract for this module (created above)
            GP_CanvasContextImpl::NodeContract* found = nullptr;
            for (auto &n : c->nodes) if (n.module_idx == module_idx) { found = &n; break; }
            if (found) {
                found->input_types.clear();
                found->output_types.clear();
                const int cap = 4096;
                std::vector<unsigned long long> keys(cap);
                int nin = gp_table_enumerate_io_keys(table, 0, keys.data(), cap);
                if (nin > 0) {
                    std::unordered_set<int> in_types;
                    for (int i = 0; i < nin; ++i) {
                        unsigned long long k = keys[i];
                        int type_id = -1, is_in=0, is_out=0;
                        gp_table_get_key_type_hint(table, k, &type_id, &is_in, &is_out);
                        if (type_id >= 0) in_types.insert(type_id);
                    }
                    found->input_types.assign(in_types.begin(), in_types.end());
                }
                int nout = gp_table_enumerate_io_keys(table, 1, keys.data(), cap);
                if (nout > 0) {
                    std::unordered_set<int> out_types;
                    for (int i = 0; i < nout; ++i) {
                        unsigned long long k = keys[i];
                        int type_id = -1, is_in=0, is_out=0;
                        gp_table_get_key_type_hint(table, k, &type_id, &is_in, &is_out);
                        if (type_id >= 0) out_types.insert(type_id);
                    }
                    found->output_types.assign(out_types.begin(), out_types.end());
                }
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_set_templates_dir(const char* dir) {
    // forward to table helper
    return gp_table_set_library_dir(dir) ? 1 : 0;
}

extern "C" int gp_canvas_get_templates_dir(char* out_buf, int out_len) {
    return gp_table_get_library_dir(out_buf, out_len);
}

extern "C" int gp_canvas_set_autosave(GP_CanvasContext* ctx_, const char* path, double interval_s) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!path || path[0] == '\0' || interval_s <= 0.0) {
        c->autosave_path.clear(); c->autosave_interval_s = 0.0; c->autosave_accum_s = 0.0; return 1;
    }
    c->autosave_path = std::string(path);
    c->autosave_interval_s = interval_s;
    c->autosave_accum_s = 0.0;
    return 1;
}

extern "C" int gp_canvas_get_autosave(GP_CanvasContext* ctx_, char* out_path, int out_len, double* out_interval_s) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (out_path && out_len > 0) {
        int32_t to_write = static_cast<int32_t>(std::min<size_t>(c->autosave_path.size(), static_cast<size_t>(out_len)));
        if (to_write > 0) memcpy(out_path, c->autosave_path.data(), static_cast<size_t>(to_write));
    }
    if (out_interval_s) *out_interval_s = c->autosave_interval_s;
    return 1;
}

extern "C" int gp_canvas_detach_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    GP_TableContext* t = c->module_tables[module_idx];
    int owned = c->module_table_owned[module_idx];
    c->module_tables[module_idx] = nullptr;
    c->module_table_owned[module_idx] = 0;
    if (t && owned) gp_table_destroy(t);
    // detach node mapping for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_node_id.size())) {
        int nid = c->module_node_id[module_idx];
        c->module_node_id[module_idx] = -1;
        for (auto &n : c->nodes) {
            if (n.node_id == nid) { n.module_idx = -1; break; }
        }
    }
    return 1;
}

extern "C" int gp_canvas_add_edge(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc) {
    // legacy: add untyped edge (type_id == 0)
    return gp_canvas_add_edge_with_type(ctx_, desc, 0);
}

// add an edge with an explicit type id. type_id == 0 means untyped/wildcard.
extern "C" int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc, int type_id) {
    if (!ctx_ || !desc) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // basic bounds
    if (desc->a_module < 0 || desc->a_module >= static_cast<int>(c->modules.size())) return -1;
    if (desc->b_module < 0 || desc->b_module >= static_cast<int>(c->modules.size())) return -1;
    // If a type is specified, validate against module contracts: a_module must provide the type as output and b_module must accept as input.
    if (type_id != 0) {
        int na = (desc->a_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->a_module] : -1;
        int nb = (desc->b_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->b_module] : -1;
        if (na < 0 || nb < 0) return -1;
        // find node contracts
        GP_CanvasContextImpl::NodeContract* nodeA = nullptr;
        GP_CanvasContextImpl::NodeContract* nodeB = nullptr;
        for (auto &n : c->nodes) {
            if (n.node_id == na) nodeA = &n;
            if (n.node_id == nb) nodeB = &n;
        }
        if (!nodeA || !nodeB) return -1;
        bool a_supports = std::find(nodeA->output_types.begin(), nodeA->output_types.end(), type_id) != nodeA->output_types.end();
        bool b_supports = std::find(nodeB->input_types.begin(), nodeB->input_types.end(), type_id) != nodeB->input_types.end();
        if (!a_supports || !b_supports) {
            printf("gp_canvas_add_edge_with_type: type %d not supported by modules %d->%d\n", type_id, desc->a_module, desc->b_module);
            return -1;
        }
    }

    GP_CanvasContextImpl::EdgeInfo ei;
    ei.desc = *desc;
    ei.type_id = type_id;
    // copy canvas-level default hues into edge if available
    if (!c->hues.empty()) {
        ei.hues = c->hues;
        ei.hue_intensity = c->hue_intensity;
    }
    c->edges.push_back(std::move(ei));
    return static_cast<int>(c->edges.size() - 1);
}

// Set the module's supported input/output type lists
extern "C" int gp_canvas_set_module_io_types(GP_CanvasContext* ctx_, int module_idx, const int* input_types, int input_count, const int* output_types, int output_count) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    int nid = -1;
    if (module_idx < static_cast<int>(c->module_node_id.size())) nid = c->module_node_id[module_idx];
    if (nid < 0) return 0;
    // find node
    for (auto &n : c->nodes) {
        if (n.node_id == nid) {
            n.input_types.clear(); n.output_types.clear();
            if (input_types && input_count > 0) n.input_types.assign(input_types, input_types + input_count);
            if (output_types && output_count > 0) n.output_types.assign(output_types, output_types + output_count);
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_get_module_node_id(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_node_id.size())) return -1;
    return c->module_node_id[module_idx];
}

// Persist canvas state to a simple line-based file format. This is intentionally
// lightweight and human readable so it's easy to edit by hand during development.
// Format (V1):
// CANVAS V1
// WIDTH HEIGHT CONTROL_BAR_H
// MODULE x y w h left_contacts right_contacts label
// EDGE a_module a_contact_idx b_module b_contact_idx type_id
// NODE node_id module_idx input_count [inputs...] output_count [outputs...]
// Lines may appear in any order; loader will reconstruct internal vectors.
extern "C" int gp_canvas_save_to_file(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::ofstream ofs(path);
    if (!ofs.good()) return 0;
    ofs << "CANVAS V1\n";
    ofs << c->width << " " << c->height << " " << c->control_bar_h << "\n";
    // modules
    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &m = c->modules[i];
        // label may contain spaces; write as remainder of line
        ofs << "MODULE " << m.x << " " << m.y << " " << m.w << " " << m.h << " " << m.left_contacts << " " << m.right_contacts << " ";
        // write label as-is but escape newlines and backslashes
        std::string lbl(m.label, m.label + sizeof(m.label));
        // trim at first null
        size_t z = lbl.find('\0'); if (z != std::string::npos) lbl.resize(z);
        for (char ch : lbl) {
            if (ch == '\\') ofs << "\\\\";
            else if (ch == '\n') ofs << "\\n";
            else ofs << ch;
        }
        ofs << "\n";
    }
    // edges
    for (const auto &ei : c->edges) {
        const auto &e = ei.desc;
        ofs << "EDGE " << e.a_module << " " << e.a_contact_idx << " " << e.b_module << " " << e.b_contact_idx << " " << ei.type_id << "\n";
    }
    // nodes/contracts
    for (const auto &n : c->nodes) {
        ofs << "NODE " << n.node_id << " " << n.module_idx << " ";
        ofs << static_cast<int>(n.input_types.size());
        for (int t : n.input_types) ofs << " " << t;
        ofs << " " << static_cast<int>(n.output_types.size());
        for (int t : n.output_types) ofs << " " << t;
        ofs << "\n";
    }
    ofs.close();
    return 1;
}

// Load canvas state from file written by `gp_canvas_save_to_file`. The loader
// will clear current modules/edges/nodes and recreate them. Any canvas-owned
// tables will be destroyed. Returns 1 on success.
extern "C" int gp_canvas_load_from_file(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::ifstream ifs(path);
    if (!ifs.good()) return 0;
    // clear existing modules/edges/nodes (destroy owned tables)
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] && c->module_table_owned[i]) gp_table_destroy(c->module_tables[i]);
    }
    c->modules.clear();
    c->module_tables.clear();
    c->module_table_owned.clear();
    c->edges.clear();
    c->nodes.clear();
    c->module_node_id.clear();

    std::string line;
    // optionally read header
    if (!std::getline(ifs, line)) return 0;
    if (line.rfind("CANVAS", 0) == 0) {
        // read dims
        if (!std::getline(ifs, line)) return 0;
        std::istringstream sh(line);
        int w,h,cbh; sh >> w >> h >> cbh;
        c->width = w; c->height = h; c->control_bar_h = cbh;
    } else {
        // no header; reset stream to beginning
        ifs.clear(); ifs.seekg(0);
    }

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "MODULE") {
            GP_CanvasModuleDesc m{};
            ss >> m.x >> m.y >> m.w >> m.h >> m.left_contacts >> m.right_contacts;
            // remainder of line is label (may be empty)
            std::string rest;
            std::getline(ss, rest);
            // trim leading spaces
            if (!rest.empty() && rest[0] == ' ') rest.erase(0,1);
            // unescape backslashes and \n
            std::string lbl; lbl.reserve(rest.size());
            for (size_t i = 0; i < rest.size(); ++i) {
                char ch = rest[i];
                if (ch == '\\' && i + 1 < rest.size()) {
                    char nx = rest[i+1];
                    if (nx == 'n') { lbl.push_back('\n'); ++i; }
                    else { lbl.push_back(nx); ++i; }
                } else lbl.push_back(ch);
            }
            // copy into fixed label buffer
            std::memset(m.label, 0, sizeof(m.label));
            std::memcpy(m.label, lbl.c_str(), std::min<size_t>(lbl.size(), sizeof(m.label)-1));
            // append module
            c->modules.push_back(m);
            c->module_tables.push_back(nullptr);
            c->module_table_owned.push_back(0);
            // assign node id for this module
            int nid = c->next_node_id++;
            c->module_node_id.push_back(nid);
            GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = static_cast<int>(c->modules.size() - 1);
            c->nodes.push_back(std::move(nc));
        } else if (tag == "EDGE") {
            GP_CanvasEdgeDesc e{}; int type_id = 0;
            ss >> e.a_module >> e.a_contact_idx >> e.b_module >> e.b_contact_idx >> type_id;
            GP_CanvasContextImpl::EdgeInfo ei; ei.desc = e; ei.type_id = type_id;
            if (!c->hues.empty()) { ei.hues = c->hues; ei.hue_intensity = c->hue_intensity; }
            c->edges.push_back(std::move(ei));
        } else if (tag == "NODE") {
            int nid = -1; int midx = -1; ss >> nid >> midx;
            GP_CanvasContextImpl::NodeContract *found = nullptr;
            for (auto &n : c->nodes) if (n.node_id == nid) { found = &n; break; }
            if (!found) { GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = midx; c->nodes.push_back(std::move(nc)); found = &c->nodes.back(); }
            int in_count = 0; ss >> in_count;
            for (int i = 0; i < in_count; ++i) { int t; ss >> t; found->input_types.push_back(t); }
            int out_count = 0; ss >> out_count;
            for (int i = 0; i < out_count; ++i) { int t; ss >> t; found->output_types.push_back(t); }
        }
    }
    // done
    return 1;
}

extern "C" int gp_canvas_set_cable_style(GP_CanvasContext* ctx_, int jacket_px, int jacket_border) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->jacket_px = std::max(1, jacket_px);
    c->jacket_border = std::max(0, jacket_border);
    return 1;
}

extern "C" int gp_canvas_set_edge_hues(GP_CanvasContext* ctx_, const float* hues, int hue_count, float hue_intensity) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->hues.clear();
    c->hue_intensity = std::clamp(hue_intensity, 0.0f, 1.0f);
    if (!hues || hue_count <= 0) return 1;
    c->hues.assign(hues, hues + hue_count);
    // propagate to existing edges as a default palette (doesn't override per-edge custom hues)
    for (auto &ei : c->edges) {
        if (ei.hues.empty()) {
            ei.hues = c->hues;
            ei.hue_intensity = c->hue_intensity;
        }
    }
    return 1;
}

extern "C" int gp_canvas_raster_rgba(GP_CanvasContext* ctx_, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!ctx_ || !out_rgba) return 0;
    auto *ctx = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int w = ctx->width;
    int h = ctx->height;
    int pitch = w * 4;
    if (out_len_bytes < w * h * 4) return 0;

    // clear
    memset(out_rgba, 0, static_cast<size_t>(w) * h * 4);
    // draw control bar at top with toggle tool buttons
    int cbh = ctx->control_bar_h;
    if (cbh > 0) {
        memset_rect(out_rgba, w, h, pitch, 0, 0, w, cbh, Color{28,28,34,255});
        // two tool groups: canvas (left) and table (right)
        const int canvas_btn_count = 3;
        const int table_btn_count = 3;
        const int spacing = 8;
        int bh = std::max(4, cbh - 8);
        int bw = bh; // square buttons
        // left (canvas) group
        int bx = 8; int by = 4;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            Color fill = (ctx->selected_tool_canvas == bi) ? Color{90,90,110,255} : Color{60,60,72,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            // left/right border
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
        }
        // right (table) group
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, w - 8 - group_width);
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            Color fill = (ctx->selected_tool_table == bi) ? Color{90,70,90,255} : Color{60,50,60,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
        }
        // IO counts (inputs / outputs) shown to the left of the table buttons
        auto draw_io_group = [&](int base_x, int byy, int in_count, const char* label) {
            // minus box, number area, plus box
            int nbw = bw;
            int num_w = std::max(24, nbw * 2);
            int gap = 10;
            int bx_minus = base_x - (nbw + gap + num_w + gap + nbw);
            int bx_num = bx_minus + nbw + gap;
            int bx_plus = bx_num + num_w + gap;
            // minus
            memset_rect(out_rgba, w, h, pitch, bx_minus, byy, nbw, bh, Color{50,50,56,255});
            // number background
            memset_rect(out_rgba, w, h, pitch, bx_num, byy, num_w, bh, Color{36,36,42,255});
            // plus
            memset_rect(out_rgba, w, h, pitch, bx_plus, byy, nbw, bh, Color{50,50,56,255});
            // render number text centered
            std::string s = std::to_string(in_count);
            auto bm = render_text_to_rgba(s, 1.0f, {255,255,255,255});
            if (!bm.pixels.empty()) {
                int tx = bx_num + (num_w - bm.width) / 2;
                int ty = byy + (bh - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            // tiny label (not required) is skipped for compactness
        };
        // compute left of table buttons start for groups placement
        int io_base_x = bx_r; // place IO groups to the left of the table buttons
        int io_by = by;
        // inputs group
        int focused = ctx->focused_module;
        int in_count = 0, out_count = 0;
        if (focused >= 0 && focused < static_cast<int>(ctx->module_io_in_count.size())) in_count = ctx->module_io_in_count[focused];
        if (focused >= 0 && focused < static_cast<int>(ctx->module_io_out_count.size())) out_count = ctx->module_io_out_count[focused];
        // outputs group on the left, inputs group to the right with extra separation
        draw_io_group(io_base_x -  (bw + spacing + 80), io_by, out_count, "out");
        draw_io_group(io_base_x, io_by, in_count, "in");
    }

    // Prepare storage for per-module table hitboxes discovered during table rendering.
    std::vector<std::vector<GP_TableHitBox>> module_hitboxes(ctx->modules.size());

    // draw modules
    for (const auto &m : ctx->modules) {
        Color bg{40,40,50,255};
        memset_rect(out_rgba, w, h, pitch, m.x, m.y, m.w, m.h, bg);
        // draw contacts — prefer counts from attached table IO keys when available
        int left_count = m.left_contacts;
        int right_count = m.right_contacts;
        int mi_idx_for_counts = static_cast<int>(&m - ctx->modules.data());
        GP_TableContext* tt = nullptr;
        if (mi_idx_for_counts >= 0 && mi_idx_for_counts < static_cast<int>(ctx->module_tables.size())) {
            tt = ctx->module_tables[mi_idx_for_counts];
            if (tt) {
                int in_ct = 0, out_ct = 0; get_table_io_counts(tt, in_ct, out_ct);
                if (in_ct > 0) left_count = in_ct;
                if (out_ct > 0) right_count = out_ct;
            }
        }
        // if this module is focused, draw a thin blue border outside its rect
        int mi_idx = static_cast<int>(&m - ctx->modules.data());
        if (mi_idx == ctx->focused_module) {
            Color fb{60,120,220,255};
            int t = 2; // thickness
            // top and bottom bands
            for (int dy = 1; dy <= t; ++dy) {
                int ytop = m.y - dy;
                int ybot = m.y + m.h - 1 + dy;
                if (ytop >= 0 && ytop < h) memset_rect(out_rgba, w, h, pitch, std::max(0, m.x - dy), ytop, std::min(w, m.w + 2*dy), 1, fb);
                if (ybot >= 0 && ybot < h) memset_rect(out_rgba, w, h, pitch, std::max(0, m.x - dy), ybot, std::min(w, m.w + 2*dy), 1, fb);
            }
            // left and right bands
            for (int dx = 1; dx <= t; ++dx) {
                int lx = m.x - dx;
                int rx = m.x + m.w - 1 + dx;
                if (lx >= 0 && lx < w) memset_rect(out_rgba, w, h, pitch, lx, std::max(0, m.y - t), 1, std::min(h, m.h + 2*t), fb);
                if (rx >= 0 && rx < w) memset_rect(out_rgba, w, h, pitch, rx, std::max(0, m.y - t), 1, std::min(h, m.h + 2*t), fb);
            }
        }
        // If module has attached table, sync its IO layout and render it into the module rect
        int mi = static_cast<int>(&m - ctx->modules.data());
        if (mi >= 0 && mi < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[mi]) {
            GP_TableContext* t = ctx->module_tables[mi];
            // ensure the table reflects any UI-specified input/output counts
            sync_module_table_io_layout(ctx, mi);
            // render table into a temporary buffer sized to module
            int tw = std::max(1, m.w);
            int th = std::max(1, m.h);
            std::vector<uint8_t> tmp(static_cast<size_t>(tw) * th * 4);
            GP_TableGeom geom{};
            gp_table_get_geom(t, &geom);
            // Adjust geom to the module size so renderer writes into our tmp buffer
            geom.width_px = tw;
            geom.height_px = th;
            // render table into tmp using table renderer and capture hitboxes
            const int hitcap = 4096;
            std::vector<GP_TableHitBox> hits(hitcap);
            int hits_written = 0;
            gp_table_render_rgba_with_state(t, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &hits_written);
            if (hits_written > 0) {
                module_hitboxes[mi].assign(hits.begin(), hits.begin() + hits_written);
            }
            // blit tmp into out_rgba at module.x,module.y (clipping)
            for (int yy = 0; yy < th; ++yy) {
                int dst_y = m.y + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                uint8_t* dst_row = out_rgba + dst_y * pitch + m.x * 4;
                uint8_t* src_row = tmp.data() + yy * tw * 4;
                int copy_w = std::min(tw, std::max(0, w - m.x));
                if (copy_w <= 0) continue;
                std::memcpy(dst_row, src_row, static_cast<size_t>(copy_w) * 4);
            }
        }
    }

    // ensure rope sim exists
    if (!ctx->rope_sim) ctx->rope_sim = rope_sim_create(1024, 64);

    // update/create ropes for edges
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        const auto &edge = ctx->edges[ei].desc;
        if (edge.a_module < 0 || edge.a_module >= static_cast<int>(ctx->modules.size())) continue;
        if (edge.b_module < 0 || edge.b_module >= static_cast<int>(ctx->modules.size())) continue;
        int ax, ay, bx, by;
        // compute contact positions. If the endpoint module has an attached table
        // and we captured hitboxes during rendering, prefer the table-provided
        // hitbox center for exact LED anchor coordinates. Fall back to legacy
        // computed positions otherwise.
        const GP_CanvasModuleDesc &ma = ctx->modules[edge.a_module];
        const GP_CanvasModuleDesc &mb = ctx->modules[edge.b_module];
        bool resolvedA = false, resolvedB = false;
        if (edge.a_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.a_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.a_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && hb.aux0 == edge.a_contact_idx) {
                    ax = ctx->modules[edge.a_module].x + (hb.x0 + hb.x1) / 2;
                    ay = ctx->modules[edge.a_module].y + (hb.y0 + hb.y1) / 2;
                    resolvedA = true; break;
                }
            }
        }
        if (edge.b_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.b_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.b_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && hb.aux0 == edge.b_contact_idx) {
                    bx = ctx->modules[edge.b_module].x + (hb.x0 + hb.x1) / 2;
                    by = ctx->modules[edge.b_module].y + (hb.y0 + hb.y1) / 2;
                    resolvedB = true; break;
                }
            }
        }
        if (!resolvedA) {
            int a_left_count = ma.left_contacts;
            if (edge.a_module < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[edge.a_module]) {
                int in_ct=0, out_ct=0; get_table_io_counts(ctx->module_tables[edge.a_module], in_ct, out_ct);
                if (in_ct > 0) a_left_count = in_ct;
            }
            compute_contact_pos_with_count(ma, true, edge.a_contact_idx, a_left_count, ax, ay);
        }
        if (!resolvedB) {
            int b_right_count = mb.right_contacts;
            if (edge.b_module < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[edge.b_module]) {
                int in_ct=0, out_ct=0; get_table_io_counts(ctx->module_tables[edge.b_module], in_ct, out_ct);
                if (out_ct > 0) b_right_count = out_ct;
            }
            compute_contact_pos_with_count(mb, false, edge.b_contact_idx, b_right_count, bx, by);
        }
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) {
            int segs = 8;
            float slack = 0.0f;
            int newr = rope_sim_add_rope(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack);
            ctx->edges[ei].rope_idx = newr;
        } else {
            rope_sim_move_endpoints(ctx->rope_sim, ridx, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by));
        }
    }

    // step sim
    rope_sim_step(ctx->rope_sim, 1.0f/60.0f, 800.0f, 8, 0.86f);

    // render edges using the table spline drawer for exact Catmull-Rom appearance
    extern void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity);
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) continue;
        int vc = rope_sim_get_vertex_count(ctx->rope_sim, ridx);
        if (vc < 2) continue;
        std::vector<float> verts(static_cast<size_t>(vc) * 2);
        int got = rope_sim_get_vertices(ctx->rope_sim, ridx, verts.data(), static_cast<int>(verts.size()));
        if (got <= 0) continue;
        // use configured jacket/core sizing and per-edge hue array if present
        int jacket_px = ctx->jacket_px;
        int jacket_border = ctx->jacket_border;
        const float* hues_ptr = ctx->edges[ei].hues.empty() ? nullptr : ctx->edges[ei].hues.data();
        int hue_count = static_cast<int>(ctx->edges[ei].hues.size());
        int samples_per_segment = 3;
        table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts.data(), got, jacket_px, jacket_border, hues_ptr, hue_count, samples_per_segment, ctx->edges[ei].hue_intensity);
    }

    // render provisional prospective rope (follows mouse) if present
    if (ctx->prospective_rope_idx >= 0) {
        int ridx = ctx->prospective_rope_idx;
        int vc = rope_sim_get_vertex_count(ctx->rope_sim, ridx);
        if (vc >= 2) {
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = rope_sim_get_vertices(ctx->rope_sim, ridx, verts.data(), static_cast<int>(verts.size()));
            if (got > 0) {
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                const float* hues_ptr = ctx->hues.empty() ? nullptr : ctx->hues.data();
                int hue_count = static_cast<int>(ctx->hues.size());
                int samples_per_segment = 3;
                table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts.data(), got, jacket_px, jacket_border, hues_ptr, hue_count, samples_per_segment, ctx->hue_intensity);
            }
        }
    }

    return 1;
}
