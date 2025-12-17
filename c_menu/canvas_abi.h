// Minimal canvas API for placing modules and drawing droopy rope edges between contacts
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GP_CanvasContext GP_CanvasContext;
typedef struct GP_TableContext GP_TableContext; // forward from table_abi

// Module descriptor
typedef struct {
    int32_t x, y; // top-left
    int32_t w, h;
    int32_t left_contacts;  // number of contacts down left side
    int32_t right_contacts; // number down right side
    char label[64];
} GP_CanvasModuleDesc;

// Edge reference: module index and contact index (0..n-1)
typedef struct {
    int32_t a_module;
    int32_t a_contact_idx; // 0 = topmost
    int32_t b_module;
    int32_t b_contact_idx;
} GP_CanvasEdgeDesc;

GP_CanvasContext* gp_canvas_create(int width, int height);
void gp_canvas_destroy(GP_CanvasContext* ctx);

// add module, returns module index or -1
int gp_canvas_add_module(GP_CanvasContext* ctx, const GP_CanvasModuleDesc* desc);
// move module
int gp_canvas_move_module(GP_CanvasContext* ctx, int module_idx, int x, int y);

// add edge, returns edge index or -1
int gp_canvas_add_edge(GP_CanvasContext* ctx, const GP_CanvasEdgeDesc* desc);

// rasterize into RGBA buffer sized width*height*4; returns 1 on success
int gp_canvas_raster_rgba(GP_CanvasContext* ctx, uint8_t* out_rgba, int32_t out_len_bytes);

// Set cable rendering style for this canvas (jacket radius and inner border)
int gp_canvas_set_cable_style(GP_CanvasContext* ctx, int jacket_px, int jacket_border);

// Set global hue array for colored core rendering. `hues` is an array of float in [0..1].
// The function copies the hue data internally. Passing nullptr clears hues.
int gp_canvas_set_edge_hues(GP_CanvasContext* ctx, const float* hues, int hue_count, float hue_intensity);

// Forward a mouse click (canvas-local coords). Returns 1 if handled.
int gp_canvas_on_click(GP_CanvasContext* ctx, int x, int y);

// Mouse drag handlers to support click-and-dragging modules.
int gp_canvas_on_mouse_down(GP_CanvasContext* ctx, int x, int y);
int gp_canvas_on_mouse_move(GP_CanvasContext* ctx, int x, int y);
int gp_canvas_on_mouse_up(GP_CanvasContext* ctx, int x, int y);

// Create/destroy a canvas-owned table attached to module. The created table
// will be owned by the canvas and destroyed when detached or when canvas
// is destroyed. Returns 1 on success.
int gp_canvas_create_table(GP_CanvasContext* ctx, int module_idx);
int gp_canvas_destroy_table(GP_CanvasContext* ctx, int module_idx);

// Step the canvas' internal simulation by dt seconds. Returns 1 on success.
int gp_canvas_step(GP_CanvasContext* ctx, float dt);

// Return the internal RopeSim pointer for this canvas (or NULL if none).
// Useful to attach table contexts to the canvas-owned simulator so tables defer
// simulation to the canvas. The returned pointer is an opaque `RopeSim*`.
void* gp_canvas_get_rope_sim(GP_CanvasContext* ctx);

// Attach a `GP_TableContext` to a canvas module so the canvas will render
// the table inside the module rectangle and forward clicks. `take_ownership`
// indicates whether the canvas should destroy the table when the module is
// removed/destroyed (1 = canvas destroys it). Returns 1 on success.
int gp_canvas_attach_table(GP_CanvasContext* ctx, int module_idx, GP_TableContext* table, int take_ownership);

// Detach any table from the given module. If the canvas owned the table,
// it will destroy it. Returns 1 on success.
int gp_canvas_detach_table(GP_CanvasContext* ctx, int module_idx);

// Register a host window pointer with the canvas so the canvas can retain
// references to windows it will handle (opaque pointer). Returns 1 on success.
int gp_canvas_register_window(GP_CanvasContext* ctx, void* window_ptr);
int gp_canvas_unregister_window(GP_CanvasContext* ctx, void* window_ptr);

// Get the backing graph node id for a registered window pointer, or -1 if none.
int gp_canvas_get_window_node_id(GP_CanvasContext* ctx, void* window_ptr);

// Typed-edge API: add an edge with an explicit type id (0 == untyped/wildcard).
int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx, const GP_CanvasEdgeDesc* desc, int type_id);

// Set the module's supported input/output type lists. Each type id is an int.
int gp_canvas_set_module_io_types(GP_CanvasContext* ctx, int module_idx, const int* input_types, int input_count, const int* output_types, int output_count);

// Return the backing graph node id for a module, or -1 if none.
int gp_canvas_get_module_node_id(GP_CanvasContext* ctx, int module_idx);

// Persist/restore canvas state to a simple text file. Returns 1 on success.
int gp_canvas_save_to_file(GP_CanvasContext* ctx, const char* path);
int gp_canvas_load_from_file(GP_CanvasContext* ctx, const char* path);

// Templates directory: set a global library directory for templates (used by
// table template APIs if they are invoked with dir==NULL).
int gp_canvas_set_templates_dir(const char* dir);
int gp_canvas_get_templates_dir(char* out_buf, int out_len);

// Per-canvas autosave: set autosave path and interval in seconds. If path is
// NULL or empty, autosave is disabled. Autosave will trigger during
// `gp_canvas_step` when enough time has accumulated.
int gp_canvas_set_autosave(GP_CanvasContext* ctx, const char* path, double interval_s);
int gp_canvas_get_autosave(GP_CanvasContext* ctx, char* out_path, int out_len, double* out_interval_s);

#ifdef __cplusplus
}
#endif
