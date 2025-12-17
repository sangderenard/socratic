#pragma once
#include "table_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Set the group id for a node (node key format matches selection key used in table code)
int gp_table_node_group_set(GP_TableContext* ctx, unsigned long long node_key, int group);
int gp_table_node_group_get(GP_TableContext* ctx, unsigned long long node_key, int* out_group);

// Manage allowed / disallowed group pairs. Pairs are directional (from_group -> to_group).
int gp_table_node_group_add_allowed(GP_TableContext* ctx, int from_group, int to_group);
int gp_table_node_group_clear_allowed(GP_TableContext* ctx);
int gp_table_node_group_add_disallowed(GP_TableContext* ctx, int from_group, int to_group);
int gp_table_node_group_clear_disallowed(GP_TableContext* ctx);

// Query if an edge between two node keys is allowed according to configured rules.
// Returns 1 if allowed, 0 if not allowed, or -1 on error.
int gp_table_node_group_is_edge_allowed(GP_TableContext* ctx, unsigned long long a, unsigned long long b);

#ifdef __cplusplus
}
#endif
