#include "table_node_groups.h"
#include <unordered_map>
#include <unordered_set>
#include <mutex>

struct NodeGroupState {
    std::unordered_map<unsigned long long, int> node_group;
    std::unordered_set<unsigned long long> allowed_pairs; // packed (from<<32)|to
    std::unordered_set<unsigned long long> disallowed_pairs;
};

static std::unordered_map<GP_TableContext*, NodeGroupState> g_states;
static std::mutex g_states_mutex;

static unsigned long long pack_pair(int a, int b) {
    return (static_cast<unsigned long long>(static_cast<uint32_t>(a)) << 32) | static_cast<unsigned long long>(static_cast<uint32_t>(b));
}

int gp_table_node_group_set(GP_TableContext* ctx, unsigned long long node_key, int group) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    NodeGroupState &s = g_states[ctx];
    s.node_group[node_key] = group;
    return 1;
}

int gp_table_node_group_get(GP_TableContext* ctx, unsigned long long node_key, int* out_group) {
    if (!ctx || !out_group) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    auto itc = g_states.find(ctx);
    if (itc == g_states.end()) return 0;
    auto it = itc->second.node_group.find(node_key);
    if (it == itc->second.node_group.end()) return 0;
    *out_group = it->second;
    return 1;
}

int gp_table_node_group_add_allowed(GP_TableContext* ctx, int from_group, int to_group) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    NodeGroupState &s = g_states[ctx];
    s.allowed_pairs.insert(pack_pair(from_group, to_group));
    return 1;
}

int gp_table_node_group_clear_allowed(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    g_states[ctx].allowed_pairs.clear();
    return 1;
}

int gp_table_node_group_add_disallowed(GP_TableContext* ctx, int from_group, int to_group) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    NodeGroupState &s = g_states[ctx];
    s.disallowed_pairs.insert(pack_pair(from_group, to_group));
    return 1;
}

int gp_table_node_group_clear_disallowed(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    g_states[ctx].disallowed_pairs.clear();
    return 1;
}

int gp_table_node_group_is_edge_allowed(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return -1;
    std::lock_guard<std::mutex> lk(g_states_mutex);
    auto itc = g_states.find(ctx);
    // If no state configured for this context, allow by default
    if (itc == g_states.end()) return 1;
    NodeGroupState &s = itc->second;
    int ga = 0, gb = 0;
    auto ita = s.node_group.find(a);
    if (ita != s.node_group.end()) ga = ita->second;
    auto itb = s.node_group.find(b);
    if (itb != s.node_group.end()) gb = itb->second;
    unsigned long long p = pack_pair(ga, gb);
    if (!s.allowed_pairs.empty()) {
        return s.allowed_pairs.count(p) ? 1 : 0;
    }
    if (s.disallowed_pairs.count(p)) return 0;
    return 1;
}
