#include "menu_waveform.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gp_menu {

static inline float clampf(float x, float lo, float hi) {
    return std::min(hi, std::max(lo, x));
}

MenuWaveform::MenuWaveform(int width_px, int height_px, int history_len)
    : m_w(std::max(1, width_px)),
      m_h(std::max(1, height_px)) {
    const int n = std::max(1, history_len);
    m_ring.resize(static_cast<std::size_t>(n));
    clear();
}

void MenuWaveform::clear() {
    for (auto& s : m_ring) {
        s.t_ns = 0;
        s.v = 0.0f;
    }
    m_head = 0;
    m_filled = false;
}

void MenuWaveform::push(std::uint64_t t_ns, float v) {
    if (m_ring.empty()) return;
    m_ring[m_head] = WaveSample{t_ns, v};
    m_head = (m_head + 1u) % static_cast<std::uint32_t>(m_ring.size());
    if (m_head == 0) m_filled = true;
}

bool MenuWaveform::raster_rgba(std::uint8_t* out_rgba, int out_len_bytes) const {
    if (!out_rgba) return false;
    const int need = m_w * m_h * 4;
    if (out_len_bytes < need) return false;

    // Clear background: opaque black.
    std::memset(out_rgba, 0, static_cast<std::size_t>(need));
    for (int i = 0; i < m_w * m_h; ++i) {
        out_rgba[i * 4 + 3] = 255;
    }

    if (m_ring.empty()) return true;

    // Render latest samples right-to-left across width.
    // Sample index 0 => rightmost pixel column.
    const int n = static_cast<int>(m_ring.size());
    const int cols = m_w;

    auto sample_at = [&](int k) -> float {
        // k=0 => most recent.
        const std::uint32_t head = m_head;
        int idx = static_cast<int>(head) - 1 - k;
        while (idx < 0) idx += n;
        idx %= n;
        return m_ring[static_cast<std::size_t>(idx)].v;
    };

    for (int x = 0; x < cols; ++x) {
        const float v = sample_at(x);
        const float v01 = 0.5f * (clampf(v, -1.0f, 1.0f) + 1.0f); // [-1,1] -> [0,1]
        const int y = static_cast<int>(std::lround((1.0f - v01) * float(m_h - 1)));

        // Draw a single bright pixel (white) for now.
        const int xi = (m_w - 1) - x;
        if (xi < 0 || xi >= m_w) continue;
        if (y < 0 || y >= m_h) continue;
        const int off = (y * m_w + xi) * 4;
        out_rgba[off + 0] = 255;
        out_rgba[off + 1] = 255;
        out_rgba[off + 2] = 255;
        out_rgba[off + 3] = 255;
    }

    return true;
}

} // namespace gp_menu
