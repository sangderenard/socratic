#pragma once

#include <cstdint>
#include <vector>

// Internal C++ implementation. Exposed to C via menu_waveform_abi.h.

namespace gp_menu {

struct WaveSample {
    std::uint64_t t_ns = 0;
    float v = 0.0f;
};

class MenuWaveform {
public:
    MenuWaveform(int width_px, int height_px, int history_len);

    void clear();
    void push(std::uint64_t t_ns, float v);

    // Writes RGBA8 pixels. Returns true on success.
    bool raster_rgba(std::uint8_t* out_rgba, int out_len_bytes) const;

    int width_px() const { return m_w; }
    int height_px() const { return m_h; }

private:
    int m_w = 0;
    int m_h = 0;

    // Ring buffer of samples.
    std::vector<WaveSample> m_ring;
    std::uint32_t m_head = 0;
    bool m_filled = false;
};

} // namespace gp_menu
