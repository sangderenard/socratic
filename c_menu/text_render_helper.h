// Simple text rendering helper using stb_easy_font.h
// Provides functions to render ASCII text into an RGBA bitmap (transparent background)
// and a bulk helper to create multiple bitmaps (string sets) for stateful work.

#pragma once

#include <string>
#include <vector>
#include <array>

struct TextBitmap {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels; // RGBA, row-major
    std::string text;
};

// Render one ASCII string to an RGBA bitmap. The returned TextBitmap.pixels will
// be width*height*4 bytes (RGBA). Transparent background where alpha==0.
// color: {r,g,b,a} 0-255. scale: scale factor for the font (1.0 == native stb pixels).
TextBitmap render_text_to_rgba(const std::string &text,
                               float scale = 1.0f,
                               const std::array<unsigned char,4> &color = {255,255,255,255});

// Bulk create a set of TextBitmap objects from multiple strings.
std::vector<TextBitmap> create_text_bitmaps(const std::vector<std::string> &texts,
                                            float scale = 1.0f,
                                            const std::array<unsigned char,4> &color = {255,255,255,255});
