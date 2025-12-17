// Implementation of simple text->RGBA bitmap helper using stb_easy_font.h

#include "text_render_helper.h"
#include <cstring>
#include <cmath>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4201)
#endif

#include "stb_easy_font.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static inline void blend_pixel(unsigned char *dst, const unsigned char *src)
{
    // src and dst are RGBA bytes. alpha blending: out = src + dst*(1-src_a)
    float sa = src[3] / 255.0f;
    float ida = 1.0f - sa;
    for (int i = 0; i < 3; ++i) {
        float s = src[i] / 255.0f;
        float d = dst[i] / 255.0f;
        float out = s * sa + d * (1.0f - sa);
        dst[i] = (unsigned char)lrintf(out * 255.0f);
    }
    float outa = sa + dst[3] / 255.0f * ida;
    dst[3] = (unsigned char)lrintf(outa * 255.0f);
}

TextBitmap render_text_to_rgba(const std::string &text, float scale, const std::array<unsigned char,4> &color)
{
    TextBitmap out;
    if (text.empty()) return out;

    // stb_easy_font expects mutable char*
    std::string s = text;
    int w = stb_easy_font_width((char*)s.c_str());
    int h = stb_easy_font_height((char*)s.c_str());
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;

    int iw = (int)std::ceil(w * scale) + 2;
    int ih = (int)std::ceil(h * scale) + 2;
    out.width = iw;
    out.height = ih;
    out.text = text;
    out.pixels.assign(iw * ih * 4, 0);

    // large enough vertex buffer (bytes)
    const int VBUF_BYTES = 65536;
    std::vector<char> vbuf(VBUF_BYTES);

    unsigned char col[4] = { color[0], color[1], color[2], color[3] };

    int num_quads = stb_easy_font_print(0.f, 0.f, (char*)s.c_str(), col, vbuf.data(), (int)vbuf.size());

    // Each quad is 4 vertices, each vertex is (float x,y,z) + color[4] --> 16 bytes per vertex
    // quad size = 4 * 16 = 64 bytes, as stb_easy_font uses
    const int VERT_STRIDE = 16;
    const int QUAD_STRIDE = 64;

    for (int q = 0; q < num_quads; ++q) {
        int base = q * QUAD_STRIDE;
        float xs[4], ys[4];
        unsigned char vcol[4];
        for (int v = 0; v < 4; ++v) {
            int off = base + v * VERT_STRIDE;
            float fx = * (float *) (vbuf.data() + off + 0);
            float fy = * (float *) (vbuf.data() + off + 4);
            xs[v] = fx;
            ys[v] = fy;
            // color is same per vertex, but we load first vertex
            if (v == 0) {
                std::memcpy(vcol, vbuf.data() + off + 12, 4);
            }
        }

        float minx = xs[0], maxx = xs[0], miny = ys[0], maxy = ys[0];
        for (int i = 1; i < 4; ++i) {
            if (xs[i] < minx) minx = xs[i];
            if (xs[i] > maxx) maxx = xs[i];
            if (ys[i] < miny) miny = ys[i];
            if (ys[i] > maxy) maxy = ys[i];
        }

        int x0 = (int)std::floor(minx * scale);
        int x1 = (int)std::ceil (maxx * scale);
        int y0 = (int)std::floor(miny * scale);
        int y1 = (int)std::ceil (maxy * scale);

        // clamp
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > iw) x1 = iw; if (y1 > ih) y1 = ih;

        unsigned char src_pixel[4] = { vcol[0], vcol[1], vcol[2], vcol[3] };

        for (int yy = y0; yy < y1; ++yy) {
            for (int xx = x0; xx < x1; ++xx) {
                unsigned char *dst = &out.pixels[(yy * iw + xx) * 4];
                // write with alpha blending
                blend_pixel(dst, src_pixel);
            }
        }
    }

    return out;
}

std::vector<TextBitmap> create_text_bitmaps(const std::vector<std::string> &texts, float scale, const std::array<unsigned char,4> &color)
{
    std::vector<TextBitmap> result;
    result.reserve(texts.size());
    for (const auto &t : texts) {
        result.push_back(render_text_to_rgba(t, scale, color));
    }
    return result;
}
