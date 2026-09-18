// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#include "Renderer.h"

#include "ForceLayout.h"
#include "spacetime/Spacetime.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numbers>
#include <random>
#include <unordered_map>
#include <vector>

namespace tessera {
namespace {

// =====================================================================
// Vec3
// =====================================================================

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 &operator+=(const Vec3 &o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    [[nodiscard]] double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

// =====================================================================
// Rotation helpers
// =====================================================================

Vec3 rotateX(Vec3 v, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}

Vec3 rotateY(Vec3 v, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}

Vec3 rotateZ(Vec3 v, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}

Vec3 applyRotation(Vec3 v, double rx, double ry, double rz) {
    return rotateZ(rotateY(rotateX(v, rx), ry), rz);
}

// =====================================================================
// Color & Image
// =====================================================================

struct Color {
    uint8_t r = 0, g = 0, b = 0;
};

class Image {
public:
    Image(int w, int h, Color bg = {20, 20, 30})
        : w_(w), h_(h), pixels_(static_cast<std::size_t>(w) * h, bg) {}

    void blendPixel(int x, int y, Color c, float a) {
        if (x < 0 || x >= w_ || y < 0 || y >= h_) return;
        auto &dst = pixels_[y * w_ + x];
        dst.r = static_cast<uint8_t>(a * c.r + (1.0f - a) * dst.r);
        dst.g = static_cast<uint8_t>(a * c.g + (1.0f - a) * dst.g);
        dst.b = static_cast<uint8_t>(a * c.b + (1.0f - a) * dst.b);
    }

    void setPixel(int x, int y, Color c) {
        if (x >= 0 && x < w_ && y >= 0 && y < h_)
            pixels_[y * w_ + x] = c;
    }

    void drawLine(int x0, int y0, int x1, int y1, Color c, float alpha = 0.7f) {
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        for (;;) {
            blendPixel(x0, y0, c, alpha);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    void fillCircle(int cx, int cy, int r, Color c) {
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                if (dx * dx + dy * dy <= r * r)
                    setPixel(cx + dx, cy + dy, c);
    }

    void drawRect(int x, int y, int w, int h, Color c) {
        for (int i = x; i < x + w; ++i) {
            setPixel(i, y, c);
            setPixel(i, y + h - 1, c);
        }
        for (int j = y; j < y + h; ++j) {
            setPixel(x, j, c);
            setPixel(x + w - 1, j, c);
        }
    }

    bool writePNG(const std::string &path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;

        // Build raw image data: filter byte (0) + RGB per row
        std::size_t rawRowBytes = 1 + static_cast<std::size_t>(w_) * 3;
        std::vector<uint8_t> raw(rawRowBytes * h_);
        for (int y = 0; y < h_; ++y) {
            raw[y * rawRowBytes] = 0; // filter: None
            for (int x = 0; x < w_; ++x) {
                const auto &p = pixels_[y * w_ + x];
                std::size_t off = y * rawRowBytes + 1 + x * 3;
                raw[off + 0] = p.r;
                raw[off + 1] = p.g;
                raw[off + 2] = p.b;
            }
        }

        // Deflate compress
        uLongf compBound = compressBound(static_cast<uLong>(raw.size()));
        std::vector<uint8_t> comp(compBound);
        int zret = compress2(comp.data(), &compBound, raw.data(),
                             static_cast<uLong>(raw.size()), 6);
        if (zret != Z_OK) return false;
        comp.resize(compBound);

        // PNG signature
        const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        f.write(reinterpret_cast<const char *>(sig), 8);

        // Helper: write a PNG chunk
        auto writeChunk = [&](const char type[4], const uint8_t *data,
                              uint32_t len) {
            uint8_t lenBE[4] = {uint8_t(len >> 24), uint8_t(len >> 16),
                                uint8_t(len >> 8), uint8_t(len)};
            f.write(reinterpret_cast<char *>(lenBE), 4);
            f.write(type, 4);
            if (len > 0)
                f.write(reinterpret_cast<const char *>(data), len);
            // CRC over type + data
            uint32_t crc = crc32(0, reinterpret_cast<const Bytef *>(type), 4);
            if (len > 0)
                crc = crc32(crc, data, len);
            uint8_t crcBE[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16),
                                uint8_t(crc >> 8), uint8_t(crc)};
            f.write(reinterpret_cast<char *>(crcBE), 4);
        };

        // IHDR: width, height, bit depth 8, color type 2 (RGB)
        uint8_t ihdr[13] = {};
        auto w32 = static_cast<uint32_t>(w_);
        auto h32 = static_cast<uint32_t>(h_);
        ihdr[0] = w32 >> 24; ihdr[1] = w32 >> 16;
        ihdr[2] = w32 >> 8;  ihdr[3] = w32;
        ihdr[4] = h32 >> 24; ihdr[5] = h32 >> 16;
        ihdr[6] = h32 >> 8;  ihdr[7] = h32;
        ihdr[8] = 8;   // bit depth
        ihdr[9] = 2;   // color type: RGB
        ihdr[10] = 0;  // compression
        ihdr[11] = 0;  // filter
        ihdr[12] = 0;  // interlace
        writeChunk("IHDR", ihdr, 13);

        // IDAT
        writeChunk("IDAT", comp.data(), static_cast<uint32_t>(comp.size()));

        // IEND
        writeChunk("IEND", nullptr, 0);

        return f.good();
    }

    int width() const { return w_; }
    int height() const { return h_; }
    const std::vector<Color> &pixels() const { return pixels_; }

private:
    int w_, h_;
    std::vector<Color> pixels_;
};

// =====================================================================
// GIF writer (animated GIF89a with LZW compression)
// =====================================================================

// 6x6x6 RGB color cube (216 colors) for GIF quantization
struct GifPalette {
    uint8_t entries[256][3];

    GifPalette() {
        std::memset(entries, 0, sizeof(entries));
        for (int r = 0; r < 6; ++r)
            for (int g = 0; g < 6; ++g)
                for (int b = 0; b < 6; ++b) {
                    int idx = r * 36 + g * 6 + b;
                    entries[idx][0] = static_cast<uint8_t>(r * 51);
                    entries[idx][1] = static_cast<uint8_t>(g * 51);
                    entries[idx][2] = static_cast<uint8_t>(b * 51);
                }
    }

    uint8_t quantize(Color c) const {
        int r = (c.r + 25) / 51;
        int g = (c.g + 25) / 51;
        int b = (c.b + 25) / 51;
        r = std::clamp(r, 0, 5);
        g = std::clamp(g, 0, 5);
        b = std::clamp(b, 0, 5);
        return static_cast<uint8_t>(r * 36 + g * 6 + b);
    }
};

std::vector<uint8_t> lzwEncode(const uint8_t *data, std::size_t len) {
    constexpr int minCodeSize = 8;
    constexpr int clearCode = 256;
    constexpr int eoiCode = 257;

    // Bit buffer
    uint32_t bitBuf = 0;
    int bitCount = 0;
    std::vector<uint8_t> bytes;
    bytes.reserve(len);

    auto emitCode = [&](int code, int nbits) {
        bitBuf |= static_cast<uint32_t>(code) << bitCount;
        bitCount += nbits;
        while (bitCount >= 8) {
            bytes.push_back(static_cast<uint8_t>(bitBuf & 0xFF));
            bitBuf >>= 8;
            bitCount -= 8;
        }
    };

    auto flushBits = [&]() {
        if (bitCount > 0) {
            bytes.push_back(static_cast<uint8_t>(bitBuf & 0xFF));
            bitBuf = 0;
            bitCount = 0;
        }
    };

    std::unordered_map<uint32_t, uint16_t> table;
    table.reserve(4096);
    int codeSize = minCodeSize + 1;
    int nextCode = eoiCode + 1;

    auto resetTable = [&]() {
        table.clear();
        codeSize = minCodeSize + 1;
        nextCode = eoiCode + 1;
    };

    emitCode(clearCode, codeSize);
    resetTable();

    if (len == 0) {
        emitCode(eoiCode, codeSize);
        flushBits();
        return {0};
    }

    int current = data[0];
    for (std::size_t i = 1; i < len; ++i) {
        int pixel = data[i];
        auto key = static_cast<uint32_t>(current) * 256 + pixel;
        auto it = table.find(key);
        if (it != table.end()) {
            current = it->second;
        } else {
            emitCode(current, codeSize);
            table[key] = static_cast<uint16_t>(nextCode);
            if (nextCode < 4096) {
                nextCode++;
                if (nextCode > (1 << codeSize) && codeSize < 12)
                    codeSize++;
            } else {
                emitCode(clearCode, codeSize);
                resetTable();
            }
            current = pixel;
        }
    }
    emitCode(current, codeSize);
    emitCode(eoiCode, codeSize);
    flushBits();

    // Pack into GIF sub-blocks (max 255 bytes each)
    std::vector<uint8_t> result;
    result.reserve(bytes.size() + bytes.size() / 255 + 2);
    std::size_t pos = 0;
    while (pos < bytes.size()) {
        auto blockLen = static_cast<uint8_t>(
            std::min(std::size_t(255), bytes.size() - pos));
        result.push_back(blockLen);
        result.insert(result.end(), bytes.begin() + pos,
                       bytes.begin() + pos + blockLen);
        pos += blockLen;
    }
    result.push_back(0); // block terminator
    return result;
}

void writeLE16(std::ofstream &f, uint16_t v) {
    uint8_t b[2] = {static_cast<uint8_t>(v & 0xFF),
                     static_cast<uint8_t>(v >> 8)};
    f.write(reinterpret_cast<char *>(b), 2);
}

bool writeGif(const std::string &path, int w, int h,
              const std::vector<Image> &frames, int delayCentiseconds) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    GifPalette palette;

    // Header
    f.write("GIF89a", 6);

    // Logical Screen Descriptor
    writeLE16(f, static_cast<uint16_t>(w));
    writeLE16(f, static_cast<uint16_t>(h));
    uint8_t packed = 0xF7; // GCT flag=1, color res=7, sort=0, GCT size=7 (256)
    f.put(static_cast<char>(packed));
    f.put(0); // background color index
    f.put(0); // pixel aspect ratio

    // Global Color Table (256 * 3 bytes)
    f.write(reinterpret_cast<const char *>(palette.entries), 256 * 3);

    // NETSCAPE2.0 Application Extension (infinite loop)
    f.put(0x21);            // extension introducer
    f.put(static_cast<char>(0xFF)); // application extension label
    f.put(0x0B);            // block size
    f.write("NETSCAPE2.0", 11);
    f.put(0x03);            // sub-block size
    f.put(0x01);            // sub-block ID
    writeLE16(f, 0);        // loop count: 0 = infinite
    f.put(0x00);            // block terminator

    // Frames
    for (const auto &frame : frames) {
        // Graphic Control Extension
        f.put(0x21);
        f.put(static_cast<char>(0xF9));
        f.put(0x04);
        f.put(0x00); // disposal=0, no transparency
        writeLE16(f, static_cast<uint16_t>(delayCentiseconds));
        f.put(0x00); // transparent index (unused)
        f.put(0x00); // block terminator

        // Image Descriptor
        f.put(0x2C);
        writeLE16(f, 0); // left
        writeLE16(f, 0); // top
        writeLE16(f, static_cast<uint16_t>(w));
        writeLE16(f, static_cast<uint16_t>(h));
        f.put(0x00); // no local color table

        // Quantize frame to palette indices
        const auto &px = frame.pixels();
        std::vector<uint8_t> indices(px.size());
        for (std::size_t i = 0; i < px.size(); ++i)
            indices[i] = palette.quantize(px[i]);

        // LZW minimum code size + compressed data
        f.put(0x08); // min code size = 8
        auto compressed = lzwEncode(indices.data(), indices.size());
        f.write(reinterpret_cast<const char *>(compressed.data()),
                static_cast<std::streamsize>(compressed.size()));
    }

    // Trailer
    f.put(0x3B);
    return f.good();
}

// =====================================================================
// Layout
// =====================================================================

struct EdgeInfo {
    std::size_t src, tgt;
    bool timelike;
};

struct LayoutData {
    std::vector<Vec3> pos;
    std::vector<EdgeInfo> edges;
};

LayoutData computeLayout(const Spacetime &st, int maxIters) {
    LayoutData layout;

    auto vertList = st.getVertexList();
    auto edgeList = st.getEdgeList();
    if (!vertList || !edgeList) return layout;

    auto verts = vertList->toVector();
    auto edges = edgeList->toVector();
    std::size_t N = verts.size();
    if (N == 0) return layout;

    // Map vertex ID -> index
    std::unordered_map<std::uint64_t, std::size_t> idToIdx;
    idToIdx.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        idToIdx[verts[i]->getId()] = i;

    // Record edges
    for (auto *e : edges) {
        auto si = idToIdx.find(e->getSource()->getId());
        auto ti = idToIdx.find(e->getTarget()->getId());
        if (si != idToIdx.end() && ti != idToIdx.end())
            layout.edges.push_back(
                {si->second, ti->second, e->isTimelike()});
    }

    // Group vertices by time slice
    std::vector<double> times(N);
    double minT = 1e18, maxT = -1e18;
    for (std::size_t i = 0; i < N; ++i) {
        times[i] = verts[i]->getTime();
        minT = std::min(minT, times[i]);
        maxT = std::max(maxT, times[i]);
    }
    double tRange = std::max(maxT - minT, 1.0);

    std::unordered_map<int, std::vector<std::size_t>> slices;
    for (std::size_t i = 0; i < N; ++i)
        slices[static_cast<int>(std::round(times[i]))].push_back(i);

    // Initialize: circle per time slice, y = normalized time
    layout.pos.resize(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.05, 0.05);

    for (auto &[t, indices] : slices) {
        double normT = (static_cast<double>(t) - minT) / tRange;
        auto n = indices.size();
        double radius = std::sqrt(static_cast<double>(n)) * 0.3;
        for (std::size_t k = 0; k < n; ++k) {
            double angle =
                2.0 * std::numbers::pi * k / std::max(n, std::size_t(1));
            layout.pos[indices[k]] = {
                radius * std::cos(angle) + jitter(rng), normT * 10.0,
                radius * std::sin(angle) + jitter(rng)};
        }
    }

    // Adaptive iteration count for large spacetimes
    int iters = maxIters;
    if (N > 20000) iters = std::min(iters, 20);
    else if (N > 5000) iters = std::min(iters, 50);
    else if (N > 1000) iters = std::min(iters, 200);

    // Solve the spatial (x, z) coordinates with the shared 2D layout engine:
    // each vertex's time slice is a repulsion group, so nodes spread only
    // against others in the same slice; the y/time coordinate stays fixed.
    std::vector<std::pair<int, int>> layoutEdges;
    layoutEdges.reserve(layout.edges.size());
    for (const auto &e : layout.edges)
        layoutEdges.emplace_back(static_cast<int>(e.src),
                                 static_cast<int>(e.tgt));

    std::vector<int> groups(N);
    std::vector<double> initPos(N * 2);
    for (std::size_t i = 0; i < N; ++i) {
        groups[i] = static_cast<int>(std::round(times[i]));
        initPos[i * 2]     = layout.pos[i].x;
        initPos[i * 2 + 1] = layout.pos[i].z;
    }

    auto solved = ForceLayout::layout2D(
        static_cast<int>(N), layoutEdges, /*targetRadii=*/{}, groups,
        /*centerIdx=*/-1, initPos, /*restLengths=*/{},
        /*springK=*/0.01, /*repulsionK=*/0.5, iters, /*cooling=*/0.995,
        /*repulsionCap=*/200, /*initialStep=*/0.5);

    for (std::size_t i = 0; i < N; ++i) {
        layout.pos[i].x = solved[i * 2];
        layout.pos[i].z = solved[i * 2 + 1];
    }

    return layout;
}

// =====================================================================
// Panel rendering
// =====================================================================

// Compute the maximum projected bounding box across all rotations in
// the GIF animation path, so every frame uses the same scale/centering.
struct BBox {
    double xmin, xmax, ymin, ymax;
};

BBox computeGifBBox(const LayoutData &layout, int nFrames,
                    double tiltRad, int spin, int precession) {
    BBox bb = {1e18, -1e18, 1e18, -1e18};
    for (int i = 0; i < nFrames; ++i) {
        double t = static_cast<double>(i) / nFrames;
        double ry = 2.0 * std::numbers::pi * spin * t;
        double rx = tiltRad * std::cos(2.0 * std::numbers::pi * precession * t);
        double rz = tiltRad * std::sin(2.0 * std::numbers::pi * precession * t);
        for (const auto &pos : layout.pos) {
            Vec3 r = applyRotation(pos, rx, ry, rz);
            bb.xmin = std::min(bb.xmin, r.x);
            bb.xmax = std::max(bb.xmax, r.x);
            bb.ymin = std::min(bb.ymin, r.y);
            bb.ymax = std::max(bb.ymax, r.y);
        }
    }
    return bb;
}

void renderPanel(Image &img, int ox, int oy, int pw, int ph,
                 const LayoutData &layout, double rx, double ry,
                 double rz, const BBox *fixedBB = nullptr) {
    if (layout.pos.empty()) return;

    // Rotate all positions
    std::vector<Vec3> rotated(layout.pos.size());
    for (std::size_t i = 0; i < layout.pos.size(); ++i)
        rotated[i] = applyRotation(layout.pos[i], rx, ry, rz);

    double xmin, xmax, ymin, ymax;
    if (fixedBB) {
        xmin = fixedBB->xmin;
        xmax = fixedBB->xmax;
        ymin = fixedBB->ymin;
        ymax = fixedBB->ymax;
    } else {
        xmin = 1e18; xmax = -1e18; ymin = 1e18; ymax = -1e18;
        for (const auto &p : rotated) {
            xmin = std::min(xmin, p.x);
            xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y);
            ymax = std::max(ymax, p.y);
        }
    }

    double xRange = std::max(xmax - xmin, 1e-6);
    double yRange = std::max(ymax - ymin, 1e-6);

    int margin = pw / 15;
    int drawW = pw - 2 * margin;
    int drawH = ph - 2 * margin;

    // Uniform scale to preserve aspect ratio
    double scale = std::min(drawW / xRange, drawH / yRange);
    double cx = (xmin + xmax) / 2.0;
    double cy = (ymin + ymax) / 2.0;

    auto toPixelX = [&](double x) -> int {
        return ox + pw / 2 + static_cast<int>((x - cx) * scale);
    };
    auto toPixelY = [&](double y) -> int {
        return oy + ph / 2 - static_cast<int>((y - cy) * scale);
    };

    // Draw edges
    Color timelikeColor = {80, 130, 255};
    Color spacelikeColor = {255, 100, 80};
    float edgeAlpha =
        layout.edges.size() > 5000 ? 0.25f
        : layout.edges.size() > 1000 ? 0.4f : 0.6f;

    for (const auto &e : layout.edges) {
        int x0 = toPixelX(rotated[e.src].x);
        int y0 = toPixelY(rotated[e.src].y);
        int x1 = toPixelX(rotated[e.tgt].x);
        int y1 = toPixelY(rotated[e.tgt].y);
        Color c = e.timelike ? timelikeColor : spacelikeColor;
        img.drawLine(x0, y0, x1, y1, c, edgeAlpha);
    }

    // Draw vertices
    Color vertexColor = {240, 240, 240};
    int r = std::max(1, std::min(3, pw / 300));
    for (const auto &p : rotated) {
        int px = toPixelX(p.x);
        int py_ = toPixelY(p.y);
        img.fillCircle(px, py_, r, vertexColor);
    }

    // Orientation gizmo (bottom-right corner)
    int gx = ox + pw - margin;
    int gy = oy + ph - margin;
    int gLen = margin * 2 / 3;
    Vec3 ax = applyRotation({1, 0, 0}, rx, ry, rz) * gLen;
    Vec3 ay = applyRotation({0, 1, 0}, rx, ry, rz) * gLen;
    Vec3 az = applyRotation({0, 0, 1}, rx, ry, rz) * gLen;
    img.drawLine(gx, gy, gx + int(ax.x), gy - int(ax.y), {255, 80, 80}, 1.0f);
    img.drawLine(gx, gy, gx + int(ay.x), gy - int(ay.y), {80, 255, 80}, 1.0f);
    img.drawLine(gx, gy, gx + int(az.x), gy - int(az.y), {80, 80, 255}, 1.0f);

    // Panel border
    img.drawRect(ox, oy, pw, ph, {60, 60, 80});
}

// =====================================================================
// Graph export (GraphML / DOT)
// =====================================================================

bool writeGraphML(const Spacetime &st, const std::string &path) {
    auto vertList = st.getVertexList();
    auto edgeList = st.getEdgeList();
    if (!vertList || !edgeList) return false;

    auto verts = vertList->toVector();
    auto edges = edgeList->toVector();

    std::ofstream f(path);
    if (!f) return false;

    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      << "<graphml xmlns=\"http://graphml.graphstudio.org\"\n"
      << "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
      << "         xsi:schemaLocation=\"http://graphml.graphstudio.org\">\n"
      << "  <key id=\"time\" for=\"node\" attr.name=\"time\" "
         "attr.type=\"double\"/>\n"
      << "  <key id=\"degree\" for=\"node\" attr.name=\"degree\" "
         "attr.type=\"int\"/>\n"
      << "  <key id=\"sq_length\" for=\"edge\" attr.name=\"squared_length\" "
         "attr.type=\"double\"/>\n"
      << "  <key id=\"sq_length_im\" for=\"edge\" "
         "attr.name=\"squared_length_im\" attr.type=\"double\"/>\n"
      << "  <key id=\"phase\" for=\"edge\" attr.name=\"phase\" "
         "attr.type=\"double\"/>\n"
      << "  <key id=\"phase_im\" for=\"edge\" attr.name=\"phase_im\" "
         "attr.type=\"double\"/>\n"
      << "  <key id=\"timelike\" for=\"edge\" attr.name=\"timelike\" "
         "attr.type=\"boolean\"/>\n"
      << "  <graph id=\"spacetime\" edgedefault=\"undirected\">\n";

    for (auto *v : verts) {
        f << "    <node id=\"" << v->getId() << "\">\n"
          << "      <data key=\"time\">" << v->getTime() << "</data>\n"
          << "      <data key=\"degree\">" << v->degree() << "</data>\n"
          << "    </node>\n";
    }

    for (std::size_t i = 0; i < edges.size(); ++i) {
        auto *e = edges[i];
        // Full complex l^2 + C* connection phase (re/im); causal character from the canonical
        // Edge::isTimelike(), not the superseded sign-of-Re test. The
        // Re key keeps its name for compatibility with existing consumers.
        const std::complex<double> sq = (e->getLength() * e->getLength());
        f << "    <edge id=\"e" << i << "\" source=\""
          << e->getSource()->getId() << "\" target=\""
          << e->getTarget()->getId() << "\">\n"
          << "      <data key=\"sq_length\">" << sq.real() << "</data>\n"
          << "      <data key=\"sq_length_im\">" << sq.imag() << "</data>\n"
          << "      <data key=\"phase\">" << e->getPhase().real()
          << "</data>\n"
          << "      <data key=\"phase_im\">" << e->getPhase().imag()
          << "</data>\n"
          << "      <data key=\"timelike\">"
          << (e->isTimelike() ? "true" : "false") << "</data>\n"
          << "    </edge>\n";
    }

    f << "  </graph>\n</graphml>\n";
    return f.good();
}

bool writeDot(const Spacetime &st, const std::string &path) {
    auto vertList = st.getVertexList();
    auto edgeList = st.getEdgeList();
    if (!vertList || !edgeList) return false;

    auto verts = vertList->toVector();
    auto edges = edgeList->toVector();

    std::ofstream f(path);
    if (!f) return false;

    f << "graph spacetime {\n"
      << "  node [shape=point];\n";

    for (auto *v : verts)
        f << "  " << v->getId()
          << " [time=" << v->getTime()
          << ", degree=" << v->degree() << "];\n";

    for (auto *e : edges) {
        // Canonical Edge::isTimelike() classification + the full complex l^2
        // and phase; squared_length stays Re for compatibility.
        const std::complex<double> sq = (e->getLength() * e->getLength());
        const bool tl = e->isTimelike();
        f << "  " << e->getSource()->getId()
          << " -- " << e->getTarget()->getId()
          << " [squared_length=" << sq.real()
          << ", squared_length_im=" << sq.imag()
          << ", phase=" << e->getPhase().real()
          << ", phase_im=" << e->getPhase().imag()
          << ", timelike=" << (tl ? "true" : "false")
          << ", color=" << (tl ? "blue" : "red") << "];\n";
    }

    f << "}\n";
    return f.good();
}

} // anonymous namespace

// =====================================================================
// Public API
// =====================================================================

bool endsWith(const std::string &s, const std::string &suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) {
                          return std::tolower(a) == std::tolower(b);
                      });
}

void renderSpacetime(const Spacetime &st, const std::string &path,
                     int panelSize, int layoutIters,
                     double tilt, int spin, int precession,
                     int nFrames, int delayCentiseconds) {
    if (endsWith(path, ".graphml")) {
        writeGraphML(st, path);
        return;
    }
    if (endsWith(path, ".dot") || endsWith(path, ".gv")) {
        writeDot(st, path);
        return;
    }

    auto layout = computeLayout(st, layoutIters);

    if (endsWith(path, ".gif")) {
        double tiltRad = tilt * std::numbers::pi / 180.0;

        auto bb = computeGifBBox(layout, nFrames, tiltRad, spin, precession);

        std::vector<Image> frames;
        frames.reserve(nFrames);
        for (int i = 0; i < nFrames; ++i) {
            double t = static_cast<double>(i) / nFrames;
            double ry = 2.0 * std::numbers::pi * spin * t;
            double rx = tiltRad *
                        std::cos(2.0 * std::numbers::pi * precession * t);
            double rz = tiltRad *
                        std::sin(2.0 * std::numbers::pi * precession * t);
            Image frame(panelSize, panelSize);
            renderPanel(frame, 0, 0, panelSize, panelSize, layout, rx, ry, rz,
                        &bb);
            frames.push_back(std::move(frame));
        }
        writeGif(path, panelSize, panelSize, frames, delayCentiseconds);
    } else {
        // Static PNG: 2x2 grid of four orientations
        int w = panelSize * 2;
        int h = panelSize * 2;
        Image img(w, h);

        double angle = 40.0 * std::numbers::pi / 180.0;

        renderPanel(img, 0, 0, panelSize, panelSize, layout, 0, 0, 0);
        renderPanel(img, panelSize, 0, panelSize, panelSize, layout, angle, 0,
                    0);
        renderPanel(img, 0, panelSize, panelSize, panelSize, layout, 0, angle,
                    0);
        renderPanel(img, panelSize, panelSize, panelSize, panelSize, layout, 0,
                    0, angle);

        img.writePNG(path);
    }
}

} // namespace tessera