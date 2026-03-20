#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <execution>
#include <numeric>
#include <vector>

namespace {

constexpr uint32_t boundary_x_min_bit = 1u << 0;
constexpr uint32_t boundary_x_max_bit = 1u << 1;
constexpr uint32_t boundary_y_min_bit = 1u << 2;
constexpr uint32_t boundary_y_max_bit = 1u << 3;
constexpr uint32_t boundary_z_min_bit = 1u << 4;
constexpr uint32_t boundary_z_max_bit = 1u << 5;

bool periodic_x_min(const uint32_t mask) {
    return (mask & boundary_x_min_bit) != 0;
}

bool periodic_x_max(const uint32_t mask) {
    return (mask & boundary_x_max_bit) != 0;
}

bool periodic_y_min(const uint32_t mask) {
    return (mask & boundary_y_min_bit) != 0;
}

bool periodic_y_max(const uint32_t mask) {
    return (mask & boundary_y_max_bit) != 0;
}

bool periodic_z_min(const uint32_t mask) {
    return (mask & boundary_z_min_bit) != 0;
}

bool periodic_z_max(const uint32_t mask) {
    return (mask & boundary_z_max_bit) != 0;
}

bool map_index_or_fail(int& value, const int size, const bool periodic_min, const bool periodic_max) {
    if (value < 0) {
        if (!periodic_min) return false;
        value %= size;
        if (value < 0) value += size;
        return true;
    }
    if (value >= size) {
        if (!periodic_max) return false;
        value %= size;
        return true;
    }
    return true;
}

float wrap_or_clamp_coordinate(float value, const int size, const bool periodic_min, const bool periodic_max) {
    if (value < 0.0f) {
        if (periodic_min) {
            const float period = static_cast<float>(size);
            value = std::fmod(value, period);
            if (value < 0.0f) value += period;
            return value;
        }
        return 0.0f;
    }
    if (value > static_cast<float>(size - 1)) {
        if (periodic_max) {
            const float period = static_cast<float>(size);
            value = std::fmod(value, period);
            if (value < 0.0f) value += period;
            return value;
        }
        return static_cast<float>(size - 1);
    }
    return value;
}

std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
    return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
}

float fetch_boundary(const float* field, int x, int y, int z, const int sx, const int sy, const int sz, const uint32_t boundary_mask) {
    if (!map_index_or_fail(x, sx, periodic_x_min(boundary_mask), periodic_x_max(boundary_mask))) x = std::clamp(x, 0, sx - 1);
    if (!map_index_or_fail(y, sy, periodic_y_min(boundary_mask), periodic_y_max(boundary_mask))) y = std::clamp(y, 0, sy - 1);
    if (!map_index_or_fail(z, sz, periodic_z_min(boundary_mask), periodic_z_max(boundary_mask))) z = std::clamp(z, 0, sz - 1);
    return field[index_3d(x, y, z, sx, sy)];
}

float sample_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz, const uint32_t boundary_mask) {
    gx = wrap_or_clamp_coordinate(gx, sx, periodic_x_min(boundary_mask), periodic_x_max(boundary_mask));
    gy = wrap_or_clamp_coordinate(gy, sy, periodic_y_min(boundary_mask), periodic_y_max(boundary_mask));
    gz = wrap_or_clamp_coordinate(gz, sz, periodic_z_min(boundary_mask), periodic_z_max(boundary_mask));
    const int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0, sx - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0, sy - 1);
    const int z0 = std::clamp(static_cast<int>(std::floor(gz)), 0, sz - 1);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);
    const float c000 = fetch_boundary(field, x0, y0, z0, sx, sy, sz, boundary_mask);
    const float c100 = fetch_boundary(field, x1, y0, z0, sx, sy, sz, boundary_mask);
    const float c010 = fetch_boundary(field, x0, y1, z0, sx, sy, sz, boundary_mask);
    const float c110 = fetch_boundary(field, x1, y1, z0, sx, sy, sz, boundary_mask);
    const float c001 = fetch_boundary(field, x0, y0, z1, sx, sy, sz, boundary_mask);
    const float c101 = fetch_boundary(field, x1, y0, z1, sx, sy, sz, boundary_mask);
    const float c011 = fetch_boundary(field, x0, y1, z1, sx, sy, sz, boundary_mask);
    const float c111 = fetch_boundary(field, x1, y1, z1, sx, sy, sz, boundary_mask);
    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0 = c00 + (c10 - c00) * ty;
    const float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

float sample_scalar(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h - 0.5f, nx, ny, nz, boundary_mask);
}

float sample_u(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
    return sample_grid(field, x / h, y / h - 0.5f, z / h - 0.5f, nx + 1, ny, nz, boundary_mask);
}

float sample_v(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
    return sample_grid(field, x / h - 0.5f, y / h, z / h - 0.5f, nx, ny + 1, nz, boundary_mask);
}

float sample_w(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h, nx, ny, nz + 1, boundary_mask);
}

void wrap_or_clamp_domain(float& x, float& y, float& z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
    auto axis = [](float value, const float extent, const bool periodic_min, const bool periodic_max) {
        if (value < 0.0f) {
            if (periodic_min) {
                value = std::fmod(value, extent);
                if (value < 0.0f) value += extent;
                return value;
            }
            return 0.0f;
        }
        if (value > extent) {
            if (periodic_max) {
                value = std::fmod(value, extent);
                if (value < 0.0f) value += extent;
                return value;
            }
            return extent;
        }
        return value;
    };
    x = axis(x, static_cast<float>(nx) * h, periodic_x_min(boundary_mask), periodic_x_max(boundary_mask));
    y = axis(y, static_cast<float>(ny) * h, periodic_y_min(boundary_mask), periodic_y_max(boundary_mask));
    z = axis(z, static_cast<float>(nz) * h, periodic_z_min(boundary_mask), periodic_z_max(boundary_mask));
}

void sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask, float& out_x, float& out_y, float& out_z) {
    wrap_or_clamp_domain(x, y, z, nx, ny, nz, h, boundary_mask);
    out_x = sample_u(velocity_x, x, y, z, nx, ny, nz, h, boundary_mask);
    out_y = sample_v(velocity_y, x, y, z, nx, ny, nz, h, boundary_mask);
    out_z = sample_w(velocity_z, x, y, z, nx, ny, nz, h, boundary_mask);
}

void advect_u(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x <= nx; ++x)
                if ((x == 0 && !periodic_x_min(boundary_mask)) || (x == nx && !periodic_x_max(boundary_mask))) destination[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                else {
                    float px = static_cast<float>(x) * h;
                    float py = (static_cast<float>(y) + 0.5f) * h;
                    float pz = (static_cast<float>(z) + 0.5f) * h;
                    float vx, vy, vz;
                    sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, boundary_mask, vx, vy, vz);
                    px -= dt * vx;
                    py -= dt * vy;
                    pz -= dt * vz;
                    wrap_or_clamp_domain(px, py, pz, nx, ny, nz, h, boundary_mask);
                    destination[index_3d(x, y, z, nx + 1, ny)] = sample_u(source_x, px, py, pz, nx, ny, nz, h, boundary_mask);
                }
    });
}

void advect_v(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y <= ny; ++y)
            for (int x = 0; x < nx; ++x)
                if ((y == 0 && !periodic_y_min(boundary_mask)) || (y == ny && !periodic_y_max(boundary_mask))) destination[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                else {
                    float px = (static_cast<float>(x) + 0.5f) * h;
                    float py = static_cast<float>(y) * h;
                    float pz = (static_cast<float>(z) + 0.5f) * h;
                    float vx, vy, vz;
                    sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, boundary_mask, vx, vy, vz);
                    px -= dt * vx;
                    py -= dt * vy;
                    pz -= dt * vz;
                    wrap_or_clamp_domain(px, py, pz, nx, ny, nz, h, boundary_mask);
                    destination[index_3d(x, y, z, nx, ny + 1)] = sample_v(source_y, px, py, pz, nx, ny, nz, h, boundary_mask);
                }
    });
}

void advect_w(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                if ((z == 0 && !periodic_z_min(boundary_mask)) || (z == nz && !periodic_z_max(boundary_mask))) destination[index_3d(x, y, z, nx, ny)] = 0.0f;
                else {
                    float px = (static_cast<float>(x) + 0.5f) * h;
                    float py = (static_cast<float>(y) + 0.5f) * h;
                    float pz = static_cast<float>(z) * h;
                    float vx, vy, vz;
                    sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, boundary_mask, vx, vy, vz);
                    px -= dt * vx;
                    py -= dt * vy;
                    pz -= dt * vz;
                    wrap_or_clamp_domain(px, py, pz, nx, ny, nz, h, boundary_mask);
                    destination[index_3d(x, y, z, nx, ny)] = sample_w(source_z, px, py, pz, nx, ny, nz, h, boundary_mask);
                }
    });
}

void advect_scalar(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * h;
                float py = (static_cast<float>(y) + 0.5f) * h;
                float pz = (static_cast<float>(z) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, nx, ny, nz, h, boundary_mask, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                wrap_or_clamp_domain(px, py, pz, nx, ny, nz, h, boundary_mask);
                destination[index_3d(x, y, z, nx, ny)] = std::max(0.0f, sample_scalar(source, px, py, pz, nx, ny, nz, h, boundary_mask));
            }
    });
}

void diffuse_grid(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int diffuse_iterations, const uint32_t boundary_mask, const std::vector<int>& slices) {
    for (int iteration = 0; iteration < diffuse_iterations; ++iteration)
        for (int parity = 0; parity < 2; ++parity)
            std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
                for (int y = 0; y < sy; ++y)
                    for (int x = 0; x < sx; ++x) {
                        if (((x + y + z) & 1) != parity) continue;
                        const float neighbors = fetch_boundary(destination, x - 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z - 1, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, sx, sy, sz, boundary_mask);
                        destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
                    }
            });
}

void diffuse_u(float* destination, const float* source, const int nx, const int ny, const int nz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x <= nx; ++x)
                if ((x == 0 && !periodic_x_min(boundary_mask)) || (x == nx && !periodic_x_max(boundary_mask))) destination[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                else if (((x + y + z) & 1) == parity) {
                    const float neighbors = fetch_boundary(destination, x - 1, y, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(destination, x, y, z - 1, nx + 1, ny, nz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, nx + 1, ny, nz, boundary_mask);
                    destination[index_3d(x, y, z, nx + 1, ny)] = (source[index_3d(x, y, z, nx + 1, ny)] + alpha * neighbors) / denom;
                }
    });
}

void diffuse_v(float* destination, const float* source, const int nx, const int ny, const int nz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y <= ny; ++y)
            for (int x = 0; x < nx; ++x)
                if ((y == 0 && !periodic_y_min(boundary_mask)) || (y == ny && !periodic_y_max(boundary_mask))) destination[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                else if (((x + y + z) & 1) == parity) {
                    const float neighbors = fetch_boundary(destination, x - 1, y, z, nx, ny + 1, nz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, nx, ny + 1, nz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, nx, ny + 1, nz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, nx, ny + 1, nz, boundary_mask) + fetch_boundary(destination, x, y, z - 1, nx, ny + 1, nz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, nx, ny + 1, nz, boundary_mask);
                    destination[index_3d(x, y, z, nx, ny + 1)] = (source[index_3d(x, y, z, nx, ny + 1)] + alpha * neighbors) / denom;
                }
    });
}

void diffuse_w(float* destination, const float* source, const int nx, const int ny, const int nz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                if ((z == 0 && !periodic_z_min(boundary_mask)) || (z == nz && !periodic_z_max(boundary_mask))) destination[index_3d(x, y, z, nx, ny)] = 0.0f;
                else if (((x + y + z) & 1) == parity) {
                    const float neighbors = fetch_boundary(destination, x - 1, y, z, nx, ny, nz + 1, boundary_mask) + fetch_boundary(destination, x + 1, y, z, nx, ny, nz + 1, boundary_mask) + fetch_boundary(destination, x, y - 1, z, nx, ny, nz + 1, boundary_mask) + fetch_boundary(destination, x, y + 1, z, nx, ny, nz + 1, boundary_mask) + fetch_boundary(destination, x, y, z - 1, nx, ny, nz + 1, boundary_mask) + fetch_boundary(destination, x, y, z + 1, nx, ny, nz + 1, boundary_mask);
                    destination[index_3d(x, y, z, nx, ny)] = (source[index_3d(x, y, z, nx, ny)] + alpha * neighbors) / denom;
                }
    });
}

void compute_divergence(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                divergence[index_3d(x, y, z, nx, ny)] = -(fetch_boundary(velocity_x, x + 1, y, z, nx + 1, ny, nz, boundary_mask) - fetch_boundary(velocity_x, x, y, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(velocity_y, x, y + 1, z, nx, ny + 1, nz, boundary_mask) - fetch_boundary(velocity_y, x, y, z, nx, ny + 1, nz, boundary_mask) + fetch_boundary(velocity_z, x, y, z + 1, nx, ny, nz + 1, boundary_mask) - fetch_boundary(velocity_z, x, y, z, nx, ny, nz + 1, boundary_mask)) * h;
    });
}

void pressure_rbgs(float* pressure, const float* divergence, const int nx, const int ny, const int nz, const int parity, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                if (((x + y + z) & 1) != parity) continue;
                float sum = 0.0f;
                int count = 0;
                if (x > 0) {
                    sum += pressure[index_3d(x - 1, y, z, nx, ny)];
                    ++count;
                } else if (periodic_x_min(boundary_mask)) {
                    sum += pressure[index_3d(nx - 1, y, z, nx, ny)];
                    ++count;
                }
                if (x + 1 < nx) {
                    sum += pressure[index_3d(x + 1, y, z, nx, ny)];
                    ++count;
                } else if (periodic_x_max(boundary_mask)) {
                    sum += pressure[index_3d(0, y, z, nx, ny)];
                    ++count;
                }
                if (y > 0) {
                    sum += pressure[index_3d(x, y - 1, z, nx, ny)];
                    ++count;
                } else if (periodic_y_min(boundary_mask)) {
                    sum += pressure[index_3d(x, ny - 1, z, nx, ny)];
                    ++count;
                }
                if (y + 1 < ny) {
                    sum += pressure[index_3d(x, y + 1, z, nx, ny)];
                    ++count;
                } else if (periodic_y_max(boundary_mask)) {
                    sum += pressure[index_3d(x, 0, z, nx, ny)];
                    ++count;
                }
                if (z > 0) {
                    sum += pressure[index_3d(x, y, z - 1, nx, ny)];
                    ++count;
                } else if (periodic_z_min(boundary_mask)) {
                    sum += pressure[index_3d(x, y, nz - 1, nx, ny)];
                    ++count;
                }
                if (z + 1 < nz) {
                    sum += pressure[index_3d(x, y, z + 1, nx, ny)];
                    ++count;
                } else if (periodic_z_max(boundary_mask)) {
                    sum += pressure[index_3d(x, y, 0, nx, ny)];
                    ++count;
                }
                pressure[index_3d(x, y, z, nx, ny)] = count > 0 ? (sum + divergence[index_3d(x, y, z, nx, ny)]) / static_cast<float>(count) : 0.0f;
            }
    });
}

void project_u(float* velocity_x, const float* pressure, const int nx, const int ny, const int, const float inv_h, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x <= nx; ++x)
                if (x == 0) {
                    if (!periodic_x_min(boundary_mask)) velocity_x[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                    else velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(0, y, z, nx, ny)] - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
                } else if (x == nx) {
                    if (!periodic_x_max(boundary_mask)) velocity_x[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                    else velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(0, y, z, nx, ny)] - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
                } else velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
    });
}

void project_v(float* velocity_y, const float* pressure, const int nx, const int ny, const int, const float inv_h, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y <= ny; ++y)
            for (int x = 0; x < nx; ++x)
                if (y == 0) {
                    if (!periodic_y_min(boundary_mask)) velocity_y[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                    else velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, 0, z, nx, ny)] - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
                } else if (y == ny) {
                    if (!periodic_y_max(boundary_mask)) velocity_y[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                    else velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, 0, z, nx, ny)] - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
                } else velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
    });
}

void project_w(float* velocity_z, const float* pressure, const int nx, const int ny, const int nz, const float inv_h, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                if (z == 0) {
                    if (!periodic_z_min(boundary_mask)) velocity_z[index_3d(x, y, z, nx, ny)] = 0.0f;
                    else velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, 0, nx, ny)] - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
                } else if (z == nz) {
                    if (!periodic_z_max(boundary_mask)) velocity_z[index_3d(x, y, z, nx, ny)] = 0.0f;
                    else velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, 0, nx, ny)] - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
                } else velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) * inv_h;
    });
}

} // namespace

extern "C" {

int32_t stable_fluids_step_parallel(const StableFluidsStepDesc* desc) {
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float viscosity = desc->viscosity;
    const float diffusion = desc->diffusion;
    const int32_t diffuse_iterations = desc->diffuse_iterations;
    const int32_t pressure_iterations = desc->pressure_iterations;
    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_min_bit : 0u) | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_max_bit : 0u) | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_max_bit : 0u);

    auto* density_field = static_cast<float*>(desc->density);
    auto* density_temporary = static_cast<float*>(desc->temporary_density);
    auto* density_previous = static_cast<float*>(desc->temporary_previous_density);
    auto* velocity_x_field = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field = static_cast<float*>(desc->velocity_z);
    auto* velocity_x_temporary = static_cast<float*>(desc->temporary_velocity_x);
    auto* velocity_y_temporary = static_cast<float*>(desc->temporary_velocity_y);
    auto* velocity_z_temporary = static_cast<float*>(desc->temporary_velocity_z);
    auto* velocity_x_previous = static_cast<float*>(desc->temporary_previous_velocity_x);
    auto* velocity_y_previous = static_cast<float*>(desc->temporary_previous_velocity_y);
    auto* velocity_z_previous = static_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure = static_cast<float*>(desc->temporary_pressure);
    auto* divergence = static_cast<float*>(desc->temporary_divergence);

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_x_field_bytes = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_y_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_z_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);

    std::vector<int> cell_slices(static_cast<std::size_t>(nz));
    std::vector<int> w_slices(static_cast<std::size_t>(nz + 1));
    std::iota(cell_slices.begin(), cell_slices.end(), 0);
    std::iota(w_slices.begin(), w_slices.end(), 0);

    std::memcpy(velocity_x_previous, velocity_x_field, velocity_x_field_bytes);
    std::memcpy(velocity_y_previous, velocity_y_field, velocity_y_field_bytes);
    std::memcpy(velocity_z_previous, velocity_z_field, velocity_z_field_bytes);
    advect_u(velocity_x_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt, boundary_mask, cell_slices);
    advect_v(velocity_y_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt, boundary_mask, cell_slices);
    advect_w(velocity_z_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt, boundary_mask, w_slices);

    if (viscosity <= 0.0f) {
        std::memcpy(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes);
        std::memcpy(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes);
        std::memcpy(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes);
    } else {
        std::memcpy(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes);
        std::memcpy(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes);
        std::memcpy(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes);
        const float alpha = dt * viscosity / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        for (int iteration = 0; iteration < diffuse_iterations; ++iteration) {
            diffuse_u(velocity_x_field, velocity_x_temporary, nx, ny, nz, alpha, denom, 0, boundary_mask, cell_slices);
            diffuse_v(velocity_y_field, velocity_y_temporary, nx, ny, nz, alpha, denom, 0, boundary_mask, cell_slices);
            diffuse_w(velocity_z_field, velocity_z_temporary, nx, ny, nz, alpha, denom, 0, boundary_mask, w_slices);
            diffuse_u(velocity_x_field, velocity_x_temporary, nx, ny, nz, alpha, denom, 1, boundary_mask, cell_slices);
            diffuse_v(velocity_y_field, velocity_y_temporary, nx, ny, nz, alpha, denom, 1, boundary_mask, cell_slices);
            diffuse_w(velocity_z_field, velocity_z_temporary, nx, ny, nz, alpha, denom, 1, boundary_mask, w_slices);
        }
    }

    std::memset(pressure, 0, static_cast<std::size_t>(cell_bytes));
    compute_divergence(divergence, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, boundary_mask, cell_slices);
    for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
        pressure_rbgs(pressure, divergence, nx, ny, nz, 0, boundary_mask, cell_slices);
        pressure_rbgs(pressure, divergence, nx, ny, nz, 1, boundary_mask, cell_slices);
    }
    project_u(velocity_x_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, cell_slices);
    project_v(velocity_y_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, cell_slices);
    project_w(velocity_z_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, w_slices);

    std::memcpy(density_previous, density_field, static_cast<std::size_t>(cell_bytes));
    advect_scalar(density_temporary, density_previous, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, dt, boundary_mask, cell_slices);

    if (diffusion <= 0.0f) std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
    else {
        std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
        const float alpha = dt * diffusion / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        diffuse_grid(density_field, density_temporary, nx, ny, nz, alpha, denom, diffuse_iterations, boundary_mask, cell_slices);
    }

    return 0;
}

} // extern "C"
