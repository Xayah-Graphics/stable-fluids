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

void diffuse_grid(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask, const std::vector<int>& slices) {
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
        for (int y = 0; y < sy; ++y)
            for (int x = 0; x < sx; ++x) {
                if (((x + y + z) & 1) != parity) continue;
                const float neighbors = fetch_boundary(destination, x - 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z - 1, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, sx, sy, sz, boundary_mask);
                destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
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

void restrict_poisson_residual(float* coarse_rhs, const float* fine_pressure, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const uint32_t boundary_mask, const std::vector<int>& coarse_slices) {
    const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
    const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
    std::for_each(std::execution::par_unseq, coarse_slices.begin(), coarse_slices.end(), [&](const int z) {
        for (int y = 0; y < coarse_ny; ++y)
            for (int x = 0; x < coarse_nx; ++x) {
                float residual_sum = 0.0f;
                int samples = 0;
                for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz)
                    for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy)
                        for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                            float neighbors = 0.0f;
                            int count = 0;
                            if (fx > 0) {
                                neighbors += fine_pressure[index_3d(fx - 1, fy, fz, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_x_min(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(fine_nx - 1, fy, fz, fine_nx, fine_ny)];
                                ++count;
                            }
                            if (fx + 1 < fine_nx) {
                                neighbors += fine_pressure[index_3d(fx + 1, fy, fz, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_x_max(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(0, fy, fz, fine_nx, fine_ny)];
                                ++count;
                            }
                            if (fy > 0) {
                                neighbors += fine_pressure[index_3d(fx, fy - 1, fz, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_y_min(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(fx, fine_ny - 1, fz, fine_nx, fine_ny)];
                                ++count;
                            }
                            if (fy + 1 < fine_ny) {
                                neighbors += fine_pressure[index_3d(fx, fy + 1, fz, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_y_max(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(fx, 0, fz, fine_nx, fine_ny)];
                                ++count;
                            }
                            if (fz > 0) {
                                neighbors += fine_pressure[index_3d(fx, fy, fz - 1, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_z_min(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(fx, fy, fine_nz - 1, fine_nx, fine_ny)];
                                ++count;
                            }
                            if (fz + 1 < fine_nz) {
                                neighbors += fine_pressure[index_3d(fx, fy, fz + 1, fine_nx, fine_ny)];
                                ++count;
                            } else if (periodic_z_max(boundary_mask)) {
                                neighbors += fine_pressure[index_3d(fx, fy, 0, fine_nx, fine_ny)];
                                ++count;
                            }
                            const float applied = static_cast<float>(count) * fine_pressure[index_3d(fx, fy, fz, fine_nx, fine_ny)] - neighbors;
                            residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                            ++samples;
                        }
                coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
            }
    });
}

void restrict_diffusion_residual(float* coarse_rhs, const float* fine_solution, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const float alpha, const uint32_t boundary_mask, const std::vector<int>& coarse_slices) {
    const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
    const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
    std::for_each(std::execution::par_unseq, coarse_slices.begin(), coarse_slices.end(), [&](const int z) {
        for (int y = 0; y < coarse_ny; ++y)
            for (int x = 0; x < coarse_nx; ++x) {
                float residual_sum = 0.0f;
                int samples = 0;
                for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz)
                    for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy)
                        for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                            const float center = fine_solution[index_3d(fx, fy, fz, fine_nx, fine_ny)];
                            const float neighbors = fetch_boundary(fine_solution, fx - 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx + 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy - 1, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy + 1, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy, fz - 1, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy, fz + 1, fine_nx, fine_ny, fine_nz, boundary_mask);
                            const float applied = (1.0f + 6.0f * alpha) * center - alpha * neighbors;
                            residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                            ++samples;
                        }
                coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
            }
    });
}

void prolongate_add(float* fine_solution, const float* coarse_solution, const int fine_nx, const int fine_ny, const int, const int coarse_nx, const int coarse_ny, const int coarse_nz, const uint32_t boundary_mask, const std::vector<int>& fine_slices) {
    std::for_each(std::execution::par_unseq, fine_slices.begin(), fine_slices.end(), [&](const int z) {
        for (int y = 0; y < fine_ny; ++y)
            for (int x = 0; x < fine_nx; ++x)
                fine_solution[index_3d(x, y, z, fine_nx, fine_ny)] += sample_grid(coarse_solution, 0.5f * static_cast<float>(x) - 0.25f, 0.5f * static_cast<float>(y) - 0.25f, 0.5f * static_cast<float>(z) - 0.25f, coarse_nx, coarse_ny, coarse_nz, boundary_mask);
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
    auto* coarse_pressure_storage = density_temporary;
    auto* coarse_rhs_storage = density_previous;

    std::vector<int> cell_slices(static_cast<std::size_t>(nz));
    std::vector<int> w_slices(static_cast<std::size_t>(nz + 1));
    std::iota(cell_slices.begin(), cell_slices.end(), 0);
    std::iota(w_slices.begin(), w_slices.end(), 0);
    constexpr int max_levels = 16;
    int level_count = 1;
    int level_nx[max_levels]{nx};
    int level_ny[max_levels]{ny};
    int level_nz[max_levels]{nz};
    float level_scale[max_levels]{1.0f};
    float* pressure_levels[max_levels]{pressure};
    float* rhs_levels[max_levels]{divergence};
    std::vector<int> level_slices[max_levels];
    level_slices[0] = cell_slices;
    std::uint64_t coarse_offset = 0;
    while (level_count < max_levels && (level_nx[level_count - 1] > 1 || level_ny[level_count - 1] > 1 || level_nz[level_count - 1] > 1)) {
        level_nx[level_count] = std::max(1, (level_nx[level_count - 1] + 1) / 2);
        level_ny[level_count] = std::max(1, (level_ny[level_count - 1] + 1) / 2);
        level_nz[level_count] = std::max(1, (level_nz[level_count - 1] + 1) / 2);
        level_scale[level_count] = level_scale[level_count - 1] * 0.25f;
        pressure_levels[level_count] = coarse_pressure_storage + coarse_offset;
        rhs_levels[level_count] = coarse_rhs_storage + coarse_offset;
        level_slices[level_count].resize(static_cast<std::size_t>(level_nz[level_count]));
        std::iota(level_slices[level_count].begin(), level_slices[level_count].end(), 0);
        coarse_offset += static_cast<std::uint64_t>(level_nx[level_count]) * static_cast<std::uint64_t>(level_ny[level_count]) * static_cast<std::uint64_t>(level_nz[level_count]);
        ++level_count;
    }
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
        auto zero_velocity_component_boundaries = [&](float* field, const int sx, const int sy, const int sz, const int axis, const std::vector<int>& slices) {
            std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](const int z) {
                if (axis == 0) {
                    if (!periodic_x_min(boundary_mask))
                        for (int y = 0; y < sy; ++y)
                            field[index_3d(0, y, z, sx, sy)] = 0.0f;
                    if (!periodic_x_max(boundary_mask))
                        for (int y = 0; y < sy; ++y)
                            field[index_3d(sx - 1, y, z, sx, sy)] = 0.0f;
                } else if (axis == 1) {
                    if (!periodic_y_min(boundary_mask))
                        for (int x = 0; x < sx; ++x)
                            field[index_3d(x, 0, z, sx, sy)] = 0.0f;
                    if (!periodic_y_max(boundary_mask))
                        for (int x = 0; x < sx; ++x)
                            field[index_3d(x, sy - 1, z, sx, sy)] = 0.0f;
                } else {
                    if (!periodic_z_min(boundary_mask) && z == 0)
                        for (int y = 0; y < sy; ++y)
                            for (int x = 0; x < sx; ++x)
                                field[index_3d(x, y, 0, sx, sy)] = 0.0f;
                    if (!periodic_z_max(boundary_mask) && z == sz - 1)
                        for (int y = 0; y < sy; ++y)
                            for (int x = 0; x < sx; ++x)
                                field[index_3d(x, y, sz - 1, sx, sy)] = 0.0f;
                }
            });
        };
        auto diffuse_velocity_component = [&](float* field, const float* source, const std::uint64_t field_bytes, const int sx, const int sy, const int sz, const int axis) {
            constexpr int component_max_levels = 16;
            int component_level_count = 1;
            int component_nx[component_max_levels]{sx};
            int component_ny[component_max_levels]{sy};
            int component_nz[component_max_levels]{sz};
            float component_scale[component_max_levels]{1.0f};
            float* solution_levels[component_max_levels]{field};
            float* rhs_levels[component_max_levels]{const_cast<float*>(source)};
            std::vector<int> component_slices[component_max_levels];
            component_slices[0].resize(static_cast<std::size_t>(sz));
            std::iota(component_slices[0].begin(), component_slices[0].end(), 0);
            std::uint64_t coarse_offset = 0;
            while (component_level_count < component_max_levels && (component_nx[component_level_count - 1] > 1 || component_ny[component_level_count - 1] > 1 || component_nz[component_level_count - 1] > 1)) {
                component_nx[component_level_count] = std::max(1, (component_nx[component_level_count - 1] + 1) / 2);
                component_ny[component_level_count] = std::max(1, (component_ny[component_level_count - 1] + 1) / 2);
                component_nz[component_level_count] = std::max(1, (component_nz[component_level_count - 1] + 1) / 2);
                component_scale[component_level_count] = component_scale[component_level_count - 1] * 0.25f;
                solution_levels[component_level_count] = density_temporary + coarse_offset;
                rhs_levels[component_level_count] = density_previous + coarse_offset;
                component_slices[component_level_count].resize(static_cast<std::size_t>(component_nz[component_level_count]));
                std::iota(component_slices[component_level_count].begin(), component_slices[component_level_count].end(), 0);
                coarse_offset += static_cast<std::uint64_t>(component_nx[component_level_count]) * static_cast<std::uint64_t>(component_ny[component_level_count]) * static_cast<std::uint64_t>(component_nz[component_level_count]);
                ++component_level_count;
            }
            std::memcpy(field, source, static_cast<std::size_t>(field_bytes));
            zero_velocity_component_boundaries(field, sx, sy, sz, axis, component_slices[0]);
            const int v_cycles = std::max(1, diffuse_iterations / 12);
            const int smoothing_steps = 1;
            const int coarse_steps = std::max(6, diffuse_iterations / 4);
            for (int cycle = 0; cycle < v_cycles; ++cycle) {
                for (int level = 0; level + 1 < component_level_count; ++level) {
                    const int lx = component_nx[level];
                    const int ly = component_ny[level];
                    const int lz = component_nz[level];
                    const float alpha = dt * viscosity / (cell_size * cell_size) * component_scale[level];
                    const float denom = 1.0f + 6.0f * alpha;
                    for (int smooth = 0; smooth < smoothing_steps; ++smooth) {
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                    }
                    const int cx = component_nx[level + 1];
                    const int cy = component_ny[level + 1];
                    const int cz = component_nz[level + 1];
                    const auto coarse_bytes = static_cast<std::uint64_t>(cx) * static_cast<std::uint64_t>(cy) * static_cast<std::uint64_t>(cz) * sizeof(float);
                    std::memset(solution_levels[level + 1], 0, static_cast<std::size_t>(coarse_bytes));
                    restrict_diffusion_residual(rhs_levels[level + 1], solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, boundary_mask, component_slices[level + 1]);
                    zero_velocity_component_boundaries(rhs_levels[level + 1], cx, cy, cz, axis, component_slices[level + 1]);
                }
                {
                    const int level = component_level_count - 1;
                    const int lx = component_nx[level];
                    const int ly = component_ny[level];
                    const int lz = component_nz[level];
                    const float alpha = dt * viscosity / (cell_size * cell_size) * component_scale[level];
                    const float denom = 1.0f + 6.0f * alpha;
                    for (int smooth = 0; smooth < coarse_steps; ++smooth) {
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                    }
                }
                for (int level = component_level_count - 2; level >= 0; --level) {
                    const int lx = component_nx[level];
                    const int ly = component_ny[level];
                    const int lz = component_nz[level];
                    const int cx = component_nx[level + 1];
                    const int cy = component_ny[level + 1];
                    const int cz = component_nz[level + 1];
                    const float alpha = dt * viscosity / (cell_size * cell_size) * component_scale[level];
                    const float denom = 1.0f + 6.0f * alpha;
                    prolongate_add(solution_levels[level], solution_levels[level + 1], lx, ly, lz, cx, cy, cz, boundary_mask, component_slices[level]);
                    zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                    for (int smooth = 0; smooth < smoothing_steps; ++smooth) {
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                        diffuse_grid(solution_levels[level], rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, component_slices[level]);
                        zero_velocity_component_boundaries(solution_levels[level], lx, ly, lz, axis, component_slices[level]);
                    }
                }
            }
        };
        diffuse_velocity_component(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, nx + 1, ny, nz, 0);
        diffuse_velocity_component(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, nx, ny + 1, nz, 1);
        diffuse_velocity_component(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, nx, ny, nz + 1, 2);
    }

    std::memset(pressure, 0, static_cast<std::size_t>(cell_bytes));
    compute_divergence(divergence, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, boundary_mask, cell_slices);
    const int pressure_v_cycles = std::max(1, pressure_iterations / 40);
    const int pressure_smoothing_steps = 1;
    const int pressure_coarse_steps = std::max(8, pressure_iterations / 10);
    for (int cycle = 0; cycle < pressure_v_cycles; ++cycle) {
        for (int level = 0; level + 1 < level_count; ++level) {
            const int lx = level_nx[level];
            const int ly = level_ny[level];
            const int lz = level_nz[level];
            for (int smooth = 0; smooth < pressure_smoothing_steps; ++smooth) {
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0, boundary_mask, level_slices[level]);
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1, boundary_mask, level_slices[level]);
            }
            const int cx = level_nx[level + 1];
            const int cy = level_ny[level + 1];
            const int cz = level_nz[level + 1];
            const auto coarse_bytes = static_cast<std::uint64_t>(cx) * static_cast<std::uint64_t>(cy) * static_cast<std::uint64_t>(cz) * sizeof(float);
            std::memset(pressure_levels[level + 1], 0, static_cast<std::size_t>(coarse_bytes));
            restrict_poisson_residual(rhs_levels[level + 1], pressure_levels[level], rhs_levels[level], lx, ly, lz, boundary_mask, level_slices[level + 1]);
        }
        {
            const int level = level_count - 1;
            const int lx = level_nx[level];
            const int ly = level_ny[level];
            const int lz = level_nz[level];
            for (int smooth = 0; smooth < pressure_coarse_steps; ++smooth) {
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0, boundary_mask, level_slices[level]);
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1, boundary_mask, level_slices[level]);
            }
        }
        for (int level = level_count - 2; level >= 0; --level) {
            const int lx = level_nx[level];
            const int ly = level_ny[level];
            const int lz = level_nz[level];
            const int cx = level_nx[level + 1];
            const int cy = level_ny[level + 1];
            const int cz = level_nz[level + 1];
            prolongate_add(pressure_levels[level], pressure_levels[level + 1], lx, ly, lz, cx, cy, cz, boundary_mask, level_slices[level]);
            for (int smooth = 0; smooth < pressure_smoothing_steps; ++smooth) {
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0, boundary_mask, level_slices[level]);
                pressure_rbgs(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1, boundary_mask, level_slices[level]);
            }
        }
    }
    project_u(velocity_x_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, cell_slices);
    project_v(velocity_y_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, cell_slices);
    project_w(velocity_z_field, pressure, nx, ny, nz, 1.0f / cell_size, boundary_mask, w_slices);

    std::memcpy(density_previous, density_field, static_cast<std::size_t>(cell_bytes));
    advect_scalar(density_temporary, density_previous, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, dt, boundary_mask, cell_slices);

    if (diffusion <= 0.0f) std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
    else {
        std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
        auto* diffusion_coarse_solution = pressure;
        auto* diffusion_coarse_rhs = divergence;
        float* diffusion_solution_levels[max_levels]{density_field};
        float* diffusion_rhs_levels[max_levels]{density_temporary};
        std::uint64_t diffusion_offset = 0;
        for (int level = 1; level < level_count; ++level) {
            diffusion_solution_levels[level] = diffusion_coarse_solution + diffusion_offset;
            diffusion_rhs_levels[level] = diffusion_coarse_rhs + diffusion_offset;
            diffusion_offset += static_cast<std::uint64_t>(level_nx[level]) * static_cast<std::uint64_t>(level_ny[level]) * static_cast<std::uint64_t>(level_nz[level]);
        }
        const int diffusion_v_cycles = std::max(1, diffuse_iterations / 12);
        const int diffusion_smoothing_steps = 1;
        const int diffusion_coarse_steps = std::max(6, diffuse_iterations / 4);
        for (int cycle = 0; cycle < diffusion_v_cycles; ++cycle) {
            for (int level = 0; level + 1 < level_count; ++level) {
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const float alpha = dt * diffusion / (cell_size * cell_size) * level_scale[level];
                const float denom = 1.0f + 6.0f * alpha;
                for (int smooth = 0; smooth < diffusion_smoothing_steps; ++smooth) {
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, level_slices[level]);
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, level_slices[level]);
                }
                const int cx = level_nx[level + 1];
                const int cy = level_ny[level + 1];
                const int cz = level_nz[level + 1];
                const auto coarse_bytes = static_cast<std::uint64_t>(cx) * static_cast<std::uint64_t>(cy) * static_cast<std::uint64_t>(cz) * sizeof(float);
                std::memset(diffusion_solution_levels[level + 1], 0, static_cast<std::size_t>(coarse_bytes));
                restrict_diffusion_residual(diffusion_rhs_levels[level + 1], diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, boundary_mask, level_slices[level + 1]);
            }
            {
                const int level = level_count - 1;
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const float alpha = dt * diffusion / (cell_size * cell_size) * level_scale[level];
                const float denom = 1.0f + 6.0f * alpha;
                for (int smooth = 0; smooth < diffusion_coarse_steps; ++smooth) {
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, level_slices[level]);
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, level_slices[level]);
                }
            }
            for (int level = level_count - 2; level >= 0; --level) {
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const int cx = level_nx[level + 1];
                const int cy = level_ny[level + 1];
                const int cz = level_nz[level + 1];
                const float alpha = dt * diffusion / (cell_size * cell_size) * level_scale[level];
                const float denom = 1.0f + 6.0f * alpha;
                prolongate_add(diffusion_solution_levels[level], diffusion_solution_levels[level + 1], lx, ly, lz, cx, cy, cz, boundary_mask, level_slices[level]);
                for (int smooth = 0; smooth < diffusion_smoothing_steps; ++smooth) {
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 0, boundary_mask, level_slices[level]);
                    diffuse_grid(diffusion_solution_levels[level], diffusion_rhs_levels[level], lx, ly, lz, alpha, denom, 1, boundary_mask, level_slices[level]);
                }
            }
        }
        {
            const float alpha = dt * diffusion / (cell_size * cell_size);
            const float denom = 1.0f + 6.0f * alpha;
            diffuse_grid(density_field, density_temporary, nx, ny, nz, alpha, denom, 0, boundary_mask, cell_slices);
            diffuse_grid(density_field, density_temporary, nx, ny, nz, alpha, denom, 1, boundary_mask, cell_slices);
        }
    }

    return 0;
}

} // extern "C"
