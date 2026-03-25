#include "stable-fluids.h"
#include <algorithm>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {
    using Stream = cudaStream_t;

    namespace {

        constexpr uint32_t boundary_x_min_bit = 1u << 0;
        constexpr uint32_t boundary_x_max_bit = 1u << 1;
        constexpr uint32_t boundary_y_min_bit = 1u << 2;
        constexpr uint32_t boundary_y_max_bit = 1u << 3;
        constexpr uint32_t boundary_z_min_bit = 1u << 4;
        constexpr uint32_t boundary_z_max_bit = 1u << 5;

        constexpr int max_levels = 16;

        enum class BoundaryAxis : int { none = -1, x = 0, y = 1, z = 2 };

        struct GridLevel {
            int nx;
            int ny;
            int nz;
            float scale;
            float* solution;
            float* rhs;
        };

        struct GridHierarchy {
            int level_count;
            GridLevel levels[max_levels];
        };

        struct VCycleConfig {
            int cycles;
            int pre_smooth;
            int post_smooth;
            int coarse_smooth;
        };

        dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
            return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
        }

        __host__ __device__ bool map_index_or_fail(int& value, const int size, const bool periodic_min, const bool periodic_max) {
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

        __host__ __device__ float wrap_or_clamp_coordinate(float value, const int size, const bool periodic_min, const bool periodic_max) {
            if (value < 0.0f) {
                if (periodic_min) {
                    const float period = static_cast<float>(size);
                    value              = fmodf(value, period);
                    if (value < 0.0f) value += period;
                    return value;
                }
                return 0.0f;
            }
            if (value > static_cast<float>(size - 1)) {
                if (periodic_max) {
                    const float period = static_cast<float>(size);
                    value              = fmodf(value, period);
                    if (value < 0.0f) value += period;
                    return value;
                }
                return static_cast<float>(size - 1);
            }
            return value;
        }

        __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
        }

        __host__ __device__ float fetch_boundary(const float* field, int x, int y, int z, const int sx, const int sy, const int sz, const uint32_t boundary_mask) {
            if (!map_index_or_fail(x, sx, (boundary_mask & boundary_x_min_bit) != 0, (boundary_mask & boundary_x_max_bit) != 0)) x = std::clamp(x, 0, sx - 1);
            if (!map_index_or_fail(y, sy, (boundary_mask & boundary_y_min_bit) != 0, (boundary_mask & boundary_y_max_bit) != 0)) y = std::clamp(y, 0, sy - 1);
            if (!map_index_or_fail(z, sz, (boundary_mask & boundary_z_min_bit) != 0, (boundary_mask & boundary_z_max_bit) != 0)) z = std::clamp(z, 0, sz - 1);
            return field[index_3d(x, y, z, sx, sy)];
        }

        __device__ float sample_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz, const uint32_t boundary_mask) {
            gx = wrap_or_clamp_coordinate(gx, sx, (boundary_mask & boundary_x_min_bit) != 0, (boundary_mask & boundary_x_max_bit) != 0);
            gy = wrap_or_clamp_coordinate(gy, sy, (boundary_mask & boundary_y_min_bit) != 0, (boundary_mask & boundary_y_max_bit) != 0);
            gz = wrap_or_clamp_coordinate(gz, sz, (boundary_mask & boundary_z_min_bit) != 0, (boundary_mask & boundary_z_max_bit) != 0);

            const int x0   = std::clamp(static_cast<int>(floorf(gx)), 0, sx - 1);
            const int y0   = std::clamp(static_cast<int>(floorf(gy)), 0, sy - 1);
            const int z0   = std::clamp(static_cast<int>(floorf(gz)), 0, sz - 1);
            const int x1   = x0 + 1;
            const int y1   = y0 + 1;
            const int z1   = z0 + 1;
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
            const float c0  = c00 + (c10 - c00) * ty;
            const float c1  = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }

        __device__ float sample_scalar(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h - 0.5f, nx, ny, nz, boundary_mask);
        }

        __device__ float sample_u(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            return sample_grid(field, pos.x / h, pos.y / h - 0.5f, pos.z / h - 0.5f, nx + 1, ny, nz, boundary_mask);
        }

        __device__ float sample_v(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h, pos.z / h - 0.5f, nx, ny + 1, nz, boundary_mask);
        }

        __device__ float sample_w(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h, nx, ny, nz + 1, boundary_mask);
        }

        __device__ float3 wrap_or_clamp_domain(const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            const float domain_x = static_cast<float>(nx) * h;
            const float domain_y = static_cast<float>(ny) * h;
            const float domain_z = static_cast<float>(nz) * h;
            auto axis            = [](float value, const float extent, const bool periodic_min, const bool periodic_max) {
                if (value < 0.0f) {
                    if (periodic_min) {
                        value = fmodf(value, extent);
                        if (value < 0.0f) value += extent;
                        return value;
                    }
                    return 0.0f;
                }
                if (value > extent) {
                    if (periodic_max) {
                        value = fmodf(value, extent);
                        if (value < 0.0f) value += extent;
                        return value;
                    }
                    return extent;
                }
                return value;
            };
            return make_float3(axis(pos.x, domain_x, (boundary_mask & boundary_x_min_bit) != 0, (boundary_mask & boundary_x_max_bit) != 0), axis(pos.y, domain_y, (boundary_mask & boundary_y_min_bit) != 0, (boundary_mask & boundary_y_max_bit) != 0), axis(pos.z, domain_z, (boundary_mask & boundary_z_min_bit) != 0, (boundary_mask & boundary_z_max_bit) != 0));
        }

        __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            pos = wrap_or_clamp_domain(pos, nx, ny, nz, h, boundary_mask);
            return make_float3(sample_u(velocity_x, pos, nx, ny, nz, h, boundary_mask), sample_v(velocity_y, pos, nx, ny, nz, h, boundary_mask), sample_w(velocity_z, pos, nx, ny, nz, h, boundary_mask));
        }

        __global__ void advect_velocity(float* velocity_x_destination, float* velocity_y_destination, float* velocity_z_destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            if (x <= nx && y < ny && z < nz) {
                if ((x == 0 && (boundary_mask & boundary_x_min_bit) == 0) || (x == nx && (boundary_mask & boundary_x_max_bit) == 0)) {
                    velocity_x_destination[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                } else {
                    const float3 pos                                      = make_float3(static_cast<float>(x) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
                    const float3 velocity                                 = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_mask);
                    velocity_x_destination[index_3d(x, y, z, nx + 1, ny)] = sample_u(source_x, wrap_or_clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h, boundary_mask), nx, ny, nz, h, boundary_mask);
                }
            }

            if (x < nx && y <= ny && z < nz) {
                if ((y == 0 && (boundary_mask & boundary_y_min_bit) == 0) || (y == ny && (boundary_mask & boundary_y_max_bit) == 0)) {
                    velocity_y_destination[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                } else {
                    const float3 pos                                      = make_float3((static_cast<float>(x) + 0.5f) * h, static_cast<float>(y) * h, (static_cast<float>(z) + 0.5f) * h);
                    const float3 velocity                                 = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_mask);
                    velocity_y_destination[index_3d(x, y, z, nx, ny + 1)] = sample_v(source_y, wrap_or_clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h, boundary_mask), nx, ny, nz, h, boundary_mask);
                }
            }

            if (x < nx && y < ny && z <= nz) {
                if ((z == 0 && (boundary_mask & boundary_z_min_bit) == 0) || (z == nz && (boundary_mask & boundary_z_max_bit) == 0)) {
                    velocity_z_destination[index_3d(x, y, z, nx, ny)] = 0.0f;
                } else {
                    const float3 pos                                  = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, static_cast<float>(z) * h);
                    const float3 velocity                             = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_mask);
                    velocity_z_destination[index_3d(x, y, z, nx, ny)] = sample_w(source_z, wrap_or_clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h, boundary_mask), nx, ny, nz, h, boundary_mask);
                }
            }
        }

        __global__ void advect_scalar_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            const float3 pos                       = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
            const float3 velocity                  = sample_velocity(velocity_x, velocity_y, velocity_z, pos, nx, ny, nz, h, boundary_mask);
            destination[index_3d(x, y, z, nx, ny)] = fmaxf(0.0f, sample_scalar(source, wrap_or_clamp_domain(make_float3(pos.x - dt * velocity.x, pos.y - dt * velocity.y, pos.z - dt * velocity.z), nx, ny, nz, h, boundary_mask), nx, ny, nz, h, boundary_mask));
        }

        __global__ void add_scalar_source_kernel(float* destination, const int sx, const int sy, const int sz, const float center_x, const float center_y, const float center_z, const float radius, const float amount, const float sample_offset_x, const float sample_offset_y, const float sample_offset_z) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz) return;

            const float px = static_cast<float>(x) + sample_offset_x;
            const float py = static_cast<float>(y) + sample_offset_y;
            const float pz = static_cast<float>(z) + sample_offset_z;
            const float dx = px - center_x;
            const float dy = py - center_y;
            const float dz = pz - center_z;
            const float radius2 = radius * radius;
            const float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2) return;
            destination[index_3d(x, y, z, sx, sy)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
        }

        __global__ void diffuse_grid_kernel(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;
            const float neighbors = fetch_boundary(destination, x - 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, sx, sy, sz, boundary_mask)
                                  + fetch_boundary(destination, x, y, z - 1, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, sx, sy, sz, boundary_mask);
            destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
        }

        __global__ void zero_velocity_component_boundaries_kernel(float* field, const int sx, const int sy, const int sz, const int axis, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz) return;
            if (axis == 0) {
                if ((x == 0 && (boundary_mask & boundary_x_min_bit) == 0) || (x == sx - 1 && (boundary_mask & boundary_x_max_bit) == 0)) field[index_3d(x, y, z, sx, sy)] = 0.0f;
            } else if (axis == 1) {
                if ((y == 0 && (boundary_mask & boundary_y_min_bit) == 0) || (y == sy - 1 && (boundary_mask & boundary_y_max_bit) == 0)) field[index_3d(x, y, z, sx, sy)] = 0.0f;
            } else {
                if ((z == 0 && (boundary_mask & boundary_z_min_bit) == 0) || (z == sz - 1 && (boundary_mask & boundary_z_max_bit) == 0)) field[index_3d(x, y, z, sx, sy)] = 0.0f;
            }
        }

        __global__ void diffuse_velocity_component_kernel(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity, const uint32_t boundary_mask, const int axis) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;
            if (axis == 0 && ((x == 0 && (boundary_mask & boundary_x_min_bit) == 0) || (x == sx - 1 && (boundary_mask & boundary_x_max_bit) == 0))) {
                destination[index_3d(x, y, z, sx, sy)] = 0.0f;
                return;
            }
            if (axis == 1 && ((y == 0 && (boundary_mask & boundary_y_min_bit) == 0) || (y == sy - 1 && (boundary_mask & boundary_y_max_bit) == 0))) {
                destination[index_3d(x, y, z, sx, sy)] = 0.0f;
                return;
            }
            if (axis == 2 && ((z == 0 && (boundary_mask & boundary_z_min_bit) == 0) || (z == sz - 1 && (boundary_mask & boundary_z_max_bit) == 0))) {
                destination[index_3d(x, y, z, sx, sy)] = 0.0f;
                return;
            }
            const float neighbors = fetch_boundary(destination, x - 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x + 1, y, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y - 1, z, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y + 1, z, sx, sy, sz, boundary_mask)
                                  + fetch_boundary(destination, x, y, z - 1, sx, sy, sz, boundary_mask) + fetch_boundary(destination, x, y, z + 1, sx, sy, sz, boundary_mask);
            destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
        }

        __global__ void compute_poisson_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;
            rhs[index_3d(x, y, z, nx, ny)] = -(fetch_boundary(velocity_x, x + 1, y, z, nx + 1, ny, nz, boundary_mask) - fetch_boundary(velocity_x, x, y, z, nx + 1, ny, nz, boundary_mask) + fetch_boundary(velocity_y, x, y + 1, z, nx, ny + 1, nz, boundary_mask) - fetch_boundary(velocity_y, x, y, z, nx, ny + 1, nz, boundary_mask)
                                                 + fetch_boundary(velocity_z, x, y, z + 1, nx, ny, nz + 1, boundary_mask) - fetch_boundary(velocity_z, x, y, z, nx, ny, nz + 1, boundary_mask))
                                           * h;
        }

        __global__ void poisson_rbgs_kernel(float* pressure, const float* rhs, const int nx, const int ny, const int nz, const int parity, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;

            const bool periodic_x_min = (boundary_mask & boundary_x_min_bit) != 0;
            const bool periodic_x_max = (boundary_mask & boundary_x_max_bit) != 0;
            const bool periodic_y_min = (boundary_mask & boundary_y_min_bit) != 0;
            const bool periodic_y_max = (boundary_mask & boundary_y_max_bit) != 0;
            const bool periodic_z_min = (boundary_mask & boundary_z_min_bit) != 0;
            const bool periodic_z_max = (boundary_mask & boundary_z_max_bit) != 0;

            float sum = 0.0f;
            int count = 0;

            if (x > 0) {
                sum += pressure[index_3d(x - 1, y, z, nx, ny)];
                ++count;
            } else if (periodic_x_min) {
                sum += pressure[index_3d(nx - 1, y, z, nx, ny)];
                ++count;
            }

            if (x + 1 < nx) {
                sum += pressure[index_3d(x + 1, y, z, nx, ny)];
                ++count;
            } else if (periodic_x_max) {
                sum += pressure[index_3d(0, y, z, nx, ny)];
                ++count;
            }

            if (y > 0) {
                sum += pressure[index_3d(x, y - 1, z, nx, ny)];
                ++count;
            } else if (periodic_y_min) {
                sum += pressure[index_3d(x, ny - 1, z, nx, ny)];
                ++count;
            }

            if (y + 1 < ny) {
                sum += pressure[index_3d(x, y + 1, z, nx, ny)];
                ++count;
            } else if (periodic_y_max) {
                sum += pressure[index_3d(x, 0, z, nx, ny)];
                ++count;
            }

            if (z > 0) {
                sum += pressure[index_3d(x, y, z - 1, nx, ny)];
                ++count;
            } else if (periodic_z_min) {
                sum += pressure[index_3d(x, y, nz - 1, nx, ny)];
                ++count;
            }

            if (z + 1 < nz) {
                sum += pressure[index_3d(x, y, z + 1, nx, ny)];
                ++count;
            } else if (periodic_z_max) {
                sum += pressure[index_3d(x, y, 0, nx, ny)];
                ++count;
            }

            pressure[index_3d(x, y, z, nx, ny)] = count > 0 ? (sum + rhs[index_3d(x, y, z, nx, ny)]) / static_cast<float>(count) : 0.0f;
        }

        __global__ void restrict_poisson_residual_kernel(float* coarse_rhs, const float* fine_pressure, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const uint32_t boundary_mask) {
            const int x         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y         = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z         = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            const int coarse_nx = max(1, (fine_nx + 1) / 2);
            const int coarse_ny = max(1, (fine_ny + 1) / 2);
            const int coarse_nz = max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            const bool periodic_x_min = (boundary_mask & boundary_x_min_bit) != 0;
            const bool periodic_x_max = (boundary_mask & boundary_x_max_bit) != 0;
            const bool periodic_y_min = (boundary_mask & boundary_y_min_bit) != 0;
            const bool periodic_y_max = (boundary_mask & boundary_y_max_bit) != 0;
            const bool periodic_z_min = (boundary_mask & boundary_z_min_bit) != 0;
            const bool periodic_z_max = (boundary_mask & boundary_z_max_bit) != 0;

            float residual_sum = 0.0f;
            int samples        = 0;

            for (int fz = 2 * z; fz < min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < min(2 * x + 2, fine_nx); ++fx) {
                        float neighbors = 0.0f;
                        int count       = 0;

                        if (fx > 0) {
                            neighbors += fine_pressure[index_3d(fx - 1, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_x_min) {
                            neighbors += fine_pressure[index_3d(fine_nx - 1, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        }

                        if (fx + 1 < fine_nx) {
                            neighbors += fine_pressure[index_3d(fx + 1, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_x_max) {
                            neighbors += fine_pressure[index_3d(0, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        }

                        if (fy > 0) {
                            neighbors += fine_pressure[index_3d(fx, fy - 1, fz, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_y_min) {
                            neighbors += fine_pressure[index_3d(fx, fine_ny - 1, fz, fine_nx, fine_ny)];
                            ++count;
                        }

                        if (fy + 1 < fine_ny) {
                            neighbors += fine_pressure[index_3d(fx, fy + 1, fz, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_y_max) {
                            neighbors += fine_pressure[index_3d(fx, 0, fz, fine_nx, fine_ny)];
                            ++count;
                        }

                        if (fz > 0) {
                            neighbors += fine_pressure[index_3d(fx, fy, fz - 1, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_z_min) {
                            neighbors += fine_pressure[index_3d(fx, fy, fine_nz - 1, fine_nx, fine_ny)];
                            ++count;
                        }

                        if (fz + 1 < fine_nz) {
                            neighbors += fine_pressure[index_3d(fx, fy, fz + 1, fine_nx, fine_ny)];
                            ++count;
                        } else if (periodic_z_max) {
                            neighbors += fine_pressure[index_3d(fx, fy, 0, fine_nx, fine_ny)];
                            ++count;
                        }

                        const float applied = static_cast<float>(count) * fine_pressure[index_3d(fx, fy, fz, fine_nx, fine_ny)] - neighbors;
                        residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                        ++samples;
                    }
                }
            }

            coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
        }

        __global__ void restrict_diffusion_residual_kernel(float* coarse_rhs, const float* fine_solution, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const float alpha, const uint32_t boundary_mask) {
            const int x         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y         = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z         = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            const int coarse_nx = max(1, (fine_nx + 1) / 2);
            const int coarse_ny = max(1, (fine_ny + 1) / 2);
            const int coarse_nz = max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            float residual_sum = 0.0f;
            int samples        = 0;

            for (int fz = 2 * z; fz < min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < min(2 * x + 2, fine_nx); ++fx) {
                        const float center    = fine_solution[index_3d(fx, fy, fz, fine_nx, fine_ny)];
                        const float neighbors = fetch_boundary(fine_solution, fx - 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx + 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy - 1, fz, fine_nx, fine_ny, fine_nz, boundary_mask)
                                              + fetch_boundary(fine_solution, fx, fy + 1, fz, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy, fz - 1, fine_nx, fine_ny, fine_nz, boundary_mask) + fetch_boundary(fine_solution, fx, fy, fz + 1, fine_nx, fine_ny, fine_nz, boundary_mask);
                        const float applied = (1.0f + 6.0f * alpha) * center - alpha * neighbors;
                        residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                        ++samples;
                    }
                }
            }

            coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
        }

        __global__ void prolongate_add_kernel(float* fine_pressure, const float* coarse_pressure, const int fine_nx, const int fine_ny, const int fine_nz, const int coarse_nx, const int coarse_ny, const int coarse_nz, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= fine_nx || y >= fine_ny || z >= fine_nz) return;
            fine_pressure[index_3d(x, y, z, fine_nx, fine_ny)] += sample_grid(coarse_pressure, 0.5f * static_cast<float>(x) - 0.25f, 0.5f * static_cast<float>(y) - 0.25f, 0.5f * static_cast<float>(z) - 0.25f, coarse_nx, coarse_ny, coarse_nz, boundary_mask);
        }

        __global__ void project_velocity_kernel(float* velocity_x, float* velocity_y, float* velocity_z, const float* pressure, const int nx, const int ny, const int nz, const float inv_h, const uint32_t boundary_mask) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            const bool periodic_x_min = (boundary_mask & boundary_x_min_bit) != 0;
            const bool periodic_x_max = (boundary_mask & boundary_x_max_bit) != 0;
            const bool periodic_y_min = (boundary_mask & boundary_y_min_bit) != 0;
            const bool periodic_y_max = (boundary_mask & boundary_y_max_bit) != 0;
            const bool periodic_z_min = (boundary_mask & boundary_z_min_bit) != 0;
            const bool periodic_z_max = (boundary_mask & boundary_z_max_bit) != 0;

            if (x <= nx && y < ny && z < nz) {
                if (x == 0) {
                    if (!periodic_x_min) {
                        velocity_x[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                    } else {
                        velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(0, y, z, nx, ny)] - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
                    }
                } else if (x == nx) {
                    if (!periodic_x_max) {
                        velocity_x[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
                    } else {
                        velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(0, y, z, nx, ny)] - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
                    }
                } else {
                    velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
                }
            }

            if (x < nx && y <= ny && z < nz) {
                if (y == 0) {
                    if (!periodic_y_min) {
                        velocity_y[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                    } else {
                        velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, 0, z, nx, ny)] - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
                    }
                } else if (y == ny) {
                    if (!periodic_y_max) {
                        velocity_y[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
                    } else {
                        velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, 0, z, nx, ny)] - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
                    }
                } else {
                    velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
                }
            }

            if (x < nx && y < ny && z <= nz) {
                if (z == 0) {
                    if (!periodic_z_min) {
                        velocity_z[index_3d(x, y, z, nx, ny)] = 0.0f;
                    } else {
                        velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, 0, nx, ny)] - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
                    }
                } else if (z == nz) {
                    if (!periodic_z_max) {
                        velocity_z[index_3d(x, y, z, nx, ny)] = 0.0f;
                    } else {
                        velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, 0, nx, ny)] - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
                    }
                } else {
                    velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) * inv_h;
                }
            }
        }

        GridHierarchy build_hierarchy(const int base_nx, const int base_ny, const int base_nz, float* base_solution, float* base_rhs, float* coarse_solution_storage, float* coarse_rhs_storage) {
            GridHierarchy hierarchy{.level_count = 1};
            hierarchy.levels[0] = GridLevel{
                .nx       = base_nx,
                .ny       = base_ny,
                .nz       = base_nz,
                .scale    = 1.0f,
                .solution = base_solution,
                .rhs      = base_rhs,
            };

            std::uint64_t offset = 0;
            while (hierarchy.level_count < max_levels) {
                const GridLevel& previous = hierarchy.levels[hierarchy.level_count - 1];
                if (previous.nx <= 1 && previous.ny <= 1 && previous.nz <= 1) break;

                const int nx = std::max(1, (previous.nx + 1) / 2);
                const int ny = std::max(1, (previous.ny + 1) / 2);
                const int nz = std::max(1, (previous.nz + 1) / 2);

                hierarchy.levels[hierarchy.level_count] = GridLevel{
                    .nx       = nx,
                    .ny       = ny,
                    .nz       = nz,
                    .scale    = previous.scale * 0.25f,
                    .solution = coarse_solution_storage + offset,
                    .rhs      = coarse_rhs_storage + offset,
                };

                offset += static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
                ++hierarchy.level_count;
            }

            return hierarchy;
        }


        struct PoissonVCycleOps {
            uint32_t boundary_mask;

            int clear_coarse_solution(const GridLevel& level, const Stream stream) const {
                const auto bytes = static_cast<std::uint64_t>(level.nx) * static_cast<std::uint64_t>(level.ny) * static_cast<std::uint64_t>(level.nz) * sizeof(float);
                return cudaMemsetAsync(level.solution, 0, bytes, stream) == cudaSuccess ? 0 : 5001;
            }

            int smooth_level(const GridLevel& level, const dim3& block, const Stream stream, const int iterations) const {
                const dim3 grid = make_grid(level.nx, level.ny, level.nz, block);
                for (int i = 0; i < iterations; ++i) {
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 0, boundary_mask);
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 1, boundary_mask);
                }
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int restrict_residual(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                restrict_poisson_residual_kernel<<<make_grid(coarse.nx, coarse.ny, coarse.nz, block), block, 0, stream>>>(coarse.rhs, fine.solution, fine.rhs, fine.nx, fine.ny, fine.nz, boundary_mask);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int prolongate_and_postprocess(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                prolongate_add_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, coarse.solution, fine.nx, fine.ny, fine.nz, coarse.nx, coarse.ny, coarse.nz, boundary_mask);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }
        };

        struct DiffusionVCycleOps {
            float coefficient;
            uint32_t boundary_mask;
            BoundaryAxis boundary_axis;

            int clear_coarse_solution(const GridLevel& level, const Stream stream) const {
                const auto bytes = static_cast<std::uint64_t>(level.nx) * static_cast<std::uint64_t>(level.ny) * static_cast<std::uint64_t>(level.nz) * sizeof(float);
                return cudaMemsetAsync(level.solution, 0, bytes, stream) == cudaSuccess ? 0 : 5001;
            }

            int smooth_level(const GridLevel& level, const dim3& block, const Stream stream, const int iterations) const {
                const dim3 grid   = make_grid(level.nx, level.ny, level.nz, block);
                const float alpha = coefficient * level.scale;
                const float denom = 1.0f + 6.0f * alpha;

                if (boundary_axis == BoundaryAxis::none) {
                    for (int i = 0; i < iterations; ++i) {
                        diffuse_grid_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 0, boundary_mask);
                        diffuse_grid_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 1, boundary_mask);
                    }
                } else {
                    const int axis = static_cast<int>(boundary_axis);
                    for (int i = 0; i < iterations; ++i) {
                        diffuse_velocity_component_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 0, boundary_mask, axis);
                        diffuse_velocity_component_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 1, boundary_mask, axis);
                    }
                }

                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int restrict_residual(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                const float alpha = coefficient * fine.scale;
                restrict_diffusion_residual_kernel<<<make_grid(coarse.nx, coarse.ny, coarse.nz, block), block, 0, stream>>>(coarse.rhs, fine.solution, fine.rhs, fine.nx, fine.ny, fine.nz, alpha, boundary_mask);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int prolongate_and_postprocess(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                prolongate_add_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, coarse.solution, fine.nx, fine.ny, fine.nz, coarse.nx, coarse.ny, coarse.nz, boundary_mask);
                if (cudaGetLastError() != cudaSuccess) return 5001;

                if (boundary_axis != BoundaryAxis::none) {
                    zero_velocity_component_boundaries_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, fine.nx, fine.ny, fine.nz, static_cast<int>(boundary_axis), boundary_mask);
                    if (cudaGetLastError() != cudaSuccess) return 5001;
                }

                return 0;
            }
        };

        template <class TOps>
        int run_v_cycle(const GridHierarchy& hierarchy, const VCycleConfig& config, const TOps& ops, const dim3& block, const Stream stream) {
            for (int cycle = 0; cycle < config.cycles; ++cycle) {
                for (int level_index = 0; level_index + 1 < hierarchy.level_count; ++level_index) {
                    const GridLevel& fine   = hierarchy.levels[level_index];
                    const GridLevel& coarse = hierarchy.levels[level_index + 1];

                    if (const int code = ops.smooth_level(fine, block, stream, config.pre_smooth); code != 0) return code;
                    if (const int code = ops.clear_coarse_solution(coarse, stream); code != 0) return code;
                    if (const int code = ops.restrict_residual(fine, coarse, block, stream); code != 0) return code;
                }

                if (const int code = ops.smooth_level(hierarchy.levels[hierarchy.level_count - 1], block, stream, config.coarse_smooth); code != 0) return code;

                for (int level_index = hierarchy.level_count - 2; level_index >= 0; --level_index) {
                    const GridLevel& fine   = hierarchy.levels[level_index];
                    const GridLevel& coarse = hierarchy.levels[level_index + 1];

                    if (const int code = ops.prolongate_and_postprocess(fine, coarse, block, stream); code != 0) return code;
                    if (const int code = ops.smooth_level(fine, block, stream, config.post_smooth); code != 0) return code;
                }
            }

            return 0;
        }

    } // namespace

} // namespace stable_fluids

extern "C" {

int32_t stable_fluids_advect_velocity_cuda(const StableFluidsAdvectVelocityDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_advect_velocity_desc(desc); code != 0) return code;

    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_min_bit : 0u) | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_max_bit : 0u)
                                 | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_max_bit : 0u);
    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const auto stream        = static_cast<Stream>(desc->stream);

    const auto velocity_x_field_bytes = static_cast<std::uint64_t>(desc->nx + 1) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    const auto velocity_y_field_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny + 1) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    const auto velocity_z_field_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz + 1) * sizeof(float);

    auto* velocity_x_field     = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field     = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field     = static_cast<float*>(desc->velocity_z);
    auto* velocity_x_temporary = static_cast<float*>(desc->temporary_velocity_x);
    auto* velocity_y_temporary = static_cast<float*>(desc->temporary_velocity_y);
    auto* velocity_z_temporary = static_cast<float*>(desc->temporary_velocity_z);
    auto* velocity_x_previous  = static_cast<float*>(desc->temporary_previous_velocity_x);
    auto* velocity_y_previous  = static_cast<float*>(desc->temporary_previous_velocity_y);
    auto* velocity_z_previous  = static_cast<float*>(desc->temporary_previous_velocity_z);

    nvtx3::scoped_range range("stable.step.advect_velocity");
    if (cudaMemcpyAsync(velocity_x_previous, velocity_x_field, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    if (cudaMemcpyAsync(velocity_y_previous, velocity_y_field, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    if (cudaMemcpyAsync(velocity_z_previous, velocity_z_field, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    advect_velocity<<<velocity_grid, block, 0, stream>>>(velocity_x_temporary, velocity_y_temporary, velocity_z_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, boundary_mask);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t stable_fluids_diffuse_velocity_cuda(const StableFluidsDiffuseVelocityDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_diffuse_velocity_desc(desc); code != 0) return code;

    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_min_bit : 0u) | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_max_bit : 0u)
                                 | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_max_bit : 0u);
    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const auto stream = static_cast<Stream>(desc->stream);

    const auto velocity_x_field_bytes = static_cast<std::uint64_t>(desc->nx + 1) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    const auto velocity_y_field_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny + 1) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    const auto velocity_z_field_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz + 1) * sizeof(float);

    auto* velocity_x_field     = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field     = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field     = static_cast<float*>(desc->velocity_z);
    auto* velocity_x_temporary = static_cast<float*>(desc->temporary_velocity_x);
    auto* velocity_y_temporary = static_cast<float*>(desc->temporary_velocity_y);
    auto* velocity_z_temporary = static_cast<float*>(desc->temporary_velocity_z);
    auto* coarse_solution      = static_cast<float*>(desc->temporary_density);
    auto* coarse_rhs           = static_cast<float*>(desc->temporary_previous_density);

    nvtx3::scoped_range range("stable.step.diffuse_velocity");
    if (desc->viscosity <= 0.0f) {
        if (cudaMemcpyAsync(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        if (cudaMemcpyAsync(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        if (cudaMemcpyAsync(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        return 0;
    }

    const float diffusion_alpha = desc->dt * desc->viscosity / (desc->cell_size * desc->cell_size);
    const VCycleConfig diffusion_config{.cycles = std::max(1, desc->diffuse_iterations / 12), .pre_smooth = 1, .post_smooth = 1, .coarse_smooth = std::max(6, desc->diffuse_iterations / 4)};
    auto diffuse_velocity_component = [&](float* field, const float* source, const std::uint64_t field_bytes, const int sx, const int sy, const int sz, const BoundaryAxis axis) -> int32_t {
        GridHierarchy hierarchy = build_hierarchy(sx, sy, sz, field, const_cast<float*>(source), coarse_solution, coarse_rhs);
        if (cudaMemcpyAsync(field, source, field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        zero_velocity_component_boundaries_kernel<<<make_grid(sx, sy, sz, block), block, 0, stream>>>(field, sx, sy, sz, static_cast<int>(axis), boundary_mask);
        if (cudaGetLastError() != cudaSuccess) return 5001;
        const DiffusionVCycleOps ops{.coefficient = diffusion_alpha, .boundary_mask = boundary_mask, .boundary_axis = axis};
        return run_v_cycle(hierarchy, diffusion_config, ops, block, stream);
    };

    if (const int32_t code = diffuse_velocity_component(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, desc->nx + 1, desc->ny, desc->nz, BoundaryAxis::x); code != 0) return code;
    if (const int32_t code = diffuse_velocity_component(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, desc->nx, desc->ny + 1, desc->nz, BoundaryAxis::y); code != 0) return code;
    if (const int32_t code = diffuse_velocity_component(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, desc->nx, desc->ny, desc->nz + 1, BoundaryAxis::z); code != 0) return code;
    return 0;
}

int32_t stable_fluids_project_cuda(const StableFluidsProjectDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_project_desc(desc); code != 0) return code;

    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_min_bit : 0u) | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_max_bit : 0u)
                                 | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_max_bit : 0u);
    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells         = make_grid(desc->nx, desc->ny, desc->nz, block);
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const auto stream        = static_cast<Stream>(desc->stream);
    const auto cell_bytes    = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);

    auto* velocity_x = static_cast<float*>(desc->velocity_x);
    auto* velocity_y = static_cast<float*>(desc->velocity_y);
    auto* velocity_z = static_cast<float*>(desc->velocity_z);
    auto* pressure   = static_cast<float*>(desc->temporary_pressure);
    auto* divergence = static_cast<float*>(desc->temporary_divergence);
    auto* coarse_pressure_storage = static_cast<float*>(desc->temporary_density);
    auto* coarse_rhs_storage      = static_cast<float*>(desc->temporary_previous_density);

    const GridHierarchy pressure_hierarchy = build_hierarchy(desc->nx, desc->ny, desc->nz, pressure, divergence, coarse_pressure_storage, coarse_rhs_storage);

    nvtx3::scoped_range range("stable.step.project");
    if (cudaMemsetAsync(pressure, 0, cell_bytes, stream) != cudaSuccess) return 5001;
    compute_poisson_rhs_kernel<<<cells, block, 0, stream>>>(divergence, velocity_x, velocity_y, velocity_z, desc->nx, desc->ny, desc->nz, desc->cell_size, boundary_mask);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    const PoissonVCycleOps ops{.boundary_mask = boundary_mask};
    const VCycleConfig config{.cycles = std::max(1, desc->pressure_iterations / 40), .pre_smooth = 1, .post_smooth = 1, .coarse_smooth = std::max(8, desc->pressure_iterations / 10)};
    if (const int32_t code = run_v_cycle(pressure_hierarchy, config, ops, block, stream); code != 0) return code;
    project_velocity_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x, velocity_y, velocity_z, pressure, desc->nx, desc->ny, desc->nz, 1.0f / desc->cell_size, boundary_mask);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t stable_fluids_advect_density_cuda(const StableFluidsAdvectDensityDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_advect_density_desc(desc); code != 0) return code;

    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_y_min_bit : 0u)
                                 | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_y_max_bit : 0u) | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? stable_fluids::boundary_z_max_bit : 0u);

    const auto cell_bytes             = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    auto* density_field               = static_cast<float*>(desc->density);
    auto* density_temporary           = static_cast<float*>(desc->temporary_density);
    auto* density_previous            = static_cast<float*>(desc->temporary_previous_density);
    auto* velocity_x_field            = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field            = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field            = static_cast<float*>(desc->velocity_z);
    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells = make_grid(desc->nx, desc->ny, desc->nz, block);
    const auto stream        = static_cast<stable_fluids::Stream>(desc->stream);

    nvtx3::scoped_range range("stable.step.advect_density");
    if (cudaMemcpyAsync(density_previous, density_field, cell_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    advect_scalar_kernel<<<cells, block, 0, stream>>>(density_temporary, density_previous, velocity_x_field, velocity_y_field, velocity_z_field, desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, boundary_mask);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t stable_fluids_diffuse_density_cuda(const StableFluidsDiffuseDensityDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_diffuse_density_desc(desc); code != 0) return code;

    const uint32_t boundary_mask = (desc->boundary_x_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_min_bit : 0u) | (desc->boundary_x_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_x_max_bit : 0u) | (desc->boundary_y_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_min_bit : 0u) | (desc->boundary_y_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_y_max_bit : 0u)
                                 | (desc->boundary_z_min == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_min_bit : 0u) | (desc->boundary_z_max == STABLE_FLUIDS_BOUNDARY_PERIODIC ? boundary_z_max_bit : 0u);
    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells = make_grid(desc->nx, desc->ny, desc->nz, block);
    const auto stream = static_cast<Stream>(desc->stream);
    const auto cell_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);

    auto* density_field     = static_cast<float*>(desc->density);
    auto* density_temporary = static_cast<float*>(desc->temporary_density);
    auto* pressure          = static_cast<float*>(desc->temporary_pressure);
    auto* divergence        = static_cast<float*>(desc->temporary_divergence);

    nvtx3::scoped_range range("stable.step.diffuse_density");
    if (cudaMemcpyAsync(density_field, density_temporary, cell_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    if (desc->diffusion <= 0.0f) return 0;

    GridHierarchy hierarchy = build_hierarchy(desc->nx, desc->ny, desc->nz, density_field, density_temporary, pressure, divergence);
    const float diffusion_alpha = desc->dt * desc->diffusion / (desc->cell_size * desc->cell_size);
    const DiffusionVCycleOps ops{.coefficient = diffusion_alpha, .boundary_mask = boundary_mask, .boundary_axis = BoundaryAxis::none};
    const VCycleConfig config{.cycles = std::max(1, desc->diffuse_iterations / 12), .pre_smooth = 1, .post_smooth = 1, .coarse_smooth = std::max(6, desc->diffuse_iterations / 4)};
    if (const int32_t code = run_v_cycle(hierarchy, config, ops, block, stream); code != 0) return code;

    const float denom = 1.0f + 6.0f * diffusion_alpha;
    diffuse_grid_kernel<<<cells, block, 0, stream>>>(density_field, density_temporary, desc->nx, desc->ny, desc->nz, diffusion_alpha, denom, 0, boundary_mask);
    diffuse_grid_kernel<<<cells, block, 0, stream>>>(density_field, density_temporary, desc->nx, desc->ny, desc->nz, diffusion_alpha, denom, 1, boundary_mask);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t stable_fluids_add_scalar_source_cuda(const StableFluidsAddScalarSourceDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_add_scalar_source_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 grid = make_grid(desc->nx, desc->ny, desc->nz, block);
    const auto stream = static_cast<Stream>(desc->stream);
    add_scalar_source_kernel<<<grid, block, 0, stream>>>(static_cast<float*>(desc->scalar), desc->nx, desc->ny, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount, desc->sample_offset_x, desc->sample_offset_y, desc->sample_offset_z);
    return cudaGetLastError() == cudaSuccess ? 0 : 5001;
}

int32_t stable_fluids_add_vector_source_cuda(const StableFluidsAddVectorSourceDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_add_vector_source_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const auto stream = static_cast<Stream>(desc->stream);
    add_scalar_source_kernel<<<make_grid(desc->nx + 1, desc->ny, desc->nz, block), block, 0, stream>>>(static_cast<float*>(desc->vector_x), desc->nx + 1, desc->ny, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_x, 0.0f, 0.5f, 0.5f);
    add_scalar_source_kernel<<<make_grid(desc->nx, desc->ny + 1, desc->nz, block), block, 0, stream>>>(static_cast<float*>(desc->vector_y), desc->nx, desc->ny + 1, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_y, 0.5f, 0.0f, 0.5f);
    add_scalar_source_kernel<<<make_grid(desc->nx, desc->ny, desc->nz + 1, block), block, 0, stream>>>(static_cast<float*>(desc->vector_z), desc->nx, desc->ny, desc->nz + 1, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_z, 0.5f, 0.5f, 0.0f);
    return cudaGetLastError() == cudaSuccess ? 0 : 5001;
}
}
