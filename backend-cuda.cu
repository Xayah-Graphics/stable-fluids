#include "stable-fluids.h"
#include <algorithm>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {
    using Stream = cudaStream_t;

    namespace {

        constexpr int boundary_x_min_face = 0;
        constexpr int boundary_x_max_face = 1;
        constexpr int boundary_y_min_face = 2;
        constexpr int boundary_y_max_face = 3;
        constexpr int boundary_z_min_face = 4;
        constexpr int boundary_z_max_face = 5;

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

        struct InflowValues {
            float x_min;
            float x_max;
            float y_min;
            float y_max;
            float z_min;
            float z_max;
        };

        dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
            return {static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))};
        }

        __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
        }

        __device__ float scalar_boundary_cell_value(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int cx = std::clamp(x, 0, sx - 1);
            const int cy = std::clamp(y, 0, sy - 1);
            const int cz = std::clamp(z, 0, sz - 1);

            float sum = 0.0f;
            int count = 0;

            if (x < 0 && ((boundary_pack >> (boundary_x_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.x_min;
                ++count;
            }
            if (x >= sx && ((boundary_pack >> (boundary_x_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.x_max;
                ++count;
            }
            if (y < 0 && ((boundary_pack >> (boundary_y_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.y_min;
                ++count;
            }
            if (y >= sy && ((boundary_pack >> (boundary_y_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.y_max;
                ++count;
            }
            if (z < 0 && ((boundary_pack >> (boundary_z_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.z_min;
                ++count;
            }
            if (z >= sz && ((boundary_pack >> (boundary_z_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.z_max;
                ++count;
            }

            if (count > 0) return sum / static_cast<float>(count);
            return field[index_3d(cx, cy, cz, sx, sy)];
        }

        __device__ float fetch_scalar_boundary(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {
                return field[index_3d(x, y, z, sx, sy)];
            }
            return scalar_boundary_cell_value(field, x, y, z, sx, sy, sz, boundary_pack, inflow);
        }

        __device__ float velocity_boundary_value_for_face(const float* field, const int component_axis, const int face, const int x, const int y, const int z, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            const uint32_t type   = (boundary_pack >> (face * 3)) & 0x7u;
            const int normal_axis = (face == boundary_x_min_face || face == boundary_x_max_face) ? 0 : (face == boundary_y_min_face || face == boundary_y_max_face) ? 1 : 2;

            const int ix         = face == boundary_x_min_face ? (sx > 1 ? 1 : 0) : face == boundary_x_max_face ? (sx > 1 ? sx - 2 : 0) : std::clamp(x, 0, sx - 1);
            const int iy         = face == boundary_y_min_face ? (sy > 1 ? 1 : 0) : face == boundary_y_max_face ? (sy > 1 ? sy - 2 : 0) : std::clamp(y, 0, sy - 1);
            const int iz         = face == boundary_z_min_face ? (sz > 1 ? 1 : 0) : face == boundary_z_max_face ? (sz > 1 ? sz - 2 : 0) : std::clamp(z, 0, sz - 1);
            const float interior = field[index_3d(ix, iy, iz, sx, sy)];

            if (component_axis == normal_axis) {
                if (type == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                    switch (face) {
                    case boundary_x_min_face: return inflow.x_min;
                    case boundary_x_max_face: return inflow.x_max;
                    case boundary_y_min_face: return inflow.y_min;
                    case boundary_y_max_face: return inflow.y_max;
                    case boundary_z_min_face: return inflow.z_min;
                    case boundary_z_max_face: return inflow.z_max;
                    default: return 0.0f;
                    }
                }
                if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW) return interior;
                return 0.0f;
            }

            if (type == STABLE_FLUIDS_BOUNDARY_NO_SLIP) return 0.0f;
            return interior;
        }

        __device__ float fetch_velocity_boundary(const float* field, const int component_axis, const int x, const int y, const int z, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {
                return field[index_3d(x, y, z, sx, sy)];
            }

            float sum = 0.0f;
            int count = 0;

            if (x < 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_x_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (x >= sx) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_x_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (y < 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_y_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (y >= sy) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_y_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (z < 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_z_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (z >= sz) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_z_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }

            if (count > 0) return sum / static_cast<float>(count);

            const int cx = std::clamp(x, 0, sx - 1);
            const int cy = std::clamp(y, 0, sy - 1);
            const int cz = std::clamp(z, 0, sz - 1);
            return field[index_3d(cx, cy, cz, sx, sy)];
        }

        __device__ bool scalar_cell_is_inflow_boundary(const int x, const int y, const int z, const int nx, const int ny, const int nz, const uint32_t boundary_pack) {

            if (x == 0 && ((boundary_pack >> (boundary_x_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            if (x == nx - 1 && ((boundary_pack >> (boundary_x_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            if (y == 0 && ((boundary_pack >> (boundary_y_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            if (y == ny - 1 && ((boundary_pack >> (boundary_y_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            if (z == 0 && ((boundary_pack >> (boundary_z_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            if (z == nz - 1 && ((boundary_pack >> (boundary_z_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) return true;
            return false;
        }

        __device__ float scalar_inflow_boundary_value(const int x, const int y, const int z, const int nx, const int ny, const int nz, const uint32_t boundary_pack, const InflowValues& inflow) {

            float sum = 0.0f;
            int count = 0;

            if (x == 0 && ((boundary_pack >> (boundary_x_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.x_min;
                ++count;
            }
            if (x == nx - 1 && ((boundary_pack >> (boundary_x_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.x_max;
                ++count;
            }
            if (y == 0 && ((boundary_pack >> (boundary_y_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.y_min;
                ++count;
            }
            if (y == ny - 1 && ((boundary_pack >> (boundary_y_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.y_max;
                ++count;
            }
            if (z == 0 && ((boundary_pack >> (boundary_z_min_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.z_min;
                ++count;
            }
            if (z == nz - 1 && ((boundary_pack >> (boundary_z_max_face * 3)) & 0x7u) == STABLE_FLUIDS_BOUNDARY_INFLOW) {
                sum += inflow.z_max;
                ++count;
            }

            return count > 0 ? sum / static_cast<float>(count) : 0.0f;
        }

        __device__ float velocity_boundary_value_at_index(const float* field, const int component_axis, const int x, const int y, const int z, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            float sum = 0.0f;
            int count = 0;

            if (x == 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_x_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (x == sx - 1) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_x_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (y == 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_y_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (y == sy - 1) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_y_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (z == 0) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_z_min_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }
            if (z == sz - 1) {
                sum += velocity_boundary_value_for_face(field, component_axis, boundary_z_max_face, x, y, z, sx, sy, sz, boundary_pack, inflow);
                ++count;
            }

            return count > 0 ? (sum / static_cast<float>(count)) : field[index_3d(x, y, z, sx, sy)];
        }

        __device__ float sample_scalar_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            gx = std::clamp(gx, -0.5f, static_cast<float>(sx) - 0.5f);
            gy = std::clamp(gy, -0.5f, static_cast<float>(sy) - 0.5f);
            gz = std::clamp(gz, -0.5f, static_cast<float>(sz) - 0.5f);

            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const int z1 = z0 + 1;

            const float tx = gx - static_cast<float>(x0);
            const float ty = gy - static_cast<float>(y0);
            const float tz = gz - static_cast<float>(z0);

            const float c000 = fetch_scalar_boundary(field, x0, y0, z0, sx, sy, sz, boundary_pack, inflow);
            const float c100 = fetch_scalar_boundary(field, x1, y0, z0, sx, sy, sz, boundary_pack, inflow);
            const float c010 = fetch_scalar_boundary(field, x0, y1, z0, sx, sy, sz, boundary_pack, inflow);
            const float c110 = fetch_scalar_boundary(field, x1, y1, z0, sx, sy, sz, boundary_pack, inflow);
            const float c001 = fetch_scalar_boundary(field, x0, y0, z1, sx, sy, sz, boundary_pack, inflow);
            const float c101 = fetch_scalar_boundary(field, x1, y0, z1, sx, sy, sz, boundary_pack, inflow);
            const float c011 = fetch_scalar_boundary(field, x0, y1, z1, sx, sy, sz, boundary_pack, inflow);
            const float c111 = fetch_scalar_boundary(field, x1, y1, z1, sx, sy, sz, boundary_pack, inflow);

            const float c00 = c000 + (c100 - c000) * tx;
            const float c10 = c010 + (c110 - c010) * tx;
            const float c01 = c001 + (c101 - c001) * tx;
            const float c11 = c011 + (c111 - c011) * tx;
            const float c0  = c00 + (c10 - c00) * ty;
            const float c1  = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }

        __device__ float sample_velocity_component_grid(const float* field, const int component_axis, float gx, float gy, float gz, const int sx, const int sy, const int sz, const uint32_t boundary_pack, const InflowValues& inflow) {

            gx = std::clamp(gx, -0.5f, static_cast<float>(sx) - 0.5f);
            gy = std::clamp(gy, -0.5f, static_cast<float>(sy) - 0.5f);
            gz = std::clamp(gz, -0.5f, static_cast<float>(sz) - 0.5f);

            const int x0 = static_cast<int>(floorf(gx));
            const int y0 = static_cast<int>(floorf(gy));
            const int z0 = static_cast<int>(floorf(gz));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const int z1 = z0 + 1;

            const float tx = gx - static_cast<float>(x0);
            const float ty = gy - static_cast<float>(y0);
            const float tz = gz - static_cast<float>(z0);

            const float c000 = fetch_velocity_boundary(field, component_axis, x0, y0, z0, sx, sy, sz, boundary_pack, inflow);
            const float c100 = fetch_velocity_boundary(field, component_axis, x1, y0, z0, sx, sy, sz, boundary_pack, inflow);
            const float c010 = fetch_velocity_boundary(field, component_axis, x0, y1, z0, sx, sy, sz, boundary_pack, inflow);
            const float c110 = fetch_velocity_boundary(field, component_axis, x1, y1, z0, sx, sy, sz, boundary_pack, inflow);
            const float c001 = fetch_velocity_boundary(field, component_axis, x0, y0, z1, sx, sy, sz, boundary_pack, inflow);
            const float c101 = fetch_velocity_boundary(field, component_axis, x1, y0, z1, sx, sy, sz, boundary_pack, inflow);
            const float c011 = fetch_velocity_boundary(field, component_axis, x0, y1, z1, sx, sy, sz, boundary_pack, inflow);
            const float c111 = fetch_velocity_boundary(field, component_axis, x1, y1, z1, sx, sy, sz, boundary_pack, inflow);

            const float c00 = c000 + (c100 - c000) * tx;
            const float c10 = c010 + (c110 - c010) * tx;
            const float c01 = c001 + (c101 - c001) * tx;
            const float c11 = c011 + (c111 - c011) * tx;
            const float c0  = c00 + (c10 - c00) * ty;
            const float c1  = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }

        __device__ float sample_scalar(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_pack, const InflowValues& inflow) {
            return sample_scalar_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h - 0.5f, nx, ny, nz, boundary_pack, inflow);
        }

        __device__ float sample_u(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_pack, const InflowValues& inflow) {
            return sample_velocity_component_grid(field, 0, pos.x / h, pos.y / h - 0.5f, pos.z / h - 0.5f, nx + 1, ny, nz, boundary_pack, inflow);
        }

        __device__ float sample_v(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_pack, const InflowValues& inflow) {
            return sample_velocity_component_grid(field, 1, pos.x / h - 0.5f, pos.y / h, pos.z / h - 0.5f, nx, ny + 1, nz, boundary_pack, inflow);
        }

        __device__ float sample_w(const float* field, const float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_pack, const InflowValues& inflow) {
            return sample_velocity_component_grid(field, 2, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h, nx, ny, nz + 1, boundary_pack, inflow);
        }

        __device__ float3 clamp_domain(const float3 pos, const int nx, const int ny, const int nz, const float h) {
            return make_float3(std::clamp(pos.x, 0.0f, static_cast<float>(nx) * h), std::clamp(pos.y, 0.0f, static_cast<float>(ny) * h), std::clamp(pos.z, 0.0f, static_cast<float>(nz) * h));
        }

        __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, float3 pos, const int nx, const int ny, const int nz, const float h, const uint32_t boundary_pack, const InflowValues& inflow) {

            pos = clamp_domain(pos, nx, ny, nz, h);
            return make_float3(sample_u(velocity_x, pos, nx, ny, nz, h, boundary_pack, inflow), sample_v(velocity_y, pos, nx, ny, nz, h, boundary_pack, inflow), sample_w(velocity_z, pos, nx, ny, nz, h, boundary_pack, inflow));
        }

        __global__ void enforce_velocity_boundaries_kernel(float* velocity_x, float* velocity_y, float* velocity_z, const int nx, const int ny, const int nz, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            if (x <= nx && y < ny && z < nz) {
                const int sx = nx + 1;
                const int sy = ny;
                const int sz = nz;
                if (x == 0 || x == sx - 1 || y == 0 || y == sy - 1 || z == 0 || z == sz - 1) {
                    velocity_x[index_3d(x, y, z, sx, sy)] = velocity_boundary_value_at_index(velocity_x, 0, x, y, z, sx, sy, sz, boundary_pack, inflow);
                }
            }

            if (x < nx && y <= ny && z < nz) {
                const int sx = nx;
                const int sy = ny + 1;
                const int sz = nz;
                if (x == 0 || x == sx - 1 || y == 0 || y == sy - 1 || z == 0 || z == sz - 1) {
                    velocity_y[index_3d(x, y, z, sx, sy)] = velocity_boundary_value_at_index(velocity_y, 1, x, y, z, sx, sy, sz, boundary_pack, inflow);
                }
            }

            if (x < nx && y < ny && z <= nz) {
                const int sx = nx;
                const int sy = ny;
                const int sz = nz + 1;
                if (x == 0 || x == sx - 1 || y == 0 || y == sy - 1 || z == 0 || z == sz - 1) {
                    velocity_z[index_3d(x, y, z, sx, sy)] = velocity_boundary_value_at_index(velocity_z, 2, x, y, z, sx, sy, sz, boundary_pack, inflow);
                }
            }
        }

        __global__ void advect_velocity(float* velocity_x_destination, float* velocity_y_destination, float* velocity_z_destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            if (x <= nx && y < ny && z < nz) {
                const float3 pos                                      = make_float3(static_cast<float>(x) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
                const float3 vel                                      = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_pack, inflow);
                const float3 back                                     = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
                velocity_x_destination[index_3d(x, y, z, nx + 1, ny)] = sample_u(source_x, back, nx, ny, nz, h, boundary_pack, inflow);
            }

            if (x < nx && y <= ny && z < nz) {
                const float3 pos                                      = make_float3((static_cast<float>(x) + 0.5f) * h, static_cast<float>(y) * h, (static_cast<float>(z) + 0.5f) * h);
                const float3 vel                                      = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_pack, inflow);
                const float3 back                                     = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
                velocity_y_destination[index_3d(x, y, z, nx, ny + 1)] = sample_v(source_y, back, nx, ny, nz, h, boundary_pack, inflow);
            }

            if (x < nx && y < ny && z <= nz) {
                const float3 pos                                  = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, static_cast<float>(z) * h);
                const float3 vel                                  = sample_velocity(source_x, source_y, source_z, pos, nx, ny, nz, h, boundary_pack, inflow);
                const float3 back                                 = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
                velocity_z_destination[index_3d(x, y, z, nx, ny)] = sample_w(source_z, back, nx, ny, nz, h, boundary_pack, inflow);
            }
        }

        __global__ void advect_scalar_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t boundary_pack, const InflowValues& inflow, const int clamp_non_negative) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;

            if (scalar_cell_is_inflow_boundary(x, y, z, nx, ny, nz, boundary_pack)) {
                destination[index_3d(x, y, z, nx, ny)] = scalar_inflow_boundary_value(x, y, z, nx, ny, nz, boundary_pack, inflow);
                return;
            }

            const float3 pos                       = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
            const float3 vel                       = sample_velocity(velocity_x, velocity_y, velocity_z, pos, nx, ny, nz, h, boundary_pack, inflow);
            const float3 back                      = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
            const float value                      = sample_scalar(source, back, nx, ny, nz, h, boundary_pack, inflow);
            destination[index_3d(x, y, z, nx, ny)] = clamp_non_negative != 0 ? fmaxf(0.0f, value) : value;
        }

        __global__ void apply_scalar_inflow_boundaries_kernel(float* scalar, const int nx, const int ny, const int nz, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;

            if (scalar_cell_is_inflow_boundary(x, y, z, nx, ny, nz, boundary_pack)) {
                scalar[index_3d(x, y, z, nx, ny)] = scalar_inflow_boundary_value(x, y, z, nx, ny, nz, boundary_pack, inflow);
            }
        }

        __global__ void add_scalar_source_kernel(float* destination, const int sx, const int sy, const int sz, const float center_x, const float center_y, const float center_z, const float radius, const float amount, const float sample_offset_x, const float sample_offset_y, const float sample_offset_z) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz) return;

            const float px      = static_cast<float>(x) + sample_offset_x;
            const float py      = static_cast<float>(y) + sample_offset_y;
            const float pz      = static_cast<float>(z) + sample_offset_z;
            const float dx      = px - center_x;
            const float dy      = py - center_y;
            const float dz      = pz - center_z;
            const float radius2 = radius * radius;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2) return;
            destination[index_3d(x, y, z, sx, sy)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
        }

        __global__ void compute_staggered_velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;

            const float vx = 0.5f * (velocity_x[index_3d(x, y, z, nx + 1, ny)] + velocity_x[index_3d(x + 1, y, z, nx + 1, ny)]);
            const float vy = 0.5f * (velocity_y[index_3d(x, y, z, nx, ny + 1)] + velocity_y[index_3d(x, y + 1, z, nx, ny + 1)]);
            const float vz = 0.5f * (velocity_z[index_3d(x, y, z, nx, ny)] + velocity_z[index_3d(x, y, z + 1, nx, ny)]);
            destination[index_3d(x, y, z, nx, ny)] = sqrtf(vx * vx + vy * vy + vz * vz);
        }

        __global__ void diffuse_grid_kernel(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;

            if (scalar_cell_is_inflow_boundary(x, y, z, sx, sy, sz, boundary_pack)) {
                destination[index_3d(x, y, z, sx, sy)] = scalar_inflow_boundary_value(x, y, z, sx, sy, sz, boundary_pack, inflow);
                return;
            }

            const float neighbors = fetch_scalar_boundary(destination, x - 1, y, z, sx, sy, sz, boundary_pack, inflow) + fetch_scalar_boundary(destination, x + 1, y, z, sx, sy, sz, boundary_pack, inflow) + fetch_scalar_boundary(destination, x, y - 1, z, sx, sy, sz, boundary_pack, inflow)
                                  + fetch_scalar_boundary(destination, x, y + 1, z, sx, sy, sz, boundary_pack, inflow) + fetch_scalar_boundary(destination, x, y, z - 1, sx, sy, sz, boundary_pack, inflow) + fetch_scalar_boundary(destination, x, y, z + 1, sx, sy, sz, boundary_pack, inflow);

            destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
        }

        __global__ void diffuse_velocity_component_kernel(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int parity, const uint32_t boundary_pack, const InflowValues& inflow, const int axis) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;

            if (x == 0 || x == sx - 1 || y == 0 || y == sy - 1 || z == 0 || z == sz - 1) {
                destination[index_3d(x, y, z, sx, sy)] = velocity_boundary_value_at_index(destination, axis, x, y, z, sx, sy, sz, boundary_pack, inflow);
                return;
            }

            const float neighbors = fetch_velocity_boundary(destination, axis, x - 1, y, z, sx, sy, sz, boundary_pack, inflow) + fetch_velocity_boundary(destination, axis, x + 1, y, z, sx, sy, sz, boundary_pack, inflow) + fetch_velocity_boundary(destination, axis, x, y - 1, z, sx, sy, sz, boundary_pack, inflow)
                                  + fetch_velocity_boundary(destination, axis, x, y + 1, z, sx, sy, sz, boundary_pack, inflow) + fetch_velocity_boundary(destination, axis, x, y, z - 1, sx, sy, sz, boundary_pack, inflow) + fetch_velocity_boundary(destination, axis, x, y, z + 1, sx, sy, sz, boundary_pack, inflow);

            destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
        }

        __device__ void pressure_row_info(const float* pressure, const int x, const int y, const int z, const int nx, const int ny, const int nz, const uint32_t boundary_pack, float& sum, int& diag) {

            sum  = 0.0f;
            diag = 0;

            auto add_neighbor = [&](const int nxp, const int nyp, const int nzp, const int face_if_outside) {
                if (nxp >= 0 && nxp < nx && nyp >= 0 && nyp < ny && nzp >= 0 && nzp < nz) {
                    sum += pressure[index_3d(nxp, nyp, nzp, nx, ny)];
                    ++diag;
                    return;
                }

                const uint32_t type = (boundary_pack >> (face_if_outside * 3)) & 0x7u;
                if (type == STABLE_FLUIDS_BOUNDARY_INFLOW || type == STABLE_FLUIDS_BOUNDARY_OUTFLOW) {
                    ++diag;
                }
            };

            add_neighbor(x - 1, y, z, boundary_x_min_face);
            add_neighbor(x + 1, y, z, boundary_x_max_face);
            add_neighbor(x, y - 1, z, boundary_y_min_face);
            add_neighbor(x, y + 1, z, boundary_y_max_face);
            add_neighbor(x, y, z - 1, boundary_z_min_face);
            add_neighbor(x, y, z + 1, boundary_z_max_face);
        }

        __global__ void compute_poisson_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;

            rhs[index_3d(x, y, z, nx, ny)] = -(velocity_x[index_3d(x + 1, y, z, nx + 1, ny)] - velocity_x[index_3d(x, y, z, nx + 1, ny)] + velocity_y[index_3d(x, y + 1, z, nx, ny + 1)] - velocity_y[index_3d(x, y, z, nx, ny + 1)] + velocity_z[index_3d(x, y, z + 1, nx, ny)] - velocity_z[index_3d(x, y, z, nx, ny)]) * h;
        }

        __global__ void poisson_rbgs_kernel(float* pressure, const float* rhs, const int nx, const int ny, const int nz, const int parity, const uint32_t boundary_pack) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;

            float sum = 0.0f;
            int diag  = 0;
            pressure_row_info(pressure, x, y, z, nx, ny, nz, boundary_pack, sum, diag);

            pressure[index_3d(x, y, z, nx, ny)] = diag > 0 ? (sum + rhs[index_3d(x, y, z, nx, ny)]) / static_cast<float>(diag) : 0.0f;
        }

        __global__ void restrict_poisson_residual_kernel(float* coarse_rhs, const float* fine_pressure, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const uint32_t boundary_pack) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
            const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
            const int coarse_nz = std::max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            float residual_sum = 0.0f;
            int samples        = 0;

            for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                        float sum = 0.0f;
                        int diag  = 0;
                        pressure_row_info(fine_pressure, fx, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack, sum, diag);
                        const float applied = static_cast<float>(diag) * fine_pressure[index_3d(fx, fy, fz, fine_nx, fine_ny)] - sum;
                        residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                        ++samples;
                    }
                }
            }

            coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
        }

        __global__ void restrict_diffusion_residual_kernel(float* coarse_rhs, const float* fine_solution, const float* fine_rhs, const int fine_nx, const int fine_ny, const int fine_nz, const float alpha, const uint32_t boundary_pack, const InflowValues& inflow, const int boundary_axis) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
            const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
            const int coarse_nz = std::max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            float residual_sum = 0.0f;
            int samples        = 0;

            for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                        if (boundary_axis < 0) {
                            if (scalar_cell_is_inflow_boundary(fx, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack)) {
                                ++samples;
                                continue;
                            }
                            const float center    = fine_solution[index_3d(fx, fy, fz, fine_nx, fine_ny)];
                            const float neighbors = fetch_scalar_boundary(fine_solution, fx - 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_scalar_boundary(fine_solution, fx + 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow)
                                                  + fetch_scalar_boundary(fine_solution, fx, fy - 1, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_scalar_boundary(fine_solution, fx, fy + 1, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow)
                                                  + fetch_scalar_boundary(fine_solution, fx, fy, fz - 1, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_scalar_boundary(fine_solution, fx, fy, fz + 1, fine_nx, fine_ny, fine_nz, boundary_pack, inflow);
                            const float applied = (1.0f + 6.0f * alpha) * center - alpha * neighbors;
                            residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                        } else {
                            if (fx == 0 || fx == fine_nx - 1 || fy == 0 || fy == fine_ny - 1 || fz == 0 || fz == fine_nz - 1) {
                                ++samples;
                                continue;
                            }
                            const float center    = fine_solution[index_3d(fx, fy, fz, fine_nx, fine_ny)];
                            const float neighbors = fetch_velocity_boundary(fine_solution, boundary_axis, fx - 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_velocity_boundary(fine_solution, boundary_axis, fx + 1, fy, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow)
                                                  + fetch_velocity_boundary(fine_solution, boundary_axis, fx, fy - 1, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_velocity_boundary(fine_solution, boundary_axis, fx, fy + 1, fz, fine_nx, fine_ny, fine_nz, boundary_pack, inflow)
                                                  + fetch_velocity_boundary(fine_solution, boundary_axis, fx, fy, fz - 1, fine_nx, fine_ny, fine_nz, boundary_pack, inflow) + fetch_velocity_boundary(fine_solution, boundary_axis, fx, fy, fz + 1, fine_nx, fine_ny, fine_nz, boundary_pack, inflow);
                            const float applied = (1.0f + 6.0f * alpha) * center - alpha * neighbors;
                            residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - applied;
                        }
                        ++samples;
                    }
                }
            }

            coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
        }

        __global__ void prolongate_add_kernel(float* fine_pressure, const float* coarse_pressure, const int fine_nx, const int fine_ny, const int fine_nz, const int coarse_nx, const int coarse_ny, const int coarse_nz) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= fine_nx || y >= fine_ny || z >= fine_nz) return;

            const float gx = 0.5f * static_cast<float>(x) - 0.25f;
            const float gy = 0.5f * static_cast<float>(y) - 0.25f;
            const float gz = 0.5f * static_cast<float>(z) - 0.25f;

            const int x0 = std::clamp(static_cast<int>(floorf(gx)), 0, coarse_nx - 1);
            const int y0 = std::clamp(static_cast<int>(floorf(gy)), 0, coarse_ny - 1);
            const int z0 = std::clamp(static_cast<int>(floorf(gz)), 0, coarse_nz - 1);
            const int x1 = std::clamp(x0 + 1, 0, coarse_nx - 1);
            const int y1 = std::clamp(y0 + 1, 0, coarse_ny - 1);
            const int z1 = std::clamp(z0 + 1, 0, coarse_nz - 1);

            const float tx = gx - floorf(gx);
            const float ty = gy - floorf(gy);
            const float tz = gz - floorf(gz);

            const float c000 = coarse_pressure[index_3d(x0, y0, z0, coarse_nx, coarse_ny)];
            const float c100 = coarse_pressure[index_3d(x1, y0, z0, coarse_nx, coarse_ny)];
            const float c010 = coarse_pressure[index_3d(x0, y1, z0, coarse_nx, coarse_ny)];
            const float c110 = coarse_pressure[index_3d(x1, y1, z0, coarse_nx, coarse_ny)];
            const float c001 = coarse_pressure[index_3d(x0, y0, z1, coarse_nx, coarse_ny)];
            const float c101 = coarse_pressure[index_3d(x1, y0, z1, coarse_nx, coarse_ny)];
            const float c011 = coarse_pressure[index_3d(x0, y1, z1, coarse_nx, coarse_ny)];
            const float c111 = coarse_pressure[index_3d(x1, y1, z1, coarse_nx, coarse_ny)];

            const float c00 = c000 + (c100 - c000) * tx;
            const float c10 = c010 + (c110 - c010) * tx;
            const float c01 = c001 + (c101 - c001) * tx;
            const float c11 = c011 + (c111 - c011) * tx;
            const float c0  = c00 + (c10 - c00) * ty;
            const float c1  = c01 + (c11 - c01) * ty;
            fine_pressure[index_3d(x, y, z, fine_nx, fine_ny)] += c0 + (c1 - c0) * tz;
        }

        __global__ void project_velocity_kernel(float* velocity_x, float* velocity_y, float* velocity_z, const float* pressure, const int nx, const int ny, const int nz, const float inv_h, const uint32_t boundary_pack, const InflowValues& inflow) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            if (x <= nx && y < ny && z < nz) {
                float& u = velocity_x[index_3d(x, y, z, nx + 1, ny)];
                if (x == 0) {
                    const uint32_t type = (boundary_pack >> (boundary_x_min_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        u = inflow.x_min;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        u -= (pressure[index_3d(0, y, z, nx, ny)] - 0.0f) * inv_h;
                    else
                        u = 0.0f;
                } else if (x == nx) {
                    const uint32_t type = (boundary_pack >> (boundary_x_max_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        u = inflow.x_max;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        u -= (0.0f - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
                    else
                        u = 0.0f;
                } else {
                    u -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
                }
            }

            if (x < nx && y <= ny && z < nz) {
                float& v = velocity_y[index_3d(x, y, z, nx, ny + 1)];
                if (y == 0) {
                    const uint32_t type = (boundary_pack >> (boundary_y_min_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        v = inflow.y_min;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        v -= (pressure[index_3d(x, 0, z, nx, ny)] - 0.0f) * inv_h;
                    else
                        v = 0.0f;
                } else if (y == ny) {
                    const uint32_t type = (boundary_pack >> (boundary_y_max_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        v = inflow.y_max;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        v -= (0.0f - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
                    else
                        v = 0.0f;
                } else {
                    v -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
                }
            }

            if (x < nx && y < ny && z <= nz) {
                float& w = velocity_z[index_3d(x, y, z, nx, ny)];
                if (z == 0) {
                    const uint32_t type = (boundary_pack >> (boundary_z_min_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        w = inflow.z_min;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        w -= (pressure[index_3d(x, y, 0, nx, ny)] - 0.0f) * inv_h;
                    else
                        w = 0.0f;
                } else if (z == nz) {
                    const uint32_t type = (boundary_pack >> (boundary_z_max_face * 3)) & 0x7u;
                    if (type == STABLE_FLUIDS_BOUNDARY_INFLOW)
                        w = inflow.z_max;
                    else if (type == STABLE_FLUIDS_BOUNDARY_OUTFLOW)
                        w -= (0.0f - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
                    else
                        w = 0.0f;
                } else {
                    w -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) * inv_h;
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
            uint32_t boundary_pack;

            static int clear_coarse_solution(const GridLevel& level, const Stream stream) {
                const auto bytes = static_cast<std::uint64_t>(level.nx) * static_cast<std::uint64_t>(level.ny) * static_cast<std::uint64_t>(level.nz) * sizeof(float);
                return cudaMemsetAsync(level.solution, 0, bytes, stream) == cudaSuccess ? 0 : 5001;
            }

            int smooth_level(const GridLevel& level, const dim3& block, const Stream stream, const int iterations) const {
                const dim3 grid = make_grid(level.nx, level.ny, level.nz, block);
                for (int i = 0; i < iterations; ++i) {
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 0, boundary_pack);
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 1, boundary_pack);
                }
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int restrict_residual(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                restrict_poisson_residual_kernel<<<make_grid(coarse.nx, coarse.ny, coarse.nz, block), block, 0, stream>>>(coarse.rhs, fine.solution, fine.rhs, fine.nx, fine.ny, fine.nz, boundary_pack);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            static int prolongate_and_postprocess(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) {
                prolongate_add_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, coarse.solution, fine.nx, fine.ny, fine.nz, coarse.nx, coarse.ny, coarse.nz);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }
        };

        struct DiffusionVCycleOps {
            float coefficient;
            uint32_t boundary_pack;
            InflowValues inflow;
            BoundaryAxis boundary_axis;

            static int clear_coarse_solution(const GridLevel& level, const Stream stream) {
                const auto bytes = static_cast<std::uint64_t>(level.nx) * static_cast<std::uint64_t>(level.ny) * static_cast<std::uint64_t>(level.nz) * sizeof(float);
                return cudaMemsetAsync(level.solution, 0, bytes, stream) == cudaSuccess ? 0 : 5001;
            }

            int smooth_level(const GridLevel& level, const dim3& block, const Stream stream, const int iterations) const {
                const dim3 grid   = make_grid(level.nx, level.ny, level.nz, block);
                const float alpha = coefficient * level.scale;
                const float denom = 1.0f + 6.0f * alpha;

                if (boundary_axis == BoundaryAxis::none) {
                    for (int i = 0; i < iterations; ++i) {
                        diffuse_grid_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 0, boundary_pack, inflow);
                        diffuse_grid_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 1, boundary_pack, inflow);
                    }
                } else {
                    const int axis = static_cast<int>(boundary_axis);
                    for (int i = 0; i < iterations; ++i) {
                        diffuse_velocity_component_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 0, boundary_pack, inflow, axis);
                        diffuse_velocity_component_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, alpha, denom, 1, boundary_pack, inflow, axis);
                    }
                }

                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int restrict_residual(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                const float alpha = coefficient * fine.scale;
                restrict_diffusion_residual_kernel<<<make_grid(coarse.nx, coarse.ny, coarse.nz, block), block, 0, stream>>>(coarse.rhs, fine.solution, fine.rhs, fine.nx, fine.ny, fine.nz, alpha, boundary_pack, inflow, static_cast<int>(boundary_axis));
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            static int prolongate_and_postprocess(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) {
                prolongate_add_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, coarse.solution, fine.nx, fine.ny, fine.nz, coarse.nx, coarse.ny, coarse.nz);
                if (cudaGetLastError() != cudaSuccess) return 5001;
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

    const uint32_t boundary_pack = (desc->boundary_x_min << (boundary_x_min_face * 3)) | (desc->boundary_x_max << (boundary_x_max_face * 3)) | (desc->boundary_y_min << (boundary_y_min_face * 3)) | (desc->boundary_y_max << (boundary_y_max_face * 3)) | (desc->boundary_z_min << (boundary_z_min_face * 3)) | (desc->boundary_z_max << (boundary_z_max_face * 3));
    const InflowValues inflow{
        .x_min = desc->inflow_velocity_x_min,
        .x_max = desc->inflow_velocity_x_max,
        .y_min = desc->inflow_velocity_y_min,
        .y_max = desc->inflow_velocity_y_max,
        .z_min = desc->inflow_velocity_z_min,
        .z_max = desc->inflow_velocity_z_max,
    };

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

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x_previous, velocity_y_previous, velocity_z_previous, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    advect_velocity<<<velocity_grid, block, 0, stream>>>(velocity_x_temporary, velocity_y_temporary, velocity_z_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x_temporary, velocity_y_temporary, velocity_z_temporary, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    return 0;
}

int32_t stable_fluids_diffuse_velocity_cuda(const StableFluidsDiffuseVelocityDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_diffuse_velocity_desc(desc); code != 0) return code;

    const uint32_t boundary_pack = (desc->boundary_x_min << (boundary_x_min_face * 3)) | (desc->boundary_x_max << (boundary_x_max_face * 3)) | (desc->boundary_y_min << (boundary_y_min_face * 3)) | (desc->boundary_y_max << (boundary_y_max_face * 3)) | (desc->boundary_z_min << (boundary_z_min_face * 3)) | (desc->boundary_z_max << (boundary_z_max_face * 3));
    const InflowValues inflow{
        .x_min = desc->inflow_velocity_x_min,
        .x_max = desc->inflow_velocity_x_max,
        .y_min = desc->inflow_velocity_y_min,
        .y_max = desc->inflow_velocity_y_max,
        .z_min = desc->inflow_velocity_z_min,
        .z_max = desc->inflow_velocity_z_max,
    };

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
    auto* coarse_solution      = static_cast<float*>(desc->temporary_density);
    auto* coarse_rhs           = static_cast<float*>(desc->temporary_previous_density);

    nvtx3::scoped_range range("stable.step.diffuse_velocity");

    if (desc->viscosity <= 0.0f) {
        if (cudaMemcpyAsync(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        if (cudaMemcpyAsync(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        if (cudaMemcpyAsync(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x_field, velocity_y_field, velocity_z_field, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
        return cudaGetLastError() == cudaSuccess ? 0 : 5001;
    }

    const float diffusion_alpha = desc->dt * desc->viscosity / (desc->cell_size * desc->cell_size);
    const VCycleConfig diffusion_config{
        .cycles        = std::max(1, desc->diffuse_iterations / 12),
        .pre_smooth    = 1,
        .post_smooth   = 1,
        .coarse_smooth = std::max(6, desc->diffuse_iterations / 4),
    };

    auto diffuse_velocity_component = [&](float* field, const float* source, const std::uint64_t field_bytes, const int sx, const int sy, const int sz, const BoundaryAxis axis) -> int32_t {
        GridHierarchy hierarchy = build_hierarchy(sx, sy, sz, field, const_cast<float*>(source), coarse_solution, coarse_rhs);
        if (cudaMemcpyAsync(field, source, field_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        const DiffusionVCycleOps ops{
            .coefficient   = diffusion_alpha,
            .boundary_pack = boundary_pack,
            .inflow        = inflow,
            .boundary_axis = axis,
        };
        const int32_t code = run_v_cycle(hierarchy, diffusion_config, ops, block, stream);
        if (code != 0) return code;
        return 0;
    };

    if (const int32_t code = diffuse_velocity_component(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes, desc->nx + 1, desc->ny, desc->nz, BoundaryAxis::x); code != 0) return code;
    if (const int32_t code = diffuse_velocity_component(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes, desc->nx, desc->ny + 1, desc->nz, BoundaryAxis::y); code != 0) return code;
    if (const int32_t code = diffuse_velocity_component(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes, desc->nx, desc->ny, desc->nz + 1, BoundaryAxis::z); code != 0) return code;

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x_field, velocity_y_field, velocity_z_field, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    return 0;
}

int32_t stable_fluids_project_cuda(const StableFluidsProjectDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_project_desc(desc); code != 0) return code;

    const uint32_t boundary_pack = (desc->boundary_x_min << (boundary_x_min_face * 3)) | (desc->boundary_x_max << (boundary_x_max_face * 3)) | (desc->boundary_y_min << (boundary_y_min_face * 3)) | (desc->boundary_y_max << (boundary_y_max_face * 3)) | (desc->boundary_z_min << (boundary_z_min_face * 3)) | (desc->boundary_z_max << (boundary_z_max_face * 3));
    const InflowValues inflow{
        .x_min = desc->inflow_velocity_x_min,
        .x_max = desc->inflow_velocity_x_max,
        .y_min = desc->inflow_velocity_y_min,
        .y_max = desc->inflow_velocity_y_max,
        .z_min = desc->inflow_velocity_z_min,
        .z_max = desc->inflow_velocity_z_max,
    };

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells         = make_grid(desc->nx, desc->ny, desc->nz, block);
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const auto stream        = static_cast<Stream>(desc->stream);
    const auto cell_bytes    = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);

    auto* velocity_x              = static_cast<float*>(desc->velocity_x);
    auto* velocity_y              = static_cast<float*>(desc->velocity_y);
    auto* velocity_z              = static_cast<float*>(desc->velocity_z);
    auto* pressure                = static_cast<float*>(desc->temporary_pressure);
    auto* divergence              = static_cast<float*>(desc->temporary_divergence);
    auto* coarse_pressure_storage = static_cast<float*>(desc->temporary_density);
    auto* coarse_rhs_storage      = static_cast<float*>(desc->temporary_previous_density);

    const GridHierarchy pressure_hierarchy = build_hierarchy(desc->nx, desc->ny, desc->nz, pressure, divergence, coarse_pressure_storage, coarse_rhs_storage);

    nvtx3::scoped_range range("stable.step.project");

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x, velocity_y, velocity_z, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    if (cudaMemsetAsync(pressure, 0, cell_bytes, stream) != cudaSuccess) return 5001;

    compute_poisson_rhs_kernel<<<cells, block, 0, stream>>>(divergence, velocity_x, velocity_y, velocity_z, desc->nx, desc->ny, desc->nz, desc->cell_size);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    const PoissonVCycleOps ops{.boundary_pack = boundary_pack};
    const VCycleConfig config{
        .cycles        = std::max(1, desc->pressure_iterations / 40),
        .pre_smooth    = 1,
        .post_smooth   = 1,
        .coarse_smooth = std::max(8, desc->pressure_iterations / 10),
    };

    if (const int32_t code = run_v_cycle(pressure_hierarchy, config, ops, block, stream); code != 0) return code;

    project_velocity_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x, velocity_y, velocity_z, pressure, desc->nx, desc->ny, desc->nz, 1.0f / desc->cell_size, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x, velocity_y, velocity_z, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    return 0;
}

int32_t stable_fluids_advect_scalar_cuda(const StableFluidsAdvectScalarDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_advect_scalar_desc(desc); code != 0) return code;

    const uint32_t boundary_pack = (desc->boundary_x_min << (boundary_x_min_face * 3)) | (desc->boundary_x_max << (boundary_x_max_face * 3)) | (desc->boundary_y_min << (boundary_y_min_face * 3)) | (desc->boundary_y_max << (boundary_y_max_face * 3)) | (desc->boundary_z_min << (boundary_z_min_face * 3)) | (desc->boundary_z_max << (boundary_z_max_face * 3));
    const InflowValues inflow{
        .x_min = desc->inflow_scalar_x_min,
        .x_max = desc->inflow_scalar_x_max,
        .y_min = desc->inflow_scalar_y_min,
        .y_max = desc->inflow_scalar_y_max,
        .z_min = desc->inflow_scalar_z_min,
        .z_max = desc->inflow_scalar_z_max,
    };

    const auto cell_bytes  = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    auto* scalar_field     = static_cast<float*>(desc->scalar);
    auto* scalar_temporary = static_cast<float*>(desc->temporary_scalar);
    auto* scalar_previous  = static_cast<float*>(desc->temporary_previous_scalar);
    auto* velocity_x_field = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field = static_cast<float*>(desc->velocity_z);

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells         = make_grid(desc->nx, desc->ny, desc->nz, block);
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const auto stream        = static_cast<Stream>(desc->stream);

    nvtx3::scoped_range range("stable.step.advect_scalar");

    if (cudaMemcpyAsync(scalar_previous, scalar_field, cell_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;

    constexpr InflowValues velocity_inflow{
        .x_min = 0.0f,
        .x_max = 0.0f,
        .y_min = 0.0f,
        .y_max = 0.0f,
        .z_min = 0.0f,
        .z_max = 0.0f,
    };

    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(velocity_x_field, velocity_y_field, velocity_z_field, desc->nx, desc->ny, desc->nz, boundary_pack, velocity_inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    advect_scalar_kernel<<<cells, block, 0, stream>>>(scalar_temporary, scalar_previous, velocity_x_field, velocity_y_field, velocity_z_field, desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, boundary_pack, inflow, static_cast<int>(desc->clamp_non_negative));
    if (cudaGetLastError() != cudaSuccess) return 5001;

    apply_scalar_inflow_boundaries_kernel<<<cells, block, 0, stream>>>(scalar_temporary, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    return 0;
}

int32_t stable_fluids_diffuse_scalar_cuda(const StableFluidsDiffuseScalarDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_diffuse_scalar_desc(desc); code != 0) return code;

    const uint32_t boundary_pack = (desc->boundary_x_min << (boundary_x_min_face * 3)) | (desc->boundary_x_max << (boundary_x_max_face * 3)) | (desc->boundary_y_min << (boundary_y_min_face * 3)) | (desc->boundary_y_max << (boundary_y_max_face * 3)) | (desc->boundary_z_min << (boundary_z_min_face * 3)) | (desc->boundary_z_max << (boundary_z_max_face * 3));
    const InflowValues inflow{
        .x_min = desc->inflow_scalar_x_min,
        .x_max = desc->inflow_scalar_x_max,
        .y_min = desc->inflow_scalar_y_min,
        .y_max = desc->inflow_scalar_y_max,
        .z_min = desc->inflow_scalar_z_min,
        .z_max = desc->inflow_scalar_z_max,
    };

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells      = make_grid(desc->nx, desc->ny, desc->nz, block);
    const auto stream     = static_cast<Stream>(desc->stream);
    const auto cell_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);

    auto* scalar_field     = static_cast<float*>(desc->scalar);
    auto* scalar_temporary = static_cast<float*>(desc->temporary_scalar);
    auto* pressure         = static_cast<float*>(desc->temporary_solution_storage);
    auto* divergence       = static_cast<float*>(desc->temporary_rhs_storage);

    nvtx3::scoped_range range("stable.step.diffuse_scalar");

    if (cudaMemcpyAsync(scalar_field, scalar_temporary, cell_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
    if (desc->diffusion <= 0.0f) {
        apply_scalar_inflow_boundaries_kernel<<<cells, block, 0, stream>>>(scalar_field, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
        return cudaGetLastError() == cudaSuccess ? 0 : 5001;
    }

    GridHierarchy hierarchy     = build_hierarchy(desc->nx, desc->ny, desc->nz, scalar_field, scalar_temporary, pressure, divergence);
    const float diffusion_alpha = desc->dt * desc->diffusion / (desc->cell_size * desc->cell_size);
    const DiffusionVCycleOps ops{
        .coefficient   = diffusion_alpha,
        .boundary_pack = boundary_pack,
        .inflow        = inflow,
        .boundary_axis = BoundaryAxis::none,
    };
    const VCycleConfig config{
        .cycles        = std::max(1, desc->diffuse_iterations / 12),
        .pre_smooth    = 1,
        .post_smooth   = 1,
        .coarse_smooth = std::max(6, desc->diffuse_iterations / 4),
    };

    if (const int32_t code = run_v_cycle(hierarchy, config, ops, block, stream); code != 0) return code;

    const float denom = 1.0f + 6.0f * diffusion_alpha;
    diffuse_grid_kernel<<<cells, block, 0, stream>>>(scalar_field, scalar_temporary, desc->nx, desc->ny, desc->nz, diffusion_alpha, denom, 0, boundary_pack, inflow);
    diffuse_grid_kernel<<<cells, block, 0, stream>>>(scalar_field, scalar_temporary, desc->nx, desc->ny, desc->nz, diffusion_alpha, denom, 1, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    apply_scalar_inflow_boundaries_kernel<<<cells, block, 0, stream>>>(scalar_field, desc->nx, desc->ny, desc->nz, boundary_pack, inflow);
    if (cudaGetLastError() != cudaSuccess) return 5001;

    return 0;
}

int32_t stable_fluids_add_scalar_source_cuda(const StableFluidsAddScalarSourceDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_add_scalar_source_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 grid   = make_grid(desc->nx, desc->ny, desc->nz, block);
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

int32_t stable_fluids_compute_staggered_velocity_magnitude_cuda(const StableFluidsComputeStaggeredVelocityMagnitudeDesc* desc) {
    using namespace stable_fluids;
    if (const int32_t code = stable_fluids_validate_compute_staggered_velocity_magnitude_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 grid = make_grid(desc->nx, desc->ny, desc->nz, block);
    const auto stream = static_cast<Stream>(desc->stream);

    compute_staggered_velocity_magnitude_kernel<<<grid, block, 0, stream>>>(static_cast<float*>(desc->destination), static_cast<const float*>(desc->velocity_x), static_cast<const float*>(desc->velocity_y), static_cast<const float*>(desc->velocity_z), desc->nx, desc->ny, desc->nz);

    return cudaGetLastError() == cudaSuccess ? 0 : 5001;
}
}
