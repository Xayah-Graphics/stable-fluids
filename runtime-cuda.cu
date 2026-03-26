#include "stable-fluids.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {

    using Stream = cudaStream_t;

    constexpr int32_t success              = 0;
    constexpr int32_t invalid_context      = 1007;
    constexpr int32_t cuda_failure         = 5001;
    constexpr uint8_t cell_fluid           = 0;
    constexpr uint8_t cell_solid           = 1;
    constexpr uint8_t face_open            = 0;
    constexpr uint8_t face_fixed           = 1;
    enum class Axis : int {
        x = 0,
        y = 1,
        z = 2,
    };

    struct DeviceBuffers {
        float* density       = nullptr;
        float* dye_r         = nullptr;
        float* dye_g         = nullptr;
        float* dye_b         = nullptr;
        float* temperature   = nullptr;
        float* velocity_x    = nullptr;
        float* velocity_y    = nullptr;
        float* velocity_z    = nullptr;
        float* temp_density  = nullptr;
        float* temp_dye_r    = nullptr;
        float* temp_dye_g    = nullptr;
        float* temp_dye_b    = nullptr;
        float* temp_velocity_x = nullptr;
        float* temp_velocity_y = nullptr;
        float* temp_velocity_z = nullptr;
        float* pressure      = nullptr;
        float* divergence    = nullptr;
        uint8_t* cell_flags  = nullptr;
        uint8_t* u_flags     = nullptr;
        uint8_t* v_flags     = nullptr;
        uint8_t* w_flags     = nullptr;
        float* u_target      = nullptr;
        float* v_target      = nullptr;
        float* w_target      = nullptr;
    };

    struct HostBoundaryAtlas {
        std::vector<uint8_t> cell_flags{};
        std::vector<uint8_t> u_flags{};
        std::vector<uint8_t> v_flags{};
        std::vector<uint8_t> w_flags{};
        std::vector<float> u_target{};
        std::vector<float> v_target{};
        std::vector<float> w_target{};
    };

    struct ContextStorage {
        StableFluidsSimulationConfig config{};
        std::vector<StableFluidsColliderDesc> colliders{};
        HostBoundaryAtlas host_atlas{};
        DeviceBuffers device{};
        Stream stream                 = nullptr;
        bool owns_stream              = false;
        bool atlas_dirty              = true;
    };

    __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
        return {
            static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)),
        };
    }

    std::uint64_t scalar_count(const StableFluidsSimulationConfig& config) {
        return static_cast<std::uint64_t>(config.nx) * static_cast<std::uint64_t>(config.ny) * static_cast<std::uint64_t>(config.nz);
    }

    std::uint64_t u_face_count(const StableFluidsSimulationConfig& config) {
        return static_cast<std::uint64_t>(config.nx + 1) * static_cast<std::uint64_t>(config.ny) * static_cast<std::uint64_t>(config.nz);
    }

    std::uint64_t v_face_count(const StableFluidsSimulationConfig& config) {
        return static_cast<std::uint64_t>(config.nx) * static_cast<std::uint64_t>(config.ny + 1) * static_cast<std::uint64_t>(config.nz);
    }

    std::uint64_t w_face_count(const StableFluidsSimulationConfig& config) {
        return static_cast<std::uint64_t>(config.nx) * static_cast<std::uint64_t>(config.ny) * static_cast<std::uint64_t>(config.nz + 1);
    }

    bool point_inside_collider(const StableFluidsColliderDesc& collider, const float x, const float y, const float z) {
        const float dx = x - collider.center_x;
        const float dy = y - collider.center_y;
        const float dz = z - collider.center_z;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_SPHERE) {
            return dx * dx + dy * dy + dz * dz <= collider.radius * collider.radius;
        }
        return std::abs(dx) <= collider.half_extent_x && std::abs(dy) <= collider.half_extent_y && std::abs(dz) <= collider.half_extent_z;
    }

    float collider_signed_distance(const StableFluidsColliderDesc& collider, const float x, const float y, const float z) {
        const float dx = x - collider.center_x;
        const float dy = y - collider.center_y;
        const float dz = z - collider.center_z;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_SPHERE) {
            return std::sqrt(dx * dx + dy * dy + dz * dz) - collider.radius;
        }

        const float qx = std::abs(dx) - collider.half_extent_x;
        const float qy = std::abs(dy) - collider.half_extent_y;
        const float qz = std::abs(dz) - collider.half_extent_z;
        const float outside = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) + std::max(qy, 0.0f) * std::max(qy, 0.0f) + std::max(qz, 0.0f) * std::max(qz, 0.0f));
        const float inside  = std::min(std::max(qx, std::max(qy, qz)), 0.0f);
        return outside + inside;
    }

    float sample_collider_velocity_axis(const std::vector<StableFluidsColliderDesc>& colliders, const float x, const float y, const float z, const Axis axis) {
        float best_distance = 1.0e30f;
        float best_value    = 0.0f;
        for (const auto& collider : colliders) {
            const float distance = collider_signed_distance(collider, x, y, z);
            if (distance > 0.75f) continue;
            if (distance >= best_distance) continue;
            best_distance = distance;
            if (axis == Axis::x) best_value = collider.linear_velocity_x;
            if (axis == Axis::y) best_value = collider.linear_velocity_y;
            if (axis == Axis::z) best_value = collider.linear_velocity_z;
        }
        return best_value;
    }

    void resize_host_atlas(ContextStorage& context) {
        context.host_atlas.cell_flags.resize(static_cast<std::size_t>(scalar_count(context.config)), cell_fluid);
        context.host_atlas.u_flags.resize(static_cast<std::size_t>(u_face_count(context.config)), face_open);
        context.host_atlas.v_flags.resize(static_cast<std::size_t>(v_face_count(context.config)), face_open);
        context.host_atlas.w_flags.resize(static_cast<std::size_t>(w_face_count(context.config)), face_open);
        context.host_atlas.u_target.resize(static_cast<std::size_t>(u_face_count(context.config)), 0.0f);
        context.host_atlas.v_target.resize(static_cast<std::size_t>(v_face_count(context.config)), 0.0f);
        context.host_atlas.w_target.resize(static_cast<std::size_t>(w_face_count(context.config)), 0.0f);
    }

    int32_t upload_boundary_atlas(ContextStorage& context) {
        const auto cell_bytes   = scalar_count(context.config) * sizeof(uint8_t);
        const auto u_flag_bytes = u_face_count(context.config) * sizeof(uint8_t);
        const auto v_flag_bytes = v_face_count(context.config) * sizeof(uint8_t);
        const auto w_flag_bytes = w_face_count(context.config) * sizeof(uint8_t);
        const auto u_val_bytes  = u_face_count(context.config) * sizeof(float);
        const auto v_val_bytes  = v_face_count(context.config) * sizeof(float);
        const auto w_val_bytes  = w_face_count(context.config) * sizeof(float);

        if (cudaMemcpyAsync(context.device.cell_flags, context.host_atlas.cell_flags.data(), cell_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.u_flags, context.host_atlas.u_flags.data(), u_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.v_flags, context.host_atlas.v_flags.data(), v_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.w_flags, context.host_atlas.w_flags.data(), w_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.u_target, context.host_atlas.u_target.data(), u_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.v_target, context.host_atlas.v_target.data(), v_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        if (cudaMemcpyAsync(context.device.w_target, context.host_atlas.w_target.data(), w_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return cuda_failure;
        return success;
    }

    void build_boundary_atlas(ContextStorage& context) {
        const int nx = context.config.nx;
        const int ny = context.config.ny;
        const int nz = context.config.nz;
        const float h = context.config.cell_size;

        resize_host_atlas(context);
        std::fill(context.host_atlas.cell_flags.begin(), context.host_atlas.cell_flags.end(), cell_fluid);
        std::fill(context.host_atlas.u_flags.begin(), context.host_atlas.u_flags.end(), face_open);
        std::fill(context.host_atlas.v_flags.begin(), context.host_atlas.v_flags.end(), face_open);
        std::fill(context.host_atlas.w_flags.begin(), context.host_atlas.w_flags.end(), face_open);
        std::fill(context.host_atlas.u_target.begin(), context.host_atlas.u_target.end(), 0.0f);
        std::fill(context.host_atlas.v_target.begin(), context.host_atlas.v_target.end(), 0.0f);
        std::fill(context.host_atlas.w_target.begin(), context.host_atlas.w_target.end(), 0.0f);

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    for (const auto& collider : context.colliders) {
                        if (!point_inside_collider(collider, px, py, pz)) continue;
                        context.host_atlas.cell_flags[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))] = cell_solid;
                        break;
                    }
                }
            }
        }

        auto apply_domain_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int x, const int y, const int z, const int sx, const int sy, const StableFluidsBoundaryFaceDesc& face) {
            if (face.type == STABLE_FLUIDS_BOUNDARY_OUTFLOW) return;
            flags[static_cast<std::size_t>(index_3d(x, y, z, sx, sy))] = face_fixed;
            values[static_cast<std::size_t>(index_3d(x, y, z, sx, sy))] = face.type == STABLE_FLUIDS_BOUNDARY_INFLOW ? face.velocity : 0.0f;
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                apply_domain_face(context.host_atlas.u_flags, context.host_atlas.u_target, 0, y, z, nx + 1, ny, context.config.domain_boundary.x_min);
                apply_domain_face(context.host_atlas.u_flags, context.host_atlas.u_target, nx, y, z, nx + 1, ny, context.config.domain_boundary.x_max);
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                apply_domain_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, 0, z, nx, ny + 1, context.config.domain_boundary.y_min);
                apply_domain_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, ny, z, nx, ny + 1, context.config.domain_boundary.y_max);
            }
        }

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                apply_domain_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, 0, nx, ny, context.config.domain_boundary.z_min);
                apply_domain_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, nz, nx, ny, context.config.domain_boundary.z_max);
            }
        }

        auto is_solid = [&](const int x, const int y, const int z) {
            if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
            return context.host_atlas.cell_flags[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))] == cell_solid;
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 1; x < nx; ++x) {
                    if (!is_solid(x - 1, y, z) && !is_solid(x, y, z)) continue;
                    const float px = static_cast<float>(x) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    context.host_atlas.u_flags[static_cast<std::size_t>(index_3d(x, y, z, nx + 1, ny))]  = face_fixed;
                    context.host_atlas.u_target[static_cast<std::size_t>(index_3d(x, y, z, nx + 1, ny))] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::x);
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 1; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    if (!is_solid(x, y - 1, z) && !is_solid(x, y, z)) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = static_cast<float>(y) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    context.host_atlas.v_flags[static_cast<std::size_t>(index_3d(x, y, z, nx, ny + 1))]  = face_fixed;
                    context.host_atlas.v_target[static_cast<std::size_t>(index_3d(x, y, z, nx, ny + 1))] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::y);
                }
            }
        }

        for (int z = 1; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    if (!is_solid(x, y, z - 1) && !is_solid(x, y, z)) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = static_cast<float>(z) * h;
                    context.host_atlas.w_flags[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))]  = face_fixed;
                    context.host_atlas.w_target[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::z);
                }
            }
        }
    }

    template <class T>
    int32_t allocate_device_array(T*& ptr, const std::uint64_t count) {
        if (count == 0) return success;
        if (cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)) != cudaSuccess) return cuda_failure;
        return success;
    }

    template <class T>
    void release_device_array(T*& ptr) {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = nullptr;
    }

    void destroy_buffers(ContextStorage& context) {
        release_device_array(context.device.density);
        release_device_array(context.device.dye_r);
        release_device_array(context.device.dye_g);
        release_device_array(context.device.dye_b);
        release_device_array(context.device.temperature);
        release_device_array(context.device.velocity_x);
        release_device_array(context.device.velocity_y);
        release_device_array(context.device.velocity_z);
        release_device_array(context.device.temp_density);
        release_device_array(context.device.temp_dye_r);
        release_device_array(context.device.temp_dye_g);
        release_device_array(context.device.temp_dye_b);
        release_device_array(context.device.temp_velocity_x);
        release_device_array(context.device.temp_velocity_y);
        release_device_array(context.device.temp_velocity_z);
        release_device_array(context.device.pressure);
        release_device_array(context.device.divergence);
        release_device_array(context.device.cell_flags);
        release_device_array(context.device.u_flags);
        release_device_array(context.device.v_flags);
        release_device_array(context.device.w_flags);
        release_device_array(context.device.u_target);
        release_device_array(context.device.v_target);
        release_device_array(context.device.w_target);
    }

    int32_t allocate_buffers(ContextStorage& context) {
        if (allocate_device_array(context.device.density, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.dye_r, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.dye_g, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.dye_b, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temperature, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.velocity_x, u_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.velocity_y, v_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.velocity_z, w_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_density, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_dye_r, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_dye_g, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_dye_b, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_velocity_x, u_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_velocity_y, v_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.temp_velocity_z, w_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.pressure, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.divergence, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.cell_flags, scalar_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.u_flags, u_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.v_flags, v_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.w_flags, w_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.u_target, u_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.v_target, v_face_count(context.config)) != 0) return cuda_failure;
        if (allocate_device_array(context.device.w_target, w_face_count(context.config)) != 0) return cuda_failure;
        return success;
    }

    __device__ bool cell_is_solid(const uint8_t* cell_flags, const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        return cell_flags[index_3d(x, y, z, nx, ny)] == cell_solid;
    }

    __device__ float clamp_world(const float value, const float max_value) {
        return fminf(fmaxf(value, 0.0f), max_value);
    }

    __device__ bool point_is_solid(const uint8_t* cell_flags, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const int ix = min(max(static_cast<int>(floorf(x / h)), 0), nx - 1);
        const int iy = min(max(static_cast<int>(floorf(y / h)), 0), ny - 1);
        const int iz = min(max(static_cast<int>(floorf(z / h)), 0), nz - 1);
        return cell_flags[index_3d(ix, iy, iz, nx, ny)] == cell_solid;
    }

    __device__ float3 clip_backtrace_to_fluid(const uint8_t* cell_flags, const float3 origin, const float3 target, const int nx, const int ny, const int nz, const float h) {
        float3 lo = origin;
        float3 hi = target;
        if (!point_is_solid(cell_flags, hi.x, hi.y, hi.z, nx, ny, nz, h)) return hi;
        for (int i = 0; i < 8; ++i) {
            const float3 mid = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
            if (point_is_solid(cell_flags, mid.x, mid.y, mid.z, nx, ny, nz, h)) hi = mid;
            else lo = mid;
        }
        return lo;
    }

    __device__ float sample_scalar_field(const float* field, const uint8_t* cell_flags, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = clamp_world(x / h - 0.5f, static_cast<float>(nx - 1));
        const float gy = clamp_world(y / h - 0.5f, static_cast<float>(ny - 1));
        const float gz = clamp_world(z / h - 0.5f, static_cast<float>(nz - 1));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = min(x0 + 1, nx - 1);
        const int y1 = min(y0 + 1, ny - 1);
        const int z1 = min(z0 + 1, nz - 1);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);

        auto load = [&](const int ix, const int iy, const int iz) {
            if (cell_flags[index_3d(ix, iy, iz, nx, ny)] == cell_solid) return 0.0f;
            return field[index_3d(ix, iy, iz, nx, ny)];
        };

        const float c000 = load(x0, y0, z0);
        const float c100 = load(x1, y0, z0);
        const float c010 = load(x0, y1, z0);
        const float c110 = load(x1, y1, z0);
        const float c001 = load(x0, y0, z1);
        const float c101 = load(x1, y0, z1);
        const float c011 = load(x0, y1, z1);
        const float c111 = load(x1, y1, z1);
        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float load_u(const float* field, const uint8_t* flags, const float* target, const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        const int ix = min(max(x, 0), nx);
        const int iy = min(max(y, 0), ny - 1);
        const int iz = min(max(z, 0), nz - 1);
        const auto index = index_3d(ix, iy, iz, nx + 1, ny);
        return flags[index] == face_fixed ? target[index] : field[index];
    }

    __device__ float load_v(const float* field, const uint8_t* flags, const float* target, const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        const int ix = min(max(x, 0), nx - 1);
        const int iy = min(max(y, 0), ny);
        const int iz = min(max(z, 0), nz - 1);
        const auto index = index_3d(ix, iy, iz, nx, ny + 1);
        return flags[index] == face_fixed ? target[index] : field[index];
    }

    __device__ float load_w(const float* field, const uint8_t* flags, const float* target, const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        const int ix = min(max(x, 0), nx - 1);
        const int iy = min(max(y, 0), ny - 1);
        const int iz = min(max(z, 0), nz);
        const auto index = index_3d(ix, iy, iz, nx, ny);
        return flags[index] == face_fixed ? target[index] : field[index];
    }

    __device__ float sample_u_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = clamp_world(x / h, static_cast<float>(nx));
        const float gy = clamp_world(y / h - 0.5f, static_cast<float>(ny - 1));
        const float gz = clamp_world(z / h - 0.5f, static_cast<float>(nz - 1));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = min(x0 + 1, nx);
        const int y1 = min(y0 + 1, ny - 1);
        const int z1 = min(z0 + 1, nz - 1);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        const float c000 = load_u(field, flags, target, x0, y0, z0, nx, ny, nz);
        const float c100 = load_u(field, flags, target, x1, y0, z0, nx, ny, nz);
        const float c010 = load_u(field, flags, target, x0, y1, z0, nx, ny, nz);
        const float c110 = load_u(field, flags, target, x1, y1, z0, nx, ny, nz);
        const float c001 = load_u(field, flags, target, x0, y0, z1, nx, ny, nz);
        const float c101 = load_u(field, flags, target, x1, y0, z1, nx, ny, nz);
        const float c011 = load_u(field, flags, target, x0, y1, z1, nx, ny, nz);
        const float c111 = load_u(field, flags, target, x1, y1, z1, nx, ny, nz);
        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_v_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = clamp_world(x / h - 0.5f, static_cast<float>(nx - 1));
        const float gy = clamp_world(y / h, static_cast<float>(ny));
        const float gz = clamp_world(z / h - 0.5f, static_cast<float>(nz - 1));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = min(x0 + 1, nx - 1);
        const int y1 = min(y0 + 1, ny);
        const int z1 = min(z0 + 1, nz - 1);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        const float c000 = load_v(field, flags, target, x0, y0, z0, nx, ny, nz);
        const float c100 = load_v(field, flags, target, x1, y0, z0, nx, ny, nz);
        const float c010 = load_v(field, flags, target, x0, y1, z0, nx, ny, nz);
        const float c110 = load_v(field, flags, target, x1, y1, z0, nx, ny, nz);
        const float c001 = load_v(field, flags, target, x0, y0, z1, nx, ny, nz);
        const float c101 = load_v(field, flags, target, x1, y0, z1, nx, ny, nz);
        const float c011 = load_v(field, flags, target, x0, y1, z1, nx, ny, nz);
        const float c111 = load_v(field, flags, target, x1, y1, z1, nx, ny, nz);
        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_w_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = clamp_world(x / h - 0.5f, static_cast<float>(nx - 1));
        const float gy = clamp_world(y / h - 0.5f, static_cast<float>(ny - 1));
        const float gz = clamp_world(z / h, static_cast<float>(nz));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = min(x0 + 1, nx - 1);
        const int y1 = min(y0 + 1, ny - 1);
        const int z1 = min(z0 + 1, nz);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        const float c000 = load_w(field, flags, target, x0, y0, z0, nx, ny, nz);
        const float c100 = load_w(field, flags, target, x1, y0, z0, nx, ny, nz);
        const float c010 = load_w(field, flags, target, x0, y1, z0, nx, ny, nz);
        const float c110 = load_w(field, flags, target, x1, y1, z0, nx, ny, nz);
        const float c001 = load_w(field, flags, target, x0, y0, z1, nx, ny, nz);
        const float c101 = load_w(field, flags, target, x1, y0, z1, nx, ny, nz);
        const float c011 = load_w(field, flags, target, x0, y1, z1, nx, ny, nz);
        const float c111 = load_w(field, flags, target, x1, y1, z1, nx, ny, nz);
        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float3 sample_velocity_field(const float* u, const float* v, const float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const float3 pos, const int nx, const int ny, const int nz, const float h) {
        return make_float3(
            sample_u_field(u, u_flags, u_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
            sample_v_field(v, v_flags, v_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
            sample_w_field(w, w_flags, w_target, pos.x, pos.y, pos.z, nx, ny, nz, h)
        );
    }

    __global__ void clear_scalar_fields_kernel(float* density, float* dye_r, float* dye_g, float* dye_b, float* temperature, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        density[index] = 0.0f;
        dye_r[index] = 0.0f;
        dye_g[index] = 0.0f;
        dye_b[index] = 0.0f;
        temperature[index] = 0.0f;
    }

    __global__ void clear_velocity_fields_kernel(float* u, float* v, float* w, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x <= nx && y < ny && z < nz) u[index_3d(x, y, z, nx + 1, ny)] = 0.0f;
        if (x < nx && y <= ny && z < nz) v[index_3d(x, y, z, nx, ny + 1)] = 0.0f;
        if (x < nx && y < ny && z <= nz) w[index_3d(x, y, z, nx, ny)] = 0.0f;
    }

    __global__ void apply_face_constraints_kernel(float* u, float* v, float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x <= nx && y < ny && z < nz) {
            const auto index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[index] == face_fixed) u[index] = u_target[index];
        }
        if (x < nx && y <= ny && z < nz) {
            const auto index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[index] == face_fixed) v[index] = v_target[index];
        }
        if (x < nx && y < ny && z <= nz) {
            const auto index = index_3d(x, y, z, nx, ny);
            if (w_flags[index] == face_fixed) w[index] = w_target[index];
        }
    }

    __global__ void apply_scalar_boundary_kernel(float* scalar, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const StableFluidsDomainBoundaryDesc domain_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            scalar[index] = 0.0f;
            return;
        }

        float sum = 0.0f;
        int count = 0;
        if (x == 0 && domain_boundary.x_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.x_min.scalar; ++count; }
        if (x == nx - 1 && domain_boundary.x_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.x_max.scalar; ++count; }
        if (y == 0 && domain_boundary.y_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.y_min.scalar; ++count; }
        if (y == ny - 1 && domain_boundary.y_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.y_max.scalar; ++count; }
        if (z == 0 && domain_boundary.z_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.z_min.scalar; ++count; }
        if (z == nz - 1 && domain_boundary.z_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { sum += domain_boundary.z_max.scalar; ++count; }
        if (count > 0) scalar[index] = sum / static_cast<float>(count);
    }

    __global__ void add_sources_kernel(float* density, float* dye_r, float* dye_g, float* dye_b, float* temperature, float* u, float* v, float* w, const uint8_t* cell_flags, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const StableFluidsSourceDesc* sources, const int source_count, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x < nx && y < ny && z < nz) {
            const float px = (static_cast<float>(x) + 0.5f) * h;
            const float py = (static_cast<float>(y) + 0.5f) * h;
            const float pz = (static_cast<float>(z) + 0.5f) * h;
            if (cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) {
                for (int source_index = 0; source_index < source_count; ++source_index) {
                    const auto& source = sources[source_index];
                    const float dx = px - source.center_x;
                    const float dy = py - source.center_y;
                    const float dz = pz - source.center_z;
                    const float distance2 = dx * dx + dy * dy + dz * dz;
                    const float radius2 = source.radius * source.radius;
                    if (distance2 > radius2) continue;
                    const float weight = fmaxf(0.0f, 1.0f - distance2 / radius2);
                    const auto index = index_3d(x, y, z, nx, ny);
                    density[index] += source.density_amount * weight;
                    dye_r[index] += source.dye_r * weight;
                    dye_g[index] += source.dye_g * weight;
                    dye_b[index] += source.dye_b * weight;
                    temperature[index] += source.temperature_amount * weight;
                }
            }
        }

        if (x <= nx && y < ny && z < nz && u_flags[index_3d(x, y, z, nx + 1, ny)] != face_fixed) {
            const float px = static_cast<float>(x) * h;
            const float py = (static_cast<float>(y) + 0.5f) * h;
            const float pz = (static_cast<float>(z) + 0.5f) * h;
            for (int source_index = 0; source_index < source_count; ++source_index) {
                const auto& source = sources[source_index];
                const float dx = px - source.center_x;
                const float dy = py - source.center_y;
                const float dz = pz - source.center_z;
                const float distance2 = dx * dx + dy * dy + dz * dz;
                const float radius2 = source.radius * source.radius;
                if (distance2 > radius2) continue;
                u[index_3d(x, y, z, nx + 1, ny)] += source.velocity_x * fmaxf(0.0f, 1.0f - distance2 / radius2);
            }
        }

        if (x < nx && y <= ny && z < nz && v_flags[index_3d(x, y, z, nx, ny + 1)] != face_fixed) {
            const float px = (static_cast<float>(x) + 0.5f) * h;
            const float py = static_cast<float>(y) * h;
            const float pz = (static_cast<float>(z) + 0.5f) * h;
            for (int source_index = 0; source_index < source_count; ++source_index) {
                const auto& source = sources[source_index];
                const float dx = px - source.center_x;
                const float dy = py - source.center_y;
                const float dz = pz - source.center_z;
                const float distance2 = dx * dx + dy * dy + dz * dz;
                const float radius2 = source.radius * source.radius;
                if (distance2 > radius2) continue;
                v[index_3d(x, y, z, nx, ny + 1)] += source.velocity_y * fmaxf(0.0f, 1.0f - distance2 / radius2);
            }
        }

        if (x < nx && y < ny && z <= nz && w_flags[index_3d(x, y, z, nx, ny)] != face_fixed) {
            const float px = (static_cast<float>(x) + 0.5f) * h;
            const float py = (static_cast<float>(y) + 0.5f) * h;
            const float pz = static_cast<float>(z) * h;
            for (int source_index = 0; source_index < source_count; ++source_index) {
                const auto& source = sources[source_index];
                const float dx = px - source.center_x;
                const float dy = py - source.center_y;
                const float dz = pz - source.center_z;
                const float distance2 = dx * dx + dy * dy + dz * dz;
                const float radius2 = source.radius * source.radius;
                if (distance2 > radius2) continue;
                w[index_3d(x, y, z, nx, ny)] += source.velocity_z * fmaxf(0.0f, 1.0f - distance2 / radius2);
            }
        }
    }

    __global__ void add_forces_kernel(float* u, float* v, float* w, const float* density, const float* temperature, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float dt, const float density_buoyancy, const float temperature_buoyancy, const float ambient_temperature, const float uniform_force_x, const float uniform_force_y, const float uniform_force_z) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x > 0 && x < nx && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[face_index] != face_fixed) u[face_index] += dt * uniform_force_x;
        }

        if (x < nx && y > 0 && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] != face_fixed) {
                const auto below = index_3d(x, y - 1, z, nx, ny);
                const auto above = index_3d(x, y, z, nx, ny);
                float buoyancy = 0.0f;
                if (cell_flags[below] != cell_solid && cell_flags[above] != cell_solid) {
                    buoyancy += density_buoyancy * 0.5f * (density[below] + density[above]);
                    buoyancy += temperature_buoyancy * (0.5f * (temperature[below] + temperature[above]) - ambient_temperature);
                }
                v[face_index] += dt * (uniform_force_y + buoyancy);
            }
        }

        if (x < nx && y < ny && z > 0 && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] != face_fixed) w[face_index] += dt * uniform_force_z;
        }
    }

    __global__ void advect_velocity_kernel(float* u_dst, float* v_dst, float* w_dst, const float* u_src, const float* v_src, const float* w_src, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float h, const float dt) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x <= nx && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[face_index] == face_fixed) u_dst[face_index] = u_target[face_index];
            else {
                const float3 pos = make_float3(static_cast<float>(x) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
                const float3 vel = sample_velocity_field(u_src, v_src, w_src, u_flags, v_flags, w_flags, u_target, v_target, w_target, pos, nx, ny, nz, h);
                float3 back = make_float3(clamp_world(pos.x - dt * vel.x, static_cast<float>(nx) * h), clamp_world(pos.y - dt * vel.y, static_cast<float>(ny) * h), clamp_world(pos.z - dt * vel.z, static_cast<float>(nz) * h));
                back = clip_backtrace_to_fluid(cell_flags, pos, back, nx, ny, nz, h);
                u_dst[face_index] = sample_u_field(u_src, u_flags, u_target, back.x, back.y, back.z, nx, ny, nz, h);
            }
        }

        if (x < nx && y <= ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] == face_fixed) v_dst[face_index] = v_target[face_index];
            else {
                const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, static_cast<float>(y) * h, (static_cast<float>(z) + 0.5f) * h);
                const float3 vel = sample_velocity_field(u_src, v_src, w_src, u_flags, v_flags, w_flags, u_target, v_target, w_target, pos, nx, ny, nz, h);
                float3 back = make_float3(clamp_world(pos.x - dt * vel.x, static_cast<float>(nx) * h), clamp_world(pos.y - dt * vel.y, static_cast<float>(ny) * h), clamp_world(pos.z - dt * vel.z, static_cast<float>(nz) * h));
                back = clip_backtrace_to_fluid(cell_flags, pos, back, nx, ny, nz, h);
                v_dst[face_index] = sample_v_field(v_src, v_flags, v_target, back.x, back.y, back.z, nx, ny, nz, h);
            }
        }

        if (x < nx && y < ny && z <= nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] == face_fixed) w_dst[face_index] = w_target[face_index];
            else {
                const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, static_cast<float>(z) * h);
                const float3 vel = sample_velocity_field(u_src, v_src, w_src, u_flags, v_flags, w_flags, u_target, v_target, w_target, pos, nx, ny, nz, h);
                float3 back = make_float3(clamp_world(pos.x - dt * vel.x, static_cast<float>(nx) * h), clamp_world(pos.y - dt * vel.y, static_cast<float>(ny) * h), clamp_world(pos.z - dt * vel.z, static_cast<float>(nz) * h));
                back = clip_backtrace_to_fluid(cell_flags, pos, back, nx, ny, nz, h);
                w_dst[face_index] = sample_w_field(w_src, w_flags, w_target, back.x, back.y, back.z, nx, ny, nz, h);
            }
        }
    }

    __global__ void advect_scalar_kernel(float* dst, const float* src, const float* u, const float* v, const float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float h, const float dt) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            dst[index] = 0.0f;
            return;
        }
        const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const float3 vel = sample_velocity_field(u, v, w, u_flags, v_flags, w_flags, u_target, v_target, w_target, pos, nx, ny, nz, h);
        float3 back = make_float3(clamp_world(pos.x - dt * vel.x, static_cast<float>(nx) * h), clamp_world(pos.y - dt * vel.y, static_cast<float>(ny) * h), clamp_world(pos.z - dt * vel.z, static_cast<float>(nz) * h));
        back = clip_backtrace_to_fluid(cell_flags, pos, back, nx, ny, nz, h);
        dst[index] = fmaxf(0.0f, sample_scalar_field(src, cell_flags, back.x, back.y, back.z, nx, ny, nz, h));
    }

    __global__ void diffuse_scalar_rbgs_kernel(float* dst, const float* src, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float alpha, const StableFluidsDomainBoundaryDesc domain_boundary, const int parity) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            dst[index] = 0.0f;
            return;
        }

        float inflow_sum = 0.0f;
        int inflow_count = 0;
        if (x == 0 && domain_boundary.x_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.x_min.scalar; ++inflow_count; }
        if (x == nx - 1 && domain_boundary.x_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.x_max.scalar; ++inflow_count; }
        if (y == 0 && domain_boundary.y_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.y_min.scalar; ++inflow_count; }
        if (y == ny - 1 && domain_boundary.y_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.y_max.scalar; ++inflow_count; }
        if (z == 0 && domain_boundary.z_min.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.z_min.scalar; ++inflow_count; }
        if (z == nz - 1 && domain_boundary.z_max.type == STABLE_FLUIDS_BOUNDARY_INFLOW) { inflow_sum += domain_boundary.z_max.scalar; ++inflow_count; }
        if (inflow_count > 0) {
            dst[index] = inflow_sum / static_cast<float>(inflow_count);
            return;
        }

        const float center = dst[index];
        const float left = x > 0 && cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid ? dst[index_3d(x - 1, y, z, nx, ny)] : center;
        const float right = x + 1 < nx && cell_flags[index_3d(x + 1, y, z, nx, ny)] != cell_solid ? dst[index_3d(x + 1, y, z, nx, ny)] : center;
        const float down = y > 0 && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid ? dst[index_3d(x, y - 1, z, nx, ny)] : center;
        const float up = y + 1 < ny && cell_flags[index_3d(x, y + 1, z, nx, ny)] != cell_solid ? dst[index_3d(x, y + 1, z, nx, ny)] : center;
        const float back = z > 0 && cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid ? dst[index_3d(x, y, z - 1, nx, ny)] : center;
        const float front = z + 1 < nz && cell_flags[index_3d(x, y, z + 1, nx, ny)] != cell_solid ? dst[index_3d(x, y, z + 1, nx, ny)] : center;
        dst[index] = (src[index] + alpha * (left + right + down + up + back + front)) / (1.0f + 6.0f * alpha);
    }

    __global__ void diffuse_velocity_rbgs_kernel(float* dst, const float* src, const uint8_t* flags, const float* target, const int sx, const int sy, const int sz, const float alpha, const int parity) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= sx || y >= sy || z >= sz || ((x + y + z) & 1) != parity) return;
        const auto index = index_3d(x, y, z, sx, sy);
        if (flags[index] == face_fixed) {
            dst[index] = target[index];
            return;
        }

        const float center = dst[index];
        const float left = x > 0 ? (flags[index_3d(x - 1, y, z, sx, sy)] == face_fixed ? target[index_3d(x - 1, y, z, sx, sy)] : dst[index_3d(x - 1, y, z, sx, sy)]) : center;
        const float right = x + 1 < sx ? (flags[index_3d(x + 1, y, z, sx, sy)] == face_fixed ? target[index_3d(x + 1, y, z, sx, sy)] : dst[index_3d(x + 1, y, z, sx, sy)]) : center;
        const float down = y > 0 ? (flags[index_3d(x, y - 1, z, sx, sy)] == face_fixed ? target[index_3d(x, y - 1, z, sx, sy)] : dst[index_3d(x, y - 1, z, sx, sy)]) : center;
        const float up = y + 1 < sy ? (flags[index_3d(x, y + 1, z, sx, sy)] == face_fixed ? target[index_3d(x, y + 1, z, sx, sy)] : dst[index_3d(x, y + 1, z, sx, sy)]) : center;
        const float back = z > 0 ? (flags[index_3d(x, y, z - 1, sx, sy)] == face_fixed ? target[index_3d(x, y, z - 1, sx, sy)] : dst[index_3d(x, y, z - 1, sx, sy)]) : center;
        const float front = z + 1 < sz ? (flags[index_3d(x, y, z + 1, sx, sy)] == face_fixed ? target[index_3d(x, y, z + 1, sx, sy)] : dst[index_3d(x, y, z + 1, sx, sy)]) : center;
        dst[index] = (src[index] + alpha * (left + right + down + up + back + front)) / (1.0f + 6.0f * alpha);
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* u, const float* v, const float* w, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            divergence[index] = 0.0f;
            return;
        }
        divergence[index] = -(u[index_3d(x + 1, y, z, nx + 1, ny)] - u[index_3d(x, y, z, nx + 1, ny)] + v[index_3d(x, y + 1, z, nx, ny + 1)] - v[index_3d(x, y, z, nx, ny + 1)] + w[index_3d(x, y, z + 1, nx, ny)] - w[index_3d(x, y, z, nx, ny)]) * h;
    }

    __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const uint8_t* cell_flags, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const int nx, const int ny, const int nz, const int parity) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            pressure[index] = 0.0f;
            return;
        }

        float sum = 0.0f;
        int diag = 0;

        if (u_flags[index_3d(x, y, z, nx + 1, ny)] != face_fixed) {
            ++diag;
            if (x > 0 && cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x - 1, y, z, nx, ny)];
        }
        if (u_flags[index_3d(x + 1, y, z, nx + 1, ny)] != face_fixed) {
            ++diag;
            if (x + 1 < nx && cell_flags[index_3d(x + 1, y, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x + 1, y, z, nx, ny)];
        }
        if (v_flags[index_3d(x, y, z, nx, ny + 1)] != face_fixed) {
            ++diag;
            if (y > 0 && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y - 1, z, nx, ny)];
        }
        if (v_flags[index_3d(x, y + 1, z, nx, ny + 1)] != face_fixed) {
            ++diag;
            if (y + 1 < ny && cell_flags[index_3d(x, y + 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y + 1, z, nx, ny)];
        }
        if (w_flags[index_3d(x, y, z, nx, ny)] != face_fixed) {
            ++diag;
            if (z > 0 && cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z - 1, nx, ny)];
        }
        if (w_flags[index_3d(x, y, z + 1, nx, ny)] != face_fixed) {
            ++diag;
            if (z + 1 < nz && cell_flags[index_3d(x, y, z + 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z + 1, nx, ny)];
        }

        pressure[index] = diag > 0 ? (sum + divergence[index]) / static_cast<float>(diag) : 0.0f;
    }

    __global__ void project_velocity_kernel(float* u, float* v, float* w, const float* pressure, const uint8_t* cell_flags, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const int nx, const int ny, const int nz, const float inv_h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x <= nx && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[face_index] == face_fixed) u[face_index] = u_target[face_index];
            else if (x == 0) u[face_index] -= (pressure[index_3d(0, y, z, nx, ny)] - 0.0f) * inv_h;
            else if (x == nx) u[face_index] -= (0.0f - pressure[index_3d(nx - 1, y, z, nx, ny)]) * inv_h;
            else if (cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) u[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y <= ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] == face_fixed) v[face_index] = v_target[face_index];
            else if (y == 0) v[face_index] -= (pressure[index_3d(x, 0, z, nx, ny)] - 0.0f) * inv_h;
            else if (y == ny) v[face_index] -= (0.0f - pressure[index_3d(x, ny - 1, z, nx, ny)]) * inv_h;
            else if (cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) v[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y < ny && z <= nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] == face_fixed) w[face_index] = w_target[face_index];
            else if (z == 0) w[face_index] -= (pressure[index_3d(x, y, 0, nx, ny)] - 0.0f) * inv_h;
            else if (z == nz) w[face_index] -= (0.0f - pressure[index_3d(x, y, nz - 1, nx, ny)]) * inv_h;
            else if (cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) w[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) * inv_h;
        }
    }

    __global__ void compute_velocity_magnitude_kernel(float* destination, const float* u, const float* v, const float* w, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            destination[index] = 0.0f;
            return;
        }
        const float ux = 0.5f * (u[index_3d(x, y, z, nx + 1, ny)] + u[index_3d(x + 1, y, z, nx + 1, ny)]);
        const float vy = 0.5f * (v[index_3d(x, y, z, nx, ny + 1)] + v[index_3d(x, y + 1, z, nx, ny + 1)]);
        const float wz = 0.5f * (w[index_3d(x, y, z, nx, ny)] + w[index_3d(x, y, z + 1, nx, ny)]);
        destination[index] = sqrtf(ux * ux + vy * vy + wz * wz);
    }

    __global__ void export_solid_mask_kernel(float* destination, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        destination[index] = cell_flags[index] == cell_solid ? 1.0f : 0.0f;
    }

    __global__ void pack_smoke_rgba_kernel(float* destination, const float* density, const float* dye_r, const float* dye_g, const float* dye_b, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        const auto base = index * 4ull;
        if (cell_flags[index] == cell_solid) {
            destination[base + 0] = 0.0f;
            destination[base + 1] = 0.0f;
            destination[base + 2] = 0.0f;
            destination[base + 3] = 0.0f;
            return;
        }
        destination[base + 0] = density[index];
        destination[base + 1] = dye_r[index];
        destination[base + 2] = dye_g[index];
        destination[base + 3] = dye_b[index];
    }

} // namespace stable_fluids

struct StableFluidsContext_t : stable_fluids::ContextStorage {
};

namespace {

    stable_fluids::ContextStorage* as_storage(StableFluidsContext context) {
        return static_cast<stable_fluids::ContextStorage*>(context);
    }

    int32_t rebuild_atlas_if_needed(stable_fluids::ContextStorage& context) {
        if (!context.atlas_dirty) return stable_fluids::success;
        stable_fluids::build_boundary_atlas(context);
        const int32_t code = stable_fluids::upload_boundary_atlas(context);
        if (code != 0) return code;
        context.atlas_dirty = false;
        return stable_fluids::success;
    }

    int32_t reset_fields(stable_fluids::ContextStorage& context) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 cells = stable_fluids::make_grid(context.config.nx, context.config.ny, context.config.nz, block);
        const dim3 faces = stable_fluids::make_grid(context.config.nx + 1, context.config.ny + 1, context.config.nz + 1, block);
        stable_fluids::clear_scalar_fields_kernel<<<cells, block, 0, context.stream>>>(context.device.density, context.device.dye_r, context.device.dye_g, context.device.dye_b, context.device.temperature, context.config.nx, context.config.ny, context.config.nz);
        stable_fluids::clear_velocity_fields_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        if (const int32_t code = rebuild_atlas_if_needed(context); code != 0) return code;
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(context.device.density, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(context.device.dye_r, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(context.device.dye_g, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(context.device.dye_b, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(context.device.temperature, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }

    int32_t diffuse_velocity(stable_fluids::ContextStorage& context, const dim3& block, const dim3& faces) {
        const float alpha = context.config.dt * context.config.viscosity / (context.config.cell_size * context.config.cell_size);
        if (alpha <= 0.0f) return stable_fluids::success;

        const auto u_bytes = stable_fluids::u_face_count(context.config) * sizeof(float);
        const auto v_bytes = stable_fluids::v_face_count(context.config) * sizeof(float);
        const auto w_bytes = stable_fluids::w_face_count(context.config) * sizeof(float);
        if (cudaMemcpyAsync(context.device.temp_velocity_x, context.device.velocity_x, u_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        if (cudaMemcpyAsync(context.device.temp_velocity_y, context.device.velocity_y, v_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        if (cudaMemcpyAsync(context.device.temp_velocity_z, context.device.velocity_z, w_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::cuda_failure;

        for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 1);
        }

        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }

    int32_t project_velocity(stable_fluids::ContextStorage& context, const dim3& block, const dim3& cells, const dim3& faces) {
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        if (cudaMemsetAsync(context.device.pressure, 0, stable_fluids::scalar_count(context.config) * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        stable_fluids::compute_divergence_kernel<<<cells, block, 0, context.stream>>>(context.device.divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, 0);
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, 1);
        }
        stable_fluids::project_velocity_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.pressure, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz, 1.0f / context.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }

    int32_t advect_and_diffuse_scalar(stable_fluids::ContextStorage& context, float* field, float* temp, const dim3& block, const dim3& cells) {
        stable_fluids::advect_scalar_kernel<<<cells, block, 0, context.stream>>>(temp, field, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.dt);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        const auto bytes = stable_fluids::scalar_count(context.config) * sizeof(float);
        if (cudaMemcpyAsync(field, temp, bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        const float alpha = context.config.dt * context.config.diffusion / (context.config.cell_size * context.config.cell_size);
        if (alpha > 0.0f) {
            for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
                stable_fluids::diffuse_scalar_rbgs_kernel<<<cells, block, 0, context.stream>>>(field, temp, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, alpha, context.config.domain_boundary, 0);
                stable_fluids::diffuse_scalar_rbgs_kernel<<<cells, block, 0, context.stream>>>(field, temp, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, alpha, context.config.domain_boundary, 1);
            }
            if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
        }
        stable_fluids::apply_scalar_boundary_kernel<<<cells, block, 0, context.stream>>>(field, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.domain_boundary);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }

} // namespace

extern "C" {

int32_t stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, StableFluidsContext* out_context) {
    if (out_context == nullptr) return stable_fluids::invalid_context;
    *out_context = nullptr;
    if (const int32_t code = stable_fluids_validate_context_create_desc(desc); code != 0) return code;

    std::unique_ptr<StableFluidsContext_t> context{new (std::nothrow) StableFluidsContext_t{}};
    if (!context) return stable_fluids::cuda_failure;
    context->config = desc->config;
    context->stream = static_cast<cudaStream_t>(desc->stream);
    if (context->stream == nullptr) {
        if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) return stable_fluids::cuda_failure;
        context->owns_stream = true;
    }
    if (const int32_t code = stable_fluids::allocate_buffers(*context); code != 0) {
        stable_fluids::destroy_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    }
    if (const int32_t code = reset_fields(*context); code != 0) {
        stable_fluids::destroy_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    }
    *out_context = context.release();
    return stable_fluids::success;
}

int32_t stable_fluids_destroy_context_cuda(StableFluidsContext context) {
    if (context == nullptr) return stable_fluids::success;
    auto* storage = as_storage(context);
    cudaStreamSynchronize(storage->stream);
    stable_fluids::destroy_buffers(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete context;
    return stable_fluids::success;
}

int32_t stable_fluids_reset_context_cuda(StableFluidsContext context) {
    if (context == nullptr) return stable_fluids::invalid_context;
    return reset_fields(*as_storage(context));
}

int32_t stable_fluids_update_scene_cuda(StableFluidsContext context, const StableFluidsSceneDesc* desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (const int32_t code = stable_fluids_validate_scene_desc(desc); code != 0) return code;
    auto* storage = as_storage(context);
    storage->colliders.assign(desc->colliders, desc->colliders + desc->collider_count);
    storage->atlas_dirty = true;
    return rebuild_atlas_if_needed(*storage);
}

int32_t stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (const int32_t code = stable_fluids_validate_step_desc(desc); code != 0) return code;
    auto& storage = *as_storage(context);
    if (const int32_t code = rebuild_atlas_if_needed(storage); code != 0) return code;

    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    const dim3 faces = stable_fluids::make_grid(storage.config.nx + 1, storage.config.ny + 1, storage.config.nz + 1, block);

    nvtx3::scoped_range range("stable.step.context");

    if (desc->source_count > 0) {
        StableFluidsSourceDesc* device_sources = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&device_sources), static_cast<std::size_t>(desc->source_count) * sizeof(StableFluidsSourceDesc)) != cudaSuccess) return stable_fluids::cuda_failure;
        const auto free_sources = [&]() { if (device_sources != nullptr) cudaFree(device_sources); };
        if (cudaMemcpyAsync(device_sources, desc->sources, static_cast<std::size_t>(desc->source_count) * sizeof(StableFluidsSourceDesc), cudaMemcpyHostToDevice, storage.stream) != cudaSuccess) {
            free_sources();
            return stable_fluids::cuda_failure;
        }
        stable_fluids::add_sources_kernel<<<faces, block, 0, storage.stream>>>(storage.device.density, storage.device.dye_r, storage.device.dye_g, storage.device.dye_b, storage.device.temperature, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, device_sources, static_cast<int>(desc->source_count), storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        free_sources();
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;
    }

    stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.config.nx, storage.config.ny, storage.config.nz);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;

    stable_fluids::add_forces_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.density, storage.device.temperature, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, storage.config.density_buoyancy, storage.config.temperature_buoyancy, storage.config.ambient_temperature, storage.config.uniform_force_x, storage.config.uniform_force_y, storage.config.uniform_force_z);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;

    if (const int32_t code = diffuse_velocity(storage, block, faces); code != 0) return code;
    if (const int32_t code = project_velocity(storage, block, cells, faces); code != 0) return code;

    stable_fluids::advect_velocity_kernel<<<faces, block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::cuda_failure;

    const auto u_bytes = stable_fluids::u_face_count(storage.config) * sizeof(float);
    const auto v_bytes = stable_fluids::v_face_count(storage.config) * sizeof(float);
    const auto w_bytes = stable_fluids::w_face_count(storage.config) * sizeof(float);
    if (cudaMemcpyAsync(storage.device.velocity_x, storage.device.temp_velocity_x, u_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;
    if (cudaMemcpyAsync(storage.device.velocity_y, storage.device.temp_velocity_y, v_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;
    if (cudaMemcpyAsync(storage.device.velocity_z, storage.device.temp_velocity_z, w_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;

    if (const int32_t code = project_velocity(storage, block, cells, faces); code != 0) return code;
    if (const int32_t code = advect_and_diffuse_scalar(storage, storage.device.density, storage.device.temp_density, block, cells); code != 0) return code;
    if (const int32_t code = advect_and_diffuse_scalar(storage, storage.device.dye_r, storage.device.temp_dye_r, block, cells); code != 0) return code;
    if (const int32_t code = advect_and_diffuse_scalar(storage, storage.device.dye_g, storage.device.temp_dye_g, block, cells); code != 0) return code;
    if (const int32_t code = advect_and_diffuse_scalar(storage, storage.device.dye_b, storage.device.temp_dye_b, block, cells); code != 0) return code;
    if (const int32_t code = advect_and_diffuse_scalar(storage, storage.device.temperature, storage.device.temp_density, block, cells); code != 0) return code;

    return stable_fluids::success;
}

int32_t stable_fluids_export_field_cuda(StableFluidsContext context, const StableFluidsExportFieldDesc* desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (const int32_t code = stable_fluids_validate_export_field_desc(desc); code != 0) return code;
    auto& storage = *as_storage(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);

    if (desc->field == STABLE_FLUIDS_EXPORT_DENSITY) {
        if (cudaMemcpyAsync(desc->destination, storage.device.density, stable_fluids::scalar_count(storage.config) * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        return stable_fluids::success;
    }
    if (desc->field == STABLE_FLUIDS_EXPORT_PRESSURE) {
        if (cudaMemcpyAsync(desc->destination, storage.device.pressure, stable_fluids::scalar_count(storage.config) * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        return stable_fluids::success;
    }
    if (desc->field == STABLE_FLUIDS_EXPORT_DIVERGENCE) {
        if (cudaMemcpyAsync(desc->destination, storage.device.divergence, stable_fluids::scalar_count(storage.config) * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::cuda_failure;
        return stable_fluids::success;
    }
    if (desc->field == STABLE_FLUIDS_EXPORT_DYE_RGBA) {
        stable_fluids::pack_smoke_rgba_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(desc->destination), storage.device.density, storage.device.dye_r, storage.device.dye_g, storage.device.dye_b, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }
    if (desc->field == STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE) {
        stable_fluids::compute_velocity_magnitude_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(desc->destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
    }
    stable_fluids::export_solid_mask_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(desc->destination), storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::cuda_failure;
}

int32_t stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc) {
    if (context == nullptr || out_desc == nullptr) return stable_fluids::invalid_context;
    out_desc->struct_size = sizeof(StableFluidsGridDesc);
    out_desc->api_version = STABLE_FLUIDS_API_VERSION;
    out_desc->nx = as_storage(context)->config.nx;
    out_desc->ny = as_storage(context)->config.ny;
    out_desc->nz = as_storage(context)->config.nz;
    out_desc->cell_size = as_storage(context)->config.cell_size;
    return stable_fluids::success;
}

} // extern "C"
