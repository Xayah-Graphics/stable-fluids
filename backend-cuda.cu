#include "stable-fluids-3d.h"

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

    constexpr StableFluidsResult success              = STABLE_FLUIDS_RESULT_OK;
    constexpr StableFluidsResult invalid_argument     = STABLE_FLUIDS_RESULT_INVALID_ARGUMENT;
    constexpr StableFluidsResult invalid_context      = STABLE_FLUIDS_RESULT_INVALID_CONTEXT;
    constexpr StableFluidsResult invalid_config       = STABLE_FLUIDS_RESULT_INVALID_CONFIG;
    constexpr StableFluidsResult invalid_field        = STABLE_FLUIDS_RESULT_INVALID_FIELD;
    constexpr StableFluidsResult invalid_scene        = STABLE_FLUIDS_RESULT_INVALID_SCENE;
    constexpr StableFluidsResult invalid_export       = STABLE_FLUIDS_RESULT_INVALID_EXPORT;
    constexpr StableFluidsResult out_of_memory        = STABLE_FLUIDS_RESULT_OUT_OF_MEMORY;
    constexpr StableFluidsResult backend_failure      = STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    constexpr uint8_t cell_fluid           = 0;
    constexpr uint8_t cell_solid           = 1;
    constexpr uint8_t face_open            = 0;
    constexpr uint8_t face_fixed           = 1;
    constexpr uint8_t face_outflow         = 2;
    enum class Axis : int {
        x = 0,
        y = 1,
        z = 2,
    };

    struct ProjectionMetricsState;

    struct DeviceBuffers {
        float* velocity_x    = nullptr;
        float* velocity_y    = nullptr;
        float* velocity_z    = nullptr;
        float* temp_velocity_x = nullptr;
        float* temp_velocity_y = nullptr;
        float* temp_velocity_z = nullptr;
        float* pressure      = nullptr;
        float* divergence    = nullptr;
        float* residual_divergence = nullptr;
        ProjectionMetricsState* projection_metrics = nullptr;
        float* scalar_scratch = nullptr;
        uint8_t* cell_flags  = nullptr;
        uint8_t* u_flags     = nullptr;
        uint8_t* v_flags     = nullptr;
        uint8_t* w_flags     = nullptr;
        float* u_target      = nullptr;
        float* v_target      = nullptr;
        float* w_target      = nullptr;
    };

    struct FieldStorage {
        StableFluidsFieldCreateDesc desc{};
        float* data = nullptr;
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
        std::vector<FieldStorage> fields{};
        std::vector<StableFluidsBuoyancyDesc> buoyancy_terms{};
        HostBoundaryAtlas host_atlas{};
        DeviceBuffers device{};
        Stream stream                 = nullptr;
        bool owns_stream              = false;
        bool atlas_dirty              = true;
        uint32_t max_field_components = 0;
    };

    struct ProjectionMetricsState {
        float max_abs_divergence = 0.0f;
        float sum_sq_divergence  = 0.0f;
        uint32_t fluid_cell_count = 0;
        uint32_t _padding         = 0;
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

    std::uint64_t field_value_count(const StableFluidsSimulationConfig& config, const uint32_t components) {
        return scalar_count(config) * static_cast<std::uint64_t>(components);
    }

    FieldStorage* find_field(ContextStorage& context, const StableFluidsFieldHandle handle) {
        if (handle == 0) return nullptr;
        const auto index = static_cast<std::size_t>(handle - 1u);
        if (index >= context.fields.size()) return nullptr;
        return &context.fields[index];
    }

    const FieldStorage* find_field(const ContextStorage& context, const StableFluidsFieldHandle handle) {
        if (handle == 0) return nullptr;
        const auto index = static_cast<std::size_t>(handle - 1u);
        if (index >= context.fields.size()) return nullptr;
        return &context.fields[index];
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

    StableFluidsResult upload_boundary_atlas(ContextStorage& context) {
        const auto cell_bytes   = scalar_count(context.config) * sizeof(uint8_t);
        const auto u_flag_bytes = u_face_count(context.config) * sizeof(uint8_t);
        const auto v_flag_bytes = v_face_count(context.config) * sizeof(uint8_t);
        const auto w_flag_bytes = w_face_count(context.config) * sizeof(uint8_t);
        const auto u_val_bytes  = u_face_count(context.config) * sizeof(float);
        const auto v_val_bytes  = v_face_count(context.config) * sizeof(float);
        const auto w_val_bytes  = w_face_count(context.config) * sizeof(float);

        if (cudaMemcpyAsync(context.device.cell_flags, context.host_atlas.cell_flags.data(), cell_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.u_flags, context.host_atlas.u_flags.data(), u_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.v_flags, context.host_atlas.v_flags.data(), v_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.w_flags, context.host_atlas.w_flags.data(), w_flag_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.u_target, context.host_atlas.u_target.data(), u_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.v_target, context.host_atlas.v_target.data(), v_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemcpyAsync(context.device.w_target, context.host_atlas.w_target.data(), w_val_bytes, cudaMemcpyHostToDevice, context.stream) != cudaSuccess) return backend_failure;
        return success;
    }

    StableFluidsResult build_boundary_atlas(ContextStorage& context) {
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

        auto set_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int x, const int y, const int z, const int sx, const int sy, const uint8_t type, const float target) {
            const auto index = static_cast<std::size_t>(index_3d(x, y, z, sx, sy));
            flags[index] = type;
            values[index] = target;
        };
        auto apply_domain_normal_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int x, const int y, const int z, const int sx, const int sy, const StableFluidsBoundaryFaceDesc& face) {
            if (face.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW) {
                set_face(flags, values, x, y, z, sx, sy, face_outflow, 0.0f);
                return;
            }
            set_face(flags, values, x, y, z, sx, sy, face_fixed, face.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? face.velocity : 0.0f);
        };
        auto apply_no_slip_tangent_faces = [&](const StableFluidsBoundaryFaceDesc& face, const Axis axis, const bool max_side) {
            if (face.type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) return;
            if (axis == Axis::x) {
                const int tangent_x = max_side ? nx - 1 : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int y = 0; y <= ny; ++y) set_face(context.host_atlas.v_flags, context.host_atlas.v_target, tangent_x, y, z, nx, ny + 1, face_fixed, 0.0f);
                    for (int y = 0; y < ny; ++y) set_face(context.host_atlas.w_flags, context.host_atlas.w_target, tangent_x, y, z, nx, ny, face_fixed, 0.0f);
                }
            }
            if (axis == Axis::y) {
                const int tangent_y = max_side ? ny - 1 : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int x = 0; x <= nx; ++x) set_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, tangent_y, z, nx + 1, ny, face_fixed, 0.0f);
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, tangent_y, z, nx, ny, face_fixed, 0.0f);
                }
            }
            if (axis == Axis::z) {
                const int tangent_z = max_side ? nz - 1 : 0;
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x <= nx; ++x) set_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, y, tangent_z, nx + 1, ny, face_fixed, 0.0f);
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, y, tangent_z, nx, ny + 1, face_fixed, 0.0f);
                }
            }
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                apply_domain_normal_face(context.host_atlas.u_flags, context.host_atlas.u_target, 0, y, z, nx + 1, ny, context.config.domain_boundary.x_min);
                apply_domain_normal_face(context.host_atlas.u_flags, context.host_atlas.u_target, nx, y, z, nx + 1, ny, context.config.domain_boundary.x_max);
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                apply_domain_normal_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, 0, z, nx, ny + 1, context.config.domain_boundary.y_min);
                apply_domain_normal_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, ny, z, nx, ny + 1, context.config.domain_boundary.y_max);
            }
        }

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                apply_domain_normal_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, 0, nx, ny, context.config.domain_boundary.z_min);
                apply_domain_normal_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, nz, nx, ny, context.config.domain_boundary.z_max);
            }
        }

        apply_no_slip_tangent_faces(context.config.domain_boundary.x_min, Axis::x, false);
        apply_no_slip_tangent_faces(context.config.domain_boundary.x_max, Axis::x, true);
        apply_no_slip_tangent_faces(context.config.domain_boundary.y_min, Axis::y, false);
        apply_no_slip_tangent_faces(context.config.domain_boundary.y_max, Axis::y, true);
        apply_no_slip_tangent_faces(context.config.domain_boundary.z_min, Axis::z, false);
        apply_no_slip_tangent_faces(context.config.domain_boundary.z_max, Axis::z, true);

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
                    const auto index = static_cast<std::size_t>(index_3d(x, y, z, nx + 1, ny));
                    if (context.host_atlas.u_flags[index] == face_open) {
                        context.host_atlas.u_flags[index]  = face_fixed;
                        context.host_atlas.u_target[index] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::x);
                    }
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
                    const auto index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny + 1));
                    if (context.host_atlas.v_flags[index] == face_open) {
                        context.host_atlas.v_flags[index]  = face_fixed;
                        context.host_atlas.v_target[index] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::y);
                    }
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
                    const auto index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny));
                    if (context.host_atlas.w_flags[index] == face_open) {
                        context.host_atlas.w_flags[index]  = face_fixed;
                        context.host_atlas.w_target[index] = sample_collider_velocity_axis(context.colliders, px, py, pz, Axis::z);
                    }
                }
            }
        }
        return success;
    }

    template <class T>
    StableFluidsResult allocate_device_array(T*& ptr, const std::uint64_t count) {
        if (count == 0) return success;
        if (cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)) != cudaSuccess) return out_of_memory;
        return success;
    }

    template <class T>
    void release_device_array(T*& ptr) {
        if (ptr != nullptr) cudaFree(ptr);
        ptr = nullptr;
    }

    void destroy_buffers(ContextStorage& context) {
        release_device_array(context.device.velocity_x);
        release_device_array(context.device.velocity_y);
        release_device_array(context.device.velocity_z);
        release_device_array(context.device.temp_velocity_x);
        release_device_array(context.device.temp_velocity_y);
        release_device_array(context.device.temp_velocity_z);
        release_device_array(context.device.pressure);
        release_device_array(context.device.divergence);
        release_device_array(context.device.residual_divergence);
        release_device_array(context.device.projection_metrics);
        release_device_array(context.device.scalar_scratch);
        release_device_array(context.device.cell_flags);
        release_device_array(context.device.u_flags);
        release_device_array(context.device.v_flags);
        release_device_array(context.device.w_flags);
        release_device_array(context.device.u_target);
        release_device_array(context.device.v_target);
        release_device_array(context.device.w_target);
        for (auto& field : context.fields) release_device_array(field.data);
    }

    StableFluidsResult allocate_buffers(ContextStorage& context) {
        if (allocate_device_array(context.device.velocity_x, u_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.velocity_y, v_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.velocity_z, w_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.temp_velocity_x, u_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.temp_velocity_y, v_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.temp_velocity_z, w_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.pressure, scalar_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.divergence, scalar_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.residual_divergence, scalar_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.projection_metrics, 1) != success) return out_of_memory;
        if (allocate_device_array(context.device.scalar_scratch, field_value_count(context.config, (std::max)(context.max_field_components, 1u))) != success) return out_of_memory;
        if (allocate_device_array(context.device.cell_flags, scalar_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.u_flags, u_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.v_flags, v_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.w_flags, w_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.u_target, u_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.v_target, v_face_count(context.config)) != success) return out_of_memory;
        if (allocate_device_array(context.device.w_target, w_face_count(context.config)) != success) return out_of_memory;
        for (auto& field : context.fields) {
            if (allocate_device_array(field.data, field_value_count(context.config, field.desc.component_count)) != success) return out_of_memory;
        }
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

    __device__ int wrap_index(const int value, const int size) {
        const int mod = value % size;
        return mod < 0 ? mod + size : mod;
    }

    __device__ float load_scalar_sample(const float* field, const uint8_t* cell_flags, int ix, int iy, int iz, const int nx, const int ny, const int nz, const uint32_t extension_mode, const float constant_value) {
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
            ix = wrap_index(ix, nx);
            iy = wrap_index(iy, ny);
            iz = wrap_index(iz, nz);
        } else if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
            if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT) return constant_value;
            ix = min(max(ix, 0), nx - 1);
            iy = min(max(iy, 0), ny - 1);
            iz = min(max(iz, 0), nz - 1);
        }
        if (cell_flags[index_3d(ix, iy, iz, nx, ny)] == cell_solid) return constant_value;
        return field[index_3d(ix, iy, iz, nx, ny)];
    }

    __device__ float sample_scalar_field(const float* field, const uint8_t* cell_flags, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t extension_mode, const float constant_value) {
        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        const float c000 = load_scalar_sample(field, cell_flags, x0, y0, z0, nx, ny, nz, extension_mode, constant_value);
        const float c100 = load_scalar_sample(field, cell_flags, x1, y0, z0, nx, ny, nz, extension_mode, constant_value);
        const float c010 = load_scalar_sample(field, cell_flags, x0, y1, z0, nx, ny, nz, extension_mode, constant_value);
        const float c110 = load_scalar_sample(field, cell_flags, x1, y1, z0, nx, ny, nz, extension_mode, constant_value);
        const float c001 = load_scalar_sample(field, cell_flags, x0, y0, z1, nx, ny, nz, extension_mode, constant_value);
        const float c101 = load_scalar_sample(field, cell_flags, x1, y0, z1, nx, ny, nz, extension_mode, constant_value);
        const float c011 = load_scalar_sample(field, cell_flags, x0, y1, z1, nx, ny, nz, extension_mode, constant_value);
        const float c111 = load_scalar_sample(field, cell_flags, x1, y1, z1, nx, ny, nz, extension_mode, constant_value);
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

    __global__ void clear_field_kernel(float* field, const int components, const float value_0, const float value_1, const float value_2, const float value_3, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto cell_index = index_3d(x, y, z, nx, ny);
        const auto cell_count_value = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (components > 0) field[cell_index] = value_0;
        if (components > 1) field[cell_count_value + cell_index] = value_1;
        if (components > 2) field[cell_count_value * 2u + cell_index] = value_2;
        if (components > 3) field[cell_count_value * 3u + cell_index] = value_3;
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

    __global__ void clear_solid_cells_kernel(float* field, const int components, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto cell_index = index_3d(x, y, z, nx, ny);
        if (cell_flags[cell_index] != cell_solid) return;
        const auto cell_count_value = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        for (int component = 0; component < components; ++component) field[static_cast<std::uint64_t>(component) * cell_count_value + cell_index] = 0.0f;
    }

    __global__ void add_velocity_source_kernel(float* u, float* v, float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float center_x, const float center_y, const float center_z, const float radius, const float velocity_x, const float velocity_y, const float velocity_z, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        const float radius2 = radius * radius;
        if (x <= nx && y < ny && z < nz && u_flags[index_3d(x, y, z, nx + 1, ny)] != face_fixed) {
            const float px = static_cast<float>(x) * h;
            const float py = (static_cast<float>(y) + 0.5f) * h;
            const float pz = (static_cast<float>(z) + 0.5f) * h;
            const float dx = px - center_x;
            const float dy = py - center_y;
            const float dz = pz - center_z;
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= radius2) {
                const float weight = fmaxf(0.0f, 1.0f - distance2 / radius2);
                const auto index = index_3d(x, y, z, nx + 1, ny);
                u[index] = u[index] * (1.0f - weight) + velocity_x * weight;
            }
        }

        if (x < nx && y <= ny && z < nz && v_flags[index_3d(x, y, z, nx, ny + 1)] != face_fixed) {
            const float px = (static_cast<float>(x) + 0.5f) * h;
            const float py = static_cast<float>(y) * h;
            const float pz = (static_cast<float>(z) + 0.5f) * h;
            const float dx = px - center_x;
            const float dy = py - center_y;
            const float dz = pz - center_z;
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= radius2) {
                const float weight = fmaxf(0.0f, 1.0f - distance2 / radius2);
                const auto index = index_3d(x, y, z, nx, ny + 1);
                v[index] = v[index] * (1.0f - weight) + velocity_y * weight;
            }
        }

        if (x < nx && y < ny && z <= nz && w_flags[index_3d(x, y, z, nx, ny)] != face_fixed) {
            const float px = (static_cast<float>(x) + 0.5f) * h;
            const float py = (static_cast<float>(y) + 0.5f) * h;
            const float pz = static_cast<float>(z) * h;
            const float dx = px - center_x;
            const float dy = py - center_y;
            const float dz = pz - center_z;
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 <= radius2) {
                const float weight = fmaxf(0.0f, 1.0f - distance2 / radius2);
                const auto index = index_3d(x, y, z, nx, ny);
                w[index] = w[index] * (1.0f - weight) + velocity_z * weight;
            }
        }
    }

    __global__ void add_field_source_kernel(float* field, const int components, const uint8_t* cell_flags, const float center_x, const float center_y, const float center_z, const float radius, const float value_0, const float value_1, const float value_2, const float value_3, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) return;
        const float px = (static_cast<float>(x) + 0.5f) * h;
        const float py = (static_cast<float>(y) + 0.5f) * h;
        const float pz = (static_cast<float>(z) + 0.5f) * h;
        const float dx = px - center_x;
        const float dy = py - center_y;
        const float dz = pz - center_z;
        const float distance2 = dx * dx + dy * dy + dz * dz;
        const float radius2 = radius * radius;
        if (distance2 > radius2) return;
        const float weight = fmaxf(0.0f, 1.0f - distance2 / radius2);
        const auto cell_count_value = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (components > 0) field[index] = field[index] * (1.0f - weight) + value_0 * weight;
        if (components > 1) field[cell_count_value + index] = field[cell_count_value + index] * (1.0f - weight) + value_1 * weight;
        if (components > 2) field[cell_count_value * 2u + index] = field[cell_count_value * 2u + index] * (1.0f - weight) + value_2 * weight;
        if (components > 3) field[cell_count_value * 3u + index] = field[cell_count_value * 3u + index] * (1.0f - weight) + value_3 * weight;
    }

    __global__ void add_uniform_forces_kernel(float* u, float* v, float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const int nx, const int ny, const int nz, const float dt, const float uniform_force_x, const float uniform_force_y, const float uniform_force_z) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x > 0 && x < nx && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[face_index] != face_fixed) u[face_index] += dt * uniform_force_x;
        }

        if (x < nx && y > 0 && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] != face_fixed) v[face_index] += dt * uniform_force_y;
        }

        if (x < nx && y < ny && z > 0 && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] != face_fixed) w[face_index] += dt * uniform_force_z;
        }
    }

    __global__ void add_buoyancy_kernel(float* v, const float* field, const uint8_t* v_flags, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float dt, const float weight, const float ambient) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y <= 0 || y >= ny || z >= nz) return;
        const auto face_index = index_3d(x, y, z, nx, ny + 1);
        if (v_flags[face_index] == face_fixed) return;
        const auto below = index_3d(x, y - 1, z, nx, ny);
        const auto above = index_3d(x, y, z, nx, ny);
        if (cell_flags[below] == cell_solid || cell_flags[above] == cell_solid) return;
        const float averaged = 0.5f * (field[below] + field[above]);
        v[face_index] += dt * weight * (averaged - ambient);
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

    __global__ void advect_scalar_kernel(float* dst, const float* src, const float* u, const float* v, const float* w, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t extension_mode, const float constant_value) {
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
        dst[index] = sample_scalar_field(src, cell_flags, back.x, back.y, back.z, nx, ny, nz, h, extension_mode, constant_value);
    }

    __global__ void diffuse_scalar_rbgs_kernel(float* dst, const float* src, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float alpha, const uint32_t extension_mode, const float constant_value, const int parity) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz || ((x + y + z) & 1) != parity) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            dst[index] = 0.0f;
            return;
        }
        const float left = load_scalar_sample(dst, cell_flags, x - 1, y, z, nx, ny, nz, extension_mode, constant_value);
        const float right = load_scalar_sample(dst, cell_flags, x + 1, y, z, nx, ny, nz, extension_mode, constant_value);
        const float down = load_scalar_sample(dst, cell_flags, x, y - 1, z, nx, ny, nz, extension_mode, constant_value);
        const float up = load_scalar_sample(dst, cell_flags, x, y + 1, z, nx, ny, nz, extension_mode, constant_value);
        const float back = load_scalar_sample(dst, cell_flags, x, y, z - 1, nx, ny, nz, extension_mode, constant_value);
        const float front = load_scalar_sample(dst, cell_flags, x, y, z + 1, nx, ny, nz, extension_mode, constant_value);
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

    __global__ void compute_divergence_kernel(float* divergence, const float* u, const float* v, const float* w, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float inv_h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            divergence[index] = 0.0f;
            return;
        }
        divergence[index] = (u[index_3d(x + 1, y, z, nx + 1, ny)] - u[index_3d(x, y, z, nx + 1, ny)] + v[index_3d(x, y + 1, z, nx, ny + 1)] - v[index_3d(x, y, z, nx, ny + 1)] + w[index_3d(x, y, z + 1, nx, ny)] - w[index_3d(x, y, z, nx, ny)]) * inv_h;
    }

    __global__ void accumulate_projection_metrics_kernel(ProjectionMetricsState* metrics, const float* divergence, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) return;
        const float value = fabsf(divergence[index]);
        atomicMax(reinterpret_cast<unsigned int*>(&metrics->max_abs_divergence), __float_as_uint(value));
        atomicAdd(&metrics->sum_sq_divergence, value * value);
        atomicAdd(&metrics->fluid_cell_count, 1u);
    }

    __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const uint8_t* cell_flags, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const int nx, const int ny, const int nz, const float h2, const int parity) {
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

        if (u_flags[index_3d(x, y, z, nx + 1, ny)] == face_open) {
            ++diag;
            if (x > 0 && cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x - 1, y, z, nx, ny)];
        }
        if (u_flags[index_3d(x + 1, y, z, nx + 1, ny)] == face_open) {
            ++diag;
            if (x + 1 < nx && cell_flags[index_3d(x + 1, y, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x + 1, y, z, nx, ny)];
        }
        if (v_flags[index_3d(x, y, z, nx, ny + 1)] == face_open) {
            ++diag;
            if (y > 0 && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y - 1, z, nx, ny)];
        }
        if (v_flags[index_3d(x, y + 1, z, nx, ny + 1)] == face_open) {
            ++diag;
            if (y + 1 < ny && cell_flags[index_3d(x, y + 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y + 1, z, nx, ny)];
        }
        if (w_flags[index_3d(x, y, z, nx, ny)] == face_open) {
            ++diag;
            if (z > 0 && cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z - 1, nx, ny)];
        }
        if (w_flags[index_3d(x, y, z + 1, nx, ny)] == face_open) {
            ++diag;
            if (z + 1 < nz && cell_flags[index_3d(x, y, z + 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z + 1, nx, ny)];
        }

        pressure[index] = diag > 0 ? (sum - divergence[index] * h2) / static_cast<float>(diag) : 0.0f;
    }

    __global__ void project_velocity_kernel(float* u, float* v, float* w, const float* pressure, const uint8_t* cell_flags, const uint8_t* u_flags, const uint8_t* v_flags, const uint8_t* w_flags, const float* u_target, const float* v_target, const float* w_target, const int nx, const int ny, const int nz, const float inv_h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

        if (x <= nx && y < ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx + 1, ny);
            if (u_flags[face_index] == face_fixed) u[face_index] = u_target[face_index];
            else if (u_flags[face_index] == face_open && x > 0 && x < nx && cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) u[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y <= ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] == face_fixed) v[face_index] = v_target[face_index];
            else if (v_flags[face_index] == face_open && y > 0 && y < ny && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) v[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y < ny && z <= nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] == face_fixed) w[face_index] = w_target[face_index];
            else if (w_flags[face_index] == face_open && z > 0 && z < nz && cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) w[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) * inv_h;
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

    __global__ void export_velocity_kernel(float* destination, const float* u, const float* v, const float* w, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        const auto base = index * 3ull;
        if (cell_flags[index] == cell_solid) {
            destination[base + 0] = 0.0f;
            destination[base + 1] = 0.0f;
            destination[base + 2] = 0.0f;
            return;
        }
        destination[base + 0] = 0.5f * (u[index_3d(x, y, z, nx + 1, ny)] + u[index_3d(x + 1, y, z, nx + 1, ny)]);
        destination[base + 1] = 0.5f * (v[index_3d(x, y, z, nx, ny + 1)] + v[index_3d(x, y + 1, z, nx, ny + 1)]);
        destination[base + 2] = 0.5f * (w[index_3d(x, y, z, nx, ny)] + w[index_3d(x, y, z + 1, nx, ny)]);
    }

    __global__ void export_solid_mask_kernel(float* destination, const uint8_t* cell_flags, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        destination[index] = cell_flags[index] == cell_solid ? 1.0f : 0.0f;
    }

    __global__ void export_field_components_kernel(float* destination, const float* field, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const int total_components, const int component_offset, const int export_components) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) {
            for (int component = 0; component < export_components; ++component) destination[index * static_cast<std::uint64_t>(export_components) + static_cast<std::uint64_t>(component)] = 0.0f;
            return;
        }
        const auto cell_count_value = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        for (int component = 0; component < export_components; ++component) {
            const auto src_component = static_cast<std::uint64_t>(component_offset + component);
            destination[index * static_cast<std::uint64_t>(export_components) + static_cast<std::uint64_t>(component)] = src_component < static_cast<std::uint64_t>(total_components) ? field[src_component * cell_count_value + index] : 0.0f;
        }
    }

    __global__ void pack_alpha_rgb_rgba_kernel(float* destination, const float* alpha_field, const float* rgb_field, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const int rgb_components) {
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
        const auto cell_count_value = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        destination[base + 0] = alpha_field[index];
        destination[base + 1] = rgb_components > 0 ? rgb_field[index] : 0.0f;
        destination[base + 2] = rgb_components > 1 ? rgb_field[cell_count_value + index] : 0.0f;
        destination[base + 3] = rgb_components > 2 ? rgb_field[cell_count_value * 2u + index] : 0.0f;
    }

} // namespace stable_fluids

struct StableFluidsContext_t : stable_fluids::ContextStorage {
};

namespace {

    stable_fluids::ContextStorage* as_storage(StableFluidsContext context) {
        return static_cast<stable_fluids::ContextStorage*>(context);
    }

    StableFluidsResult rebuild_atlas_if_needed(stable_fluids::ContextStorage& context) {
        if (!context.atlas_dirty) return stable_fluids::success;
        const StableFluidsResult build_code = stable_fluids::build_boundary_atlas(context);
        if (build_code != stable_fluids::success) return build_code;
        const StableFluidsResult code = stable_fluids::upload_boundary_atlas(context);
        if (code != stable_fluids::success) return code;
        context.atlas_dirty = false;
        return stable_fluids::success;
    }

    StableFluidsResult reset_fields(stable_fluids::ContextStorage& context) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 cells = stable_fluids::make_grid(context.config.nx, context.config.ny, context.config.nz, block);
        const dim3 faces = stable_fluids::make_grid(context.config.nx + 1, context.config.ny + 1, context.config.nz + 1, block);
        stable_fluids::clear_velocity_fields_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (const StableFluidsResult code = rebuild_atlas_if_needed(context); code != stable_fluids::success) return code;
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        for (const auto& field : context.fields) {
            stable_fluids::clear_field_kernel<<<cells, block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), field.desc.default_value_0, field.desc.default_value_1, field.desc.default_value_2, field.desc.default_value_3, context.config.nx, context.config.ny, context.config.nz);
            stable_fluids::clear_solid_cells_kernel<<<cells, block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz);
        }
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult diffuse_velocity(stable_fluids::ContextStorage& context, const dim3& block, const dim3& faces) {
        const float alpha = context.config.dt * context.config.viscosity / (context.config.cell_size * context.config.cell_size);
        if (alpha <= 0.0f) return stable_fluids::success;

        const auto u_bytes = stable_fluids::u_face_count(context.config) * sizeof(float);
        const auto v_bytes = stable_fluids::v_face_count(context.config) * sizeof(float);
        const auto w_bytes = stable_fluids::w_face_count(context.config) * sizeof(float);
        if (cudaMemcpyAsync(context.device.temp_velocity_x, context.device.velocity_x, u_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemcpyAsync(context.device.temp_velocity_y, context.device.velocity_y, v_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemcpyAsync(context.device.temp_velocity_z, context.device.velocity_z, w_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;

        for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 1);
        }

        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult project_velocity(stable_fluids::ContextStorage& context, const dim3& block, const dim3& cells, const dim3& faces) {
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        const float inv_h = 1.0f / context.config.cell_size;
        const float h2 = context.config.cell_size * context.config.cell_size;
        if (cudaMemsetAsync(context.device.pressure, 0, stable_fluids::scalar_count(context.config) * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::compute_divergence_kernel<<<cells, block, 0, context.stream>>>(context.device.divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 0);
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 1);
        }
        stable_fluids::project_velocity_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.pressure, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::compute_divergence_kernel<<<cells, block, 0, context.stream>>>(context.device.residual_divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::accumulate_projection_metrics_kernel<<<cells, block, 0, context.stream>>>(context.device.projection_metrics, context.device.residual_divergence, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult advect_and_diffuse_scalar(stable_fluids::ContextStorage& context, const stable_fluids::FieldStorage& field, const dim3& block, const dim3& cells) {
        const auto cell_count_value = stable_fluids::scalar_count(context.config);
        const auto bytes = cell_count_value * sizeof(float);
        for (uint32_t component = 0; component < field.desc.component_count; ++component) {
            float* field_component = field.data + static_cast<std::uint64_t>(component) * cell_count_value;
            float* temp_component = context.device.scalar_scratch + static_cast<std::uint64_t>(component) * cell_count_value;
            const float constant_value = component == 0 ? field.desc.default_value_0 : (component == 1 ? field.desc.default_value_1 : (component == 2 ? field.desc.default_value_2 : field.desc.default_value_3));
            if ((field.desc.flags & STABLE_FLUIDS_FIELD_ADVECT) != 0u) {
                stable_fluids::advect_scalar_kernel<<<cells, block, 0, context.stream>>>(temp_component, field_component, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.dt, field.desc.extension_mode, constant_value);
                if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
                if (cudaMemcpyAsync(field_component, temp_component, bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
            }
            if ((field.desc.flags & STABLE_FLUIDS_FIELD_DIFFUSE) != 0u) {
                const float alpha = context.config.dt * field.desc.diffusion / (context.config.cell_size * context.config.cell_size);
                if (alpha > 0.0f) {
                    if (cudaMemcpyAsync(temp_component, field_component, bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
                    for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
                        stable_fluids::diffuse_scalar_rbgs_kernel<<<cells, block, 0, context.stream>>>(field_component, temp_component, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, alpha, field.desc.extension_mode, constant_value, 0);
                        stable_fluids::diffuse_scalar_rbgs_kernel<<<cells, block, 0, context.stream>>>(field_component, temp_component, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, alpha, field.desc.extension_mode, constant_value, 1);
                    }
                    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
                }
            }
        }
        stable_fluids::clear_solid_cells_kernel<<<cells, block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

} // namespace

extern "C" {

StableFluidsResult stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, StableFluidsContext* out_context, StableFluidsFieldHandle* out_field_handles, const uint32_t out_field_handle_capacity) {
    if (out_context == nullptr) return stable_fluids::invalid_argument;
    *out_context = nullptr;
    if (desc == nullptr) return stable_fluids::invalid_argument;
    if (desc->config.nx <= 0 || desc->config.ny <= 0 || desc->config.nz <= 0) return stable_fluids::invalid_config;
    if (desc->config.cell_size <= 0.0f) return stable_fluids::invalid_config;
    if (desc->config.dt <= 0.0f) return stable_fluids::invalid_config;
    if (desc->config.diffuse_iterations <= 0 || desc->config.pressure_iterations <= 0) return stable_fluids::invalid_config;
    const auto validate_boundary_face = [](const StableFluidsBoundaryFaceDesc& face) {
        return face.type <= STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW;
    };
    if (!validate_boundary_face(desc->config.domain_boundary.x_min)) return stable_fluids::invalid_config;
    if (!validate_boundary_face(desc->config.domain_boundary.x_max)) return stable_fluids::invalid_config;
    if (!validate_boundary_face(desc->config.domain_boundary.y_min)) return stable_fluids::invalid_config;
    if (!validate_boundary_face(desc->config.domain_boundary.y_max)) return stable_fluids::invalid_config;
    if (!validate_boundary_face(desc->config.domain_boundary.z_min)) return stable_fluids::invalid_config;
    if (!validate_boundary_face(desc->config.domain_boundary.z_max)) return stable_fluids::invalid_config;
    if (desc->field_count > 0 && desc->fields == nullptr) return stable_fluids::invalid_argument;
    if (desc->field_count > 0 && (out_field_handles == nullptr || out_field_handle_capacity < desc->field_count)) return stable_fluids::invalid_argument;
    if (desc->buoyancy_term_count > 0 && desc->buoyancy_terms == nullptr) return stable_fluids::invalid_argument;
    for (uint32_t index = 0; index < desc->field_count; ++index) {
        const auto& field = desc->fields[index];
        if (field.component_count == 0 || field.component_count > 4) return stable_fluids::invalid_field;
        if ((field.flags & ~(STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE)) != 0u) return stable_fluids::invalid_field;
        if (field.extension_mode > STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE) return stable_fluids::invalid_field;
        if (field.diffusion < 0.0f) return stable_fluids::invalid_field;
    }
    for (uint32_t index = 0; index < desc->buoyancy_term_count; ++index) {
        const auto& term = desc->buoyancy_terms[index];
        if (term.field_index >= desc->field_count) return stable_fluids::invalid_field;
    }

    std::unique_ptr<StableFluidsContext_t> context{new (std::nothrow) StableFluidsContext_t{}};
    if (!context) return stable_fluids::out_of_memory;
    context->config = desc->config;
    context->stream = static_cast<cudaStream_t>(desc->stream);
    if (context->stream == nullptr) {
        if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) return stable_fluids::backend_failure;
        context->owns_stream = true;
    }
    context->fields.reserve(desc->field_count);
    for (uint32_t index = 0; index < desc->field_count; ++index) {
        context->fields.push_back(stable_fluids::FieldStorage{
            .desc = desc->fields[index],
        });
        context->max_field_components = (std::max)(context->max_field_components, desc->fields[index].component_count);
        if (out_field_handles != nullptr) out_field_handles[index] = index + 1u;
    }
    if (desc->buoyancy_term_count > 0) context->buoyancy_terms.assign(desc->buoyancy_terms, desc->buoyancy_terms + desc->buoyancy_term_count);
    else context->buoyancy_terms.clear();
    for (const auto& term : context->buoyancy_terms) {
        const auto* field = stable_fluids::find_field(*context, term.field_index + 1u);
        if (field == nullptr || field->desc.component_count == 0) {
            if (context->owns_stream) cudaStreamDestroy(context->stream);
            return stable_fluids::invalid_field;
        }
    }
    if (const StableFluidsResult code = stable_fluids::allocate_buffers(*context); code != stable_fluids::success) {
        stable_fluids::destroy_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    }
    if (const StableFluidsResult code = reset_fields(*context); code != stable_fluids::success) {
        stable_fluids::destroy_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    }
    *out_context = context.release();
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_destroy_context_cuda(StableFluidsContext context) {
    if (context == nullptr) return stable_fluids::success;
    auto* storage = as_storage(context);
    cudaStreamSynchronize(storage->stream);
    stable_fluids::destroy_buffers(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete context;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_reset_context_cuda(StableFluidsContext context) {
    if (context == nullptr) return stable_fluids::invalid_context;
    return reset_fields(*as_storage(context));
}

StableFluidsResult stable_fluids_update_scene_cuda(StableFluidsContext context, const StableFluidsSceneDesc* desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (desc == nullptr) return stable_fluids::invalid_argument;
    if (desc->collider_count > 0 && desc->colliders == nullptr) return stable_fluids::invalid_scene;
    for (uint32_t index = 0; index < desc->collider_count; ++index) {
        const auto& collider = desc->colliders[index];
        if (collider.collider_type > STABLE_FLUIDS_COLLIDER_BOX) return stable_fluids::invalid_scene;
        if (collider.velocity_boundary_type > STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP) return stable_fluids::invalid_scene;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_SPHERE && collider.radius <= 0.0f) return stable_fluids::invalid_scene;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_BOX && (collider.half_extent_x <= 0.0f || collider.half_extent_y <= 0.0f || collider.half_extent_z <= 0.0f)) return stable_fluids::invalid_scene;
    }
    auto* storage = as_storage(context);
    if (desc->collider_count > 0) storage->colliders.assign(desc->colliders, desc->colliders + desc->collider_count);
    else storage->colliders.clear();
    storage->atlas_dirty = true;
    return rebuild_atlas_if_needed(*storage);
}

StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (desc == nullptr) return stable_fluids::invalid_argument;
    auto& storage = *as_storage(context);
    if (desc->velocity_source_count > 0 && desc->velocity_sources == nullptr) return stable_fluids::invalid_argument;
    if (desc->field_source_count > 0 && desc->field_sources == nullptr) return stable_fluids::invalid_argument;
    for (uint32_t index = 0; index < desc->velocity_source_count; ++index) {
        if (desc->velocity_sources[index].radius <= 0.0f) return stable_fluids::invalid_argument;
    }
    for (uint32_t index = 0; index < desc->field_source_count; ++index) {
        const auto& source = desc->field_sources[index];
        if (source.radius <= 0.0f) return stable_fluids::invalid_argument;
        if (stable_fluids::find_field(storage, source.field) == nullptr) return stable_fluids::invalid_field;
    }
    if (const StableFluidsResult code = rebuild_atlas_if_needed(storage); code != stable_fluids::success) return code;

    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    const dim3 faces = stable_fluids::make_grid(storage.config.nx + 1, storage.config.ny + 1, storage.config.nz + 1, block);

    nvtx3::scoped_range range("stable.step.context");

    for (uint32_t index = 0; index < desc->velocity_source_count; ++index) {
        const auto& source = desc->velocity_sources[index];
        stable_fluids::add_velocity_source_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_x, source.velocity_y, source.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    for (uint32_t index = 0; index < desc->field_source_count; ++index) {
        const auto& source = desc->field_sources[index];
        const auto* field = stable_fluids::find_field(storage, source.field);
        if (field == nullptr) return stable_fluids::invalid_field;
        stable_fluids::add_field_source_kernel<<<cells, block, 0, storage.stream>>>(field->data, static_cast<int>(field->desc.component_count), storage.device.cell_flags, source.center_x, source.center_y, source.center_z, source.radius, source.value_0, source.value_1, source.value_2, source.value_3, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.config.nx, storage.config.ny, storage.config.nz);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    stable_fluids::add_uniform_forces_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, storage.config.uniform_force_x, storage.config.uniform_force_y, storage.config.uniform_force_z);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    for (const auto& term : storage.buoyancy_terms) {
        const auto* field = stable_fluids::find_field(storage, term.field_index + 1u);
        if (field == nullptr) return stable_fluids::invalid_field;
        stable_fluids::add_buoyancy_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_y, field->data, storage.device.v_flags, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, term.weight, term.ambient);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    if (const StableFluidsResult code = diffuse_velocity(storage, block, faces); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = project_velocity(storage, block, cells, faces); code != stable_fluids::success) return code;

    stable_fluids::advect_velocity_kernel<<<faces, block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    const auto u_bytes = stable_fluids::u_face_count(storage.config) * sizeof(float);
    const auto v_bytes = stable_fluids::v_face_count(storage.config) * sizeof(float);
    const auto w_bytes = stable_fluids::w_face_count(storage.config) * sizeof(float);
    if (cudaMemcpyAsync(storage.device.velocity_x, storage.device.temp_velocity_x, u_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpyAsync(storage.device.velocity_y, storage.device.temp_velocity_y, v_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpyAsync(storage.device.velocity_z, storage.device.temp_velocity_z, w_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;

    if (const StableFluidsResult code = project_velocity(storage, block, cells, faces); code != stable_fluids::success) return code;
    for (const auto& field : storage.fields) {
        if (const StableFluidsResult code = advect_and_diffuse_scalar(storage, field, block, cells); code != stable_fluids::success) return code;
    }

    return stable_fluids::success;
}

StableFluidsResult stable_fluids_export_field_components_cuda(StableFluidsContext context, const StableFluidsFieldHandle field_handle, const uint32_t component_offset, const uint32_t component_count, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    const auto* field = stable_fluids::find_field(storage, field_handle);
    if (field == nullptr || component_count == 0 || component_count > 4 || component_offset + component_count > field->desc.component_count) return stable_fluids::invalid_export;
    stable_fluids::export_field_components_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(field->desc.component_count), static_cast<int>(component_offset), static_cast<int>(component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_alpha_rgb_rgba_cuda(StableFluidsContext context, const StableFluidsFieldHandle alpha_field_handle, const StableFluidsFieldHandle rgb_field_handle, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    const auto* alpha_field = stable_fluids::find_field(storage, alpha_field_handle);
    const auto* rgb_field = stable_fluids::find_field(storage, rgb_field_handle);
    if (alpha_field == nullptr || rgb_field == nullptr || alpha_field->desc.component_count < 1 || rgb_field->desc.component_count < 3) return stable_fluids::invalid_export;
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    stable_fluids::pack_alpha_rgb_rgba_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), alpha_field->data, rgb_field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(rgb_field->desc.component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_cuda(StableFluidsContext context, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    stable_fluids::export_velocity_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_magnitude_cuda(StableFluidsContext context, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    stable_fluids::compute_velocity_magnitude_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_solid_mask_cuda(StableFluidsContext context, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells = stable_fluids::make_grid(storage.config.nx, storage.config.ny, storage.config.nz, block);
    stable_fluids::export_solid_mask_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_pressure_cuda(StableFluidsContext context, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    if (cudaMemcpyAsync(destination, storage.device.pressure, stable_fluids::scalar_count(storage.config) * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_export_divergence_cuda(StableFluidsContext context, void* destination) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (destination == nullptr) return stable_fluids::invalid_export;
    auto& storage = *as_storage(context);
    if (cudaMemcpyAsync(destination, storage.device.divergence, stable_fluids::scalar_count(storage.config) * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_get_projection_metrics_cuda(StableFluidsContext context, StableFluidsProjectionMetrics* out_metrics) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (out_metrics == nullptr) return stable_fluids::invalid_argument;
    auto& storage = *as_storage(context);
    stable_fluids::ProjectionMetricsState state{};
    if (cudaStreamSynchronize(storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpy(&state, storage.device.projection_metrics, sizeof(state), cudaMemcpyDeviceToHost) != cudaSuccess) return stable_fluids::backend_failure;
    out_metrics->max_abs_divergence = state.max_abs_divergence;
    out_metrics->rms_divergence = state.fluid_cell_count > 0 ? std::sqrt(state.sum_sq_divergence / static_cast<float>(state.fluid_cell_count)) : 0.0f;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc) {
    if (context == nullptr) return stable_fluids::invalid_context;
    if (out_desc == nullptr) return stable_fluids::invalid_argument;
    out_desc->nx = as_storage(context)->config.nx;
    out_desc->ny = as_storage(context)->config.ny;
    out_desc->nz = as_storage(context)->config.nz;
    out_desc->cell_size = as_storage(context)->config.cell_size;
    return stable_fluids::success;
}

} // extern "C"
