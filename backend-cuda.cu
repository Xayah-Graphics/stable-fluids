#include "stable-fluids-3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <new>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {

    using Stream = cudaStream_t;

    constexpr StableFluidsResult success              = STABLE_FLUIDS_RESULT_OK;
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

    struct LaunchGeometry {
        dim3 block{};
        dim3 cells{};
        dim3 faces{};
    };

    struct DeviceBuffers {
        float* velocity_x    = nullptr;
        float* velocity_y    = nullptr;
        float* velocity_z    = nullptr;
        float* temp_velocity_x = nullptr;
        float* temp_velocity_y = nullptr;
        float* temp_velocity_z = nullptr;
        float* pressure      = nullptr;
        float* divergence    = nullptr;
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
        std::vector<int32_t> cell_owner{};
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

    LaunchGeometry make_launch_geometry(const StableFluidsSimulationConfig& config) {
        const dim3 block(
            static_cast<unsigned>(std::max(config.block_x, 1)),
            static_cast<unsigned>(std::max(config.block_y, 1)),
            static_cast<unsigned>(std::max(config.block_z, 1))
        );
        return {
            .block = block,
            .cells = make_grid(config.nx, config.ny, config.nz, block),
            .faces = make_grid(config.nx + 1, config.ny + 1, config.nz + 1, block),
        };
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

    void resize_host_atlas(ContextStorage& context) {
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto u_count = static_cast<std::uint64_t>(context.config.nx + 1) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto v_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny + 1) * static_cast<std::uint64_t>(context.config.nz);
        const auto w_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz + 1);
        context.host_atlas.cell_flags.resize(static_cast<std::size_t>(cell_count), cell_fluid);
        context.host_atlas.cell_owner.resize(static_cast<std::size_t>(cell_count), -1);
        context.host_atlas.u_flags.resize(static_cast<std::size_t>(u_count), face_open);
        context.host_atlas.v_flags.resize(static_cast<std::size_t>(v_count), face_open);
        context.host_atlas.w_flags.resize(static_cast<std::size_t>(w_count), face_open);
        context.host_atlas.u_target.resize(static_cast<std::size_t>(u_count), 0.0f);
        context.host_atlas.v_target.resize(static_cast<std::size_t>(v_count), 0.0f);
        context.host_atlas.w_target.resize(static_cast<std::size_t>(w_count), 0.0f);
    }

    StableFluidsResult upload_boundary_atlas(ContextStorage& context) {
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto u_count = static_cast<std::uint64_t>(context.config.nx + 1) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto v_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny + 1) * static_cast<std::uint64_t>(context.config.nz);
        const auto w_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz + 1);
        const auto cell_bytes   = cell_count * sizeof(uint8_t);
        const auto u_flag_bytes = u_count * sizeof(uint8_t);
        const auto v_flag_bytes = v_count * sizeof(uint8_t);
        const auto w_flag_bytes = w_count * sizeof(uint8_t);
        const auto u_val_bytes  = u_count * sizeof(float);
        const auto v_val_bytes  = v_count * sizeof(float);
        const auto w_val_bytes  = w_count * sizeof(float);

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
        constexpr float collider_touch_radius = 0.75f;
        constexpr float huge_distance = 1.0e30f;

        struct DomainSide {
            Axis axis;
            bool max_side;
            StableFluidsBoundaryFaceDesc face;
        };

        const std::array domain_sides{
            DomainSide{ .axis = Axis::x, .max_side = false, .face = context.config.domain_boundary.x_min, },
            DomainSide{ .axis = Axis::x, .max_side = true,  .face = context.config.domain_boundary.x_max, },
            DomainSide{ .axis = Axis::y, .max_side = false, .face = context.config.domain_boundary.y_min, },
            DomainSide{ .axis = Axis::y, .max_side = true,  .face = context.config.domain_boundary.y_max, },
            DomainSide{ .axis = Axis::z, .max_side = false, .face = context.config.domain_boundary.z_min, },
            DomainSide{ .axis = Axis::z, .max_side = true,  .face = context.config.domain_boundary.z_max, },
        };

        resize_host_atlas(context);
        std::ranges::fill(context.host_atlas.cell_flags, cell_fluid);
        std::ranges::fill(context.host_atlas.cell_owner.begin(), context.host_atlas.cell_owner.end(), -1);
        std::ranges::fill(context.host_atlas.u_flags.begin(), context.host_atlas.u_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.v_flags.begin(), context.host_atlas.v_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.w_flags.begin(), context.host_atlas.w_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.u_target.begin(), context.host_atlas.u_target.end(), 0.0f);
        std::ranges::fill(context.host_atlas.v_target.begin(), context.host_atlas.v_target.end(), 0.0f);
        std::ranges::fill(context.host_atlas.w_target.begin(), context.host_atlas.w_target.end(), 0.0f);

        auto set_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int x, const int y, const int z, const int sx, const int sy, const uint8_t type, const float target) {
            const auto index = static_cast<std::size_t>(index_3d(x, y, z, sx, sy));
            flags[index] = type;
            values[index] = target;
        };
        auto cell_owner = [&](const int x, const int y, const int z) {
            if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return -1;
            return context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))];
        };
        auto collider_axis_velocity = [&](const StableFluidsColliderDesc& collider, const Axis axis) {
            if (axis == Axis::x) return collider.linear_velocity_x;
            if (axis == Axis::y) return collider.linear_velocity_y;
            return collider.linear_velocity_z;
        };
        auto find_best_collider = [&](const float px, const float py, const float pz, const bool require_inside, const bool require_no_slip, const float max_distance) {
            float best_distance = huge_distance;
            int best_owner      = -1;
            for (std::size_t collider_index = 0; collider_index < context.colliders.size(); ++collider_index) {
                const auto& collider = context.colliders[collider_index];
                if (require_inside && !point_inside_collider(collider, px, py, pz)) continue;
                if (require_no_slip && collider.velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) continue;
                const float distance = collider_signed_distance(collider, px, py, pz);
                if (distance > max_distance || distance >= best_distance) continue;
                best_distance = distance;
                best_owner = static_cast<int>(collider_index);
            }
            return best_owner;
        };
        auto boundary_face_type = [&](const StableFluidsBoundaryFaceDesc& face) {
            return face.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
        };
        auto boundary_face_target = [&](const StableFluidsBoundaryFaceDesc& face) {
            return face.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? face.velocity : 0.0f;
        };
        auto pick_face_owner = [&](const int owner_a, const int owner_b, const float px, const float py, const float pz) {
            if (owner_a >= 0 && owner_b < 0) return owner_a;
            if (owner_b >= 0 && owner_a < 0) return owner_b;
            if (owner_a >= 0 && owner_b >= 0) {
                const float distance_a = collider_signed_distance(context.colliders[static_cast<std::size_t>(owner_a)], px, py, pz);
                const float distance_b = collider_signed_distance(context.colliders[static_cast<std::size_t>(owner_b)], px, py, pz);
                return distance_a <= distance_b ? owner_a : owner_b;
            }
            return find_best_collider(px, py, pz, false, false, collider_touch_radius);
        };
        auto set_interface_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int face_index_x, const int face_index_y, const int face_index_z, const int sx, const int sy, const float px, const float py, const float pz, const Axis axis, const int owner_a, const int owner_b) {
            const auto face_index = static_cast<std::size_t>(index_3d(face_index_x, face_index_y, face_index_z, sx, sy));
            if (flags[face_index] != face_open) return;
            const int owner = pick_face_owner(owner_a, owner_b, px, py, pz);
            if (owner < 0) return;
            flags[face_index] = face_fixed;
            values[face_index] = collider_axis_velocity(context.colliders[static_cast<std::size_t>(owner)], axis);
        };
        auto set_no_slip_touch_face = [&](std::vector<uint8_t>& flags, std::vector<float>& values, const int face_index_x, const int face_index_y, const int face_index_z, const int sx, const int sy, const float px, const float py, const float pz, const Axis axis, const int owner) {
            if (owner < 0) return;
            const auto face_index = static_cast<std::size_t>(index_3d(face_index_x, face_index_y, face_index_z, sx, sy));
            if (flags[face_index] != face_open) return;
            if (context.colliders[static_cast<std::size_t>(owner)].velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) return;
            flags[face_index] = face_fixed;
            values[face_index] = collider_axis_velocity(context.colliders[static_cast<std::size_t>(owner)], axis);
        };
        auto pick_touch_owner = [&](const float px, const float py, const float pz, const int x_begin, const int x_end, const int y_begin, const int y_end, const int z_begin, const int z_end) {
            float best_distance = huge_distance;
            int best_owner      = -1;
            for (int cell_z = z_begin; cell_z <= z_end; ++cell_z) {
                for (int cell_y = y_begin; cell_y <= y_end; ++cell_y) {
                    for (int cell_x = x_begin; cell_x <= x_end; ++cell_x) {
                        const int owner = cell_owner(cell_x, cell_y, cell_z);
                        if (owner < 0) continue;
                        const auto& collider = context.colliders[static_cast<std::size_t>(owner)];
                        if (collider.velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) continue;
                        const float distance = collider_signed_distance(collider, px, py, pz);
                        if (distance >= best_distance) continue;
                        best_distance = distance;
                        best_owner = owner;
                    }
                }
            }
            return best_owner;
        };
        auto apply_domain_side = [&](const DomainSide& side) {
            const uint8_t type = boundary_face_type(side.face);
            const float target = boundary_face_target(side.face);
            switch (side.axis) {
            case Axis::x: {
                const int normal_x = side.max_side ? nx : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int y = 0; y < ny; ++y) set_face(context.host_atlas.u_flags, context.host_atlas.u_target, normal_x, y, z, nx + 1, ny, type, target);
                }
                if (side.face.type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) break;
                const int tangent_x = side.max_side ? nx - 1 : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int y = 0; y <= ny; ++y) set_face(context.host_atlas.v_flags, context.host_atlas.v_target, tangent_x, y, z, nx, ny + 1, face_fixed, 0.0f);
                    for (int y = 0; y < ny; ++y) set_face(context.host_atlas.w_flags, context.host_atlas.w_target, tangent_x, y, z, nx, ny, face_fixed, 0.0f);
                }
                break;
            }
            case Axis::y: {
                const int normal_y = side.max_side ? ny : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, normal_y, z, nx, ny + 1, type, target);
                }
                if (side.face.type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) break;
                const int tangent_y = side.max_side ? ny - 1 : 0;
                for (int z = 0; z < nz; ++z) {
                    for (int x = 0; x <= nx; ++x) set_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, tangent_y, z, nx + 1, ny, face_fixed, 0.0f);
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, tangent_y, z, nx, ny, face_fixed, 0.0f);
                }
                break;
            }
            case Axis::z: {
                const int normal_z = side.max_side ? nz : 0;
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, normal_z, nx, ny, type, target);
                }
                if (side.face.type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) break;
                const int tangent_z = side.max_side ? nz - 1 : 0;
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x <= nx; ++x) set_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, y, tangent_z, nx + 1, ny, face_fixed, 0.0f);
                    for (int x = 0; x < nx; ++x) set_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, y, tangent_z, nx, ny + 1, face_fixed, 0.0f);
                }
                break;
            }
            }
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const auto cell_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny));
                    const int owner = find_best_collider(px, py, pz, true, false, huge_distance);
                    if (owner < 0) continue;
                    context.host_atlas.cell_flags[cell_index] = cell_solid;
                    context.host_atlas.cell_owner[cell_index] = owner;
                }
            }
        }

        for (const auto& side : domain_sides) apply_domain_side(side);

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 1; x < nx; ++x) {
                    const int left_owner = cell_owner(x - 1, y, z);
                    const int right_owner = cell_owner(x, y, z);
                    if (left_owner < 0 && right_owner < 0) continue;
                    const float px = static_cast<float>(x) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    set_interface_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, y, z, nx + 1, ny, px, py, pz, Axis::x, left_owner, right_owner);
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 1; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int below_owner = cell_owner(x, y - 1, z);
                    const int above_owner = cell_owner(x, y, z);
                    if (below_owner < 0 && above_owner < 0) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = static_cast<float>(y) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    set_interface_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, y, z, nx, ny + 1, px, py, pz, Axis::y, below_owner, above_owner);
                }
            }
        }

        for (int z = 1; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int back_owner = cell_owner(x, y, z - 1);
                    const int front_owner = cell_owner(x, y, z);
                    if (back_owner < 0 && front_owner < 0) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = static_cast<float>(z) * h;
                    set_interface_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, z, nx, ny, px, py, pz, Axis::z, back_owner, front_owner);
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x <= nx; ++x) {
                    const float px = static_cast<float>(x) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const int owner = pick_touch_owner(px, py, pz, x - 1, x, y - 1, y, z - 1, z);
                    set_no_slip_touch_face(context.host_atlas.u_flags, context.host_atlas.u_target, x, y, z, nx + 1, ny, px, py, pz, Axis::x, owner);
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y <= ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = static_cast<float>(y) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const int owner = pick_touch_owner(px, py, pz, x - 1, x, y - 1, y, z - 1, z);
                    set_no_slip_touch_face(context.host_atlas.v_flags, context.host_atlas.v_target, x, y, z, nx, ny + 1, px, py, pz, Axis::y, owner);
                }
            }
        }

        for (int z = 0; z <= nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = static_cast<float>(z) * h;
                    const int owner = pick_touch_owner(px, py, pz, x - 1, x, y - 1, y, z - 1, z);
                    set_no_slip_touch_face(context.host_atlas.w_flags, context.host_atlas.w_target, x, y, z, nx, ny, px, py, pz, Axis::z, owner);
                }
            }
        }
        return success;
    }

    void destroy_buffers(ContextStorage& context) {
        if (context.device.velocity_x != nullptr) cudaFree(context.device.velocity_x);
        if (context.device.velocity_y != nullptr) cudaFree(context.device.velocity_y);
        if (context.device.velocity_z != nullptr) cudaFree(context.device.velocity_z);
        if (context.device.temp_velocity_x != nullptr) cudaFree(context.device.temp_velocity_x);
        if (context.device.temp_velocity_y != nullptr) cudaFree(context.device.temp_velocity_y);
        if (context.device.temp_velocity_z != nullptr) cudaFree(context.device.temp_velocity_z);
        if (context.device.pressure != nullptr) cudaFree(context.device.pressure);
        if (context.device.divergence != nullptr) cudaFree(context.device.divergence);
        if (context.device.projection_metrics != nullptr) cudaFree(context.device.projection_metrics);
        if (context.device.scalar_scratch != nullptr) cudaFree(context.device.scalar_scratch);
        if (context.device.cell_flags != nullptr) cudaFree(context.device.cell_flags);
        if (context.device.u_flags != nullptr) cudaFree(context.device.u_flags);
        if (context.device.v_flags != nullptr) cudaFree(context.device.v_flags);
        if (context.device.w_flags != nullptr) cudaFree(context.device.w_flags);
        if (context.device.u_target != nullptr) cudaFree(context.device.u_target);
        if (context.device.v_target != nullptr) cudaFree(context.device.v_target);
        if (context.device.w_target != nullptr) cudaFree(context.device.w_target);
        context.device.velocity_x = nullptr;
        context.device.velocity_y = nullptr;
        context.device.velocity_z = nullptr;
        context.device.temp_velocity_x = nullptr;
        context.device.temp_velocity_y = nullptr;
        context.device.temp_velocity_z = nullptr;
        context.device.pressure = nullptr;
        context.device.divergence = nullptr;
        context.device.projection_metrics = nullptr;
        context.device.scalar_scratch = nullptr;
        context.device.cell_flags = nullptr;
        context.device.u_flags = nullptr;
        context.device.v_flags = nullptr;
        context.device.w_flags = nullptr;
        context.device.u_target = nullptr;
        context.device.v_target = nullptr;
        context.device.w_target = nullptr;
        for (auto& field : context.fields) {
            if (field.data != nullptr) cudaFree(field.data);
            field.data = nullptr;
        }
    }

    StableFluidsResult allocate_buffers(ContextStorage& context) {
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto u_count = static_cast<std::uint64_t>(context.config.nx + 1) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto v_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny + 1) * static_cast<std::uint64_t>(context.config.nz);
        const auto w_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz + 1);
        const auto scratch_count = cell_count * static_cast<std::uint64_t>((std::max)(context.max_field_components, 1u));
        if (u_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_x), u_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (v_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_y), v_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (w_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_z), w_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (u_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_x), u_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (v_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_y), v_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (w_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_z), w_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.pressure), cell_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.divergence), cell_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (cudaMalloc(reinterpret_cast<void**>(&context.device.projection_metrics), sizeof(ProjectionMetricsState)) != cudaSuccess) return out_of_memory;
        if (scratch_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.scalar_scratch), scratch_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.cell_flags), cell_count * sizeof(uint8_t)) != cudaSuccess) return out_of_memory;
        if (u_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.u_flags), u_count * sizeof(uint8_t)) != cudaSuccess) return out_of_memory;
        if (v_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.v_flags), v_count * sizeof(uint8_t)) != cudaSuccess) return out_of_memory;
        if (w_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.w_flags), w_count * sizeof(uint8_t)) != cudaSuccess) return out_of_memory;
        if (u_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.u_target), u_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (v_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.v_target), v_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        if (w_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.w_target), w_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        for (auto& field : context.fields) {
            const auto field_count = cell_count * static_cast<std::uint64_t>(field.desc.component_count);
            if (field_count > 0 && cudaMalloc(reinterpret_cast<void**>(&field.data), field_count * sizeof(float)) != cudaSuccess) return out_of_memory;
        }
        return success;
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

    __device__ bool point_inside_domain(const float3 point, const int nx, const int ny, const int nz, const float h) {
        return point.x >= 0.0f && point.x <= static_cast<float>(nx) * h && point.y >= 0.0f && point.y <= static_cast<float>(ny) * h && point.z >= 0.0f && point.z <= static_cast<float>(nz) * h;
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

    struct ScalarAxisSample {
        int i0;
        int i1;
        float t;
    };

    __device__ ScalarAxisSample resolve_scalar_axis(const float g, const int size, const uint32_t extension_mode) {
        if (size <= 1) return { .i0 = 0, .i1 = 0, .t = 0.0f, };
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
            const int i0 = static_cast<int>(floorf(g));
            const int i1 = i0 + 1;
            return {
                .i0 = wrap_index(i0, size),
                .i1 = wrap_index(i1, size),
                .t = g - static_cast<float>(i0),
            };
        }
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            const float clamped = fminf(fmaxf(g, 0.0f), static_cast<float>(size - 1));
            const int i0 = static_cast<int>(floorf(clamped));
            const int i1 = min(i0 + 1, size - 1);
            return {
                .i0 = i0,
                .i1 = i1,
                .t = clamped - static_cast<float>(i0),
            };
        }
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE) {
            if (g <= 0.0f) return { .i0 = 0, .i1 = 1, .t = g, };
            if (g >= static_cast<float>(size - 1)) return { .i0 = size - 2, .i1 = size - 1, .t = g - static_cast<float>(size - 2), };
            const int i0 = static_cast<int>(floorf(g));
            return {
                .i0 = i0,
                .i1 = i0 + 1,
                .t = g - static_cast<float>(i0),
            };
        }
        const int i0 = static_cast<int>(floorf(g));
        const int i1 = i0 + 1;
        return {
            .i0 = i0,
            .i1 = i1,
            .t = g - static_cast<float>(i0),
        };
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
        const ScalarAxisSample xs = resolve_scalar_axis(gx, nx, extension_mode);
        const ScalarAxisSample ys = resolve_scalar_axis(gy, ny, extension_mode);
        const ScalarAxisSample zs = resolve_scalar_axis(gz, nz, extension_mode);
        const float c000 = load_scalar_sample(field, cell_flags, xs.i0, ys.i0, zs.i0, nx, ny, nz, extension_mode, constant_value);
        const float c100 = load_scalar_sample(field, cell_flags, xs.i1, ys.i0, zs.i0, nx, ny, nz, extension_mode, constant_value);
        const float c010 = load_scalar_sample(field, cell_flags, xs.i0, ys.i1, zs.i0, nx, ny, nz, extension_mode, constant_value);
        const float c110 = load_scalar_sample(field, cell_flags, xs.i1, ys.i1, zs.i0, nx, ny, nz, extension_mode, constant_value);
        const float c001 = load_scalar_sample(field, cell_flags, xs.i0, ys.i0, zs.i1, nx, ny, nz, extension_mode, constant_value);
        const float c101 = load_scalar_sample(field, cell_flags, xs.i1, ys.i0, zs.i1, nx, ny, nz, extension_mode, constant_value);
        const float c011 = load_scalar_sample(field, cell_flags, xs.i0, ys.i1, zs.i1, nx, ny, nz, extension_mode, constant_value);
        const float c111 = load_scalar_sample(field, cell_flags, xs.i1, ys.i1, zs.i1, nx, ny, nz, extension_mode, constant_value);
        const float c00 = c000 + (c100 - c000) * xs.t;
        const float c10 = c010 + (c110 - c010) * xs.t;
        const float c01 = c001 + (c101 - c001) * xs.t;
        const float c11 = c011 + (c111 - c011) * xs.t;
        const float c0 = c00 + (c10 - c00) * ys.t;
        const float c1 = c01 + (c11 - c01) * ys.t;
        return c0 + (c1 - c0) * zs.t;
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

    __device__ float clamp_u_outflow(const float value, const int x, const int nx) {
        if (x == 0) return fminf(value, 0.0f);
        if (x == nx) return fmaxf(value, 0.0f);
        return value;
    }

    __device__ float clamp_v_outflow(const float value, const int y, const int ny) {
        if (y == 0) return fminf(value, 0.0f);
        if (y == ny) return fmaxf(value, 0.0f);
        return value;
    }

    __device__ float clamp_w_outflow(const float value, const int z, const int nz) {
        if (z == 0) return fminf(value, 0.0f);
        if (z == nz) return fmaxf(value, 0.0f);
        return value;
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
            else if (u_flags[index] == face_outflow) u[index] = clamp_u_outflow(u[index], x, nx);
        }
        if (x < nx && y <= ny && z < nz) {
            const auto index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[index] == face_fixed) v[index] = v_target[index];
            else if (v_flags[index] == face_outflow) v[index] = clamp_v_outflow(v[index], y, ny);
        }
        if (x < nx && y < ny && z <= nz) {
            const auto index = index_3d(x, y, z, nx, ny);
            if (w_flags[index] == face_fixed) w[index] = w_target[index];
            else if (w_flags[index] == face_outflow) w[index] = clamp_w_outflow(w[index], z, nz);
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
        const float3 raw_back = make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z);
        float3 back = raw_back;
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT || extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            back = make_float3(clamp_world(back.x, static_cast<float>(nx) * h), clamp_world(back.y, static_cast<float>(ny) * h), clamp_world(back.z, static_cast<float>(nz) * h));
        }
        if (point_inside_domain(back, nx, ny, nz, h)) back = clip_backtrace_to_fluid(cell_flags, pos, back, nx, ny, nz, h);
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

    __global__ void accumulate_projection_metrics_kernel(ProjectionMetricsState* metrics, const float* u, const float* v, const float* w, const uint8_t* cell_flags, const int nx, const int ny, const int nz, const float inv_h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (cell_flags[index] == cell_solid) return;
        const float value = fabsf((u[index_3d(x + 1, y, z, nx + 1, ny)] - u[index_3d(x, y, z, nx + 1, ny)] + v[index_3d(x, y + 1, z, nx, ny + 1)] - v[index_3d(x, y, z, nx, ny + 1)] + w[index_3d(x, y, z + 1, nx, ny)] - w[index_3d(x, y, z, nx, ny)]) * inv_h);
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
        } else if (u_flags[index_3d(x, y, z, nx + 1, ny)] == face_outflow) {
            ++diag;
        }
        if (u_flags[index_3d(x + 1, y, z, nx + 1, ny)] == face_open) {
            ++diag;
            if (x + 1 < nx && cell_flags[index_3d(x + 1, y, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x + 1, y, z, nx, ny)];
        } else if (u_flags[index_3d(x + 1, y, z, nx + 1, ny)] == face_outflow) {
            ++diag;
        }
        if (v_flags[index_3d(x, y, z, nx, ny + 1)] == face_open) {
            ++diag;
            if (y > 0 && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y - 1, z, nx, ny)];
        } else if (v_flags[index_3d(x, y, z, nx, ny + 1)] == face_outflow) {
            ++diag;
        }
        if (v_flags[index_3d(x, y + 1, z, nx, ny + 1)] == face_open) {
            ++diag;
            if (y + 1 < ny && cell_flags[index_3d(x, y + 1, z, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y + 1, z, nx, ny)];
        } else if (v_flags[index_3d(x, y + 1, z, nx, ny + 1)] == face_outflow) {
            ++diag;
        }
        if (w_flags[index_3d(x, y, z, nx, ny)] == face_open) {
            ++diag;
            if (z > 0 && cell_flags[index_3d(x, y, z - 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z - 1, nx, ny)];
        } else if (w_flags[index_3d(x, y, z, nx, ny)] == face_outflow) {
            ++diag;
        }
        if (w_flags[index_3d(x, y, z + 1, nx, ny)] == face_open) {
            ++diag;
            if (z + 1 < nz && cell_flags[index_3d(x, y, z + 1, nx, ny)] != cell_solid) sum += pressure[index_3d(x, y, z + 1, nx, ny)];
        } else if (w_flags[index_3d(x, y, z + 1, nx, ny)] == face_outflow) {
            ++diag;
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
            else if (u_flags[face_index] == face_outflow) {
                if (x == 0 && cell_flags[index_3d(0, y, z, nx, ny)] != cell_solid) u[face_index] -= pressure[index_3d(0, y, z, nx, ny)] * inv_h;
                else if (x == nx && cell_flags[index_3d(nx - 1, y, z, nx, ny)] != cell_solid) u[face_index] += pressure[index_3d(nx - 1, y, z, nx, ny)] * inv_h;
            }
            else if (u_flags[face_index] == face_open && x > 0 && x < nx && cell_flags[index_3d(x - 1, y, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) u[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y <= ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] == face_fixed) v[face_index] = v_target[face_index];
            else if (v_flags[face_index] == face_outflow) {
                if (y == 0 && cell_flags[index_3d(x, 0, z, nx, ny)] != cell_solid) v[face_index] -= pressure[index_3d(x, 0, z, nx, ny)] * inv_h;
                else if (y == ny && cell_flags[index_3d(x, ny - 1, z, nx, ny)] != cell_solid) v[face_index] += pressure[index_3d(x, ny - 1, z, nx, ny)] * inv_h;
            }
            else if (v_flags[face_index] == face_open && y > 0 && y < ny && cell_flags[index_3d(x, y - 1, z, nx, ny)] != cell_solid && cell_flags[index_3d(x, y, z, nx, ny)] != cell_solid) v[face_index] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) * inv_h;
        }

        if (x < nx && y < ny && z <= nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] == face_fixed) w[face_index] = w_target[face_index];
            else if (w_flags[face_index] == face_outflow) {
                if (z == 0 && cell_flags[index_3d(x, y, 0, nx, ny)] != cell_solid) w[face_index] -= pressure[index_3d(x, y, 0, nx, ny)] * inv_h;
                else if (z == nz && cell_flags[index_3d(x, y, nz - 1, nx, ny)] != cell_solid) w[face_index] += pressure[index_3d(x, y, nz - 1, nx, ny)] * inv_h;
            }
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

    StableFluidsResult clear_projection_state(stable_fluids::ContextStorage& context) {
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        if (cudaMemsetAsync(context.device.pressure, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemsetAsync(context.device.divergence, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        return stable_fluids::success;
    }

    StableFluidsResult apply_velocity_constraints(stable_fluids::ContextStorage& context, const stable_fluids::LaunchGeometry& launch) {
        stable_fluids::apply_face_constraints_kernel<<<launch.faces, launch.block, 0, context.stream>>>(
            context.device.velocity_x,
            context.device.velocity_y,
            context.device.velocity_z,
            context.device.u_flags,
            context.device.v_flags,
            context.device.w_flags,
            context.device.u_target,
            context.device.v_target,
            context.device.w_target,
            context.config.nx,
            context.config.ny,
            context.config.nz
        );
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult copy_velocity_fields(
        stable_fluids::ContextStorage& context,
        float* dst_x,
        float* dst_y,
        float* dst_z,
        const float* src_x,
        const float* src_y,
        const float* src_z
    ) {
        const auto u_bytes = static_cast<std::uint64_t>(context.config.nx + 1) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz) * sizeof(float);
        const auto v_bytes = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny + 1) * static_cast<std::uint64_t>(context.config.nz) * sizeof(float);
        const auto w_bytes = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz + 1) * sizeof(float);
        if (cudaMemcpyAsync(dst_x, src_x, u_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemcpyAsync(dst_y, src_y, v_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemcpyAsync(dst_z, src_z, w_bytes, cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        return stable_fluids::success;
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
        const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(context.config);
        if (const StableFluidsResult code = clear_projection_state(context); code != stable_fluids::success) return code;
        stable_fluids::clear_velocity_fields_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (const StableFluidsResult code = rebuild_atlas_if_needed(context); code != stable_fluids::success) return code;
        if (const StableFluidsResult code = apply_velocity_constraints(context, launch); code != stable_fluids::success) return code;
        for (const auto& field : context.fields) {
            stable_fluids::clear_field_kernel<<<launch.cells, launch.block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), field.desc.default_value_0, field.desc.default_value_1, field.desc.default_value_2, field.desc.default_value_3, context.config.nx, context.config.ny, context.config.nz);
            stable_fluids::clear_solid_cells_kernel<<<launch.cells, launch.block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz);
        }
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult diffuse_velocity(stable_fluids::ContextStorage& context, const stable_fluids::LaunchGeometry& launch) {
        const float alpha = context.config.dt * context.config.viscosity / (context.config.cell_size * context.config.cell_size);
        if (alpha <= 0.0f) return stable_fluids::success;

        if (const StableFluidsResult code = copy_velocity_fields(context, context.device.temp_velocity_x, context.device.temp_velocity_y, context.device.temp_velocity_z, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z); code != stable_fluids::success) return code;

        for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_x, context.device.temp_velocity_x, context.device.u_flags, context.device.u_target, context.config.nx + 1, context.config.ny, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_y, context.device.temp_velocity_y, context.device.v_flags, context.device.v_target, context.config.nx, context.config.ny + 1, context.config.nz, alpha, 1);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 0);
            stable_fluids::diffuse_velocity_rbgs_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_z, context.device.temp_velocity_z, context.device.w_flags, context.device.w_target, context.config.nx, context.config.ny, context.config.nz + 1, alpha, 1);
        }

        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult project_velocity(stable_fluids::ContextStorage& context, const stable_fluids::LaunchGeometry& launch) {
        if (const StableFluidsResult code = apply_velocity_constraints(context, launch); code != stable_fluids::success) return code;
        const float inv_h = 1.0f / context.config.cell_size;
        const float h2 = context.config.cell_size * context.config.cell_size;
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        if (cudaMemsetAsync(context.device.pressure, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::compute_divergence_kernel<<<launch.cells, launch.block, 0, context.stream>>>(context.device.divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            stable_fluids::pressure_rbgs_kernel<<<launch.cells, launch.block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 0);
            stable_fluids::pressure_rbgs_kernel<<<launch.cells, launch.block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 1);
        }
        stable_fluids::project_velocity_kernel<<<launch.faces, launch.block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.pressure, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (const StableFluidsResult code = apply_velocity_constraints(context, launch); code != stable_fluids::success) return code;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::accumulate_projection_metrics_kernel<<<launch.cells, launch.block, 0, context.stream>>>(context.device.projection_metrics, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult advect_and_diffuse_scalar(stable_fluids::ContextStorage& context, const stable_fluids::FieldStorage& field, const dim3& block, const dim3& cells) {
        const auto cell_count_value = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
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
    *out_context = nullptr;
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
        if (index < out_field_handle_capacity) out_field_handles[index] = index + 1u;
    }
    if (desc->buoyancy_term_count > 0) context->buoyancy_terms.assign(desc->buoyancy_terms, desc->buoyancy_terms + desc->buoyancy_term_count);
    else context->buoyancy_terms.clear();
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
    auto* storage = static_cast<stable_fluids::ContextStorage*>(context);
    cudaStreamSynchronize(storage->stream);
    stable_fluids::destroy_buffers(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete context;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_reset_context_cuda(StableFluidsContext context) {
    return reset_fields(*static_cast<stable_fluids::ContextStorage*>(context));
}

StableFluidsResult stable_fluids_update_scene_cuda(StableFluidsContext context, const StableFluidsSceneDesc* desc) {
    auto* storage = static_cast<stable_fluids::ContextStorage*>(context);
    if (desc->collider_count > 0) storage->colliders.assign(desc->colliders, desc->colliders + desc->collider_count);
    else storage->colliders.clear();
    storage->atlas_dirty = true;
    return rebuild_atlas_if_needed(*storage);
}

StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    if (const StableFluidsResult code = rebuild_atlas_if_needed(storage); code != stable_fluids::success) return code;

    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);

    nvtx3::scoped_range range("stable.step.context");

    for (uint32_t index = 0; index < desc->velocity_source_count; ++index) {
        const auto& source = desc->velocity_sources[index];
        stable_fluids::add_velocity_source_kernel<<<launch.faces, launch.block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_x, source.velocity_y, source.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    for (uint32_t index = 0; index < desc->field_source_count; ++index) {
        const auto& source = desc->field_sources[index];
        const auto* field = &storage.fields[static_cast<std::size_t>(source.field - 1u)];
        stable_fluids::add_field_source_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(field->data, static_cast<int>(field->desc.component_count), storage.device.cell_flags, source.center_x, source.center_y, source.center_z, source.radius, source.value_0, source.value_1, source.value_2, source.value_3, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    if (const StableFluidsResult code = apply_velocity_constraints(storage, launch); code != stable_fluids::success) return code;

    stable_fluids::add_uniform_forces_kernel<<<launch.faces, launch.block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, storage.config.uniform_force_x, storage.config.uniform_force_y, storage.config.uniform_force_z);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    for (const auto& term : storage.buoyancy_terms) {
        const auto* field = &storage.fields[static_cast<std::size_t>(term.field_index)];
        stable_fluids::add_buoyancy_kernel<<<launch.faces, launch.block, 0, storage.stream>>>(storage.device.velocity_y, field->data, storage.device.v_flags, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, term.weight, term.ambient);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    if (const StableFluidsResult code = diffuse_velocity(storage, launch); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = project_velocity(storage, launch); code != stable_fluids::success) return code;

    stable_fluids::advect_velocity_kernel<<<launch.faces, launch.block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    if (const StableFluidsResult code = copy_velocity_fields(storage, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z); code != stable_fluids::success) return code;

    if (const StableFluidsResult code = project_velocity(storage, launch); code != stable_fluids::success) return code;
    for (const auto& field : storage.fields) {
        if (const StableFluidsResult code = advect_and_diffuse_scalar(storage, field, launch.block, launch.cells); code != stable_fluids::success) return code;
    }

    return stable_fluids::success;
}

StableFluidsResult stable_fluids_export_field_components_cuda(StableFluidsContext context, const StableFluidsFieldHandle field_handle, const uint32_t component_offset, const uint32_t component_count, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);
    const auto* field = &storage.fields[static_cast<std::size_t>(field_handle - 1u)];
    stable_fluids::export_field_components_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(static_cast<float*>(destination), field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(field->desc.component_count), static_cast<int>(component_offset), static_cast<int>(component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_alpha_rgb_rgba_cuda(StableFluidsContext context, const StableFluidsFieldHandle alpha_field_handle, const StableFluidsFieldHandle rgb_field_handle, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const auto* alpha_field = &storage.fields[static_cast<std::size_t>(alpha_field_handle - 1u)];
    const auto* rgb_field = &storage.fields[static_cast<std::size_t>(rgb_field_handle - 1u)];
    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);
    stable_fluids::pack_alpha_rgb_rgba_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(static_cast<float*>(destination), alpha_field->data, rgb_field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(rgb_field->desc.component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);
    stable_fluids::export_velocity_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_magnitude_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);
    stable_fluids::compute_velocity_magnitude_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_solid_mask_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const stable_fluids::LaunchGeometry launch = stable_fluids::make_launch_geometry(storage.config);
    stable_fluids::export_solid_mask_kernel<<<launch.cells, launch.block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_pressure_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const auto cell_count = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    if (cudaMemcpyAsync(destination, storage.device.pressure, cell_count * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_export_divergence_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const auto cell_count = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    if (cudaMemcpyAsync(destination, storage.device.divergence, cell_count * sizeof(float), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_get_projection_metrics_cuda(StableFluidsContext context, StableFluidsProjectionMetrics* out_metrics) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    stable_fluids::ProjectionMetricsState state{};
    if (cudaStreamSynchronize(storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpy(&state, storage.device.projection_metrics, sizeof(state), cudaMemcpyDeviceToHost) != cudaSuccess) return stable_fluids::backend_failure;
    out_metrics->max_abs_divergence = state.max_abs_divergence;
    out_metrics->rms_divergence = state.fluid_cell_count > 0 ? std::sqrt(state.sum_sq_divergence / static_cast<float>(state.fluid_cell_count)) : 0.0f;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc) {
    const auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    out_desc->nx = storage.config.nx;
    out_desc->ny = storage.config.ny;
    out_desc->nz = storage.config.nz;
    out_desc->cell_size = storage.config.cell_size;
    return stable_fluids::success;
}

} // extern "C"
