#include "stable-fluids-3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <new>
#include <vector>

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
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

    struct ProjectionMetricsState {
        float max_abs_divergence = 0.0f;
        float sum_sq_divergence  = 0.0f;
        uint32_t fluid_cell_count = 0;
        uint32_t _padding         = 0;
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

    __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
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

    StableFluidsResult build_boundary_atlas(ContextStorage& context) {
        const int nx = context.config.nx;
        const int ny = context.config.ny;
        const int nz = context.config.nz;
        const float h = context.config.cell_size;
        const auto cell_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        const auto u_count = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        const auto v_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz);
        const auto w_count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1);
        constexpr float huge_distance = 1.0e30f;

        context.host_atlas.cell_flags.resize(static_cast<std::size_t>(cell_count), cell_fluid);
        context.host_atlas.cell_owner.resize(static_cast<std::size_t>(cell_count), -1);
        context.host_atlas.u_flags.resize(static_cast<std::size_t>(u_count), face_open);
        context.host_atlas.v_flags.resize(static_cast<std::size_t>(v_count), face_open);
        context.host_atlas.w_flags.resize(static_cast<std::size_t>(w_count), face_open);
        context.host_atlas.u_target.resize(static_cast<std::size_t>(u_count), 0.0f);
        context.host_atlas.v_target.resize(static_cast<std::size_t>(v_count), 0.0f);
        context.host_atlas.w_target.resize(static_cast<std::size_t>(w_count), 0.0f);
        std::ranges::fill(context.host_atlas.cell_flags, cell_fluid);
        std::ranges::fill(context.host_atlas.cell_owner.begin(), context.host_atlas.cell_owner.end(), -1);
        std::ranges::fill(context.host_atlas.u_flags.begin(), context.host_atlas.u_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.v_flags.begin(), context.host_atlas.v_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.w_flags.begin(), context.host_atlas.w_flags.end(), face_open);
        std::ranges::fill(context.host_atlas.u_target.begin(), context.host_atlas.u_target.end(), 0.0f);
        std::ranges::fill(context.host_atlas.v_target.begin(), context.host_atlas.v_target.end(), 0.0f);
        std::ranges::fill(context.host_atlas.w_target.begin(), context.host_atlas.w_target.end(), 0.0f);

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const auto cell_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny));
                    float best_distance = huge_distance;
                    int owner = -1;
                    for (std::size_t collider_index = 0; collider_index < context.colliders.size(); ++collider_index) {
                        const auto& collider = context.colliders[collider_index];
                        if (!point_inside_collider(collider, px, py, pz)) continue;
                        const float distance = collider_signed_distance(collider, px, py, pz);
                        if (distance > huge_distance || distance >= best_distance) continue;
                        best_distance = distance;
                        owner = static_cast<int>(collider_index);
                    }
                    if (owner < 0) continue;
                    context.host_atlas.cell_flags[cell_index] = cell_solid;
                    context.host_atlas.cell_owner[cell_index] = owner;
                }
            }
        }

        {
            const uint8_t type = context.config.domain_boundary.x_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.x_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.x_min.velocity : 0.0f;
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    const auto face_index = static_cast<std::size_t>(index_3d(0, y, z, nx + 1, ny));
                    context.host_atlas.u_flags[face_index] = type;
                    context.host_atlas.u_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.x_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int z = 0; z < nz; ++z) {
                    for (int y = 0; y <= ny; ++y) {
                        const auto face_index = static_cast<std::size_t>(index_3d(0, y, z, nx, ny + 1));
                        context.host_atlas.v_flags[face_index] = face_fixed;
                        context.host_atlas.v_target[face_index] = 0.0f;
                    }
                    for (int y = 0; y < ny; ++y) {
                        const auto face_index = static_cast<std::size_t>(index_3d(0, y, z, nx, ny));
                        context.host_atlas.w_flags[face_index] = face_fixed;
                        context.host_atlas.w_target[face_index] = 0.0f;
                    }
                }
            }
        }
        {
            const uint8_t type = context.config.domain_boundary.x_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.x_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.x_max.velocity : 0.0f;
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    const auto face_index = static_cast<std::size_t>(index_3d(nx, y, z, nx + 1, ny));
                    context.host_atlas.u_flags[face_index] = type;
                    context.host_atlas.u_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.x_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int z = 0; z < nz; ++z) {
                    for (int y = 0; y <= ny; ++y) {
                        const auto face_index = static_cast<std::size_t>(index_3d(nx - 1, y, z, nx, ny + 1));
                        context.host_atlas.v_flags[face_index] = face_fixed;
                        context.host_atlas.v_target[face_index] = 0.0f;
                    }
                    for (int y = 0; y < ny; ++y) {
                        const auto face_index = static_cast<std::size_t>(index_3d(nx - 1, y, z, nx, ny));
                        context.host_atlas.w_flags[face_index] = face_fixed;
                        context.host_atlas.w_target[face_index] = 0.0f;
                    }
                }
            }
        }
        {
            const uint8_t type = context.config.domain_boundary.y_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.y_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.y_min.velocity : 0.0f;
            for (int z = 0; z < nz; ++z) {
                for (int x = 0; x < nx; ++x) {
                    const auto face_index = static_cast<std::size_t>(index_3d(x, 0, z, nx, ny + 1));
                    context.host_atlas.v_flags[face_index] = type;
                    context.host_atlas.v_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.y_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int z = 0; z < nz; ++z) {
                    for (int x = 0; x <= nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, 0, z, nx + 1, ny));
                        context.host_atlas.u_flags[face_index] = face_fixed;
                        context.host_atlas.u_target[face_index] = 0.0f;
                    }
                    for (int x = 0; x < nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, 0, z, nx, ny));
                        context.host_atlas.w_flags[face_index] = face_fixed;
                        context.host_atlas.w_target[face_index] = 0.0f;
                    }
                }
            }
        }
        {
            const uint8_t type = context.config.domain_boundary.y_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.y_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.y_max.velocity : 0.0f;
            for (int z = 0; z < nz; ++z) {
                for (int x = 0; x < nx; ++x) {
                    const auto face_index = static_cast<std::size_t>(index_3d(x, ny, z, nx, ny + 1));
                    context.host_atlas.v_flags[face_index] = type;
                    context.host_atlas.v_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.y_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int z = 0; z < nz; ++z) {
                    for (int x = 0; x <= nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, ny - 1, z, nx + 1, ny));
                        context.host_atlas.u_flags[face_index] = face_fixed;
                        context.host_atlas.u_target[face_index] = 0.0f;
                    }
                    for (int x = 0; x < nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, ny - 1, z, nx, ny));
                        context.host_atlas.w_flags[face_index] = face_fixed;
                        context.host_atlas.w_target[face_index] = 0.0f;
                    }
                }
            }
        }
        {
            const uint8_t type = context.config.domain_boundary.z_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.z_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.z_min.velocity : 0.0f;
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, 0, nx, ny));
                    context.host_atlas.w_flags[face_index] = type;
                    context.host_atlas.w_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.z_min.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x <= nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, y, 0, nx + 1, ny));
                        context.host_atlas.u_flags[face_index] = face_fixed;
                        context.host_atlas.u_target[face_index] = 0.0f;
                    }
                    for (int x = 0; x < nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, y, 0, nx, ny + 1));
                        context.host_atlas.v_flags[face_index] = face_fixed;
                        context.host_atlas.v_target[face_index] = 0.0f;
                    }
                }
            }
        }
        {
            const uint8_t type = context.config.domain_boundary.z_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW ? face_outflow : face_fixed;
            const float target = context.config.domain_boundary.z_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW ? context.config.domain_boundary.z_max.velocity : 0.0f;
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, nz, nx, ny));
                    context.host_atlas.w_flags[face_index] = type;
                    context.host_atlas.w_target[face_index] = target;
                }
            }
            if (context.config.domain_boundary.z_max.type == STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x <= nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, y, nz - 1, nx + 1, ny));
                        context.host_atlas.u_flags[face_index] = face_fixed;
                        context.host_atlas.u_target[face_index] = 0.0f;
                    }
                    for (int x = 0; x < nx; ++x) {
                        const auto face_index = static_cast<std::size_t>(index_3d(x, y, nz - 1, nx, ny + 1));
                        context.host_atlas.v_flags[face_index] = face_fixed;
                        context.host_atlas.v_target[face_index] = 0.0f;
                    }
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 1; x < nx; ++x) {
                    const int left_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x - 1, y, z, nx, ny))];
                    const int right_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))];
                    if (left_owner < 0 && right_owner < 0) continue;
                    const float px = static_cast<float>(x) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx + 1, ny));
                    if (context.host_atlas.u_flags[face_index] != face_open) continue;
                    int owner = -1;
                    if (left_owner >= 0 && right_owner < 0) owner = left_owner;
                    if (right_owner >= 0 && left_owner < 0) owner = right_owner;
                    if (left_owner >= 0 && right_owner >= 0) {
                        const float left_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(left_owner)], px, py, pz);
                        const float right_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(right_owner)], px, py, pz);
                        owner = left_distance <= right_distance ? left_owner : right_owner;
                    }
                    if (owner < 0) continue;
                    context.host_atlas.u_flags[face_index] = face_fixed;
                    context.host_atlas.u_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_x;
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 1; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int below_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y - 1, z, nx, ny))];
                    const int above_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))];
                    if (below_owner < 0 && above_owner < 0) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = static_cast<float>(y) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny + 1));
                    if (context.host_atlas.v_flags[face_index] != face_open) continue;
                    int owner = -1;
                    if (below_owner >= 0 && above_owner < 0) owner = below_owner;
                    if (above_owner >= 0 && below_owner < 0) owner = above_owner;
                    if (below_owner >= 0 && above_owner >= 0) {
                        const float below_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(below_owner)], px, py, pz);
                        const float above_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(above_owner)], px, py, pz);
                        owner = below_distance <= above_distance ? below_owner : above_owner;
                    }
                    if (owner < 0) continue;
                    context.host_atlas.v_flags[face_index] = face_fixed;
                    context.host_atlas.v_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_y;
                }
            }
        }

        for (int z = 1; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int back_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y, z - 1, nx, ny))];
                    const int front_owner = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(x, y, z, nx, ny))];
                    if (back_owner < 0 && front_owner < 0) continue;
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = static_cast<float>(z) * h;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny));
                    if (context.host_atlas.w_flags[face_index] != face_open) continue;
                    int owner = -1;
                    if (back_owner >= 0 && front_owner < 0) owner = back_owner;
                    if (front_owner >= 0 && back_owner < 0) owner = front_owner;
                    if (back_owner >= 0 && front_owner >= 0) {
                        const float back_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(back_owner)], px, py, pz);
                        const float front_distance = collider_signed_distance(context.colliders[static_cast<std::size_t>(front_owner)], px, py, pz);
                        owner = back_distance <= front_distance ? back_owner : front_owner;
                    }
                    if (owner < 0) continue;
                    context.host_atlas.w_flags[face_index] = face_fixed;
                    context.host_atlas.w_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_z;
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x <= nx; ++x) {
                    const float px = static_cast<float>(x) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    int owner = -1;
                    float best_distance = huge_distance;
                    for (int cell_z = z - 1; cell_z <= z; ++cell_z) {
                        for (int cell_y = y - 1; cell_y <= y; ++cell_y) {
                            for (int cell_x = x - 1; cell_x <= x; ++cell_x) {
                                if (cell_x < 0 || cell_y < 0 || cell_z < 0 || cell_x >= nx || cell_y >= ny || cell_z >= nz) continue;
                                const int candidate = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(cell_x, cell_y, cell_z, nx, ny))];
                                if (candidate < 0) continue;
                                const auto& collider = context.colliders[static_cast<std::size_t>(candidate)];
                                if (collider.velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) continue;
                                const float distance = collider_signed_distance(collider, px, py, pz);
                                if (distance >= best_distance) continue;
                                best_distance = distance;
                                owner = candidate;
                            }
                        }
                    }
                    if (owner < 0) continue;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx + 1, ny));
                    if (context.host_atlas.u_flags[face_index] != face_open) continue;
                    context.host_atlas.u_flags[face_index] = face_fixed;
                    context.host_atlas.u_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_x;
                }
            }
        }

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y <= ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = static_cast<float>(y) * h;
                    const float pz = (static_cast<float>(z) + 0.5f) * h;
                    int owner = -1;
                    float best_distance = huge_distance;
                    for (int cell_z = z - 1; cell_z <= z; ++cell_z) {
                        for (int cell_y = y - 1; cell_y <= y; ++cell_y) {
                            for (int cell_x = x - 1; cell_x <= x; ++cell_x) {
                                if (cell_x < 0 || cell_y < 0 || cell_z < 0 || cell_x >= nx || cell_y >= ny || cell_z >= nz) continue;
                                const int candidate = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(cell_x, cell_y, cell_z, nx, ny))];
                                if (candidate < 0) continue;
                                const auto& collider = context.colliders[static_cast<std::size_t>(candidate)];
                                if (collider.velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) continue;
                                const float distance = collider_signed_distance(collider, px, py, pz);
                                if (distance >= best_distance) continue;
                                best_distance = distance;
                                owner = candidate;
                            }
                        }
                    }
                    if (owner < 0) continue;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny + 1));
                    if (context.host_atlas.v_flags[face_index] != face_open) continue;
                    context.host_atlas.v_flags[face_index] = face_fixed;
                    context.host_atlas.v_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_y;
                }
            }
        }

        for (int z = 0; z <= nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float px = (static_cast<float>(x) + 0.5f) * h;
                    const float py = (static_cast<float>(y) + 0.5f) * h;
                    const float pz = static_cast<float>(z) * h;
                    int owner = -1;
                    float best_distance = huge_distance;
                    for (int cell_z = z - 1; cell_z <= z; ++cell_z) {
                        for (int cell_y = y - 1; cell_y <= y; ++cell_y) {
                            for (int cell_x = x - 1; cell_x <= x; ++cell_x) {
                                if (cell_x < 0 || cell_y < 0 || cell_z < 0 || cell_x >= nx || cell_y >= ny || cell_z >= nz) continue;
                                const int candidate = context.host_atlas.cell_owner[static_cast<std::size_t>(index_3d(cell_x, cell_y, cell_z, nx, ny))];
                                if (candidate < 0) continue;
                                const auto& collider = context.colliders[static_cast<std::size_t>(candidate)];
                                if (collider.velocity_boundary_type != STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP) continue;
                                const float distance = collider_signed_distance(collider, px, py, pz);
                                if (distance >= best_distance) continue;
                                best_distance = distance;
                                owner = candidate;
                            }
                        }
                    }
                    if (owner < 0) continue;
                    const auto face_index = static_cast<std::size_t>(index_3d(x, y, z, nx, ny));
                    if (context.host_atlas.w_flags[face_index] != face_open) continue;
                    context.host_atlas.w_flags[face_index] = face_fixed;
                    context.host_atlas.w_target[face_index] = context.colliders[static_cast<std::size_t>(owner)].linear_velocity_z;
                }
            }
        }
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
        context.atlas_dirty = false;
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

    __device__ float sample_scalar_field(const float* field, const uint8_t* cell_flags, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const uint32_t extension_mode, const float constant_value) {
        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        int xs[2]{};
        int ys[2]{};
        int zs[2]{};
        float tx = 0.0f;
        float ty = 0.0f;
        float tz = 0.0f;

        if (nx <= 1) {
            xs[0] = 0;
            xs[1] = 0;
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
            const int i0 = static_cast<int>(floorf(gx));
            const int i1 = i0 + 1;
            xs[0] = i0 % nx;
            xs[1] = i1 % nx;
            if (xs[0] < 0) xs[0] += nx;
            if (xs[1] < 0) xs[1] += nx;
            tx = gx - static_cast<float>(i0);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            const float clamped = cuda::std::clamp(gx, 0.0f, static_cast<float>(nx - 1));
            xs[0] = static_cast<int>(floorf(clamped));
            xs[1] = cuda::std::min(xs[0] + 1, nx - 1);
            tx = clamped - static_cast<float>(xs[0]);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE) {
            if (gx <= 0.0f) {
                xs[0] = 0;
                xs[1] = 1;
                tx = gx;
            } else if (gx >= static_cast<float>(nx - 1)) {
                xs[0] = nx - 2;
                xs[1] = nx - 1;
                tx = gx - static_cast<float>(nx - 2);
            } else {
                xs[0] = static_cast<int>(floorf(gx));
                xs[1] = xs[0] + 1;
                tx = gx - static_cast<float>(xs[0]);
            }
        } else {
            xs[0] = static_cast<int>(floorf(gx));
            xs[1] = xs[0] + 1;
            tx = gx - static_cast<float>(xs[0]);
        }

        if (ny <= 1) {
            ys[0] = 0;
            ys[1] = 0;
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
            const int i0 = static_cast<int>(floorf(gy));
            const int i1 = i0 + 1;
            ys[0] = i0 % ny;
            ys[1] = i1 % ny;
            if (ys[0] < 0) ys[0] += ny;
            if (ys[1] < 0) ys[1] += ny;
            ty = gy - static_cast<float>(i0);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            const float clamped = cuda::std::clamp(gy, 0.0f, static_cast<float>(ny - 1));
            ys[0] = static_cast<int>(floorf(clamped));
            ys[1] = cuda::std::min(ys[0] + 1, ny - 1);
            ty = clamped - static_cast<float>(ys[0]);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE) {
            if (gy <= 0.0f) {
                ys[0] = 0;
                ys[1] = 1;
                ty = gy;
            } else if (gy >= static_cast<float>(ny - 1)) {
                ys[0] = ny - 2;
                ys[1] = ny - 1;
                ty = gy - static_cast<float>(ny - 2);
            } else {
                ys[0] = static_cast<int>(floorf(gy));
                ys[1] = ys[0] + 1;
                ty = gy - static_cast<float>(ys[0]);
            }
        } else {
            ys[0] = static_cast<int>(floorf(gy));
            ys[1] = ys[0] + 1;
            ty = gy - static_cast<float>(ys[0]);
        }

        if (nz <= 1) {
            zs[0] = 0;
            zs[1] = 0;
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
            const int i0 = static_cast<int>(floorf(gz));
            const int i1 = i0 + 1;
            zs[0] = i0 % nz;
            zs[1] = i1 % nz;
            if (zs[0] < 0) zs[0] += nz;
            if (zs[1] < 0) zs[1] += nz;
            tz = gz - static_cast<float>(i0);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            const float clamped = cuda::std::clamp(gz, 0.0f, static_cast<float>(nz - 1));
            zs[0] = static_cast<int>(floorf(clamped));
            zs[1] = cuda::std::min(zs[0] + 1, nz - 1);
            tz = clamped - static_cast<float>(zs[0]);
        } else if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE) {
            if (gz <= 0.0f) {
                zs[0] = 0;
                zs[1] = 1;
                tz = gz;
            } else if (gz >= static_cast<float>(nz - 1)) {
                zs[0] = nz - 2;
                zs[1] = nz - 1;
                tz = gz - static_cast<float>(nz - 2);
            } else {
                zs[0] = static_cast<int>(floorf(gz));
                zs[1] = zs[0] + 1;
                tz = gz - static_cast<float>(zs[0]);
            }
        } else {
            zs[0] = static_cast<int>(floorf(gz));
            zs[1] = zs[0] + 1;
            tz = gz - static_cast<float>(zs[0]);
        }

        float c[2][2][2]{};
        for (int ax = 0; ax < 2; ++ax) {
            for (int ay = 0; ay < 2; ++ay) {
                for (int az = 0; az < 2; ++az) {
                    int ix = xs[ax];
                    int iy = ys[ay];
                    int iz = zs[az];
                    if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
                        ix %= nx;
                        iy %= ny;
                        iz %= nz;
                        if (ix < 0) ix += nx;
                        if (iy < 0) iy += ny;
                        if (iz < 0) iz += nz;
                    } else if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT) {
                            c[ax][ay][az] = constant_value;
                            continue;
                        }
                        ix = cuda::std::clamp(ix, 0, nx - 1);
                        iy = cuda::std::clamp(iy, 0, ny - 1);
                        iz = cuda::std::clamp(iz, 0, nz - 1);
                    }
                    const auto sample_index = index_3d(ix, iy, iz, nx, ny);
                    c[ax][ay][az] = cell_flags[sample_index] == cell_solid ? constant_value : field[sample_index];
                }
            }
        }

        const float c00 = c[0][0][0] + (c[1][0][0] - c[0][0][0]) * tx;
        const float c10 = c[0][1][0] + (c[1][1][0] - c[0][1][0]) * tx;
        const float c01 = c[0][0][1] + (c[1][0][1] - c[0][0][1]) * tx;
        const float c11 = c[0][1][1] + (c[1][1][1] - c[0][1][1]) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_u_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = cuda::std::clamp(x / h, 0.0f, static_cast<float>(nx));
        const float gy = cuda::std::clamp(y / h - 0.5f, 0.0f, static_cast<float>(ny - 1));
        const float gz = cuda::std::clamp(z / h - 0.5f, 0.0f, static_cast<float>(nz - 1));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = cuda::std::min(x0 + 1, nx);
        const int y1 = cuda::std::min(y0 + 1, ny - 1);
        const int z1 = cuda::std::min(z0 + 1, nz - 1);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        float c[2][2][2]{};
        for (int ax = 0; ax < 2; ++ax) {
            for (int ay = 0; ay < 2; ++ay) {
                for (int az = 0; az < 2; ++az) {
                    const int ix = ax == 0 ? x0 : x1;
                    const int iy = ay == 0 ? y0 : y1;
                    const int iz = az == 0 ? z0 : z1;
                    const auto sample_index = index_3d(ix, iy, iz, nx + 1, ny);
                    c[ax][ay][az] = flags[sample_index] == face_fixed ? target[sample_index] : field[sample_index];
                }
            }
        }
        const float c00 = c[0][0][0] + (c[1][0][0] - c[0][0][0]) * tx;
        const float c10 = c[0][1][0] + (c[1][1][0] - c[0][1][0]) * tx;
        const float c01 = c[0][0][1] + (c[1][0][1] - c[0][0][1]) * tx;
        const float c11 = c[0][1][1] + (c[1][1][1] - c[0][1][1]) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_v_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = cuda::std::clamp(x / h - 0.5f, 0.0f, static_cast<float>(nx - 1));
        const float gy = cuda::std::clamp(y / h, 0.0f, static_cast<float>(ny));
        const float gz = cuda::std::clamp(z / h - 0.5f, 0.0f, static_cast<float>(nz - 1));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = cuda::std::min(x0 + 1, nx - 1);
        const int y1 = cuda::std::min(y0 + 1, ny);
        const int z1 = cuda::std::min(z0 + 1, nz - 1);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        float c[2][2][2]{};
        for (int ax = 0; ax < 2; ++ax) {
            for (int ay = 0; ay < 2; ++ay) {
                for (int az = 0; az < 2; ++az) {
                    const int ix = ax == 0 ? x0 : x1;
                    const int iy = ay == 0 ? y0 : y1;
                    const int iz = az == 0 ? z0 : z1;
                    const auto sample_index = index_3d(ix, iy, iz, nx, ny + 1);
                    c[ax][ay][az] = flags[sample_index] == face_fixed ? target[sample_index] : field[sample_index];
                }
            }
        }
        const float c00 = c[0][0][0] + (c[1][0][0] - c[0][0][0]) * tx;
        const float c10 = c[0][1][0] + (c[1][1][0] - c[0][1][0]) * tx;
        const float c01 = c[0][0][1] + (c[1][0][1] - c[0][0][1]) * tx;
        const float c11 = c[0][1][1] + (c[1][1][1] - c[0][1][1]) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_w_field(const float* field, const uint8_t* flags, const float* target, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
        const float gx = cuda::std::clamp(x / h - 0.5f, 0.0f, static_cast<float>(nx - 1));
        const float gy = cuda::std::clamp(y / h - 0.5f, 0.0f, static_cast<float>(ny - 1));
        const float gz = cuda::std::clamp(z / h, 0.0f, static_cast<float>(nz));
        const int x0 = static_cast<int>(floorf(gx));
        const int y0 = static_cast<int>(floorf(gy));
        const int z0 = static_cast<int>(floorf(gz));
        const int x1 = cuda::std::min(x0 + 1, nx - 1);
        const int y1 = cuda::std::min(y0 + 1, ny - 1);
        const int z1 = cuda::std::min(z0 + 1, nz);
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);
        float c[2][2][2]{};
        for (int ax = 0; ax < 2; ++ax) {
            for (int ay = 0; ay < 2; ++ay) {
                for (int az = 0; az < 2; ++az) {
                    const int ix = ax == 0 ? x0 : x1;
                    const int iy = ay == 0 ? y0 : y1;
                    const int iz = az == 0 ? z0 : z1;
                    const auto sample_index = index_3d(ix, iy, iz, nx, ny);
                    c[ax][ay][az] = flags[sample_index] == face_fixed ? target[sample_index] : field[sample_index];
                }
            }
        }
        const float c00 = c[0][0][0] + (c[1][0][0] - c[0][0][0]) * tx;
        const float c10 = c[0][1][0] + (c[1][1][0] - c[0][1][0]) * tx;
        const float c01 = c[0][0][1] + (c[1][0][1] - c[0][0][1]) * tx;
        const float c11 = c[0][1][1] + (c[1][1][1] - c[0][1][1]) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
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
            else if (u_flags[index] == face_outflow) {
                if (x == 0) u[index] = cuda::std::min(u[index], 0.0f);
                else if (x == nx) u[index] = cuda::std::max(u[index], 0.0f);
            }
        }
        if (x < nx && y <= ny && z < nz) {
            const auto index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[index] == face_fixed) v[index] = v_target[index];
            else if (v_flags[index] == face_outflow) {
                if (y == 0) v[index] = cuda::std::min(v[index], 0.0f);
                else if (y == ny) v[index] = cuda::std::max(v[index], 0.0f);
            }
        }
        if (x < nx && y < ny && z <= nz) {
            const auto index = index_3d(x, y, z, nx, ny);
            if (w_flags[index] == face_fixed) w[index] = w_target[index];
            else if (w_flags[index] == face_outflow) {
                if (z == 0) w[index] = cuda::std::min(w[index], 0.0f);
                else if (z == nz) w[index] = cuda::std::max(w[index], 0.0f);
            }
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
                const float weight = cuda::std::clamp(1.0f - distance2 / radius2, 0.0f, 1.0f);
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
                const float weight = cuda::std::clamp(1.0f - distance2 / radius2, 0.0f, 1.0f);
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
                const float weight = cuda::std::clamp(1.0f - distance2 / radius2, 0.0f, 1.0f);
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
        const float weight = cuda::std::clamp(1.0f - distance2 / radius2, 0.0f, 1.0f);
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
                const float3 vel = make_float3(
                    sample_u_field(u_src, u_flags, u_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_v_field(v_src, v_flags, v_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_w_field(w_src, w_flags, w_target, pos.x, pos.y, pos.z, nx, ny, nz, h)
                );
                float3 back = make_float3(
                    cuda::std::clamp(pos.x - dt * vel.x, 0.0f, static_cast<float>(nx) * h),
                    cuda::std::clamp(pos.y - dt * vel.y, 0.0f, static_cast<float>(ny) * h),
                    cuda::std::clamp(pos.z - dt * vel.z, 0.0f, static_cast<float>(nz) * h)
                );
                const int back_ix = cuda::std::clamp(static_cast<int>(floorf(back.x / h)), 0, nx - 1);
                const int back_iy = cuda::std::clamp(static_cast<int>(floorf(back.y / h)), 0, ny - 1);
                const int back_iz = cuda::std::clamp(static_cast<int>(floorf(back.z / h)), 0, nz - 1);
                if (cell_flags[index_3d(back_ix, back_iy, back_iz, nx, ny)] == cell_solid) {
                    float3 lo = pos;
                    float3 hi = back;
                    for (int iteration = 0; iteration < 8; ++iteration) {
                        const float3 mid = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
                        const int mid_ix = cuda::std::clamp(static_cast<int>(floorf(mid.x / h)), 0, nx - 1);
                        const int mid_iy = cuda::std::clamp(static_cast<int>(floorf(mid.y / h)), 0, ny - 1);
                        const int mid_iz = cuda::std::clamp(static_cast<int>(floorf(mid.z / h)), 0, nz - 1);
                        if (cell_flags[index_3d(mid_ix, mid_iy, mid_iz, nx, ny)] == cell_solid) hi = mid;
                        else lo = mid;
                    }
                    back = lo;
                }
                u_dst[face_index] = sample_u_field(u_src, u_flags, u_target, back.x, back.y, back.z, nx, ny, nz, h);
            }
        }

        if (x < nx && y <= ny && z < nz) {
            const auto face_index = index_3d(x, y, z, nx, ny + 1);
            if (v_flags[face_index] == face_fixed) v_dst[face_index] = v_target[face_index];
            else {
                const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, static_cast<float>(y) * h, (static_cast<float>(z) + 0.5f) * h);
                const float3 vel = make_float3(
                    sample_u_field(u_src, u_flags, u_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_v_field(v_src, v_flags, v_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_w_field(w_src, w_flags, w_target, pos.x, pos.y, pos.z, nx, ny, nz, h)
                );
                float3 back = make_float3(
                    cuda::std::clamp(pos.x - dt * vel.x, 0.0f, static_cast<float>(nx) * h),
                    cuda::std::clamp(pos.y - dt * vel.y, 0.0f, static_cast<float>(ny) * h),
                    cuda::std::clamp(pos.z - dt * vel.z, 0.0f, static_cast<float>(nz) * h)
                );
                const int back_ix = cuda::std::clamp(static_cast<int>(floorf(back.x / h)), 0, nx - 1);
                const int back_iy = cuda::std::clamp(static_cast<int>(floorf(back.y / h)), 0, ny - 1);
                const int back_iz = cuda::std::clamp(static_cast<int>(floorf(back.z / h)), 0, nz - 1);
                if (cell_flags[index_3d(back_ix, back_iy, back_iz, nx, ny)] == cell_solid) {
                    float3 lo = pos;
                    float3 hi = back;
                    for (int iteration = 0; iteration < 8; ++iteration) {
                        const float3 mid = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
                        const int mid_ix = cuda::std::clamp(static_cast<int>(floorf(mid.x / h)), 0, nx - 1);
                        const int mid_iy = cuda::std::clamp(static_cast<int>(floorf(mid.y / h)), 0, ny - 1);
                        const int mid_iz = cuda::std::clamp(static_cast<int>(floorf(mid.z / h)), 0, nz - 1);
                        if (cell_flags[index_3d(mid_ix, mid_iy, mid_iz, nx, ny)] == cell_solid) hi = mid;
                        else lo = mid;
                    }
                    back = lo;
                }
                v_dst[face_index] = sample_v_field(v_src, v_flags, v_target, back.x, back.y, back.z, nx, ny, nz, h);
            }
        }

        if (x < nx && y < ny && z <= nz) {
            const auto face_index = index_3d(x, y, z, nx, ny);
            if (w_flags[face_index] == face_fixed) w_dst[face_index] = w_target[face_index];
            else {
                const float3 pos = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, static_cast<float>(z) * h);
                const float3 vel = make_float3(
                    sample_u_field(u_src, u_flags, u_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_v_field(v_src, v_flags, v_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
                    sample_w_field(w_src, w_flags, w_target, pos.x, pos.y, pos.z, nx, ny, nz, h)
                );
                float3 back = make_float3(
                    cuda::std::clamp(pos.x - dt * vel.x, 0.0f, static_cast<float>(nx) * h),
                    cuda::std::clamp(pos.y - dt * vel.y, 0.0f, static_cast<float>(ny) * h),
                    cuda::std::clamp(pos.z - dt * vel.z, 0.0f, static_cast<float>(nz) * h)
                );
                const int back_ix = cuda::std::clamp(static_cast<int>(floorf(back.x / h)), 0, nx - 1);
                const int back_iy = cuda::std::clamp(static_cast<int>(floorf(back.y / h)), 0, ny - 1);
                const int back_iz = cuda::std::clamp(static_cast<int>(floorf(back.z / h)), 0, nz - 1);
                if (cell_flags[index_3d(back_ix, back_iy, back_iz, nx, ny)] == cell_solid) {
                    float3 lo = pos;
                    float3 hi = back;
                    for (int iteration = 0; iteration < 8; ++iteration) {
                        const float3 mid = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
                        const int mid_ix = cuda::std::clamp(static_cast<int>(floorf(mid.x / h)), 0, nx - 1);
                        const int mid_iy = cuda::std::clamp(static_cast<int>(floorf(mid.y / h)), 0, ny - 1);
                        const int mid_iz = cuda::std::clamp(static_cast<int>(floorf(mid.z / h)), 0, nz - 1);
                        if (cell_flags[index_3d(mid_ix, mid_iy, mid_iz, nx, ny)] == cell_solid) hi = mid;
                        else lo = mid;
                    }
                    back = lo;
                }
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
        const float3 vel = make_float3(
            sample_u_field(u, u_flags, u_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
            sample_v_field(v, v_flags, v_target, pos.x, pos.y, pos.z, nx, ny, nz, h),
            sample_w_field(w, w_flags, w_target, pos.x, pos.y, pos.z, nx, ny, nz, h)
        );
        const float3 raw_back = make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z);
        float3 back = raw_back;
        if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT || extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_STREAK) {
            back = make_float3(
                cuda::std::clamp(back.x, 0.0f, static_cast<float>(nx) * h),
                cuda::std::clamp(back.y, 0.0f, static_cast<float>(ny) * h),
                cuda::std::clamp(back.z, 0.0f, static_cast<float>(nz) * h)
            );
        }
        if (back.x >= 0.0f && back.x <= static_cast<float>(nx) * h && back.y >= 0.0f && back.y <= static_cast<float>(ny) * h && back.z >= 0.0f && back.z <= static_cast<float>(nz) * h) {
            const int back_ix = cuda::std::clamp(static_cast<int>(floorf(back.x / h)), 0, nx - 1);
            const int back_iy = cuda::std::clamp(static_cast<int>(floorf(back.y / h)), 0, ny - 1);
            const int back_iz = cuda::std::clamp(static_cast<int>(floorf(back.z / h)), 0, nz - 1);
            if (cell_flags[index_3d(back_ix, back_iy, back_iz, nx, ny)] == cell_solid) {
                float3 lo = pos;
                float3 hi = back;
                for (int iteration = 0; iteration < 8; ++iteration) {
                    const float3 mid = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
                    const int mid_ix = cuda::std::clamp(static_cast<int>(floorf(mid.x / h)), 0, nx - 1);
                    const int mid_iy = cuda::std::clamp(static_cast<int>(floorf(mid.y / h)), 0, ny - 1);
                    const int mid_iz = cuda::std::clamp(static_cast<int>(floorf(mid.z / h)), 0, nz - 1);
                    if (cell_flags[index_3d(mid_ix, mid_iy, mid_iz, nx, ny)] == cell_solid) hi = mid;
                    else lo = mid;
                }
                back = lo;
            }
        }
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
        float sum = 0.0f;
        const int sample_x[6] = { x - 1, x + 1, x, x, x, x, };
        const int sample_y[6] = { y, y, y - 1, y + 1, y, y, };
        const int sample_z[6] = { z, z, z, z, z - 1, z + 1, };
        for (int neighbor = 0; neighbor < 6; ++neighbor) {
            int ix = sample_x[neighbor];
            int iy = sample_y[neighbor];
            int iz = sample_z[neighbor];
            if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_REPEAT) {
                ix %= nx;
                iy %= ny;
                iz %= nz;
                if (ix < 0) ix += nx;
                if (iy < 0) iy += ny;
                if (iz < 0) iz += nz;
            } else if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                if (extension_mode == STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT) {
                    sum += constant_value;
                    continue;
                }
                ix = cuda::std::clamp(ix, 0, nx - 1);
                iy = cuda::std::clamp(iy, 0, ny - 1);
                iz = cuda::std::clamp(iz, 0, nz - 1);
            }
            const auto sample_index = index_3d(ix, iy, iz, nx, ny);
            if (cell_flags[sample_index] == cell_solid) sum += constant_value;
            else sum += dst[sample_index];
        }
        dst[index] = (src[index] + alpha * sum) / (1.0f + 6.0f * alpha);
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

    StableFluidsResult apply_velocity_constraints(stable_fluids::ContextStorage& context, const dim3& block, const dim3& faces) {
        stable_fluids::apply_face_constraints_kernel<<<faces, block, 0, context.stream>>>(
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

    StableFluidsResult reset_fields(stable_fluids::ContextStorage& context) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 cells(
            static_cast<unsigned>((context.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
        const dim3 faces(
            static_cast<unsigned>((context.config.nx + 1 + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + 1 + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + 1 + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        if (cudaMemsetAsync(context.device.pressure, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemsetAsync(context.device.divergence, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::clear_velocity_fields_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (context.atlas_dirty) {
            if (const StableFluidsResult code = stable_fluids::build_boundary_atlas(context); code != stable_fluids::success) return code;
        }
        if (const StableFluidsResult code = apply_velocity_constraints(context, block, faces); code != stable_fluids::success) return code;
        for (const auto& field : context.fields) {
            stable_fluids::clear_field_kernel<<<cells, block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), field.desc.default_value_0, field.desc.default_value_1, field.desc.default_value_2, field.desc.default_value_3, context.config.nx, context.config.ny, context.config.nz);
            stable_fluids::clear_solid_cells_kernel<<<cells, block, 0, context.stream>>>(field.data, static_cast<int>(field.desc.component_count), context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz);
        }
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult diffuse_velocity(stable_fluids::ContextStorage& context) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 faces(
            static_cast<unsigned>((context.config.nx + 1 + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + 1 + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + 1 + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
        const float alpha = context.config.dt * context.config.viscosity / (context.config.cell_size * context.config.cell_size);
        if (alpha <= 0.0f) return stable_fluids::success;
        const auto u_bytes = static_cast<std::uint64_t>(context.config.nx + 1) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz) * sizeof(float);
        const auto v_bytes = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny + 1) * static_cast<std::uint64_t>(context.config.nz) * sizeof(float);
        const auto w_bytes = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz + 1) * sizeof(float);
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

    StableFluidsResult project_velocity(stable_fluids::ContextStorage& context) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 cells(
            static_cast<unsigned>((context.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
        const dim3 faces(
            static_cast<unsigned>((context.config.nx + 1 + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + 1 + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + 1 + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
        if (const StableFluidsResult code = apply_velocity_constraints(context, block, faces); code != stable_fluids::success) return code;
        const float inv_h = 1.0f / context.config.cell_size;
        const float h2 = context.config.cell_size * context.config.cell_size;
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        if (cudaMemsetAsync(context.device.pressure, 0, cell_count * sizeof(float), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::compute_divergence_kernel<<<cells, block, 0, context.stream>>>(context.device.divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 0);
            stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.config.nx, context.config.ny, context.config.nz, h2, 1);
        }
        stable_fluids::project_velocity_kernel<<<faces, block, 0, context.stream>>>(context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.pressure, context.device.cell_flags, context.device.u_flags, context.device.v_flags, context.device.w_flags, context.device.u_target, context.device.v_target, context.device.w_target, context.config.nx, context.config.ny, context.config.nz, inv_h);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        if (const StableFluidsResult code = apply_velocity_constraints(context, block, faces); code != stable_fluids::success) return code;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context.stream) != cudaSuccess) return stable_fluids::backend_failure;
        stable_fluids::accumulate_projection_metrics_kernel<<<cells, block, 0, context.stream>>>(context.device.projection_metrics, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.cell_flags, context.config.nx, context.config.ny, context.config.nz, inv_h);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }

    StableFluidsResult advect_and_diffuse_scalar(stable_fluids::ContextStorage& context, const stable_fluids::FieldStorage& field) {
        const dim3 block(
            static_cast<unsigned>(std::max(context.config.block_x, 1)),
            static_cast<unsigned>(std::max(context.config.block_y, 1)),
            static_cast<unsigned>(std::max(context.config.block_z, 1))
        );
        const dim3 cells(
            static_cast<unsigned>((context.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
        );
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
    return stable_fluids::build_boundary_atlas(*storage);
}

StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    if (storage.atlas_dirty) {
        if (const StableFluidsResult code = stable_fluids::build_boundary_atlas(storage); code != stable_fluids::success) return code;
    }
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    const dim3 faces(
        static_cast<unsigned>((storage.config.nx + 1 + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + 1 + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + 1 + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    const auto u_bytes = static_cast<std::uint64_t>(storage.config.nx + 1) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz) * sizeof(float);
    const auto v_bytes = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny + 1) * static_cast<std::uint64_t>(storage.config.nz) * sizeof(float);
    const auto w_bytes = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz + 1) * sizeof(float);

    nvtx3::scoped_range range("stable.step.context");

    for (uint32_t index = 0; index < desc->velocity_source_count; ++index) {
        const auto& source = desc->velocity_sources[index];
        stable_fluids::add_velocity_source_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_x, source.velocity_y, source.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    for (uint32_t index = 0; index < desc->field_source_count; ++index) {
        const auto& source = desc->field_sources[index];
        const auto* field = &storage.fields[static_cast<std::size_t>(source.field - 1u)];
        stable_fluids::add_field_source_kernel<<<cells, block, 0, storage.stream>>>(field->data, static_cast<int>(field->desc.component_count), storage.device.cell_flags, source.center_x, source.center_y, source.center_z, source.radius, source.value_0, source.value_1, source.value_2, source.value_3, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    if (const StableFluidsResult code = apply_velocity_constraints(storage, block, faces); code != stable_fluids::success) return code;

    stable_fluids::add_uniform_forces_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, storage.config.uniform_force_x, storage.config.uniform_force_y, storage.config.uniform_force_z);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    for (const auto& term : storage.buoyancy_terms) {
        const auto* field = &storage.fields[static_cast<std::size_t>(term.field_index)];
        stable_fluids::add_buoyancy_kernel<<<faces, block, 0, storage.stream>>>(storage.device.velocity_y, field->data, storage.device.v_flags, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt, term.weight, term.ambient);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }

    if (const StableFluidsResult code = diffuse_velocity(storage); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = project_velocity(storage); code != stable_fluids::success) return code;

    stable_fluids::advect_velocity_kernel<<<faces, block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.u_flags, storage.device.v_flags, storage.device.w_flags, storage.device.u_target, storage.device.v_target, storage.device.w_target, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpyAsync(storage.device.velocity_x, storage.device.temp_velocity_x, u_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpyAsync(storage.device.velocity_y, storage.device.temp_velocity_y, v_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemcpyAsync(storage.device.velocity_z, storage.device.temp_velocity_z, w_bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;

    if (const StableFluidsResult code = project_velocity(storage); code != stable_fluids::success) return code;
    for (const auto& field : storage.fields) {
        if (const StableFluidsResult code = advect_and_diffuse_scalar(storage, field); code != stable_fluids::success) return code;
    }

    return stable_fluids::success;
}

StableFluidsResult stable_fluids_export_field_components_cuda(StableFluidsContext context, const StableFluidsFieldHandle field_handle, const uint32_t component_offset, const uint32_t component_count, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    const auto* field = &storage.fields[static_cast<std::size_t>(field_handle - 1u)];
    stable_fluids::export_field_components_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(field->desc.component_count), static_cast<int>(component_offset), static_cast<int>(component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_alpha_rgb_rgba_cuda(StableFluidsContext context, const StableFluidsFieldHandle alpha_field_handle, const StableFluidsFieldHandle rgb_field_handle, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const auto* alpha_field = &storage.fields[static_cast<std::size_t>(alpha_field_handle - 1u)];
    const auto* rgb_field = &storage.fields[static_cast<std::size_t>(rgb_field_handle - 1u)];
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    stable_fluids::pack_alpha_rgb_rgba_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), alpha_field->data, rgb_field->data, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz, static_cast<int>(rgb_field->desc.component_count));
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    stable_fluids::export_velocity_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_velocity_magnitude_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    stable_fluids::compute_velocity_magnitude_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_solid_mask_cuda(StableFluidsContext context, void* destination) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(
        static_cast<unsigned>(std::max(storage.config.block_x, 1)),
        static_cast<unsigned>(std::max(storage.config.block_y, 1)),
        static_cast<unsigned>(std::max(storage.config.block_z, 1))
    );
    const dim3 cells(
        static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
        static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
        static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z))
    );
    stable_fluids::export_solid_mask_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.cell_flags, storage.config.nx, storage.config.ny, storage.config.nz);
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
