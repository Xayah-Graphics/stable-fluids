#include "stable-fluids-3d.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>
#include <array>

namespace {
    bool cuda_ok(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return false;
    }

    bool stable_ok(const StableFluidsResult code, const char* what) {
        if (code == STABLE_FLUIDS_RESULT_OK) return true;
        std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(code));
        return false;
    }

} // namespace

int main() {
    constexpr int32_t nx = 100;
    constexpr int32_t ny = 100;
    constexpr int32_t nz = 100;
    constexpr int frames = 24;
    constexpr float cell_size = 0.01f;
    constexpr float extent_x = static_cast<float>(nx) * cell_size;
    constexpr float extent_y = static_cast<float>(ny) * cell_size;
    constexpr float extent_z = static_cast<float>(nz) * cell_size;
    constexpr float gravity_y = -9.81f;
    constexpr float buoyancy_beta = 0.35f;
    constexpr float buoyancy_weight = -gravity_y * buoyancy_beta;

    StableFluidsSimulationConfig config{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .cell_size = cell_size,
        .dt = 1.0f / 120.0f,
        .viscosity = 0.00015f,
        .diffuse_iterations = 24,
        .pressure_iterations = 96,
        .uniform_force_x = 0.0f,
        .uniform_force_y = 0.0f,
        .uniform_force_z = 0.0f,
        .domain_boundary = {
            .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
            .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
        },
        .block_x = 8,
        .block_y = 8,
        .block_z = 4,
    };

    std::array fields{
        StableFluidsFieldCreateDesc{
            .name = "density",
            .component_count = 1,
            .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
            .diffusion = 0.00005f,
            .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
            .default_value_0 = 0.0f,
            .default_value_1 = 0.0f,
            .default_value_2 = 0.0f,
            .default_value_3 = 0.0f,
        },
        StableFluidsFieldCreateDesc{
            .name = "dye",
            .component_count = 3,
            .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
            .diffusion = 0.00002f,
            .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
            .default_value_0 = 0.0f,
            .default_value_1 = 0.0f,
            .default_value_2 = 0.0f,
            .default_value_3 = 0.0f,
        },
    };
    std::array buoyancy_terms{
        StableFluidsBuoyancyDesc{
            .field_index = 0,
            .weight = buoyancy_weight,
            .ambient = 0.0f,
        },
    };
    std::array<StableFluidsFieldHandle, 2> field_handles{};

    StableFluidsContextCreateDesc create_desc{
        .config = config,
        .stream = nullptr,
        .fields = fields.data(),
        .field_count = static_cast<uint32_t>(fields.size()),
        .buoyancy_terms = buoyancy_terms.data(),
        .buoyancy_term_count = static_cast<uint32_t>(buoyancy_terms.size()),
    };

    StableFluidsContext context = nullptr;
    if (!stable_ok(stable_fluids_create_context_cuda(&create_desc, &context, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda")) return EXIT_FAILURE;

    const StableFluidsColliderDesc collider{
        .collider_type = static_cast<uint32_t>(STABLE_FLUIDS_COLLIDER_SPHERE),
        .velocity_boundary_type = static_cast<uint32_t>(STABLE_FLUIDS_COLLIDER_VELOCITY_BOUNDARY_NO_SLIP),
        .center_x = extent_x * 0.5f,
        .center_y = extent_y * 0.36f,
        .center_z = extent_z * 0.5f,
        .radius = 0.08f,
        .half_extent_x = 0.0f,
        .half_extent_y = 0.0f,
        .half_extent_z = 0.0f,
        .linear_velocity_x = 0.0f,
        .linear_velocity_y = 0.0f,
        .linear_velocity_z = 0.0f,
    };
    const StableFluidsSceneDesc scene_desc{
        .colliders = &collider,
        .collider_count = 1,
    };
    if (!stable_ok(stable_fluids_update_scene_cuda(context, &scene_desc), "stable_fluids_update_scene_cuda")) {
        stable_fluids_destroy_context_cuda(context);
        return EXIT_FAILURE;
    }

    const auto begin = std::chrono::steady_clock::now();
    for (int frame = 0; frame < frames; ++frame) {
        const float center_x = extent_x * 0.18f;
        const float center_y = extent_y * 0.14f;
        const float center_z = extent_z * 0.28f + static_cast<float>(frame & 1) * 0.005f;
        const StableFluidsVelocitySourceDesc velocity_source{
            .center_x = center_x,
            .center_y = center_y,
            .center_z = center_z,
            .radius = 0.045f,
            .velocity_x = 0.18f,
            .velocity_y = 0.42f,
            .velocity_z = 0.12f,
        };
        const std::array field_sources{
            StableFluidsFieldSourceDesc{
                .field = field_handles[0],
                .center_x = center_x,
                .center_y = center_y,
                .center_z = center_z,
                .radius = 0.045f,
                .value_0 = 0.55f,
                .value_1 = 0.0f,
                .value_2 = 0.0f,
                .value_3 = 0.0f,
            },
            StableFluidsFieldSourceDesc{
                .field = field_handles[1],
                .center_x = center_x,
                .center_y = center_y,
                .center_z = center_z,
                .radius = 0.045f,
                .value_0 = 0.85f,
                .value_1 = 0.22f,
                .value_2 = 1.10f,
                .value_3 = 0.0f,
            },
        };
        const StableFluidsStepDesc step_desc{
            .velocity_sources = &velocity_source,
            .velocity_source_count = 1,
            .field_sources = field_sources.data(),
            .field_source_count = static_cast<uint32_t>(field_sources.size()),
        };
        if (!stable_ok(stable_fluids_step_cuda(context, &step_desc), "stable_fluids_step_cuda")) {
            stable_fluids_destroy_context_cuda(context);
            return EXIT_FAILURE;
        }
    }

    std::vector<float> density(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz), 0.0f);
    float* device_density = nullptr;
    const auto scalar_bytes = density.size() * sizeof(float);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_density), scalar_bytes), "cudaMalloc export density")) {
        stable_fluids_destroy_context_cuda(context);
        return EXIT_FAILURE;
    }

    const StableFluidsExportDesc export_desc{
        .kind = STABLE_FLUIDS_EXPORT_FIELD_COMPONENTS,
        .field_a = field_handles[0],
        .field_b = 0,
        .component_offset = 0,
        .component_count = 1,
    };
    if (!stable_ok(stable_fluids_export_cuda(context, &export_desc, device_density), "stable_fluids_export_cuda")) {
        cudaFree(device_density);
        stable_fluids_destroy_context_cuda(context);
        return EXIT_FAILURE;
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(device_density);
        stable_fluids_destroy_context_cuda(context);
        return EXIT_FAILURE;
    }
    if (!cuda_ok(cudaMemcpy(density.data(), device_density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) {
        cudaFree(device_density);
        stable_fluids_destroy_context_cuda(context);
        return EXIT_FAILURE;
    }

    cudaFree(device_density);
    stable_fluids_destroy_context_cuda(context);

    const float total_density = std::accumulate(density.begin(), density.end(), 0.0f);
    const float peak_density = density.empty() ? 0.0f : *std::max_element(density.begin(), density.end());
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::printf("frames=%d total_density=%.6f peak_density=%.6f elapsed_ms=%.3f\n", frames, total_density, peak_density, elapsed_ms);
    return EXIT_SUCCESS;
}
