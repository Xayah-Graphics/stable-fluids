#include "stable-fluids.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

namespace {

    bool cuda_ok(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return false;
    }

    bool stable_ok(const int32_t code, const char* what) {
        if (code == 0) return true;
        std::fprintf(stderr, "%s failed: %d\n", what, code);
        return false;
    }

} // namespace

int main() {
    constexpr int32_t nx = 64;
    constexpr int32_t ny = 96;
    constexpr int32_t nz = 64;
    constexpr int frames = 24;

    StableFluidsSimulationConfig config{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .cell_size = 1.0f,
        .dt = 1.0f / 90.0f,
        .viscosity = 0.0001f,
        .diffusion = 0.00005f,
        .diffuse_iterations = 20,
        .pressure_iterations = 80,
        .ambient_temperature = 0.0f,
        .density_buoyancy = 0.35f,
        .temperature_buoyancy = 0.0f,
        .uniform_force_x = 0.0f,
        .uniform_force_y = 0.0f,
        .uniform_force_z = 0.0f,
        .domain_boundary = {
            .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, .scalar = 0.0f, },
            .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, .scalar = 0.0f, },
            .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_NO_SLIP), .velocity = 0.0f, .scalar = 0.0f, },
            .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, .scalar = 0.0f, },
            .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, .scalar = 0.0f, },
            .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_OUTFLOW), .velocity = 0.0f, .scalar = 0.0f, },
        },
        .block_x = 8,
        .block_y = 8,
        .block_z = 4,
    };

    StableFluidsContextCreateDesc create_desc{
        .config = config,
        .stream = nullptr,
    };

    StableFluidsContext context = nullptr;
    if (!stable_ok(stable_fluids_create_context_cuda(&create_desc, &context), "stable_fluids_create_context_cuda")) return EXIT_FAILURE;

    const StableFluidsColliderDesc collider{
        .collider_type = static_cast<uint32_t>(STABLE_FLUIDS_COLLIDER_SPHERE),
        .boundary_type = static_cast<uint32_t>(STABLE_FLUIDS_BOUNDARY_NO_SLIP),
        .center_x = static_cast<float>(nx) * 0.5f,
        .center_y = static_cast<float>(ny) * 0.36f,
        .center_z = static_cast<float>(nz) * 0.5f,
        .radius = 8.0f,
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
        const float center_x = static_cast<float>(nx) * 0.18f;
        const float center_y = static_cast<float>(ny) * 0.14f;
        const float center_z = static_cast<float>(nz) * 0.28f + static_cast<float>(frame & 1) * 0.35f;
        const StableFluidsSourceDesc source{
            .center_x = center_x,
            .center_y = center_y,
            .center_z = center_z,
            .radius = 4.5f,
            .density_amount = 0.55f,
            .dye_r = 0.85f,
            .dye_g = 0.22f,
            .dye_b = 1.10f,
            .temperature_amount = 0.0f,
            .velocity_x = 1.8f,
            .velocity_y = 3.8f,
            .velocity_z = 1.2f,
        };
        const StableFluidsStepDesc step_desc{
            .sources = &source,
            .source_count = 1,
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

    const StableFluidsExportFieldDesc export_desc{
        .field = static_cast<uint32_t>(STABLE_FLUIDS_EXPORT_DENSITY),
        .destination = device_density,
    };
    if (!stable_ok(stable_fluids_export_field_cuda(context, &export_desc), "stable_fluids_export_field_cuda")) {
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
