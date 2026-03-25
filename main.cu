#include "stable-fluids.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {} // namespace

int main() {
    auto cuda_ok = [](const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
        return false;
    };
    auto stable_ok = [](const int32_t code, const char* what) {
        if (code == 0) return true;
        std::cerr << what << " failed (" << code << ")\n";
        return false;
    };

    constexpr int32_t nx                  = 48;
    constexpr int32_t ny                  = 72;
    constexpr int32_t nz                  = 48;
    constexpr float cell_size             = 1.0f;
    constexpr float dt                    = 1.0f / 90.0f;
    constexpr float viscosity             = 0.00015f;
    constexpr float diffusion             = 0.00005f;
    constexpr int32_t diffuse_iterations  = 24;
    constexpr int32_t pressure_iterations = 80;
    constexpr int32_t block_x             = 8;
    constexpr int32_t block_y             = 8;
    constexpr int32_t block_z             = 4;
    constexpr int32_t frames              = 16;

    const uint64_t scalar_bytes     = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_x_bytes = static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_y_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_z_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float);
    const std::size_t scalar_count  = static_cast<std::size_t>(scalar_bytes / sizeof(float));

    float* density                       = nullptr;
    float* velocity_x                    = nullptr;
    float* velocity_y                    = nullptr;
    float* velocity_z                    = nullptr;
    float* temporary_density             = nullptr;
    float* temporary_velocity_x          = nullptr;
    float* temporary_velocity_y          = nullptr;
    float* temporary_velocity_z          = nullptr;
    float* temporary_previous_density    = nullptr;
    float* temporary_previous_velocity_x = nullptr;
    float* temporary_previous_velocity_y = nullptr;
    float* temporary_previous_velocity_z = nullptr;
    float* temporary_pressure            = nullptr;
    float* temporary_divergence          = nullptr;
    cudaStream_t stream                  = nullptr;
    int exit_code                        = EXIT_SUCCESS;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_density), scalar_bytes), "cudaMalloc temporary_density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_x), velocity_x_bytes), "cudaMalloc temporary_velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_y), velocity_y_bytes), "cudaMalloc temporary_velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_velocity_z), velocity_z_bytes), "cudaMalloc temporary_velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_density), scalar_bytes), "cudaMalloc temporary_previous_density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_x), velocity_x_bytes), "cudaMalloc temporary_previous_velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_y), velocity_y_bytes), "cudaMalloc temporary_previous_velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_z), velocity_z_bytes), "cudaMalloc temporary_previous_velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_pressure), scalar_bytes), "cudaMalloc temporary_pressure")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_divergence), scalar_bytes), "cudaMalloc temporary_divergence")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) exit_code = EXIT_FAILURE;

    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(density, 0, scalar_bytes, stream), "cudaMemsetAsync density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_x, 0, velocity_x_bytes, stream), "cudaMemsetAsync velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_y, 0, velocity_y_bytes, stream), "cudaMemsetAsync velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_z, 0, velocity_z_bytes, stream), "cudaMemsetAsync velocity_z")) exit_code = EXIT_FAILURE;

    const auto cuda_begin = std::chrono::steady_clock::now();
    for (int frame = 0; exit_code == EXIT_SUCCESS && frame < frames; ++frame) {
        StableFluidsAddScalarSourceDesc scalar_source_desc{
            .struct_size     = sizeof(StableFluidsAddScalarSourceDesc),
            .api_version     = STABLE_FLUIDS_API_VERSION,
            .nx              = nx,
            .ny              = ny,
            .nz              = nz,
            .scalar          = density,
            .center_x        = static_cast<float>(nx) * 0.5f,
            .center_y        = static_cast<float>(ny) * 0.18f,
            .center_z        = static_cast<float>(nz) * 0.5f,
            .radius          = 4.5f,
            .amount          = 0.85f,
            .sample_offset_x = 0.5f,
            .sample_offset_y = 0.5f,
            .sample_offset_z = 0.5f,
            .block_x         = block_x,
            .block_y         = block_y,
            .block_z         = block_z,
            .stream          = stream,
        };

        StableFluidsAddVectorSourceDesc vector_source_desc{
            .struct_size = sizeof(StableFluidsAddVectorSourceDesc),
            .api_version = STABLE_FLUIDS_API_VERSION,
            .nx          = nx,
            .ny          = ny,
            .nz          = nz,
            .vector_x    = velocity_x,
            .vector_y    = velocity_y,
            .vector_z    = velocity_z,
            .center_x    = scalar_source_desc.center_x,
            .center_y    = scalar_source_desc.center_y,
            .center_z    = scalar_source_desc.center_z,
            .radius      = scalar_source_desc.radius,
            .amount_x    = 0.0f,
            .amount_y    = 1.2f,
            .amount_z    = 0.0f,
            .block_x     = block_x,
            .block_y     = block_y,
            .block_z     = block_z,
            .stream      = stream,
        };

        if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_add_scalar_source_cuda(&scalar_source_desc), "stable_fluids_add_scalar_source_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_add_vector_source_cuda(&vector_source_desc), "stable_fluids_add_vector_source_cuda")) exit_code = EXIT_FAILURE;

        StableFluidsAdvectVelocityDesc advect_velocity_desc{
            .struct_size                   = sizeof(StableFluidsAdvectVelocityDesc),
            .api_version                   = STABLE_FLUIDS_API_VERSION,
            .nx                            = nx,
            .ny                            = ny,
            .nz                            = nz,
            .cell_size                     = cell_size,
            .dt                            = dt,
            .velocity_x                    = velocity_x,
            .velocity_y                    = velocity_y,
            .velocity_z                    = velocity_z,
            .temporary_velocity_x          = temporary_velocity_x,
            .temporary_velocity_y          = temporary_velocity_y,
            .temporary_velocity_z          = temporary_velocity_z,
            .temporary_previous_velocity_x = temporary_previous_velocity_x,
            .temporary_previous_velocity_y = temporary_previous_velocity_y,
            .temporary_previous_velocity_z = temporary_previous_velocity_z,
            .block_x                       = block_x,
            .block_y                       = block_y,
            .block_z                       = block_z,
            .stream                        = stream,
        };

        StableFluidsDiffuseVelocityDesc diffuse_velocity_desc{
            .struct_size                = sizeof(StableFluidsDiffuseVelocityDesc),
            .api_version                = STABLE_FLUIDS_API_VERSION,
            .nx                         = nx,
            .ny                         = ny,
            .nz                         = nz,
            .cell_size                  = cell_size,
            .dt                         = dt,
            .viscosity                  = viscosity,
            .diffuse_iterations         = diffuse_iterations,
            .velocity_x                 = velocity_x,
            .velocity_y                 = velocity_y,
            .velocity_z                 = velocity_z,
            .temporary_velocity_x       = temporary_velocity_x,
            .temporary_velocity_y       = temporary_velocity_y,
            .temporary_velocity_z       = temporary_velocity_z,
            .temporary_density          = temporary_density,
            .temporary_previous_density = temporary_previous_density,
            .block_x                    = block_x,
            .block_y                    = block_y,
            .block_z                    = block_z,
            .stream                     = stream,
        };

        StableFluidsProjectDesc project_desc{
            .struct_size                = sizeof(StableFluidsProjectDesc),
            .api_version                = STABLE_FLUIDS_API_VERSION,
            .nx                         = nx,
            .ny                         = ny,
            .nz                         = nz,
            .cell_size                  = cell_size,
            .pressure_iterations        = pressure_iterations,
            .velocity_x                 = velocity_x,
            .velocity_y                 = velocity_y,
            .velocity_z                 = velocity_z,
            .temporary_pressure         = temporary_pressure,
            .temporary_divergence       = temporary_divergence,
            .temporary_density          = temporary_density,
            .temporary_previous_density = temporary_previous_density,
            .block_x                    = block_x,
            .block_y                    = block_y,
            .block_z                    = block_z,
            .stream                     = stream,
        };

        struct ScalarFieldBatch {
            float* field;
            float* temporary_field;
            float* previous_field;
            uint32_t clamp_non_negative;
        };
        const std::array scalar_fields{
            ScalarFieldBatch{density, temporary_density, temporary_previous_density, 1u},
        };

        if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_advect_velocity_cuda(&advect_velocity_desc), "stable_fluids_advect_velocity_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_diffuse_velocity_cuda(&diffuse_velocity_desc), "stable_fluids_diffuse_velocity_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_project_cuda(&project_desc), "stable_fluids_project_cuda")) exit_code = EXIT_FAILURE;
        for (const auto& scalar_field : scalar_fields) {
            StableFluidsAdvectScalarDesc advect_scalar_desc{
                .struct_size               = sizeof(StableFluidsAdvectScalarDesc),
                .api_version               = STABLE_FLUIDS_API_VERSION,
                .nx                        = nx,
                .ny                        = ny,
                .nz                        = nz,
                .cell_size                 = cell_size,
                .dt                        = dt,
                .scalar                    = scalar_field.field,
                .temporary_scalar          = scalar_field.temporary_field,
                .temporary_previous_scalar = scalar_field.previous_field,
                .velocity_x                = velocity_x,
                .velocity_y                = velocity_y,
                .velocity_z                = velocity_z,
                .clamp_non_negative        = scalar_field.clamp_non_negative,
                .block_x                   = block_x,
                .block_y                   = block_y,
                .block_z                   = block_z,
                .stream                    = stream,
            };

            StableFluidsDiffuseScalarDesc diffuse_scalar_desc{
                .struct_size                = sizeof(StableFluidsDiffuseScalarDesc),
                .api_version                = STABLE_FLUIDS_API_VERSION,
                .nx                         = nx,
                .ny                         = ny,
                .nz                         = nz,
                .cell_size                  = cell_size,
                .dt                         = dt,
                .diffusion                  = diffusion,
                .diffuse_iterations         = diffuse_iterations,
                .scalar                     = scalar_field.field,
                .temporary_scalar           = scalar_field.temporary_field,
                .temporary_solution_storage = temporary_pressure,
                .temporary_rhs_storage      = temporary_divergence,
                .clamp_non_negative         = scalar_field.clamp_non_negative,
                .block_x                    = block_x,
                .block_y                    = block_y,
                .block_z                    = block_z,
                .stream                     = stream,
            };

            if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_advect_scalar_cuda(&advect_scalar_desc), "stable_fluids_advect_scalar_cuda")) exit_code = EXIT_FAILURE;
            if (exit_code == EXIT_SUCCESS && !stable_ok(stable_fluids_diffuse_scalar_cuda(&diffuse_scalar_desc), "stable_fluids_diffuse_scalar_cuda")) exit_code = EXIT_FAILURE;
        }
    }
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) exit_code = EXIT_FAILURE;
    const auto cuda_end = std::chrono::steady_clock::now();

    std::vector<float> host_density(scalar_count, 0.0f);
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) exit_code = EXIT_FAILURE;

    const float cuda_total_density = exit_code == EXIT_SUCCESS ? std::accumulate(host_density.begin(), host_density.end(), 0.0f) : 0.0f;
    const float cuda_peak_density  = exit_code == EXIT_SUCCESS && !host_density.empty() ? *std::max_element(host_density.begin(), host_density.end()) : 0.0f;

    cudaStreamDestroy(stream);
    cudaFree(density);
    cudaFree(velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_z);
    cudaFree(temporary_density);
    cudaFree(temporary_velocity_x);
    cudaFree(temporary_velocity_y);
    cudaFree(temporary_velocity_z);
    cudaFree(temporary_previous_density);
    cudaFree(temporary_previous_velocity_x);
    cudaFree(temporary_previous_velocity_y);
    cudaFree(temporary_previous_velocity_z);
    cudaFree(temporary_pressure);
    cudaFree(temporary_divergence);
    if (exit_code != EXIT_SUCCESS) return exit_code;

    const double cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_begin).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "stable-fluids benchmark\n";
    std::cout << "grid: " << nx << " x " << ny << " x " << nz << '\n';
    std::cout << "frames: " << frames << '\n';
    std::cout << "| metric | cuda |\n";
    std::cout << "|---|---:|\n";
    std::cout << "| total_ms | " << cuda_ms << " |\n";
    std::cout << "| step_ms | " << cuda_ms / static_cast<double>(frames) << " |\n";
    std::cout << "| total_density | " << cuda_total_density << " |\n";
    std::cout << "| peak_density | " << cuda_peak_density << " |\n";
    return EXIT_SUCCESS;
}
