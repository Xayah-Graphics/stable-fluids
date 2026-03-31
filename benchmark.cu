#include "stable-fluids-3d.h"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    int nx                  = 128;
    int ny                  = 160;
    int nz                  = 128;
    int warmup_steps        = 48;
    int benchmark_steps     = 256;
    int diffuse_iterations  = 24;
    int pressure_iterations = 96;
    float cell_size         = 0.01f;
    float dt                = 1.0f / 90.0f;
    float viscosity         = 0.00012f;
    float field_diffusion   = 0.00005f;
    float field_dissipation = 0.35f;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next_value = [&](const char* const flag) {
            if (i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "missing value for %s\n", flag);
            std::exit(EXIT_FAILURE);
        };
        if (arg == "--nx") nx = std::atoi(next_value("--nx"));
        else if (arg == "--ny") ny = std::atoi(next_value("--ny"));
        else if (arg == "--nz") nz = std::atoi(next_value("--nz"));
        else if (arg == "--warmup") warmup_steps = std::atoi(next_value("--warmup"));
        else if (arg == "--steps") benchmark_steps = std::atoi(next_value("--steps"));
        else if (arg == "--diffuse-iters") diffuse_iterations = std::atoi(next_value("--diffuse-iters"));
        else if (arg == "--pressure-iters") pressure_iterations = std::atoi(next_value("--pressure-iters"));
        else if (arg == "--cell-size") cell_size = std::strtof(next_value("--cell-size"), nullptr);
        else if (arg == "--dt") dt = std::strtof(next_value("--dt"), nullptr);
        else if (arg == "--viscosity") viscosity = std::strtof(next_value("--viscosity"), nullptr);
        else if (arg == "--field-diffusion") field_diffusion = std::strtof(next_value("--field-diffusion"), nullptr);
        else if (arg == "--field-dissipation") field_dissipation = std::strtof(next_value("--field-dissipation"), nullptr);
        else if (arg == "--help") {
            std::printf(
                "stable-fluids-benchmark [--nx N] [--ny N] [--nz N] [--warmup N] [--steps N]\n"
                "                        [--diffuse-iters N] [--pressure-iters N]\n"
                "                        [--cell-size H] [--dt DT] [--viscosity V]\n"
                "                        [--field-diffusion D] [--field-dissipation K]\n");
            return EXIT_SUCCESS;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    auto check_cuda = [&](const cudaError_t status, const char* const what) {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return false;
    };
    auto check_stable = [&](const StableFluidsResult code, const char* const what) {
        if (code == STABLE_FLUIDS_RESULT_OK) return true;
        std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(code));
        return false;
    };

    const float extent_x  = static_cast<float>(nx) * cell_size;
    const float extent_y  = static_cast<float>(ny) * cell_size;
    const float extent_z  = static_cast<float>(nz) * cell_size;
    const float source_x  = extent_x * 0.50f;
    const float source_y  = extent_y * 0.16f;
    const float source_z  = extent_z * 0.50f;
    const float source_rx = extent_x * 0.07f;
    const float source_ry = extent_y * 0.06f;
    const float source_rz = extent_z * 0.07f;

    const StableFluidsSimulationConfig config{
        .nx                  = nx,
        .ny                  = ny,
        .nz                  = nz,
        .cell_size           = cell_size,
        .dt                  = dt,
        .viscosity           = viscosity,
        .diffuse_iterations  = diffuse_iterations,
        .pressure_iterations = pressure_iterations,
        .boundary =
            {
                .x_minus = {.type = STABLE_FLUIDS_BOUNDARY_PERIODIC, .value = 0.0f},
                .x_plus  = {.type = STABLE_FLUIDS_BOUNDARY_PERIODIC, .value = 0.0f},
                .y_minus = {.type = STABLE_FLUIDS_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                .y_plus  = {.type = STABLE_FLUIDS_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                .z_minus = {.type = STABLE_FLUIDS_BOUNDARY_PERIODIC, .value = 0.0f},
                .z_plus  = {.type = STABLE_FLUIDS_BOUNDARY_PERIODIC, .value = 0.0f},
            },
    };

    const std::array scalar_fields{
        StableFluidsScalarFieldDesc{
            .name          = "density",
            .diffusion     = field_diffusion,
            .dissipation   = field_dissipation,
            .initial_value = 0.0f,
        },
    };
    const std::array vector_fields{
        StableFluidsVectorFieldDesc{
            .name            = "force",
            .usage           = STABLE_FLUIDS_VECTOR_FIELD_FORCE,
            .initial_value_x = 0.0f,
            .initial_value_y = 0.0f,
            .initial_value_z = 0.0f,
        },
    };
    std::array<StableFluidsScalarFieldHandle, 1> scalar_field_handles{};
    std::array<StableFluidsVectorFieldHandle, 1> vector_field_handles{};

    const auto cell_count  = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const auto scalar_bytes = cell_count * sizeof(float);
    std::vector<float> force_x_host(cell_count, 0.0f);
    std::vector<float> force_y_host(cell_count, 0.0f);
    std::vector<float> force_z_host(cell_count, 0.0f);
    std::vector<float> density_source_host(cell_count, 0.0f);

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const auto index = static_cast<std::size_t>(x) + static_cast<std::size_t>(nx) * (static_cast<std::size_t>(y) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
                const float px   = (static_cast<float>(x) + 0.5f) * cell_size;
                const float py   = (static_cast<float>(y) + 0.5f) * cell_size;
                const float pz   = (static_cast<float>(z) + 0.5f) * cell_size;
                const float dx   = (px - source_x) / source_rx;
                const float dy   = (py - source_y) / source_ry;
                const float dz   = (pz - source_z) / source_rz;
                const float r2   = dx * dx + dy * dy + dz * dz;
                if (r2 > 1.0f) continue;
                const float plume = std::exp(-2.2f * r2);
                density_source_host[index] = 28.0f * plume;
                force_x_host[index]        = 1.8f * plume * std::sin((pz / (extent_z + 1.0e-6f)) * 6.2831853f);
                force_y_host[index]        = 7.5f * plume;
                force_z_host[index]        = 1.4f * plume * std::cos((px / (extent_x + 1.0e-6f)) * 6.2831853f);
            }
        }
    }

    cudaStream_t stream = nullptr;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) return EXIT_FAILURE;

    void* context = nullptr;
    const StableFluidsContextCreateDesc create_desc{
        .config             = config,
        .stream             = stream,
        .scalar_fields      = scalar_fields.data(),
        .scalar_field_count = static_cast<uint32_t>(scalar_fields.size()),
        .vector_fields      = vector_fields.data(),
        .vector_field_count = static_cast<uint32_t>(vector_fields.size()),
    };
    if (!check_stable(stable_fluids_create_context_cuda(&create_desc, &context, scalar_field_handles.data(), static_cast<uint32_t>(scalar_field_handles.size()), vector_field_handles.data(), static_cast<uint32_t>(vector_field_handles.size())),
            "stable_fluids_create_context_cuda")) {
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    float* force_x_device        = nullptr;
    float* force_y_device        = nullptr;
    float* force_z_device        = nullptr;
    float* density_source_device = nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_x_device), scalar_bytes), "cudaMalloc force_x_device")) return EXIT_FAILURE;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_y_device), scalar_bytes), "cudaMalloc force_y_device")) return EXIT_FAILURE;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_z_device), scalar_bytes), "cudaMalloc force_z_device")) return EXIT_FAILURE;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&density_source_device), scalar_bytes), "cudaMalloc density_source_device")) return EXIT_FAILURE;

    if (!check_cuda(cudaMemcpyAsync(force_x_device, force_x_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync force_x")) return EXIT_FAILURE;
    if (!check_cuda(cudaMemcpyAsync(force_y_device, force_y_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync force_y")) return EXIT_FAILURE;
    if (!check_cuda(cudaMemcpyAsync(force_z_device, force_z_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync force_z")) return EXIT_FAILURE;
    if (!check_cuda(cudaMemcpyAsync(density_source_device, density_source_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync density_source")) return EXIT_FAILURE;

    if (!check_stable(stable_fluids_update_vector_field_cuda(context, vector_field_handles[0], force_x_device, force_y_device, force_z_device), "stable_fluids_update_vector_field_cuda")) return EXIT_FAILURE;
    if (!check_stable(stable_fluids_update_scalar_field_source_cuda(context, scalar_field_handles[0], density_source_device), "stable_fluids_update_scalar_field_source_cuda")) return EXIT_FAILURE;

    cudaEvent_t step_begin = nullptr;
    cudaEvent_t step_end   = nullptr;
    if (!check_cuda(cudaEventCreate(&step_begin), "cudaEventCreate step_begin")) return EXIT_FAILURE;
    if (!check_cuda(cudaEventCreate(&step_end), "cudaEventCreate step_end")) return EXIT_FAILURE;

    {
        nvtx3::scoped_range range("benchmark.warmup");
        for (int step = 0; step < warmup_steps; ++step) {
            if (!check_stable(stable_fluids_step_cuda(context), "stable_fluids_step_cuda warmup")) return EXIT_FAILURE;
        }
        if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup")) return EXIT_FAILURE;
    }

    float elapsed_ms = 0.0f;
    {
        nvtx3::scoped_range range("benchmark.measure");
        if (!check_cuda(cudaEventRecord(step_begin, stream), "cudaEventRecord step_begin")) return EXIT_FAILURE;
        for (int step = 0; step < benchmark_steps; ++step) {
            if (!check_stable(stable_fluids_step_cuda(context), "stable_fluids_step_cuda")) return EXIT_FAILURE;
        }
        if (!check_cuda(cudaEventRecord(step_end, stream), "cudaEventRecord step_end")) return EXIT_FAILURE;
        if (!check_cuda(cudaEventSynchronize(step_end), "cudaEventSynchronize step_end")) return EXIT_FAILURE;
        if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, step_begin, step_end), "cudaEventElapsedTime")) return EXIT_FAILURE;
    }

    std::vector<float> density_host(cell_count, 0.0f);
    {
        nvtx3::scoped_range range("benchmark.export");
        StableFluidsView view{};
        const StableFluidsViewRequest request{
            .kind            = STABLE_FLUIDS_VIEW_SCALAR_FIELD_DATA,
            .scalar_field    = scalar_field_handles[0],
            .vector_field    = 0,
            .consumer_stream = stream,
        };
        if (!check_stable(stable_fluids_get_view_cuda(context, &request, &view), "stable_fluids_get_view_cuda")) return EXIT_FAILURE;
        if (!check_cuda(cudaMemcpyAsync(density_host.data(), view.data0, scalar_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync density_view")) return EXIT_FAILURE;
        if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize export")) return EXIT_FAILURE;
    }

    const double avg_step_ms = benchmark_steps > 0 ? static_cast<double>(elapsed_ms) / static_cast<double>(benchmark_steps) : 0.0;
    const double mlups       = elapsed_ms > 0.0f ? static_cast<double>(cell_count) * static_cast<double>(benchmark_steps) / (static_cast<double>(elapsed_ms) * 1000.0) : 0.0;
    const float total_density = std::accumulate(density_host.begin(), density_host.end(), 0.0f);
    const float peak_density  = density_host.empty() ? 0.0f : *std::max_element(density_host.begin(), density_host.end());

    std::printf(
        "benchmark grid=%dx%dx%d warmup=%d steps=%d dt=%.6f viscosity=%.7f field_diffusion=%.7f field_dissipation=%.4f\n",
        nx,
        ny,
        nz,
        warmup_steps,
        benchmark_steps,
        dt,
        viscosity,
        field_diffusion,
        field_dissipation);
    std::printf(
        "timing total_ms=%.3f avg_step_ms=%.6f mlups=%.3f total_density=%.6f peak_density=%.6f\n",
        static_cast<double>(elapsed_ms),
        avg_step_ms,
        mlups,
        total_density,
        peak_density);

    cudaEventDestroy(step_end);
    cudaEventDestroy(step_begin);
    cudaFree(density_source_device);
    cudaFree(force_z_device);
    cudaFree(force_y_device);
    cudaFree(force_x_device);
    stable_fluids_destroy_context_cuda(context);
    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
