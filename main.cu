#include "stable-fluids-3d.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

    bool stable_ok(const StableFluidsResult code, const char* what) {
        if (code == STABLE_FLUIDS_RESULT_OK) return true;
        std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(code));
        return false;
    }

    __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __global__ void fill_kernel(float* field, const float value, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        field[index_3d(x, y, z, nx, ny)] = value;
    }

    __global__ void add_blob_kernel(float* field, const float amplitude, const float center_x, const float center_y, const float center_z, const float radius, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float px      = (static_cast<float>(x) + 0.5f) * h;
        const float py      = (static_cast<float>(y) + 0.5f) * h;
        const float pz      = (static_cast<float>(z) + 0.5f) * h;
        const float dx      = px - center_x;
        const float dy      = py - center_y;
        const float dz      = pz - center_z;
        const float radius2 = radius * radius;
        const float dist2   = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;
        const float weight = radius2 > 0.0f ? 1.0f - dist2 / radius2 : 0.0f;
        field[index_3d(x, y, z, nx, ny)] += amplitude * weight;
    }

} // namespace

int main() {
    constexpr int32_t nx        = 96;
    constexpr int32_t ny        = 96;
    constexpr int32_t nz        = 96;
    constexpr int frames        = 32;
    constexpr float cell_size   = 0.01f;
    constexpr float extent_x    = static_cast<float>(nx) * cell_size;
    constexpr float extent_y    = static_cast<float>(ny) * cell_size;
    constexpr float extent_z    = static_cast<float>(nz) * cell_size;
    constexpr float source_r    = 0.055f;
    constexpr float source_x    = extent_x * 0.35f;
    constexpr float source_y    = extent_y * 0.16f;
    constexpr float source_z    = extent_z * 0.50f;

    const StableFluidsSimulationConfig config{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .cell_size = cell_size,
        .dt = 1.0f / 90.0f,
        .viscosity = 0.00012f,
        .diffuse_iterations = 24,
        .pressure_iterations = 96,
        .boundary = {
            .x = STABLE_FLUIDS_BOUNDARY_PERIODIC,
            .y = STABLE_FLUIDS_BOUNDARY_FIXED,
            .z = STABLE_FLUIDS_BOUNDARY_PERIODIC,
        },
        .block_x = 8,
        .block_y = 8,
        .block_z = 4,
    };

    const std::array fields{
        StableFluidsFieldCreateDesc{
            .name = "density",
            .diffusion = 0.00005f,
            .dissipation = 0.35f,
            .initial_value = 0.0f,
        },
    };
    std::array<StableFluidsFieldHandle, 1> field_handles{};

    const StableFluidsContextCreateDesc create_desc{
        .config = config,
        .stream = nullptr,
        .fields = fields.data(),
        .field_count = static_cast<uint32_t>(fields.size()),
    };

    StableFluidsContext context = nullptr;
    if (!stable_ok(stable_fluids_create_context_cuda(&create_desc, &context, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda")) return EXIT_FAILURE;

    const dim3 block(static_cast<unsigned>(config.block_x), static_cast<unsigned>(config.block_y), static_cast<unsigned>(config.block_z));
    const dim3 cells(
        static_cast<unsigned>((config.nx + config.block_x - 1) / config.block_x),
        static_cast<unsigned>((config.ny + config.block_y - 1) / config.block_y),
        static_cast<unsigned>((config.nz + config.block_z - 1) / config.block_z));

    const auto cell_count  = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const auto scalar_size = cell_count * sizeof(float);

    float* force_x = nullptr;
    float* force_y = nullptr;
    float* force_z = nullptr;
    float* density_source = nullptr;
    float* device_density = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&force_x), scalar_size), "cudaMalloc force_x")) return EXIT_FAILURE;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&force_y), scalar_size), "cudaMalloc force_y")) return EXIT_FAILURE;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&force_z), scalar_size), "cudaMalloc force_z")) return EXIT_FAILURE;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density_source), scalar_size), "cudaMalloc density_source")) return EXIT_FAILURE;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_density), scalar_size), "cudaMalloc device_density")) return EXIT_FAILURE;

    const auto begin = std::chrono::steady_clock::now();
    for (int frame = 0; frame < frames; ++frame) {
        fill_kernel<<<cells, block>>>(force_x, 0.0f, nx, ny, nz);
        fill_kernel<<<cells, block>>>(force_y, 0.0f, nx, ny, nz);
        fill_kernel<<<cells, block>>>(force_z, 0.0f, nx, ny, nz);
        fill_kernel<<<cells, block>>>(density_source, 0.0f, nx, ny, nz);
        if (!cuda_ok(cudaGetLastError(), "fill_kernel")) return EXIT_FAILURE;

        const float lateral = std::sin(static_cast<float>(frame) * 0.35f);
        const float swirl   = std::cos(static_cast<float>(frame) * 0.27f);
        add_blob_kernel<<<cells, block>>>(density_source, 32.0f, source_x, source_y, source_z, source_r, nx, ny, nz, cell_size);
        add_blob_kernel<<<cells, block>>>(force_x, 2.2f * lateral, source_x, source_y, source_z, source_r, nx, ny, nz, cell_size);
        add_blob_kernel<<<cells, block>>>(force_y, 7.5f, source_x, source_y, source_z, source_r, nx, ny, nz, cell_size);
        add_blob_kernel<<<cells, block>>>(force_z, 1.8f * swirl, source_x, source_y, source_z, source_r, nx, ny, nz, cell_size);
        if (!cuda_ok(cudaGetLastError(), "add_blob_kernel")) return EXIT_FAILURE;

        const StableFluidsFieldSourceDesc field_source{
            .field = field_handles[0],
            .values = density_source,
        };
        const StableFluidsStepDesc step_desc{
            .force_x = force_x,
            .force_y = force_y,
            .force_z = force_z,
            .field_sources = &field_source,
            .field_source_count = 1,
        };
        if (!stable_ok(stable_fluids_step_cuda(context, &step_desc), "stable_fluids_step_cuda")) return EXIT_FAILURE;
    }

    const StableFluidsExportDesc export_desc{
        .kind = STABLE_FLUIDS_EXPORT_FIELD,
        .field = field_handles[0],
    };
    if (!stable_ok(stable_fluids_export_cuda(context, &export_desc, device_density), "stable_fluids_export_cuda")) return EXIT_FAILURE;
    if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return EXIT_FAILURE;

    std::vector<float> density(cell_count, 0.0f);
    if (!cuda_ok(cudaMemcpy(density.data(), device_density, scalar_size, cudaMemcpyDeviceToHost), "cudaMemcpy density")) return EXIT_FAILURE;

    cudaFree(force_x);
    cudaFree(force_y);
    cudaFree(force_z);
    cudaFree(density_source);
    cudaFree(device_density);
    stable_fluids_destroy_context_cuda(context);

    const float total_density = std::accumulate(density.begin(), density.end(), 0.0f);
    const float peak_density  = density.empty() ? 0.0f : *std::max_element(density.begin(), density.end());
    const auto elapsed_ms     = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::printf("frames=%d total_density=%.6f peak_density=%.6f elapsed_ms=%.3f\n", frames, total_density, peak_density, elapsed_ms);
    return EXIT_SUCCESS;
}
