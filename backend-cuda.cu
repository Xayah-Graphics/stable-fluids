#include "stable-fluids-3d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {

    using Stream = cudaStream_t;

    constexpr StableFluidsResult success         = STABLE_FLUIDS_RESULT_OK;
    constexpr StableFluidsResult out_of_memory   = STABLE_FLUIDS_RESULT_OUT_OF_MEMORY;
    constexpr StableFluidsResult backend_failure = STABLE_FLUIDS_RESULT_BACKEND_FAILURE;

    struct ProjectionMetricsState {
        float max_abs_divergence  = 0.0f;
        float sum_sq_divergence   = 0.0f;
        uint32_t cell_count       = 0;
        uint32_t _padding         = 0;
    };

    struct DeviceBuffers {
        float* velocity_x                          = nullptr;
        float* velocity_y                          = nullptr;
        float* velocity_z                          = nullptr;
        float* temp_velocity_x                     = nullptr;
        float* temp_velocity_y                     = nullptr;
        float* temp_velocity_z                     = nullptr;
        float* pressure                            = nullptr;
        float* divergence                          = nullptr;
        ProjectionMetricsState* projection_metrics = nullptr;
    };

    struct FieldStorage {
        StableFluidsFieldCreateDesc desc{};
        float* data = nullptr;
        float* temp = nullptr;
    };

    struct ContextStorage {
        StableFluidsSimulationConfig config{};
        std::vector<FieldStorage> fields{};
        DeviceBuffers device{};
        Stream stream    = nullptr;
        bool owns_stream = false;
    };

    __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __device__ bool remap_index(int& coordinate, const int extent, const uint32_t mode) {
        if (coordinate >= 0 && coordinate < extent) return true;
        if (mode != STABLE_FLUIDS_BOUNDARY_PERIODIC || extent <= 0) return false;
        coordinate %= extent;
        if (coordinate < 0) coordinate += extent;
        return true;
    }

    __device__ float wrap_position(float value, const float extent) {
        if (extent <= 0.0f) return 0.0f;
        value = fmodf(value, extent);
        if (value < 0.0f) value += extent;
        return value;
    }

    __device__ float load_cell(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const StableFluidsBoundaryConfig boundary) {
        if (!remap_index(x, nx, boundary.x)) return 0.0f;
        if (!remap_index(y, ny, boundary.y)) return 0.0f;
        if (!remap_index(z, nz, boundary.z)) return 0.0f;
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float sample_linear(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const float extent_x = static_cast<float>(nx) * h;
        const float extent_y = static_cast<float>(ny) * h;
        const float extent_z = static_cast<float>(nz) * h;
        if (boundary.x == STABLE_FLUIDS_BOUNDARY_PERIODIC) x = wrap_position(x, extent_x);
        if (boundary.y == STABLE_FLUIDS_BOUNDARY_PERIODIC) y = wrap_position(y, extent_y);
        if (boundary.z == STABLE_FLUIDS_BOUNDARY_PERIODIC) z = wrap_position(z, extent_z);

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

        const float c000 = load_cell(field, x0, y0, z0, nx, ny, nz, boundary);
        const float c100 = load_cell(field, x1, y0, z0, nx, ny, nz, boundary);
        const float c010 = load_cell(field, x0, y1, z0, nx, ny, nz, boundary);
        const float c110 = load_cell(field, x1, y1, z0, nx, ny, nz, boundary);
        const float c001 = load_cell(field, x0, y0, z1, nx, ny, nz, boundary);
        const float c101 = load_cell(field, x1, y0, z1, nx, ny, nz, boundary);
        const float c011 = load_cell(field, x0, y1, z1, nx, ny, nz, boundary);
        const float c111 = load_cell(field, x1, y1, z1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        return make_float3(sample_linear(velocity_x, x, y, z, nx, ny, nz, h, boundary), sample_linear(velocity_y, x, y, z, nx, ny, nz, h, boundary), sample_linear(velocity_z, x, y, z, nx, ny, nz, h, boundary));
    }

    __device__ float3 trace_particle_rk2(const float x, const float y, const float z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float dt, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const float3 velocity_0 = sample_velocity(velocity_x, velocity_y, velocity_z, x, y, z, nx, ny, nz, h, boundary);
        const float3 mid        = make_float3(x - 0.5f * dt * velocity_0.x, y - 0.5f * dt * velocity_0.y, z - 0.5f * dt * velocity_0.z);
        const float3 velocity_1 = sample_velocity(velocity_x, velocity_y, velocity_z, mid.x, mid.y, mid.z, nx, ny, nz, h, boundary);
        return make_float3(x - dt * velocity_1.x, y - dt * velocity_1.y, z - dt * velocity_1.z);
    }

    __device__ void atomic_max_float(float* destination, const float value) {
        auto* bits    = reinterpret_cast<unsigned int*>(destination);
        unsigned prev = *bits;
        while (__uint_as_float(prev) < value) {
            const unsigned next = atomicCAS(bits, prev, __float_as_uint(value));
            if (next == prev) break;
            prev = next;
        }
    }

    __global__ void fill_kernel(float* field, const float value, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        field[index_3d(x, y, z, nx, ny)] = value;
    }

    __global__ void add_force_kernel(float* velocity_x, float* velocity_y, float* velocity_z, const float* force_x, const float* force_y, const float* force_z, const float dt, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (force_x != nullptr) velocity_x[index] += dt * force_x[index];
        if (force_y != nullptr) velocity_y[index] += dt * force_y[index];
        if (force_z != nullptr) velocity_z[index] += dt * force_z[index];
    }

    __global__ void add_field_source_kernel(float* field, const float* source, const float dt, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        field[index] += dt * source[index];
    }

    __global__ void advect_component_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float dt, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float px       = (static_cast<float>(x) + 0.5f) * h;
        const float py       = (static_cast<float>(y) + 0.5f) * h;
        const float pz       = (static_cast<float>(z) + 0.5f) * h;
        const float3 traced  = trace_particle_rk2(px, py, pz, velocity_x, velocity_y, velocity_z, dt, nx, ny, nz, h, boundary);
        destination[index_3d(x, y, z, nx, ny)] = sample_linear(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void diffuse_rbgs_kernel(float* destination, const float* source, const float alpha, const int parity, const int nx, const int ny, const int nz, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors =
            load_cell(destination, x - 1, y, z, nx, ny, nz, boundary) +
            load_cell(destination, x + 1, y, z, nx, ny, nz, boundary) +
            load_cell(destination, x, y - 1, z, nx, ny, nz, boundary) +
            load_cell(destination, x, y + 1, z, nx, ny, nz, boundary) +
            load_cell(destination, x, y, z - 1, nx, ny, nz, boundary) +
            load_cell(destination, x, y, z + 1, nx, ny, nz, boundary);
        const auto index      = index_3d(x, y, z, nx, ny);
        destination[index]    = (source[index] + alpha * neighbors) / (1.0f + 6.0f * alpha);
    }

    __global__ void dissipate_kernel(float* destination, const float* source, const float factor, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        destination[index] = source[index] * factor;
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h = 0.5f / h;
        const float ddx    = (load_cell(velocity_x, x + 1, y, z, nx, ny, nz, boundary) - load_cell(velocity_x, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddy    = (load_cell(velocity_y, x, y + 1, z, nx, ny, nz, boundary) - load_cell(velocity_y, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddz    = (load_cell(velocity_z, x, y, z + 1, nx, ny, nz, boundary) - load_cell(velocity_z, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        divergence[index_3d(x, y, z, nx, ny)] = ddx + ddy + ddz;
    }

    __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const int parity, const int nx, const int ny, const int nz, const float h2, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors =
            load_cell(pressure, x - 1, y, z, nx, ny, nz, boundary) +
            load_cell(pressure, x + 1, y, z, nx, ny, nz, boundary) +
            load_cell(pressure, x, y - 1, z, nx, ny, nz, boundary) +
            load_cell(pressure, x, y + 1, z, nx, ny, nz, boundary) +
            load_cell(pressure, x, y, z - 1, nx, ny, nz, boundary) +
            load_cell(pressure, x, y, z + 1, nx, ny, nz, boundary);
        const auto index   = index_3d(x, y, z, nx, ny);
        pressure[index]    = (neighbors - h2 * divergence[index]) / 6.0f;
    }

    __global__ void project_velocity_kernel(float* destination_x, float* destination_y, float* destination_z, const float* source_x, const float* source_y, const float* source_z, const float* pressure, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h = 0.5f / h;
        const auto index   = index_3d(x, y, z, nx, ny);
        const float grad_x = (load_cell(pressure, x + 1, y, z, nx, ny, nz, boundary) - load_cell(pressure, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_y = (load_cell(pressure, x, y + 1, z, nx, ny, nz, boundary) - load_cell(pressure, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_z = (load_cell(pressure, x, y, z + 1, nx, ny, nz, boundary) - load_cell(pressure, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        destination_x[index] = source_x[index] - grad_x;
        destination_y[index] = source_y[index] - grad_y;
        destination_z[index] = source_z[index] - grad_z;
    }

    __global__ void accumulate_projection_metrics_kernel(ProjectionMetricsState* metrics, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h = 0.5f / h;
        const float ddx    = (load_cell(velocity_x, x + 1, y, z, nx, ny, nz, boundary) - load_cell(velocity_x, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddy    = (load_cell(velocity_y, x, y + 1, z, nx, ny, nz, boundary) - load_cell(velocity_y, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddz    = (load_cell(velocity_z, x, y, z + 1, nx, ny, nz, boundary) - load_cell(velocity_z, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        const float value  = fabsf(ddx + ddy + ddz);
        atomic_max_float(&metrics->max_abs_divergence, value);
        atomicAdd(&metrics->sum_sq_divergence, value * value);
        atomicAdd(&metrics->cell_count, 1u);
    }

    __global__ void pack_velocity_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index       = index_3d(x, y, z, nx, ny);
        const auto cell_count  = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        destination[index]                 = velocity_x[index];
        destination[cell_count + index]    = velocity_y[index];
        destination[cell_count * 2u + index] = velocity_z[index];
    }

    __global__ void velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        const float vx   = velocity_x[index];
        const float vy   = velocity_y[index];
        const float vz   = velocity_z[index];
        destination[index] = sqrtf(vx * vx + vy * vy + vz * vz);
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
        context.device.velocity_x         = nullptr;
        context.device.velocity_y         = nullptr;
        context.device.velocity_z         = nullptr;
        context.device.temp_velocity_x    = nullptr;
        context.device.temp_velocity_y    = nullptr;
        context.device.temp_velocity_z    = nullptr;
        context.device.pressure           = nullptr;
        context.device.divergence         = nullptr;
        context.device.projection_metrics = nullptr;
        for (auto& field : context.fields) {
            if (field.data != nullptr) cudaFree(field.data);
            if (field.temp != nullptr) cudaFree(field.temp);
            field.data = nullptr;
            field.temp = nullptr;
        }
    }

    StableFluidsResult allocate_buffers(ContextStorage& context) {
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto bytes      = cell_count * sizeof(float);
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_x), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_y), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.velocity_z), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_x), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_y), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.temp_velocity_z), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.pressure), bytes) != cudaSuccess) return out_of_memory;
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context.device.divergence), bytes) != cudaSuccess) return out_of_memory;
        if (cudaMalloc(reinterpret_cast<void**>(&context.device.projection_metrics), sizeof(ProjectionMetricsState)) != cudaSuccess) return out_of_memory;
        for (auto& field : context.fields) {
            if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&field.data), bytes) != cudaSuccess) return out_of_memory;
            if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&field.temp), bytes) != cudaSuccess) return out_of_memory;
        }
        return success;
    }

    dim3 make_block(const ContextStorage& context) {
        return dim3(static_cast<unsigned>((std::max)(context.config.block_x, 1)), static_cast<unsigned>((std::max)(context.config.block_y, 1)), static_cast<unsigned>((std::max)(context.config.block_z, 1)));
    }

    dim3 make_cells(const ContextStorage& context, const dim3& block) {
        return dim3(
            static_cast<unsigned>((context.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((context.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((context.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    StableFluidsResult reset_fields(ContextStorage& context) {
        const dim3 block      = make_block(context);
        const dim3 cells      = make_cells(context, block);
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto bytes      = cell_count * sizeof(float);

        if (bytes > 0 && cudaMemsetAsync(context.device.velocity_x, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.velocity_y, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.velocity_z, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.temp_velocity_x, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.temp_velocity_y, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.temp_velocity_z, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.pressure, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.divergence, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(ProjectionMetricsState), context.stream) != cudaSuccess) return backend_failure;

        for (auto& field : context.fields) {
            fill_kernel<<<cells, block, 0, context.stream>>>(field.data, field.desc.initial_value, context.config.nx, context.config.ny, context.config.nz);
            fill_kernel<<<cells, block, 0, context.stream>>>(field.temp, field.desc.initial_value, context.config.nx, context.config.ny, context.config.nz);
            if (cudaGetLastError() != cudaSuccess) return backend_failure;
        }
        return success;
    }

    StableFluidsResult diffuse_component(ContextStorage& context, float* destination, const float* source, const float diffusion) {
        const float alpha = context.config.dt * diffusion / (context.config.cell_size * context.config.cell_size);
        if (alpha <= 0.0f) {
            const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
            if (cell_count == 0) return success;
            return cudaMemcpyAsync(destination, source, cell_count * sizeof(float), cudaMemcpyDeviceToDevice, context.stream) == cudaSuccess ? success : backend_failure;
        }
        const dim3 block = make_block(context);
        const dim3 cells = make_cells(context, block);
        if (cudaMemcpyAsync(destination, source, static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz) * sizeof(float), cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) return backend_failure;
        for (int iteration = 0; iteration < context.config.diffuse_iterations; ++iteration) {
            diffuse_rbgs_kernel<<<cells, block, 0, context.stream>>>(destination, source, alpha, 0, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
            diffuse_rbgs_kernel<<<cells, block, 0, context.stream>>>(destination, source, alpha, 1, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        }
        return cudaGetLastError() == cudaSuccess ? success : backend_failure;
    }

    StableFluidsResult project_velocity(ContextStorage& context) {
        const dim3 block      = make_block(context);
        const dim3 cells      = make_cells(context, block);
        const auto cell_count = static_cast<std::uint64_t>(context.config.nx) * static_cast<std::uint64_t>(context.config.ny) * static_cast<std::uint64_t>(context.config.nz);
        const auto bytes      = cell_count * sizeof(float);

        stable_fluids::compute_divergence_kernel<<<cells, block, 0, context.stream>>>(context.device.divergence, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
        if (cudaGetLastError() != cudaSuccess) return backend_failure;
        if (bytes > 0 && cudaMemsetAsync(context.device.pressure, 0, bytes, context.stream) != cudaSuccess) return backend_failure;
        const float h2 = context.config.cell_size * context.config.cell_size;
        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, 0, context.config.nx, context.config.ny, context.config.nz, h2, context.config.boundary);
            pressure_rbgs_kernel<<<cells, block, 0, context.stream>>>(context.device.pressure, context.device.divergence, 1, context.config.nx, context.config.ny, context.config.nz, h2, context.config.boundary);
        }
        project_velocity_kernel<<<cells, block, 0, context.stream>>>(
            context.device.temp_velocity_x, context.device.temp_velocity_y, context.device.temp_velocity_z, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.device.pressure, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
        if (cudaGetLastError() != cudaSuccess) return backend_failure;
        std::swap(context.device.velocity_x, context.device.temp_velocity_x);
        std::swap(context.device.velocity_y, context.device.temp_velocity_y);
        std::swap(context.device.velocity_z, context.device.temp_velocity_z);

        if (cudaMemsetAsync(context.device.projection_metrics, 0, sizeof(ProjectionMetricsState), context.stream) != cudaSuccess) return backend_failure;
        accumulate_projection_metrics_kernel<<<cells, block, 0, context.stream>>>(context.device.projection_metrics, context.device.velocity_x, context.device.velocity_y, context.device.velocity_z, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
        return cudaGetLastError() == cudaSuccess ? success : backend_failure;
    }

} // namespace stable_fluids

struct StableFluidsContext_t : stable_fluids::ContextStorage {};

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
        if (index < out_field_handle_capacity) out_field_handles[index] = index + 1u;
    }

    if (const StableFluidsResult code = stable_fluids::allocate_buffers(*context); code != stable_fluids::success) {
        stable_fluids::destroy_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    }

    if (const StableFluidsResult code = stable_fluids::reset_fields(*context); code != stable_fluids::success) {
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
    return stable_fluids::reset_fields(*static_cast<stable_fluids::ContextStorage*>(context));
}

StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    auto& storage          = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block       = stable_fluids::make_block(storage);
    const dim3 cells       = stable_fluids::make_cells(storage, block);
    const auto cell_count  = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    const auto bytes       = cell_count * sizeof(float);

    nvtx3::scoped_range range("stable.step");

    stable_fluids::add_force_kernel<<<cells, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, desc->force_x, desc->force_y, desc->force_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.velocity_x, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_y, storage.device.velocity_y, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_z, storage.device.velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    if (const StableFluidsResult code = stable_fluids::diffuse_component(storage, storage.device.velocity_x, storage.device.temp_velocity_x, storage.config.viscosity); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = stable_fluids::diffuse_component(storage, storage.device.velocity_y, storage.device.temp_velocity_y, storage.config.viscosity); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = stable_fluids::diffuse_component(storage, storage.device.velocity_z, storage.device.temp_velocity_z, storage.config.viscosity); code != stable_fluids::success) return code;

    if (const StableFluidsResult code = stable_fluids::project_velocity(storage); code != stable_fluids::success) return code;

    for (std::size_t field_index = 0; field_index < storage.fields.size(); ++field_index) {
        auto& field         = storage.fields[field_index];
        const float* source = nullptr;
        for (uint32_t source_index = 0; source_index < desc->field_source_count; ++source_index) {
            if (desc->field_sources[source_index].field != static_cast<StableFluidsFieldHandle>(field_index + 1u)) continue;
            source = desc->field_sources[source_index].values;
            break;
        }

        if (source != nullptr) {
            stable_fluids::add_field_source_kernel<<<cells, block, 0, storage.stream>>>(field.data, source, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz);
            if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
        }

        stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(field.temp, field.data, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

        if (const StableFluidsResult code = stable_fluids::diffuse_component(storage, field.data, field.temp, field.desc.diffusion); code != stable_fluids::success) return code;

        if (field.desc.dissipation > 0.0f) {
            const float factor = 1.0f / (1.0f + storage.config.dt * field.desc.dissipation);
            stable_fluids::dissipate_kernel<<<cells, block, 0, storage.stream>>>(field.temp, field.data, factor, storage.config.nx, storage.config.ny, storage.config.nz);
            if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
            std::swap(field.data, field.temp);
        }
    }

    if (bytes > 0 && cudaMemsetAsync(storage.device.divergence, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    stable_fluids::compute_divergence_kernel<<<cells, block, 0, storage.stream>>>(storage.device.divergence, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
}

StableFluidsResult stable_fluids_export_cuda(StableFluidsContext context, const StableFluidsExportDesc* desc, void* destination) {
    auto& storage          = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block       = stable_fluids::make_block(storage);
    const dim3 cells       = stable_fluids::make_cells(storage, block);
    const auto cell_count  = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    const auto scalar_bytes = cell_count * sizeof(float);

    switch (desc->kind) {
    case STABLE_FLUIDS_EXPORT_FIELD: {
        const auto* field = &storage.fields[static_cast<std::size_t>(desc->field - 1u)];
        return cudaMemcpyAsync(destination, field->data, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    }
    case STABLE_FLUIDS_EXPORT_VELOCITY:
        stable_fluids::pack_velocity_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE:
        stable_fluids::velocity_magnitude_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_PRESSURE:
        return cudaMemcpyAsync(destination, storage.device.pressure, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_DIVERGENCE:
        return cudaMemcpyAsync(destination, storage.device.divergence, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    default: return stable_fluids::backend_failure;
    }
}

StableFluidsResult stable_fluids_get_projection_metrics_cuda(StableFluidsContext context, StableFluidsProjectionMetrics* out_metrics) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    stable_fluids::ProjectionMetricsState state{};
    if (cudaMemcpy(&state, storage.device.projection_metrics, sizeof(state), cudaMemcpyDeviceToHost) != cudaSuccess) return stable_fluids::backend_failure;
    out_metrics->max_abs_divergence = state.max_abs_divergence;
    out_metrics->rms_divergence     = state.cell_count > 0 ? std::sqrt(state.sum_sq_divergence / static_cast<float>(state.cell_count)) : 0.0f;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc) {
    const auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    *out_desc = StableFluidsGridDesc{
        .nx = storage.config.nx,
        .ny = storage.config.ny,
        .nz = storage.config.nz,
        .cell_size = storage.config.cell_size,
    };
    return stable_fluids::success;
}

} // extern "C"
