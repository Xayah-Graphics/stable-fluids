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
        float max_abs_divergence = 0.0f;
        float sum_sq_divergence  = 0.0f;
        uint32_t cell_count      = 0;
        uint32_t _padding        = 0;
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

    __device__ float load(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const StableFluidsBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            if (boundary.x != STABLE_FLUIDS_BOUNDARY_PERIODIC || nx <= 0) return 0.0f;
            x %= nx;
            if (x < 0) x += nx;
        }
        if (y < 0 || y >= ny) {
            if (boundary.y != STABLE_FLUIDS_BOUNDARY_PERIODIC || ny <= 0) return 0.0f;
            y %= ny;
            if (y < 0) y += ny;
        }
        if (z < 0 || z >= nz) {
            if (boundary.z != STABLE_FLUIDS_BOUNDARY_PERIODIC || nz <= 0) return 0.0f;
            z %= nz;
            if (z < 0) z += nz;
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float sample_linear(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const float extent_x = static_cast<float>(nx) * h;
        const float extent_y = static_cast<float>(ny) * h;
        const float extent_z = static_cast<float>(nz) * h;
        if (boundary.x == STABLE_FLUIDS_BOUNDARY_PERIODIC) {
            x = extent_x <= 0.0f ? 0.0f : fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == STABLE_FLUIDS_BOUNDARY_PERIODIC) {
            y = extent_y <= 0.0f ? 0.0f : fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == STABLE_FLUIDS_BOUNDARY_PERIODIC) {
            z = extent_z <= 0.0f ? 0.0f : fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

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

        const float c000 = load(field, x0, y0, z0, nx, ny, nz, boundary);
        const float c100 = load(field, x1, y0, z0, nx, ny, nz, boundary);
        const float c010 = load(field, x0, y1, z0, nx, ny, nz, boundary);
        const float c110 = load(field, x1, y1, z0, nx, ny, nz, boundary);
        const float c001 = load(field, x0, y0, z1, nx, ny, nz, boundary);
        const float c101 = load(field, x1, y0, z1, nx, ny, nz, boundary);
        const float c011 = load(field, x0, y1, z1, nx, ny, nz, boundary);
        const float c111 = load(field, x1, y1, z1, nx, ny, nz, boundary);

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
        const float px                         = (static_cast<float>(x) + 0.5f) * h;
        const float py                         = (static_cast<float>(y) + 0.5f) * h;
        const float pz                         = (static_cast<float>(z) + 0.5f) * h;
        const float3 traced                    = trace_particle_rk2(px, py, pz, velocity_x, velocity_y, velocity_z, dt, nx, ny, nz, h, boundary);
        destination[index_3d(x, y, z, nx, ny)] = sample_linear(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void diffuse_rbgs_kernel(float* destination, const float* source, const float alpha, const int parity, const int nx, const int ny, const int nz, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors = load(destination, x - 1, y, z, nx, ny, nz, boundary) + load(destination, x + 1, y, z, nx, ny, nz, boundary) + load(destination, x, y - 1, z, nx, ny, nz, boundary) + load(destination, x, y + 1, z, nx, ny, nz, boundary) + load(destination, x, y, z - 1, nx, ny, nz, boundary) + load(destination, x, y, z + 1, nx, ny, nz, boundary);
        const auto index      = index_3d(x, y, z, nx, ny);
        destination[index]    = (source[index] + alpha * neighbors) / (1.0f + 6.0f * alpha);
    }

    __global__ void dissipate_kernel(float* destination, const float* source, const float factor, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index   = index_3d(x, y, z, nx, ny);
        destination[index] = source[index] * factor;
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h                    = 0.5f / h;
        const float ddx                       = (load(velocity_x, x + 1, y, z, nx, ny, nz, boundary) - load(velocity_x, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddy                       = (load(velocity_y, x, y + 1, z, nx, ny, nz, boundary) - load(velocity_y, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddz                       = (load(velocity_z, x, y, z + 1, nx, ny, nz, boundary) - load(velocity_z, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        divergence[index_3d(x, y, z, nx, ny)] = ddx + ddy + ddz;
    }

    __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const int parity, const int nx, const int ny, const int nz, const float h2, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors = load(pressure, x - 1, y, z, nx, ny, nz, boundary) + load(pressure, x + 1, y, z, nx, ny, nz, boundary) + load(pressure, x, y - 1, z, nx, ny, nz, boundary) + load(pressure, x, y + 1, z, nx, ny, nz, boundary) + load(pressure, x, y, z - 1, nx, ny, nz, boundary) + load(pressure, x, y, z + 1, nx, ny, nz, boundary);
        const auto index      = index_3d(x, y, z, nx, ny);
        pressure[index]       = (neighbors - h2 * divergence[index]) / 6.0f;
    }

    __global__ void project_velocity_kernel(float* destination_x, float* destination_y, float* destination_z, const float* source_x, const float* source_y, const float* source_z, const float* pressure, const int nx, const int ny, const int nz, const float h, const StableFluidsBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h   = 0.5f / h;
        const auto index     = index_3d(x, y, z, nx, ny);
        const float grad_x   = (load(pressure, x + 1, y, z, nx, ny, nz, boundary) - load(pressure, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_y   = (load(pressure, x, y + 1, z, nx, ny, nz, boundary) - load(pressure, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_z   = (load(pressure, x, y, z + 1, nx, ny, nz, boundary) - load(pressure, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
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
        const float ddx    = (load(velocity_x, x + 1, y, z, nx, ny, nz, boundary) - load(velocity_x, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddy    = (load(velocity_y, x, y + 1, z, nx, ny, nz, boundary) - load(velocity_y, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddz    = (load(velocity_z, x, y, z + 1, nx, ny, nz, boundary) - load(velocity_z, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        const float value  = fabsf(ddx + ddy + ddz);
        auto* bits         = reinterpret_cast<unsigned int*>(&metrics->max_abs_divergence);
        unsigned prev      = *bits;
        while (__uint_as_float(prev) < value) {
            const unsigned next = atomicCAS(bits, prev, __float_as_uint(value));
            if (next == prev) break;
            prev = next;
        }
        atomicAdd(&metrics->sum_sq_divergence, value * value);
        atomicAdd(&metrics->cell_count, 1u);
    }

    __global__ void pack_velocity_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index                     = index_3d(x, y, z, nx, ny);
        const auto cell_count                = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        destination[index]                   = velocity_x[index];
        destination[cell_count + index]      = velocity_y[index];
        destination[cell_count * 2u + index] = velocity_z[index];
    }

    __global__ void velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index   = index_3d(x, y, z, nx, ny);
        const float vx     = velocity_x[index];
        const float vy     = velocity_y[index];
        const float vz     = velocity_z[index];
        destination[index] = sqrtf(vx * vx + vy * vy + vz * vz);
    }

    void destroy_context_buffers(ContextStorage& context) {
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

    const auto cell_count        = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
    const auto bytes             = cell_count * sizeof(float);
    auto destroy_context_buffers = [&]() {
        if (context->device.velocity_x != nullptr) cudaFree(context->device.velocity_x);
        if (context->device.velocity_y != nullptr) cudaFree(context->device.velocity_y);
        if (context->device.velocity_z != nullptr) cudaFree(context->device.velocity_z);
        if (context->device.temp_velocity_x != nullptr) cudaFree(context->device.temp_velocity_x);
        if (context->device.temp_velocity_y != nullptr) cudaFree(context->device.temp_velocity_y);
        if (context->device.temp_velocity_z != nullptr) cudaFree(context->device.temp_velocity_z);
        if (context->device.pressure != nullptr) cudaFree(context->device.pressure);
        if (context->device.divergence != nullptr) cudaFree(context->device.divergence);
        if (context->device.projection_metrics != nullptr) cudaFree(context->device.projection_metrics);
        for (auto& field : context->fields) {
            if (field.data != nullptr) cudaFree(field.data);
            if (field.temp != nullptr) cudaFree(field.temp);
            field.data = nullptr;
            field.temp = nullptr;
        }
    };

    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.velocity_x), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.velocity_y), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.velocity_z), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.temp_velocity_x), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.temp_velocity_y), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.temp_velocity_z), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.pressure), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&context->device.divergence), bytes) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&context->device.projection_metrics), sizeof(stable_fluids::ProjectionMetricsState)) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::out_of_memory;
    }
    for (auto& field : context->fields) {
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&field.data), bytes) != cudaSuccess) {
            destroy_context_buffers();
            if (context->owns_stream) cudaStreamDestroy(context->stream);
            return stable_fluids::out_of_memory;
        }
        if (cell_count > 0 && cudaMalloc(reinterpret_cast<void**>(&field.temp), bytes) != cudaSuccess) {
            destroy_context_buffers();
            if (context->owns_stream) cudaStreamDestroy(context->stream);
            return stable_fluids::out_of_memory;
        }
    }

    const dim3 block(static_cast<unsigned>((std::max) (context->config.block_x, 1)), static_cast<unsigned>((std::max) (context->config.block_y, 1)), static_cast<unsigned>((std::max) (context->config.block_z, 1)));
    const dim3 cells(static_cast<unsigned>((context->config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((context->config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    if (bytes > 0 && cudaMemsetAsync(context->device.velocity_x, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.velocity_y, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.velocity_z, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.temp_velocity_x, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.temp_velocity_y, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.temp_velocity_z, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.pressure, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (bytes > 0 && cudaMemsetAsync(context->device.divergence, 0, bytes, context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    if (cudaMemsetAsync(context->device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), context->stream) != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }
    for (auto& field : context->fields) {
        stable_fluids::fill_kernel<<<cells, block, 0, context->stream>>>(field.data, field.desc.initial_value, context->config.nx, context->config.ny, context->config.nz);
        stable_fluids::fill_kernel<<<cells, block, 0, context->stream>>>(field.temp, field.desc.initial_value, context->config.nx, context->config.ny, context->config.nz);
        if (cudaGetLastError() != cudaSuccess) {
            destroy_context_buffers();
            if (context->owns_stream) cudaStreamDestroy(context->stream);
            return stable_fluids::backend_failure;
        }
    }

    if (cudaGetLastError() != cudaSuccess) {
        destroy_context_buffers();
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return stable_fluids::backend_failure;
    }

    *out_context = context.release();
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_destroy_context_cuda(StableFluidsContext context) {
    auto* storage = static_cast<stable_fluids::ContextStorage*>(context);
    cudaStreamSynchronize(storage->stream);
    if (storage->device.velocity_x != nullptr) cudaFree(storage->device.velocity_x);
    if (storage->device.velocity_y != nullptr) cudaFree(storage->device.velocity_y);
    if (storage->device.velocity_z != nullptr) cudaFree(storage->device.velocity_z);
    if (storage->device.temp_velocity_x != nullptr) cudaFree(storage->device.temp_velocity_x);
    if (storage->device.temp_velocity_y != nullptr) cudaFree(storage->device.temp_velocity_y);
    if (storage->device.temp_velocity_z != nullptr) cudaFree(storage->device.temp_velocity_z);
    if (storage->device.pressure != nullptr) cudaFree(storage->device.pressure);
    if (storage->device.divergence != nullptr) cudaFree(storage->device.divergence);
    if (storage->device.projection_metrics != nullptr) cudaFree(storage->device.projection_metrics);
    for (auto& field : storage->fields) {
        if (field.data != nullptr) cudaFree(field.data);
        if (field.temp != nullptr) cudaFree(field.temp);
    }
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete context;
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_reset_context_cuda(StableFluidsContext context) {
    auto& storage         = *static_cast<stable_fluids::ContextStorage*>(context);
    const auto cell_count = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    const auto bytes      = cell_count * sizeof(float);
    const dim3 block(static_cast<unsigned>((std::max) (storage.config.block_x, 1)), static_cast<unsigned>((std::max) (storage.config.block_y, 1)), static_cast<unsigned>((std::max) (storage.config.block_z, 1)));
    const dim3 cells(static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    if (bytes > 0 && cudaMemsetAsync(storage.device.velocity_x, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.velocity_y, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.velocity_z, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.temp_velocity_x, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.temp_velocity_y, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.temp_velocity_z, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.pressure, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.divergence, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    if (cudaMemsetAsync(storage.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    for (auto& field : storage.fields) {
        stable_fluids::fill_kernel<<<cells, block, 0, storage.stream>>>(field.data, field.desc.initial_value, storage.config.nx, storage.config.ny, storage.config.nz);
        stable_fluids::fill_kernel<<<cells, block, 0, storage.stream>>>(field.temp, field.desc.initial_value, storage.config.nx, storage.config.ny, storage.config.nz);
        if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    }
    return stable_fluids::success;
}

StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(static_cast<unsigned>((std::max) (storage.config.block_x, 1)), static_cast<unsigned>((std::max) (storage.config.block_y, 1)), static_cast<unsigned>((std::max) (storage.config.block_z, 1)));
    const dim3 cells(static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    const auto cell_count = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    const auto bytes      = cell_count * sizeof(float);

    nvtx3::scoped_range range("stable.step");

    stable_fluids::add_force_kernel<<<cells, block, 0, storage.stream>>>(storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, desc->force_x, desc->force_y, desc->force_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_x, storage.device.velocity_x, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_y, storage.device.velocity_y, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    stable_fluids::advect_component_kernel<<<cells, block, 0, storage.stream>>>(storage.device.temp_velocity_z, storage.device.velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.dt, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

    auto diffuse_component = [&](float* destination, const float* source, const float diffusion) {
        const float alpha = storage.config.dt * diffusion / (storage.config.cell_size * storage.config.cell_size);
        if (alpha <= 0.0f) {
            if (cell_count == 0) return stable_fluids::success;
            return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
        }
        if (cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
        for (int iteration = 0; iteration < storage.config.diffuse_iterations; ++iteration) {
            stable_fluids::diffuse_rbgs_kernel<<<cells, block, 0, storage.stream>>>(destination, source, alpha, 0, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.boundary);
            stable_fluids::diffuse_rbgs_kernel<<<cells, block, 0, storage.stream>>>(destination, source, alpha, 1, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.boundary);
        }
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    };

    if (const StableFluidsResult code = diffuse_component(storage.device.velocity_x, storage.device.temp_velocity_x, storage.config.viscosity); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = diffuse_component(storage.device.velocity_y, storage.device.temp_velocity_y, storage.config.viscosity); code != stable_fluids::success) return code;
    if (const StableFluidsResult code = diffuse_component(storage.device.velocity_z, storage.device.temp_velocity_z, storage.config.viscosity); code != stable_fluids::success) return code;

    stable_fluids::compute_divergence_kernel<<<cells, block, 0, storage.stream>>>(storage.device.divergence, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    if (bytes > 0 && cudaMemsetAsync(storage.device.pressure, 0, bytes, storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    const float h2 = storage.config.cell_size * storage.config.cell_size;
    for (int iteration = 0; iteration < storage.config.pressure_iterations; ++iteration) {
        stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, storage.stream>>>(storage.device.pressure, storage.device.divergence, 0, storage.config.nx, storage.config.ny, storage.config.nz, h2, storage.config.boundary);
        stable_fluids::pressure_rbgs_kernel<<<cells, block, 0, storage.stream>>>(storage.device.pressure, storage.device.divergence, 1, storage.config.nx, storage.config.ny, storage.config.nz, h2, storage.config.boundary);
    }
    stable_fluids::project_velocity_kernel<<<cells, block, 0, storage.stream>>>(
        storage.device.temp_velocity_x, storage.device.temp_velocity_y, storage.device.temp_velocity_z, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.device.pressure, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;
    std::swap(storage.device.velocity_x, storage.device.temp_velocity_x);
    std::swap(storage.device.velocity_y, storage.device.temp_velocity_y);
    std::swap(storage.device.velocity_z, storage.device.temp_velocity_z);
    if (cudaMemsetAsync(storage.device.projection_metrics, 0, sizeof(stable_fluids::ProjectionMetricsState), storage.stream) != cudaSuccess) return stable_fluids::backend_failure;
    stable_fluids::accumulate_projection_metrics_kernel<<<cells, block, 0, storage.stream>>>(storage.device.projection_metrics, storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
    if (cudaGetLastError() != cudaSuccess) return stable_fluids::backend_failure;

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

        if (const StableFluidsResult code = diffuse_component(field.data, field.temp, field.desc.diffusion); code != stable_fluids::success) return code;

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
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    const dim3 block(static_cast<unsigned>((std::max) (storage.config.block_x, 1)), static_cast<unsigned>((std::max) (storage.config.block_y, 1)), static_cast<unsigned>((std::max) (storage.config.block_z, 1)));
    const dim3 cells(static_cast<unsigned>((storage.config.nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((storage.config.ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((storage.config.nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    const auto cell_count   = static_cast<std::uint64_t>(storage.config.nx) * static_cast<std::uint64_t>(storage.config.ny) * static_cast<std::uint64_t>(storage.config.nz);
    const auto scalar_bytes = cell_count * sizeof(float);

    switch (desc->kind) {
    case STABLE_FLUIDS_EXPORT_FIELD:
        {
            const auto* field = &storage.fields[static_cast<std::size_t>(desc->field - 1u)];
            return cudaMemcpyAsync(destination, field->data, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
        }
    case STABLE_FLUIDS_EXPORT_VELOCITY:
        stable_fluids::pack_velocity_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE:
        stable_fluids::velocity_magnitude_kernel<<<cells, block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.velocity_x, storage.device.velocity_y, storage.device.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_PRESSURE: return cudaMemcpyAsync(destination, storage.device.pressure, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
    case STABLE_FLUIDS_EXPORT_DIVERGENCE: return cudaMemcpyAsync(destination, storage.device.divergence, scalar_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? stable_fluids::success : stable_fluids::backend_failure;
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
    *out_desc           = StableFluidsGridDesc{
                  .nx        = storage.config.nx,
                  .ny        = storage.config.ny,
                  .nz        = storage.config.nz,
                  .cell_size = storage.config.cell_size,
    };
    return stable_fluids::success;
}

} // extern "C"
