#include "stable-fluids-3d.h"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {
    struct ContextStorage {
        StableFluidsSimulationConfig config{};
        std::vector<const float*> bound_field_sources{};
        cudaStream_t stream = nullptr;
        dim3 block{};
        dim3 cells{};
        std::uint64_t cell_count = 0;
        std::size_t bytes        = 0;
        bool owns_stream         = false;

        struct StepGraphStorage {
            cudaGraph_t graph         = nullptr;
            cudaGraphExec_t exec      = nullptr;
            cudaGraphNode_t add_force = nullptr;
            std::vector<cudaGraphNode_t> add_field_source_nodes{};
        } step_graph{};

        struct DeviceBuffers {
            struct Flow {
                float* velocity_x      = nullptr;
                float* velocity_y      = nullptr;
                float* velocity_z      = nullptr;
                float* temp_velocity_x = nullptr;
                float* temp_velocity_y = nullptr;
                float* temp_velocity_z = nullptr;
                float* pressure        = nullptr;
                float* divergence      = nullptr;
            } flow;

            struct ScalarFields {
                StableFluidsFieldCreateDesc desc{};
                float* data = nullptr;
                float* temp = nullptr;
            };

            std::vector<ScalarFields> scalar_fields{};

        } device;
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

    void destroy_context_graph(ContextStorage& context) {
        if (context.step_graph.exec != nullptr) cudaGraphExecDestroy(context.step_graph.exec);
        if (context.step_graph.graph != nullptr) cudaGraphDestroy(context.step_graph.graph);
        context.step_graph.exec      = nullptr;
        context.step_graph.graph     = nullptr;
        context.step_graph.add_force = nullptr;
        context.step_graph.add_field_source_nodes.clear();
    }

    void destroy_context_buffers(ContextStorage& context) {
        if (context.device.flow.velocity_x != nullptr) cudaFree(context.device.flow.velocity_x);
        if (context.device.flow.velocity_y != nullptr) cudaFree(context.device.flow.velocity_y);
        if (context.device.flow.velocity_z != nullptr) cudaFree(context.device.flow.velocity_z);
        if (context.device.flow.temp_velocity_x != nullptr) cudaFree(context.device.flow.temp_velocity_x);
        if (context.device.flow.temp_velocity_y != nullptr) cudaFree(context.device.flow.temp_velocity_y);
        if (context.device.flow.temp_velocity_z != nullptr) cudaFree(context.device.flow.temp_velocity_z);
        if (context.device.flow.pressure != nullptr) cudaFree(context.device.flow.pressure);
        if (context.device.flow.divergence != nullptr) cudaFree(context.device.flow.divergence);
        context.device.flow.velocity_x      = nullptr;
        context.device.flow.velocity_y      = nullptr;
        context.device.flow.velocity_z      = nullptr;
        context.device.flow.temp_velocity_x = nullptr;
        context.device.flow.temp_velocity_y = nullptr;
        context.device.flow.temp_velocity_z = nullptr;
        context.device.flow.pressure        = nullptr;
        context.device.flow.divergence      = nullptr;
        for (auto& field : context.device.scalar_fields) {
            if (field.data != nullptr) cudaFree(field.data);
            if (field.temp != nullptr) cudaFree(field.temp);
            field.data = nullptr;
            field.temp = nullptr;
        }
    }

} // namespace stable_fluids

extern "C" {

StableFluidsResult stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, void** out_context, StableFluidsFieldHandle* out_field_handles, const uint32_t out_field_handle_capacity) {
    nvtx3::scoped_range range("stable.create_context");
    *out_context = nullptr;
    std::unique_ptr<stable_fluids::ContextStorage> context{new (std::nothrow) stable_fluids::ContextStorage{}};
    if (!context) return STABLE_FLUIDS_RESULT_OUT_OF_MEMORY;
    context->config     = desc->config;
    context->stream     = static_cast<cudaStream_t>(desc->stream);
    context->cell_count = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
    context->bytes      = context->cell_count * sizeof(float);
    context->block      = dim3(static_cast<unsigned>((std::max) (context->config.block_x, 1)), static_cast<unsigned>((std::max) (context->config.block_y, 1)), static_cast<unsigned>((std::max) (context->config.block_z, 1)));
    context->cells      = dim3(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)),
             static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));
    if (context->stream == nullptr) {
        if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        context->owns_stream = true;
    }

    context->device.scalar_fields.reserve(desc->field_count);
    context->bound_field_sources.resize(desc->field_count);
    for (uint32_t index = 0; index < desc->field_count; ++index) {
        context->device.scalar_fields.push_back(stable_fluids::ContextStorage::DeviceBuffers::ScalarFields{
            .desc = desc->fields[index],
        });
        if (index < out_field_handle_capacity) out_field_handles[index] = index + 1u;
    }

    auto fail = [&](const StableFluidsResult code) {
        stable_fluids::destroy_context_graph(*context);
        stable_fluids::destroy_context_buffers(*context);
        if (context->owns_stream) cudaStreamDestroy(context->stream);
        return code;
    };

    {
        nvtx3::scoped_range alloc_range("stable.create_context.alloc");
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_x), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_y), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_z), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_x), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_y), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_z), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.pressure), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.divergence), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        for (auto& field : context->device.scalar_fields) {
            if (cudaMalloc(reinterpret_cast<void**>(&field.data), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
            if (cudaMalloc(reinterpret_cast<void**>(&field.temp), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        }
    }

    {
        nvtx3::scoped_range init_range("stable.create_context.init");
        if (cudaMemsetAsync(context->device.flow.velocity_x, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.velocity_y, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.velocity_z, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.pressure, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.divergence, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        for (auto& field : context->device.scalar_fields) {
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(field.data, field.desc.initial_value, context->config.nx, context->config.ny, context->config.nz);
            if (cudaGetLastError() != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        }
    }

    {
        nvtx3::scoped_range graph_range("stable.create_context.graph");
        if (cudaGraphCreate(&context->step_graph.graph, 0) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        context->step_graph.add_field_source_nodes.resize(context->device.scalar_fields.size());

        auto add_kernel_node = [&](cudaGraphNode_t& node, const cudaGraphNode_t* dependencies, const std::size_t dependency_count, void* const func, void** const args) {
            cudaKernelNodeParams params{
                .func           = func,
                .gridDim        = context->cells,
                .blockDim       = context->block,
                .sharedMemBytes = 0,
                .kernelParams   = args,
                .extra          = nullptr,
            };
            return cudaGraphAddKernelNode(&node, context->step_graph.graph, dependencies, dependency_count, &params) == cudaSuccess;
        };
        auto add_memcpy_node = [&](cudaGraphNode_t& node, const cudaGraphNode_t* dependencies, const std::size_t dependency_count, void* const destination, const void* const source) {
            return cudaGraphAddMemcpyNode1D(&node, context->step_graph.graph, dependencies, dependency_count, destination, source, context->bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
        };
        auto add_zero_node = [&](cudaGraphNode_t& node, const cudaGraphNode_t* dependencies, const std::size_t dependency_count, void* const destination) {
            cudaMemsetParams params{
                .dst         = destination,
                .pitch       = 0,
                .value       = 0u,
                .elementSize = sizeof(float),
                .width       = context->cell_count,
                .height      = 1,
            };
            return cudaGraphAddMemsetNode(&node, context->step_graph.graph, dependencies, dependency_count, &params) == cudaSuccess;
        };
        auto add_diffuse_chain = [&](cudaGraphNode_t& tail, float* const destination, const float* const source, const float diffusion, const cudaGraphNode_t* dependencies, const std::size_t dependency_count) {
            if (!add_memcpy_node(tail, dependencies, dependency_count, destination, source)) return false;
            float alpha = context->config.dt * diffusion / (context->config.cell_size * context->config.cell_size);
            if (alpha <= 0.0f) return true;
            for (int iteration = 0; iteration < context->config.diffuse_iterations; ++iteration) {
                cudaGraphNode_t parity0{};
                cudaGraphNode_t parity1{};
                float* destination_ptr  = destination;
                const float* source_ptr = source;
                int parity0_value       = 0;
                int parity1_value       = 1;
                void* parity0_args[]{&destination_ptr, &source_ptr, &alpha, &parity0_value, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.boundary};
                void* parity1_args[]{&destination_ptr, &source_ptr, &alpha, &parity1_value, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.boundary};
                if (!add_kernel_node(parity0, &tail, 1, reinterpret_cast<void*>(stable_fluids::diffuse_rbgs_kernel), parity0_args)) return false;
                if (!add_kernel_node(parity1, &parity0, 1, reinterpret_cast<void*>(stable_fluids::diffuse_rbgs_kernel), parity1_args)) return false;
                tail = parity1;
            }
            return true;
        };

        {
            cudaGraphNode_t add_force_node{};
            const float* force_x = nullptr;
            const float* force_y = nullptr;
            const float* force_z = nullptr;
            void* add_force_args[]{&context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, reinterpret_cast<void*>(&force_x), reinterpret_cast<void*>(&force_y), reinterpret_cast<void*>(&force_z), &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz};
            if (!add_kernel_node(add_force_node, nullptr, 0, reinterpret_cast<void*>(stable_fluids::add_force_kernel), add_force_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            context->step_graph.add_force = add_force_node;

            cudaGraphNode_t advect_velocity_x{};
            cudaGraphNode_t advect_velocity_y{};
            cudaGraphNode_t advect_velocity_z{};
            void* advect_velocity_x_args[]{&context->device.flow.temp_velocity_x, &context->device.flow.velocity_x, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.boundary};
            void* advect_velocity_y_args[]{&context->device.flow.temp_velocity_y, &context->device.flow.velocity_y, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.boundary};
            void* advect_velocity_z_args[]{&context->device.flow.temp_velocity_z, &context->device.flow.velocity_z, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.boundary};
            if (!add_kernel_node(advect_velocity_x, &add_force_node, 1, reinterpret_cast<void*>(stable_fluids::advect_component_kernel), advect_velocity_x_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (!add_kernel_node(advect_velocity_y, &add_force_node, 1, reinterpret_cast<void*>(stable_fluids::advect_component_kernel), advect_velocity_y_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (!add_kernel_node(advect_velocity_z, &add_force_node, 1, reinterpret_cast<void*>(stable_fluids::advect_component_kernel), advect_velocity_z_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            std::array<cudaGraphNode_t, 3> diffuse_velocity_tails{};
            if (!add_diffuse_chain(diffuse_velocity_tails[0], context->device.flow.velocity_x, context->device.flow.temp_velocity_x, context->config.viscosity, &advect_velocity_x, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (!add_diffuse_chain(diffuse_velocity_tails[1], context->device.flow.velocity_y, context->device.flow.temp_velocity_y, context->config.viscosity, &advect_velocity_y, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (!add_diffuse_chain(diffuse_velocity_tails[2], context->device.flow.velocity_z, context->device.flow.temp_velocity_z, context->config.viscosity, &advect_velocity_z, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t divergence_node{};
            void* divergence_args[]{&context->device.flow.divergence, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.boundary};
            if (!add_kernel_node(divergence_node, diffuse_velocity_tails.data(), diffuse_velocity_tails.size(), reinterpret_cast<void*>(stable_fluids::compute_divergence_kernel), divergence_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t zero_pressure_node{};
            if (!add_zero_node(zero_pressure_node, &divergence_node, 1, context->device.flow.pressure)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t pressure_tail = zero_pressure_node;
            float h2                      = context->config.cell_size * context->config.cell_size;
            for (int iteration = 0; iteration < context->config.pressure_iterations; ++iteration) {
                cudaGraphNode_t parity0{};
                cudaGraphNode_t parity1{};
                int parity0_value = 0;
                int parity1_value = 1;
                void* parity0_args[]{&context->device.flow.pressure, &context->device.flow.divergence, &parity0_value, &context->config.nx, &context->config.ny, &context->config.nz, &h2, &context->config.boundary};
                void* parity1_args[]{&context->device.flow.pressure, &context->device.flow.divergence, &parity1_value, &context->config.nx, &context->config.ny, &context->config.nz, &h2, &context->config.boundary};
                if (!add_kernel_node(parity0, &pressure_tail, 1, reinterpret_cast<void*>(stable_fluids::pressure_rbgs_kernel), parity0_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
                if (!add_kernel_node(parity1, &parity0, 1, reinterpret_cast<void*>(stable_fluids::pressure_rbgs_kernel), parity1_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
                pressure_tail = parity1;
            }

            cudaGraphNode_t project_velocity_node{};
            void* project_velocity_args[]{&context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->device.flow.pressure, &context->config.nx, &context->config.ny, &context->config.nz,
                &context->config.cell_size, &context->config.boundary};
            if (!add_kernel_node(project_velocity_node, &pressure_tail, 1, reinterpret_cast<void*>(stable_fluids::project_velocity_kernel), project_velocity_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t final_divergence_node{};
            if (!add_kernel_node(final_divergence_node, &project_velocity_node, 1, reinterpret_cast<void*>(stable_fluids::compute_divergence_kernel), divergence_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            for (std::size_t field_index = 0; field_index < context->device.scalar_fields.size(); ++field_index) {
                auto& field = context->device.scalar_fields[field_index];
                cudaGraphNode_t add_field_source_node{};
                const float* source = nullptr;
                void* add_field_source_args[]{&field.data, reinterpret_cast<void*>(&source), &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz};
                if (!add_kernel_node(add_field_source_node, &project_velocity_node, 1, reinterpret_cast<void*>(stable_fluids::add_field_source_kernel), add_field_source_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
                context->step_graph.add_field_source_nodes[field_index] = add_field_source_node;

                cudaGraphNode_t advect_field_node{};
                void* advect_field_args[]{&field.temp, &field.data, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.boundary};
                if (!add_kernel_node(advect_field_node, &add_field_source_node, 1, reinterpret_cast<void*>(stable_fluids::advect_component_kernel), advect_field_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

                cudaGraphNode_t field_tail{};
                if (!add_diffuse_chain(field_tail, field.data, field.temp, field.desc.diffusion, &advect_field_node, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
                if (field.desc.dissipation > 0.0f) {
                    float factor = 1.0f / (1.0f + context->config.dt * field.desc.dissipation);
                    cudaGraphNode_t dissipate_node{};
                    void* dissipate_args[]{&field.data, &field.data, &factor, &context->config.nx, &context->config.ny, &context->config.nz};
                    if (!add_kernel_node(dissipate_node, &field_tail, 1, reinterpret_cast<void*>(stable_fluids::dissipate_kernel), dissipate_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
                }
            }
            (void) final_divergence_node;
        }

        if (cudaGraphInstantiate(&context->step_graph.exec, context->step_graph.graph) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
    }

    if (cudaGetLastError() != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

    *out_context = context.release();
    return STABLE_FLUIDS_RESULT_OK;
}

StableFluidsResult stable_fluids_destroy_context_cuda(void* context) {
    nvtx3::scoped_range range("stable.destroy_context");
    auto* storage = static_cast<stable_fluids::ContextStorage*>(context);
    cudaStreamSynchronize(storage->stream);
    stable_fluids::destroy_context_graph(*storage);
    stable_fluids::destroy_context_buffers(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete static_cast<stable_fluids::ContextStorage*>(context);
    return STABLE_FLUIDS_RESULT_OK;
}

StableFluidsResult stable_fluids_step_cuda(void* context, const StableFluidsStepDesc* desc) {
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    nvtx3::scoped_range range("stable.step");

    const float* force_x = desc != nullptr ? desc->force_x : nullptr;
    const float* force_y = desc != nullptr ? desc->force_y : nullptr;
    const float* force_z = desc != nullptr ? desc->force_z : nullptr;

    {
        nvtx3::scoped_range bind_range("stable.step.bind");
        std::fill(storage.bound_field_sources.begin(), storage.bound_field_sources.end(), nullptr);
        if (desc != nullptr) {
            for (uint32_t source_index = 0; source_index < desc->field_source_count; ++source_index) {
                const auto handle = desc->field_sources[source_index].field;
                if (handle == 0 || handle > storage.device.scalar_fields.size()) continue;
                auto& bound_source = storage.bound_field_sources[static_cast<std::size_t>(handle - 1u)];
                if (bound_source == nullptr) bound_source = desc->field_sources[source_index].values;
            }
        }
        if (storage.step_graph.add_force != nullptr) {
            void* add_force_args[]{&storage.device.flow.velocity_x, &storage.device.flow.velocity_y, &storage.device.flow.velocity_z, reinterpret_cast<void*>(&force_x), reinterpret_cast<void*>(&force_y), reinterpret_cast<void*>(&force_z), &storage.config.dt, &storage.config.nx, &storage.config.ny, &storage.config.nz};
            cudaKernelNodeParams params{
                .func           = reinterpret_cast<void*>(stable_fluids::add_force_kernel),
                .gridDim        = storage.cells,
                .blockDim       = storage.block,
                .sharedMemBytes = 0,
                .kernelParams   = add_force_args,
                .extra          = nullptr,
            };
            if (cudaGraphExecKernelNodeSetParams(storage.step_graph.exec, storage.step_graph.add_force, &params) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        }
        for (std::size_t field_index = 0; field_index < storage.device.scalar_fields.size(); ++field_index) {
            const auto node = storage.step_graph.add_field_source_nodes[field_index];
            if (node == nullptr) continue;
            auto& field               = storage.device.scalar_fields[field_index];
            const float* field_source = storage.bound_field_sources[field_index];
            void* add_field_source_args[]{&field.data, reinterpret_cast<void*>(&field_source), &storage.config.dt, &storage.config.nx, &storage.config.ny, &storage.config.nz};
            cudaKernelNodeParams params{
                .func           = reinterpret_cast<void*>(stable_fluids::add_field_source_kernel),
                .gridDim        = storage.cells,
                .blockDim       = storage.block,
                .sharedMemBytes = 0,
                .kernelParams   = add_field_source_args,
                .extra          = nullptr,
            };
            if (cudaGraphExecKernelNodeSetParams(storage.step_graph.exec, node, &params) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        }
    }

    {
        nvtx3::scoped_range launch_range("stable.step.graph_launch");
        return cudaGraphLaunch(storage.step_graph.exec, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    }
}

StableFluidsResult stable_fluids_export_cuda(void* context, const StableFluidsExportDesc* desc, void* destination) {
    nvtx3::scoped_range range("stable.export");
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);

    switch (desc->kind) {
    case STABLE_FLUIDS_EXPORT_FIELD:
        {
            const auto* field = &storage.device.scalar_fields[static_cast<std::size_t>(desc->field - 1u)];
            return cudaMemcpyAsync(destination, field->data, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        }
    case STABLE_FLUIDS_EXPORT_VELOCITY:
        stable_fluids::pack_velocity_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    case STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE:
        stable_fluids::velocity_magnitude_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(static_cast<float*>(destination), storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
        return cudaGetLastError() == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    case STABLE_FLUIDS_EXPORT_PRESSURE: return cudaMemcpyAsync(destination, storage.device.flow.pressure, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    case STABLE_FLUIDS_EXPORT_DIVERGENCE: return cudaMemcpyAsync(destination, storage.device.flow.divergence, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    default: return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    }
}

} // extern "C"
