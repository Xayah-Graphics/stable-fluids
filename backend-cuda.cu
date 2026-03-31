#include "stable-fluids-3d.h"
#include <algorithm>
#include <array>
#include <cuda/std/algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace stable_fluids {
    struct ContextStorage {
        StableFluidsSimulationConfig config{};
        cudaStream_t stream = nullptr;
        dim3 block{};
        dim3 cells{};
        std::uint64_t cell_count = 0;
        std::size_t bytes        = 0;
        bool owns_stream         = false;

        struct StepGraphStorage {
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
        } step_graph{};

        struct DeviceBuffers {
            struct Flow {
                float* velocity_x         = nullptr;
                float* velocity_y         = nullptr;
                float* velocity_z         = nullptr;
                float* temp_velocity_x    = nullptr;
                float* temp_velocity_y    = nullptr;
                float* temp_velocity_z    = nullptr;
                float* pressure           = nullptr;
                float* divergence         = nullptr;
                float* velocity_magnitude = nullptr;
            } flow;

            struct ScalarField {
                StableFluidsScalarFieldDesc desc{};
                float* data   = nullptr;
                float* temp   = nullptr;
                float* source = nullptr;
            };

            struct VectorField {
                StableFluidsVectorFieldDesc desc{};
                float* data_x = nullptr;
                float* data_y = nullptr;
                float* data_z = nullptr;
            };

            std::vector<ScalarField> scalar_fields{};
            std::vector<VectorField> vector_fields{};

        } device;
    };

    __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    bool is_valid_flow_boundary_type(const uint32_t type) {
        return type == STABLE_FLUIDS_FLOW_BOUNDARY_NO_SLIP_WALL || type == STABLE_FLUIDS_FLOW_BOUNDARY_FREE_SLIP_WALL || type == STABLE_FLUIDS_FLOW_BOUNDARY_INFLOW || type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW || type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC;
    }

    bool is_valid_scalar_boundary_type(const uint32_t type) {
        return type == STABLE_FLUIDS_SCALAR_BOUNDARY_FIXED_VALUE || type == STABLE_FLUIDS_SCALAR_BOUNDARY_ZERO_FLUX || type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC;
    }

    __device__ float load_scalar(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const StableFluidsScalarBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            const auto [type, value] = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && nx > 0) {
                x %= nx;
                if (x < 0) x += nx;
            } else if (type == STABLE_FLUIDS_SCALAR_BOUNDARY_ZERO_FLUX && nx > 0)
                x = x < 0 ? 0 : nx - 1;
            else
                return value;
        }
        if (y < 0 || y >= ny) {
            const auto [type, value] = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && ny > 0) {
                y %= ny;
                if (y < 0) y += ny;
            } else if (type == STABLE_FLUIDS_SCALAR_BOUNDARY_ZERO_FLUX && ny > 0)
                y = y < 0 ? 0 : ny - 1;
            else
                return value;
        }
        if (z < 0 || z >= nz) {
            const auto [type, value] = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && nz > 0) {
                z %= nz;
                if (z < 0) z += nz;
            } else if (type == STABLE_FLUIDS_SCALAR_BOUNDARY_ZERO_FLUX && nz > 0)
                z = z < 0 ? 0 : nz - 1;
            else
                return value;
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_pressure(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const StableFluidsFlowBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            const auto face = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                x %= nx;
                if (x < 0) x += nx;
            } else if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW)
                return face.pressure;
            else
                x = x < 0 ? 0 : nx - 1;
        }
        if (y < 0 || y >= ny) {
            const auto face = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                y %= ny;
                if (y < 0) y += ny;
            } else if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW)
                return face.pressure;
            else
                y = y < 0 ? 0 : ny - 1;
        }
        if (z < 0 || z >= nz) {
            const auto face = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                z %= nz;
                if (z < 0) z += nz;
            } else if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW)
                return face.pressure;
            else
                z = z < 0 ? 0 : nz - 1;
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_velocity_component(const float* field, const int component_axis, int x, int y, int z, const int nx, const int ny, const int nz, const StableFluidsFlowBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            const auto face = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                x %= nx;
                if (x < 0) x += nx;
            } else {
                const float interior = field[index_3d(x < 0 ? 0 : nx - 1, cuda::std::clamp(y, 0, ny - 1), cuda::std::clamp(z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                switch (component_axis) {
                case 0: prescribed = face.velocity_x; break;
                case 1: prescribed = face.velocity_y; break;
                case 2: prescribed = face.velocity_z; break;
                default: asm("trap;");
                }
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 0) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (y < 0 || y >= ny) {
            const auto face = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                y %= ny;
                if (y < 0) y += ny;
            } else {
                const float interior = field[index_3d(cuda::std::clamp(x, 0, nx - 1), y < 0 ? 0 : ny - 1, cuda::std::clamp(z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                switch (component_axis) {
                case 0: prescribed = face.velocity_x; break;
                case 1: prescribed = face.velocity_y; break;
                case 2: prescribed = face.velocity_z; break;
                default: asm("trap;");
                }
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 1) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (z < 0 || z >= nz) {
            const auto face = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                z %= nz;
                if (z < 0) z += nz;
            } else {
                const float interior = field[index_3d(cuda::std::clamp(x, 0, nx - 1), cuda::std::clamp(y, 0, ny - 1), z < 0 ? 0 : nz - 1, nx, ny)];
                float prescribed     = 0.0f;
                switch (component_axis) {
                case 0: prescribed = face.velocity_x; break;
                case 1: prescribed = face.velocity_y; break;
                case 2: prescribed = face.velocity_z; break;
                default: asm("trap;");
                }
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == STABLE_FLUIDS_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 2) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float sample_scalar_linear(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const StableFluidsScalarBoundaryConfig boundary) {
        const float extent_x = static_cast<float>(nx) * h;
        const float extent_y = static_cast<float>(ny) * h;
        const float extent_z = static_cast<float>(nz) * h;
        if (boundary.x_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC) {
            x = extent_x <= 0.0f ? 0.0f : fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC) {
            y = extent_y <= 0.0f ? 0.0f : fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC) {
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

        const float c000 = load_scalar(field, x0, y0, z0, nx, ny, nz, boundary);
        const float c100 = load_scalar(field, x1, y0, z0, nx, ny, nz, boundary);
        const float c010 = load_scalar(field, x0, y1, z0, nx, ny, nz, boundary);
        const float c110 = load_scalar(field, x1, y1, z0, nx, ny, nz, boundary);
        const float c001 = load_scalar(field, x0, y0, z1, nx, ny, nz, boundary);
        const float c101 = load_scalar(field, x1, y0, z1, nx, ny, nz, boundary);
        const float c011 = load_scalar(field, x0, y1, z1, nx, ny, nz, boundary);
        const float c111 = load_scalar(field, x1, y1, z1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_velocity_linear(const float* field, const int component_axis, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        const float extent_x = static_cast<float>(nx) * h;
        const float extent_y = static_cast<float>(ny) * h;
        const float extent_z = static_cast<float>(nz) * h;
        if (boundary.x_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC) {
            x = extent_x <= 0.0f ? 0.0f : fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC) {
            y = extent_y <= 0.0f ? 0.0f : fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC) {
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

        const float c000 = load_velocity_component(field, component_axis, x0, y0, z0, nx, ny, nz, boundary);
        const float c100 = load_velocity_component(field, component_axis, x1, y0, z0, nx, ny, nz, boundary);
        const float c010 = load_velocity_component(field, component_axis, x0, y1, z0, nx, ny, nz, boundary);
        const float c110 = load_velocity_component(field, component_axis, x1, y1, z0, nx, ny, nz, boundary);
        const float c001 = load_velocity_component(field, component_axis, x0, y0, z1, nx, ny, nz, boundary);
        const float c101 = load_velocity_component(field, component_axis, x1, y0, z1, nx, ny, nz, boundary);
        const float c011 = load_velocity_component(field, component_axis, x0, y1, z1, nx, ny, nz, boundary);
        const float c111 = load_velocity_component(field, component_axis, x1, y1, z1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        return make_float3(sample_velocity_linear(velocity_x, 0, x, y, z, nx, ny, nz, h, boundary), sample_velocity_linear(velocity_y, 1, x, y, z, nx, ny, nz, h, boundary), sample_velocity_linear(velocity_z, 2, x, y, z, nx, ny, nz, h, boundary));
    }

    __device__ float3 trace_particle_rk2(const float x, const float y, const float z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float dt, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        const auto [v0_x, v0_y, v0_z] = sample_velocity(velocity_x, velocity_y, velocity_z, x, y, z, nx, ny, nz, h, boundary);
        const auto [mid_x, mid_y, mid_z]        = make_float3(x - 0.5f * dt * v0_x, y - 0.5f * dt * v0_y, z - 0.5f * dt * v0_z);
        const auto [v1_x, v1_y, v1_z] = sample_velocity(velocity_x, velocity_y, velocity_z, mid_x, mid_y, mid_z, nx, ny, nz, h, boundary);
        return make_float3(x - dt * v1_x, y - dt * v1_y, z - dt * v1_z);
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

    __global__ void advect_velocity_component_kernel(float* destination, const float* source, const int component_axis, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float dt, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float px                         = (static_cast<float>(x) + 0.5f) * h;
        const float py                         = (static_cast<float>(y) + 0.5f) * h;
        const float pz                         = (static_cast<float>(z) + 0.5f) * h;
        const float3 traced                    = trace_particle_rk2(px, py, pz, velocity_x, velocity_y, velocity_z, dt, nx, ny, nz, h, boundary);
        destination[index_3d(x, y, z, nx, ny)] = sample_velocity_linear(source, component_axis, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_scalar_component_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const float dt, const int nx, const int ny, const int nz, const float h, const StableFluidsScalarBoundaryConfig scalar_boundary, const StableFluidsFlowBoundaryConfig flow_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float px                         = (static_cast<float>(x) + 0.5f) * h;
        const float py                         = (static_cast<float>(y) + 0.5f) * h;
        const float pz                         = (static_cast<float>(z) + 0.5f) * h;
        const float3 traced                    = trace_particle_rk2(px, py, pz, velocity_x, velocity_y, velocity_z, dt, nx, ny, nz, h, flow_boundary);
        destination[index_3d(x, y, z, nx, ny)] = sample_scalar_linear(source, traced.x, traced.y, traced.z, nx, ny, nz, h, scalar_boundary);
    }

    __global__ void diffuse_velocity_rbgs_kernel(float* destination, const float* source, const float alpha, const int parity, const int component_axis, const int nx, const int ny, const int nz, const StableFluidsFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors = load_velocity_component(destination, component_axis, x - 1, y, z, nx, ny, nz, boundary) + load_velocity_component(destination, component_axis, x + 1, y, z, nx, ny, nz, boundary) + load_velocity_component(destination, component_axis, x, y - 1, z, nx, ny, nz, boundary)
                              + load_velocity_component(destination, component_axis, x, y + 1, z, nx, ny, nz, boundary) + load_velocity_component(destination, component_axis, x, y, z - 1, nx, ny, nz, boundary) + load_velocity_component(destination, component_axis, x, y, z + 1, nx, ny, nz, boundary);
        const auto index   = index_3d(x, y, z, nx, ny);
        destination[index] = (source[index] + alpha * neighbors) / (1.0f + 6.0f * alpha);
    }

    __global__ void diffuse_scalar_rbgs_kernel(float* destination, const float* source, const float alpha, const int parity, const int nx, const int ny, const int nz, const StableFluidsScalarBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors = load_scalar(destination, x - 1, y, z, nx, ny, nz, boundary) + load_scalar(destination, x + 1, y, z, nx, ny, nz, boundary) + load_scalar(destination, x, y - 1, z, nx, ny, nz, boundary) + load_scalar(destination, x, y + 1, z, nx, ny, nz, boundary) + load_scalar(destination, x, y, z - 1, nx, ny, nz, boundary)
                              + load_scalar(destination, x, y, z + 1, nx, ny, nz, boundary);
        const auto index   = index_3d(x, y, z, nx, ny);
        destination[index] = (source[index] + alpha * neighbors) / (1.0f + 6.0f * alpha);
    }

    __global__ void dissipate_kernel(float* destination, const float* source, const float factor, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index   = index_3d(x, y, z, nx, ny);
        destination[index] = source[index] * factor;
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h                    = 0.5f / h;
        const float ddx                       = (load_velocity_component(velocity_x, 0, x + 1, y, z, nx, ny, nz, boundary) - load_velocity_component(velocity_x, 0, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddy                       = (load_velocity_component(velocity_y, 1, x, y + 1, z, nx, ny, nz, boundary) - load_velocity_component(velocity_y, 1, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float ddz                       = (load_velocity_component(velocity_z, 2, x, y, z + 1, nx, ny, nz, boundary) - load_velocity_component(velocity_z, 2, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        divergence[index_3d(x, y, z, nx, ny)] = ddx + ddy + ddz;
    }

    __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, const int parity, const int nx, const int ny, const int nz, const float h2, const StableFluidsFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;
        const float neighbors = load_pressure(pressure, x - 1, y, z, nx, ny, nz, boundary) + load_pressure(pressure, x + 1, y, z, nx, ny, nz, boundary) + load_pressure(pressure, x, y - 1, z, nx, ny, nz, boundary) + load_pressure(pressure, x, y + 1, z, nx, ny, nz, boundary) + load_pressure(pressure, x, y, z - 1, nx, ny, nz, boundary)
                              + load_pressure(pressure, x, y, z + 1, nx, ny, nz, boundary);
        const auto index = index_3d(x, y, z, nx, ny);
        pressure[index]  = (neighbors - h2 * divergence[index]) / 6.0f;
    }

    __global__ void project_velocity_kernel(float* destination_x, float* destination_y, float* destination_z, const float* source_x, const float* source_y, const float* source_z, const float* pressure, const int nx, const int ny, const int nz, const float h, const StableFluidsFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const float inv_2h   = 0.5f / h;
        const auto index     = index_3d(x, y, z, nx, ny);
        const float grad_x   = (load_pressure(pressure, x + 1, y, z, nx, ny, nz, boundary) - load_pressure(pressure, x - 1, y, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_y   = (load_pressure(pressure, x, y + 1, z, nx, ny, nz, boundary) - load_pressure(pressure, x, y - 1, z, nx, ny, nz, boundary)) * inv_2h;
        const float grad_z   = (load_pressure(pressure, x, y, z + 1, nx, ny, nz, boundary) - load_pressure(pressure, x, y, z - 1, nx, ny, nz, boundary)) * inv_2h;
        destination_x[index] = source_x[index] - grad_x;
        destination_y[index] = source_y[index] - grad_y;
        destination_z[index] = source_z[index] - grad_z;
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

    void destroy_context_resources(ContextStorage& context) {
        if (context.step_graph.exec != nullptr) cudaGraphExecDestroy(context.step_graph.exec);
        if (context.step_graph.graph != nullptr) cudaGraphDestroy(context.step_graph.graph);
        context.step_graph.exec  = nullptr;
        context.step_graph.graph = nullptr;
        if (context.device.flow.velocity_x != nullptr) cudaFree(context.device.flow.velocity_x);
        if (context.device.flow.velocity_y != nullptr) cudaFree(context.device.flow.velocity_y);
        if (context.device.flow.velocity_z != nullptr) cudaFree(context.device.flow.velocity_z);
        if (context.device.flow.temp_velocity_x != nullptr) cudaFree(context.device.flow.temp_velocity_x);
        if (context.device.flow.temp_velocity_y != nullptr) cudaFree(context.device.flow.temp_velocity_y);
        if (context.device.flow.temp_velocity_z != nullptr) cudaFree(context.device.flow.temp_velocity_z);
        if (context.device.flow.pressure != nullptr) cudaFree(context.device.flow.pressure);
        if (context.device.flow.divergence != nullptr) cudaFree(context.device.flow.divergence);
        if (context.device.flow.velocity_magnitude != nullptr) cudaFree(context.device.flow.velocity_magnitude);
        context.device.flow.velocity_x         = nullptr;
        context.device.flow.velocity_y         = nullptr;
        context.device.flow.velocity_z         = nullptr;
        context.device.flow.temp_velocity_x    = nullptr;
        context.device.flow.temp_velocity_y    = nullptr;
        context.device.flow.temp_velocity_z    = nullptr;
        context.device.flow.pressure           = nullptr;
        context.device.flow.divergence         = nullptr;
        context.device.flow.velocity_magnitude = nullptr;
        for (auto& field : context.device.scalar_fields) {
            if (field.data != nullptr) cudaFree(field.data);
            if (field.temp != nullptr) cudaFree(field.temp);
            if (field.source != nullptr) cudaFree(field.source);
            field.data   = nullptr;
            field.temp   = nullptr;
            field.source = nullptr;
        }
        for (auto& field : context.device.vector_fields) {
            if (field.data_x != nullptr) cudaFree(field.data_x);
            if (field.data_y != nullptr) cudaFree(field.data_y);
            if (field.data_z != nullptr) cudaFree(field.data_z);
            field.data_x = nullptr;
            field.data_y = nullptr;
            field.data_z = nullptr;
        }
    }

} // namespace stable_fluids

extern "C" {

StableFluidsResult stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, void** out_context, StableFluidsScalarFieldHandle* out_scalar_field_handles, const uint32_t out_scalar_field_handle_capacity, StableFluidsVectorFieldHandle* out_vector_field_handles, const uint32_t out_vector_field_handle_capacity) {
    nvtx3::scoped_range range("stable.create_context");
    *out_context = nullptr;
    std::unique_ptr<stable_fluids::ContextStorage> context{new (std::nothrow) stable_fluids::ContextStorage{}};
    if (!context) return STABLE_FLUIDS_RESULT_OUT_OF_MEMORY;
    context->config     = desc->config;
    context->stream     = static_cast<cudaStream_t>(desc->stream);
    context->cell_count = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
    context->bytes      = context->cell_count * sizeof(float);
    auto choose_block   = [&]() {
        int min_grid_size = 0;
        int block_size    = 0;
        if (cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, stable_fluids::advect_velocity_component_kernel, 0, 0) != cudaSuccess) return dim3(8u, 8u, 4u);
        if (block_size <= 0) return dim3(8u, 8u, 4u);
        unsigned block_z = block_size >= 256 ? 4u : block_size >= 128 ? 2u : 1u;
        unsigned block_y = block_size / static_cast<int>(block_z) >= 64 ? 8u : block_size / static_cast<int>(block_z) >= 32 ? 4u : 2u;
        unsigned block_x = static_cast<unsigned>((std::max) (block_size / static_cast<int>(block_y * block_z), 1));
        if (block_x > 16u) block_x = 16u;
        while (block_x * block_y * block_z > static_cast<unsigned>(block_size)) {
            if (block_x >= block_y && block_x > 1u) {
                --block_x;
                continue;
            }
            if (block_y >= block_z && block_y > 1u) {
                --block_y;
                continue;
            }
            if (block_z > 1u) {
                --block_z;
                continue;
            }
            break;
        }
        return dim3(block_x, block_y, block_z);
    };
    auto validate_flow_axis = [&](const StableFluidsFlowBoundaryFaceDesc minus_face, const StableFluidsFlowBoundaryFaceDesc plus_face) {
        if (!stable_fluids::is_valid_flow_boundary_type(minus_face.type) || !stable_fluids::is_valid_flow_boundary_type(plus_face.type)) return false;
        return (minus_face.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC) == (plus_face.type == STABLE_FLUIDS_FLOW_BOUNDARY_PERIODIC);
    };
    auto validate_scalar_axis = [&](const StableFluidsScalarBoundaryFaceDesc minus_face, const StableFluidsScalarBoundaryFaceDesc plus_face) {
        if (!stable_fluids::is_valid_scalar_boundary_type(minus_face.type) || !stable_fluids::is_valid_scalar_boundary_type(plus_face.type)) return false;
        return (minus_face.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC) == (plus_face.type == STABLE_FLUIDS_SCALAR_BOUNDARY_PERIODIC);
    };
    if (!validate_flow_axis(context->config.flow_boundary.x_minus, context->config.flow_boundary.x_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    if (!validate_flow_axis(context->config.flow_boundary.y_minus, context->config.flow_boundary.y_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    if (!validate_flow_axis(context->config.flow_boundary.z_minus, context->config.flow_boundary.z_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    context->block = choose_block();
    context->cells = dim3(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)),
        static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));
    if (context->stream == nullptr) {
        if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        context->owns_stream = true;
    }

    context->device.scalar_fields.reserve(desc->scalar_field_count);
    for (uint32_t index = 0; index < desc->scalar_field_count; ++index) {
        if (!validate_scalar_axis(desc->scalar_fields[index].boundary.x_minus, desc->scalar_fields[index].boundary.x_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        if (!validate_scalar_axis(desc->scalar_fields[index].boundary.y_minus, desc->scalar_fields[index].boundary.y_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        if (!validate_scalar_axis(desc->scalar_fields[index].boundary.z_minus, desc->scalar_fields[index].boundary.z_plus)) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        context->device.scalar_fields.push_back(stable_fluids::ContextStorage::DeviceBuffers::ScalarField{
            .desc = desc->scalar_fields[index],
        });
        if (out_scalar_field_handles != nullptr && index < out_scalar_field_handle_capacity) out_scalar_field_handles[index] = index + 1u;
    }
    context->device.vector_fields.reserve(desc->vector_field_count);
    for (uint32_t index = 0; index < desc->vector_field_count; ++index) {
        context->device.vector_fields.push_back(stable_fluids::ContextStorage::DeviceBuffers::VectorField{
            .desc = desc->vector_fields[index],
        });
        if (out_vector_field_handles != nullptr && index < out_vector_field_handle_capacity) out_vector_field_handles[index] = index + 1u;
    }

    auto fail = [&](const StableFluidsResult code) {
        stable_fluids::destroy_context_resources(*context);
        if (context->owns_stream && context->stream != nullptr) cudaStreamDestroy(context->stream);
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
        if (cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_magnitude), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        for (auto& field : context->device.scalar_fields) {
            if (cudaMalloc(reinterpret_cast<void**>(&field.data), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
            if (cudaMalloc(reinterpret_cast<void**>(&field.temp), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
            if (cudaMalloc(reinterpret_cast<void**>(&field.source), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        }
        for (auto& field : context->device.vector_fields) {
            if (cudaMalloc(reinterpret_cast<void**>(&field.data_x), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
            if (cudaMalloc(reinterpret_cast<void**>(&field.data_y), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
            if (cudaMalloc(reinterpret_cast<void**>(&field.data_z), context->bytes) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_OUT_OF_MEMORY);
        }
    }

    {
        nvtx3::scoped_range init_range("stable.create_context.init");
        if (cudaMemsetAsync(context->device.flow.velocity_x, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.velocity_y, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.velocity_z, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.temp_velocity_x, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.temp_velocity_y, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.temp_velocity_z, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.pressure, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.divergence, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (cudaMemsetAsync(context->device.flow.velocity_magnitude, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        for (auto& [desc, data, temp, source] : context->device.scalar_fields) {
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(data, desc.initial_value, context->config.nx, context->config.ny, context->config.nz);
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(temp, desc.initial_value, context->config.nx, context->config.ny, context->config.nz);
            if (cudaMemsetAsync(source, 0, context->bytes, context->stream) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (cudaGetLastError() != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        }
        for (auto& [desc, data_x, data_y, data_z] : context->device.vector_fields) {
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(data_x, desc.initial_value_x, context->config.nx, context->config.ny, context->config.nz);
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(data_y, desc.initial_value_y, context->config.nx, context->config.ny, context->config.nz);
            stable_fluids::fill_kernel<<<context->cells, context->block, 0, context->stream>>>(data_z, desc.initial_value_z, context->config.nx, context->config.ny, context->config.nz);
            if (cudaGetLastError() != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        }
    }

    {
        nvtx3::scoped_range graph_range("stable.create_context.graph");
        if (cudaGraphCreate(&context->step_graph.graph, 0) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

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
            const cudaMemsetParams params{
                .dst         = destination,
                .pitch       = 0,
                .value       = 0u,
                .elementSize = sizeof(float),
                .width       = context->cell_count,
                .height      = 1,
            };
            return cudaGraphAddMemsetNode(&node, context->step_graph.graph, dependencies, dependency_count, &params) == cudaSuccess;
        };
        auto add_velocity_diffuse_chain = [&](cudaGraphNode_t& tail, float* const destination, const float* const source, const float diffusion, const int component_axis, const cudaGraphNode_t* dependencies, const std::size_t dependency_count) {
            if (!add_memcpy_node(tail, dependencies, dependency_count, destination, source)) return false;
            float alpha = context->config.dt * diffusion / (context->config.cell_size * context->config.cell_size);
            if (alpha <= 0.0f) return true;
            for (int iteration = 0; iteration < context->config.diffuse_iterations; ++iteration) {
                cudaGraphNode_t parity0{};
                cudaGraphNode_t parity1{};
                float* destination_ptr   = destination;
                const float* source_ptr  = source;
                int parity0_value        = 0;
                int parity1_value        = 1;
                int component_axis_value = component_axis;
                void* parity0_args[]{&destination_ptr, &source_ptr, &alpha, &parity0_value, &component_axis_value, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.flow_boundary};
                void* parity1_args[]{&destination_ptr, &source_ptr, &alpha, &parity1_value, &component_axis_value, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.flow_boundary};
                if (!add_kernel_node(parity0, &tail, 1, reinterpret_cast<void*>(stable_fluids::diffuse_velocity_rbgs_kernel), parity0_args)) return false;
                if (!add_kernel_node(parity1, &parity0, 1, reinterpret_cast<void*>(stable_fluids::diffuse_velocity_rbgs_kernel), parity1_args)) return false;
                tail = parity1;
            }
            return true;
        };
        auto add_scalar_diffuse_chain = [&](cudaGraphNode_t& tail, float* const destination, const float* const source, const float diffusion, const StableFluidsScalarBoundaryConfig& boundary, const cudaGraphNode_t* dependencies, const std::size_t dependency_count) {
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
                void* parity0_args[]{&destination_ptr, &source_ptr, &alpha, &parity0_value, &context->config.nx, &context->config.ny, &context->config.nz, const_cast<StableFluidsScalarBoundaryConfig*>(&boundary)};
                void* parity1_args[]{&destination_ptr, &source_ptr, &alpha, &parity1_value, &context->config.nx, &context->config.ny, &context->config.nz, const_cast<StableFluidsScalarBoundaryConfig*>(&boundary)};
                if (!add_kernel_node(parity0, &tail, 1, reinterpret_cast<void*>(stable_fluids::diffuse_scalar_rbgs_kernel), parity0_args)) return false;
                if (!add_kernel_node(parity1, &parity0, 1, reinterpret_cast<void*>(stable_fluids::diffuse_scalar_rbgs_kernel), parity1_args)) return false;
                tail = parity1;
            }
            return true;
        };

        cudaGraphNode_t force_tail{};
        bool has_force_tail = false;
        for (auto& [desc, data_x, data_y, data_z] : context->device.vector_fields) {
            if (desc.usage != STABLE_FLUIDS_VECTOR_FIELD_FORCE) continue;
            cudaGraphNode_t add_force_node{};
            void* add_force_args[]{&context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &data_x, &data_y, &data_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz};
            if (!add_kernel_node(add_force_node, has_force_tail ? &force_tail : nullptr, has_force_tail ? 1u : 0u, reinterpret_cast<void*>(stable_fluids::add_force_kernel), add_force_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            force_tail     = add_force_node;
            has_force_tail = true;
        }

        cudaGraphNode_t advect_velocity_x{};
        cudaGraphNode_t advect_velocity_y{};
        cudaGraphNode_t advect_velocity_z{};
        int velocity_x_axis = 0;
        int velocity_y_axis = 1;
        int velocity_z_axis = 2;
        void* advect_velocity_x_args[]{
            &context->device.flow.temp_velocity_x, &context->device.flow.velocity_x, &velocity_x_axis, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.flow_boundary};
        void* advect_velocity_y_args[]{
            &context->device.flow.temp_velocity_y, &context->device.flow.velocity_y, &velocity_y_axis, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.flow_boundary};
        void* advect_velocity_z_args[]{
            &context->device.flow.temp_velocity_z, &context->device.flow.velocity_z, &velocity_z_axis, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.flow_boundary};
        if (!add_kernel_node(advect_velocity_x, has_force_tail ? &force_tail : nullptr, has_force_tail ? 1u : 0u, reinterpret_cast<void*>(stable_fluids::advect_velocity_component_kernel), advect_velocity_x_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (!add_kernel_node(advect_velocity_y, has_force_tail ? &force_tail : nullptr, has_force_tail ? 1u : 0u, reinterpret_cast<void*>(stable_fluids::advect_velocity_component_kernel), advect_velocity_y_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (!add_kernel_node(advect_velocity_z, has_force_tail ? &force_tail : nullptr, has_force_tail ? 1u : 0u, reinterpret_cast<void*>(stable_fluids::advect_velocity_component_kernel), advect_velocity_z_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

        std::array<cudaGraphNode_t, 3> diffuse_velocity_tails{};
        if (!add_velocity_diffuse_chain(diffuse_velocity_tails[0], context->device.flow.velocity_x, context->device.flow.temp_velocity_x, context->config.viscosity, 0, &advect_velocity_x, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (!add_velocity_diffuse_chain(diffuse_velocity_tails[1], context->device.flow.velocity_y, context->device.flow.temp_velocity_y, context->config.viscosity, 1, &advect_velocity_y, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        if (!add_velocity_diffuse_chain(diffuse_velocity_tails[2], context->device.flow.velocity_z, context->device.flow.temp_velocity_z, context->config.viscosity, 2, &advect_velocity_z, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

        cudaGraphNode_t divergence_node{};
        void* divergence_args[]{&context->device.flow.divergence, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &context->config.flow_boundary};
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
            void* parity0_args[]{&context->device.flow.pressure, &context->device.flow.divergence, &parity0_value, &context->config.nx, &context->config.ny, &context->config.nz, &h2, &context->config.flow_boundary};
            void* parity1_args[]{&context->device.flow.pressure, &context->device.flow.divergence, &parity1_value, &context->config.nx, &context->config.ny, &context->config.nz, &h2, &context->config.flow_boundary};
            if (!add_kernel_node(parity0, &pressure_tail, 1, reinterpret_cast<void*>(stable_fluids::pressure_rbgs_kernel), parity0_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (!add_kernel_node(parity1, &parity0, 1, reinterpret_cast<void*>(stable_fluids::pressure_rbgs_kernel), parity1_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            pressure_tail = parity1;
        }

        cudaGraphNode_t project_velocity_node{};
        void* project_velocity_args[]{&context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->device.flow.pressure, &context->config.nx, &context->config.ny, &context->config.nz,
            &context->config.cell_size, &context->config.flow_boundary};
        if (!add_kernel_node(project_velocity_node, &pressure_tail, 1, reinterpret_cast<void*>(stable_fluids::project_velocity_kernel), project_velocity_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

        cudaGraphNode_t final_divergence_node{};
        if (!add_kernel_node(final_divergence_node, &project_velocity_node, 1, reinterpret_cast<void*>(stable_fluids::compute_divergence_kernel), divergence_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
        cudaGraphNode_t velocity_magnitude_node{};
        void* velocity_magnitude_args[]{&context->device.flow.velocity_magnitude, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.nx, &context->config.ny, &context->config.nz};
        if (!add_kernel_node(velocity_magnitude_node, &final_divergence_node, 1, reinterpret_cast<void*>(stable_fluids::velocity_magnitude_kernel), velocity_magnitude_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

        for (auto& [desc, data, temp, source] : context->device.scalar_fields) {
            cudaGraphNode_t add_field_source_node{};
            void* add_field_source_args[]{&data, &source, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz};
            if (!add_kernel_node(add_field_source_node, &project_velocity_node, 1, reinterpret_cast<void*>(stable_fluids::add_field_source_kernel), add_field_source_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t advect_field_node{};
            void* advect_field_args[]{&temp, &data, &context->device.flow.velocity_x, &context->device.flow.velocity_y, &context->device.flow.velocity_z, &context->config.dt, &context->config.nx, &context->config.ny, &context->config.nz, &context->config.cell_size, &desc.boundary, &context->config.flow_boundary};
            if (!add_kernel_node(advect_field_node, &add_field_source_node, 1, reinterpret_cast<void*>(stable_fluids::advect_scalar_component_kernel), advect_field_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);

            cudaGraphNode_t field_tail{};
            if (!add_scalar_diffuse_chain(field_tail, data, temp, desc.diffusion, desc.boundary, &advect_field_node, 1)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            if (desc.dissipation > 0.0f) {
                float factor = 1.0f / (1.0f + context->config.dt * desc.dissipation);
                cudaGraphNode_t dissipate_node{};
                void* dissipate_args[]{&data, &data, &factor, &context->config.nx, &context->config.ny, &context->config.nz};
                if (!add_kernel_node(dissipate_node, &field_tail, 1, reinterpret_cast<void*>(stable_fluids::dissipate_kernel), dissipate_args)) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
            }
        }
        (void) velocity_magnitude_node;

        if (cudaGraphInstantiate(&context->step_graph.exec, context->step_graph.graph) != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
    }

    if (cudaGetLastError() != cudaSuccess) return fail(STABLE_FLUIDS_RESULT_BACKEND_FAILURE);
    *out_context = context.release();
    return STABLE_FLUIDS_RESULT_OK;
}

StableFluidsResult stable_fluids_destroy_context_cuda(void* context) {
    nvtx3::scoped_range range("stable.destroy_context");
    auto* storage = static_cast<stable_fluids::ContextStorage*>(context);
    if (storage->stream != nullptr) cudaStreamSynchronize(storage->stream);
    stable_fluids::destroy_context_resources(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete storage;
    return STABLE_FLUIDS_RESULT_OK;
}

StableFluidsResult stable_fluids_update_scalar_field_cuda(void* context, const StableFluidsScalarFieldHandle field, const float* values) {
    nvtx3::scoped_range range("stable.update_scalar_field");
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    if (field > storage.device.scalar_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    auto& scalar_field = storage.device.scalar_fields[static_cast<std::size_t>(field - 1u)];
    if (values == nullptr) return cudaMemsetAsync(scalar_field.data, 0, storage.bytes, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(scalar_field.data, values, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
}

StableFluidsResult stable_fluids_update_scalar_field_source_cuda(void* context, const StableFluidsScalarFieldHandle field, const float* values) {
    nvtx3::scoped_range range("stable.update_scalar_field_source");
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    if (field > storage.device.scalar_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    auto& scalar_field = storage.device.scalar_fields[static_cast<std::size_t>(field - 1u)];
    if (values == nullptr) return cudaMemsetAsync(scalar_field.source, 0, storage.bytes, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(scalar_field.source, values, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
}

StableFluidsResult stable_fluids_update_vector_field_cuda(void* context, const StableFluidsVectorFieldHandle field, const float* values_x, const float* values_y, const float* values_z) {
    nvtx3::scoped_range range("stable.update_vector_field");
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    if (field > storage.device.vector_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    auto& vector_field  = storage.device.vector_fields[static_cast<std::size_t>(field - 1u)];
    auto copy_component = [&](float* const destination, const float* const source) {
        if (source == nullptr) return cudaMemsetAsync(destination, 0, storage.bytes, storage.stream);
        return cudaMemcpyAsync(destination, source, storage.bytes, cudaMemcpyDeviceToDevice, storage.stream);
    };
    if (copy_component(vector_field.data_x, values_x) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    if (copy_component(vector_field.data_y, values_y) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    if (copy_component(vector_field.data_z, values_z) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    return STABLE_FLUIDS_RESULT_OK;
}

StableFluidsResult stable_fluids_step_cuda(void* context) {
    nvtx3::scoped_range range("stable.step");
    if (context == nullptr) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    return cudaGraphLaunch(storage.step_graph.exec, storage.stream) == cudaSuccess ? STABLE_FLUIDS_RESULT_OK : STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
}

StableFluidsResult stable_fluids_get_view_cuda(void* context, const StableFluidsViewRequest* request, StableFluidsView* out_view) {
    nvtx3::scoped_range range("stable.get_view");
    if (context == nullptr || request == nullptr || out_view == nullptr) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<stable_fluids::ContextStorage*>(context);
    *out_view     = StableFluidsView{
            .layout             = STABLE_FLUIDS_VIEW_LAYOUT_F32_3D,
            .nx                 = storage.config.nx,
            .ny                 = storage.config.ny,
            .nz                 = storage.config.nz,
            .row_stride_bytes   = static_cast<uint64_t>(storage.config.nx) * sizeof(float),
            .slice_stride_bytes = static_cast<uint64_t>(storage.config.nx) * static_cast<uint64_t>(storage.config.ny) * sizeof(float),
            .data0              = nullptr,
            .data1              = nullptr,
            .data2              = nullptr,
    };
    auto sync_consumer_stream = [&]() {
        if (request->consumer_stream == nullptr) return STABLE_FLUIDS_RESULT_OK;
        auto consumer_stream = static_cast<cudaStream_t>(request->consumer_stream);
        if (consumer_stream == storage.stream) return STABLE_FLUIDS_RESULT_OK;
        cudaEvent_t ready_event = nullptr;
        if (cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming) != cudaSuccess) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        if (cudaEventRecord(ready_event, storage.stream) != cudaSuccess) {
            cudaEventDestroy(ready_event);
            return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        }
        if (cudaStreamWaitEvent(consumer_stream, ready_event) != cudaSuccess) {
            cudaEventDestroy(ready_event);
            return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        }
        cudaEventDestroy(ready_event);
        return STABLE_FLUIDS_RESULT_OK;
    };

    if (request->kind == STABLE_FLUIDS_VIEW_SCALAR_FIELD_DATA) {
        if (request->scalar_field == 0 || request->scalar_field > storage.device.scalar_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        out_view->data0 = storage.device.scalar_fields[static_cast<std::size_t>(request->scalar_field - 1u)].data;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_SCALAR_FIELD_SOURCE) {
        if (request->scalar_field == 0 || request->scalar_field > storage.device.scalar_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        out_view->data0 = storage.device.scalar_fields[static_cast<std::size_t>(request->scalar_field - 1u)].source;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_VECTOR_FIELD) {
        if (request->vector_field == 0 || request->vector_field > storage.device.vector_fields.size()) return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
        auto& vector_field = storage.device.vector_fields[static_cast<std::size_t>(request->vector_field - 1u)];
        out_view->layout   = STABLE_FLUIDS_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0    = vector_field.data_x;
        out_view->data1    = vector_field.data_y;
        out_view->data2    = vector_field.data_z;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_FLOW_VELOCITY) {
        out_view->layout = STABLE_FLUIDS_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0  = storage.device.flow.velocity_x;
        out_view->data1  = storage.device.flow.velocity_y;
        out_view->data2  = storage.device.flow.velocity_z;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_FLOW_VELOCITY_MAGNITUDE) {
        out_view->data0 = storage.device.flow.velocity_magnitude;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_FLOW_PRESSURE) {
        out_view->data0 = storage.device.flow.pressure;
        return sync_consumer_stream();
    }
    if (request->kind == STABLE_FLUIDS_VIEW_FLOW_DIVERGENCE) {
        out_view->data0 = storage.device.flow.divergence;
        return sync_consumer_stream();
    }
    return STABLE_FLUIDS_RESULT_BACKEND_FAILURE;
}

} // extern "C"
