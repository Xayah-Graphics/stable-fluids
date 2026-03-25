#ifndef STABLE_FLUIDS_H
#define STABLE_FLUIDS_H

#include <stdint.h>

#ifdef _WIN32
#ifdef STABLE_FLUIDS_BUILD_SHARED
#define STABLE_FLUIDS_API __declspec(dllexport)
#else
#define STABLE_FLUIDS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define STABLE_FLUIDS_API __attribute__((visibility("default")))
#else
#define STABLE_FLUIDS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
Error code scheme:
0     : success
1xxx  : scalar/grid/step parameter errors
1000  : invalid step descriptor
1001  : invalid grid dimensions
1002  : invalid cell size
1003  : invalid dt
1004  : invalid iteration count
1005  : invalid boundary type
1006  : unsupported api_version
2xxx  : buffer errors
2001  : invalid density buffer
2002  : invalid destination buffer
2006  : invalid dye buffer
2017  : invalid temperature buffer
2018  : invalid force_x buffer
2019  : invalid force_y buffer
2020  : invalid force_z buffer
2003  : invalid velocity_x buffer
2004  : invalid velocity_y buffer
2005  : invalid velocity_z buffer
2007  : invalid temporary density buffer
2008  : invalid temporary velocity_x buffer
2009  : invalid temporary velocity_y buffer
2010  : invalid temporary velocity_z buffer
2011  : invalid temporary previous density buffer
2012  : invalid temporary previous velocity_x buffer
2013  : invalid temporary previous velocity_y buffer
2014  : invalid temporary previous velocity_z buffer
2015  : invalid temporary pressure buffer
2016  : invalid temporary divergence buffer
5xxx  : CUDA runtime or kernel launch failure
5001  : CUDA call failed
*/

#define STABLE_FLUIDS_API_VERSION 1u

typedef enum StableFluidsBoundaryType {
    STABLE_FLUIDS_BOUNDARY_NO_SLIP   = 0,
    STABLE_FLUIDS_BOUNDARY_FREE_SLIP = 1,
    STABLE_FLUIDS_BOUNDARY_INFLOW    = 2,
    STABLE_FLUIDS_BOUNDARY_OUTFLOW   = 3,
} StableFluidsBoundaryType;

typedef struct StableFluidsAdvectVelocityDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_velocity_x_min;
    float inflow_velocity_x_max;
    float inflow_velocity_y_min;
    float inflow_velocity_y_max;
    float inflow_velocity_z_min;
    float inflow_velocity_z_max;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* temporary_velocity_x;
    void* temporary_velocity_y;
    void* temporary_velocity_z;
    void* temporary_previous_velocity_x;
    void* temporary_previous_velocity_y;
    void* temporary_previous_velocity_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAdvectVelocityDesc;

typedef struct StableFluidsDiffuseVelocityDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float viscosity;
    int32_t diffuse_iterations;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_velocity_x_min;
    float inflow_velocity_x_max;
    float inflow_velocity_y_min;
    float inflow_velocity_y_max;
    float inflow_velocity_z_min;
    float inflow_velocity_z_max;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* temporary_velocity_x;
    void* temporary_velocity_y;
    void* temporary_velocity_z;
    void* temporary_density;
    void* temporary_previous_density;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsDiffuseVelocityDesc;

typedef struct StableFluidsProjectDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    int32_t pressure_iterations;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_velocity_x_min;
    float inflow_velocity_x_max;
    float inflow_velocity_y_min;
    float inflow_velocity_y_max;
    float inflow_velocity_z_min;
    float inflow_velocity_z_max;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* temporary_pressure;
    void* temporary_divergence;
    void* temporary_density;
    void* temporary_previous_density;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsProjectDesc;

typedef struct StableFluidsAdvectScalarDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_scalar_x_min;
    float inflow_scalar_x_max;
    float inflow_scalar_y_min;
    float inflow_scalar_y_max;
    float inflow_scalar_z_min;
    float inflow_scalar_z_max;
    void* scalar;
    void* temporary_scalar;
    void* temporary_previous_scalar;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    uint32_t clamp_non_negative;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAdvectScalarDesc;

typedef struct StableFluidsDiffuseScalarDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float diffusion;
    int32_t diffuse_iterations;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_scalar_x_min;
    float inflow_scalar_x_max;
    float inflow_scalar_y_min;
    float inflow_scalar_y_max;
    float inflow_scalar_z_min;
    float inflow_scalar_z_max;
    void* scalar;
    void* temporary_scalar;
    void* temporary_solution_storage;
    void* temporary_rhs_storage;
    uint32_t clamp_non_negative;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsDiffuseScalarDesc;

typedef struct StableFluidsAddScalarSourceDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    void* scalar;
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float amount;
    float sample_offset_x;
    float sample_offset_y;
    float sample_offset_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAddScalarSourceDesc;

typedef struct StableFluidsAddVectorSourceDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    void* vector_x;
    void* vector_y;
    void* vector_z;
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float amount_x;
    float amount_y;
    float amount_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAddVectorSourceDesc;

typedef struct StableFluidsAddForceDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float dt;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    float inflow_velocity_x_min;
    float inflow_velocity_x_max;
    float inflow_velocity_y_min;
    float inflow_velocity_y_max;
    float inflow_velocity_z_min;
    float inflow_velocity_z_max;
    float ambient_temperature;
    float density_buoyancy;
    float temperature_buoyancy;
    float uniform_force_x;
    float uniform_force_y;
    float uniform_force_z;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* density;
    void* temperature;
    void* force_x;
    void* force_y;
    void* force_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAddForceDesc;

typedef struct StableFluidsComputeStaggeredVelocityMagnitudeDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    void* destination;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsComputeStaggeredVelocityMagnitudeDesc;

typedef struct StableFluidsPackSmokeRgbaDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    void* destination_rgba;
    void* density;
    void* dye_r;
    void* dye_g;
    void* dye_b;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsPackSmokeRgbaDesc;

STABLE_FLUIDS_API int32_t stable_fluids_validate_advect_velocity_desc(const StableFluidsAdvectVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_diffuse_velocity_desc(const StableFluidsDiffuseVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_project_desc(const StableFluidsProjectDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_advect_scalar_desc(const StableFluidsAdvectScalarDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_diffuse_scalar_desc(const StableFluidsDiffuseScalarDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_add_scalar_source_desc(const StableFluidsAddScalarSourceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_add_vector_source_desc(const StableFluidsAddVectorSourceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_add_force_desc(const StableFluidsAddForceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_compute_staggered_velocity_magnitude_desc(const StableFluidsComputeStaggeredVelocityMagnitudeDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_pack_smoke_rgba_desc(const StableFluidsPackSmokeRgbaDesc* desc);

STABLE_FLUIDS_API int32_t stable_fluids_advect_velocity_cuda(const StableFluidsAdvectVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_diffuse_velocity_cuda(const StableFluidsDiffuseVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_project_cuda(const StableFluidsProjectDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_advect_scalar_cuda(const StableFluidsAdvectScalarDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_diffuse_scalar_cuda(const StableFluidsDiffuseScalarDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_add_scalar_source_cuda(const StableFluidsAddScalarSourceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_add_vector_source_cuda(const StableFluidsAddVectorSourceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_add_force_cuda(const StableFluidsAddForceDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_compute_staggered_velocity_magnitude_cuda(const StableFluidsComputeStaggeredVelocityMagnitudeDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_pack_smoke_rgba_cuda(const StableFluidsPackSmokeRgbaDesc* desc);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
