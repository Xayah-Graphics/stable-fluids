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

typedef enum StableFluidsBoundaryType { STABLE_FLUIDS_BOUNDARY_FIXED = 0, STABLE_FLUIDS_BOUNDARY_PERIODIC = 1 } StableFluidsBoundaryType;

typedef struct StableFluidsStepDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float viscosity;
    float diffusion;
    int32_t diffuse_iterations;
    int32_t pressure_iterations;
    uint32_t boundary_x_min;
    uint32_t boundary_x_max;
    uint32_t boundary_y_min;
    uint32_t boundary_y_max;
    uint32_t boundary_z_min;
    uint32_t boundary_z_max;
    void* density;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* temporary_density;
    void* temporary_velocity_x;
    void* temporary_velocity_y;
    void* temporary_velocity_z;
    void* temporary_previous_density;
    void* temporary_previous_velocity_x;
    void* temporary_previous_velocity_y;
    void* temporary_previous_velocity_z;
    void* temporary_pressure;
    void* temporary_divergence;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsStepDesc;

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

typedef struct StableFluidsAdvectDensityDesc {
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
    void* density;
    void* temporary_density;
    void* temporary_previous_density;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsAdvectDensityDesc;

typedef struct StableFluidsDiffuseDensityDesc {
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
    void* density;
    void* temporary_density;
    void* temporary_pressure;
    void* temporary_divergence;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} StableFluidsDiffuseDensityDesc;

STABLE_FLUIDS_API int32_t stable_fluids_validate_desc(const StableFluidsStepDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_advect_velocity_desc(const StableFluidsAdvectVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_diffuse_velocity_desc(const StableFluidsDiffuseVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_project_desc(const StableFluidsProjectDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_advect_density_desc(const StableFluidsAdvectDensityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_diffuse_density_desc(const StableFluidsDiffuseDensityDesc* desc);

STABLE_FLUIDS_API int32_t stable_fluids_advect_velocity_cuda(const StableFluidsAdvectVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_diffuse_velocity_cuda(const StableFluidsDiffuseVelocityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_project_cuda(const StableFluidsProjectDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_advect_density_cuda(const StableFluidsAdvectDensityDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_diffuse_density_cuda(const StableFluidsDiffuseDensityDesc* desc);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
