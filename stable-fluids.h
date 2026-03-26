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

#define STABLE_FLUIDS_API_VERSION 2u

typedef struct StableFluidsContext_t* StableFluidsContext;

typedef enum StableFluidsBoundaryType {
    STABLE_FLUIDS_BOUNDARY_NO_SLIP   = 0,
    STABLE_FLUIDS_BOUNDARY_FREE_SLIP = 1,
    STABLE_FLUIDS_BOUNDARY_INFLOW    = 2,
    STABLE_FLUIDS_BOUNDARY_OUTFLOW   = 3,
} StableFluidsBoundaryType;

typedef enum StableFluidsColliderType {
    STABLE_FLUIDS_COLLIDER_SPHERE = 0,
    STABLE_FLUIDS_COLLIDER_BOX    = 1,
} StableFluidsColliderType;

typedef enum StableFluidsExportField {
    STABLE_FLUIDS_EXPORT_DENSITY            = 0,
    STABLE_FLUIDS_EXPORT_DYE_RGBA           = 1,
    STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE = 2,
    STABLE_FLUIDS_EXPORT_SOLID_MASK         = 3,
    STABLE_FLUIDS_EXPORT_PRESSURE           = 4,
    STABLE_FLUIDS_EXPORT_DIVERGENCE         = 5,
} StableFluidsExportField;

typedef struct StableFluidsBoundaryFaceDesc {
    uint32_t type;
    float velocity;
    float scalar;
} StableFluidsBoundaryFaceDesc;

typedef struct StableFluidsDomainBoundaryDesc {
    StableFluidsBoundaryFaceDesc x_min;
    StableFluidsBoundaryFaceDesc x_max;
    StableFluidsBoundaryFaceDesc y_min;
    StableFluidsBoundaryFaceDesc y_max;
    StableFluidsBoundaryFaceDesc z_min;
    StableFluidsBoundaryFaceDesc z_max;
} StableFluidsDomainBoundaryDesc;

typedef struct StableFluidsSimulationConfig {
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
    float ambient_temperature;
    float density_buoyancy;
    float temperature_buoyancy;
    float uniform_force_x;
    float uniform_force_y;
    float uniform_force_z;
    StableFluidsDomainBoundaryDesc domain_boundary;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
} StableFluidsSimulationConfig;

typedef struct StableFluidsContextCreateDesc {
    uint32_t struct_size;
    uint32_t api_version;
    StableFluidsSimulationConfig config;
    void* stream;
} StableFluidsContextCreateDesc;

typedef struct StableFluidsColliderDesc {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t collider_type;
    uint32_t boundary_type;
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float half_extent_x;
    float half_extent_y;
    float half_extent_z;
    float linear_velocity_x;
    float linear_velocity_y;
    float linear_velocity_z;
} StableFluidsColliderDesc;

typedef struct StableFluidsSceneDesc {
    uint32_t struct_size;
    uint32_t api_version;
    const StableFluidsColliderDesc* colliders;
    uint32_t collider_count;
} StableFluidsSceneDesc;

typedef struct StableFluidsSourceDesc {
    uint32_t struct_size;
    uint32_t api_version;
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float density_amount;
    float dye_r;
    float dye_g;
    float dye_b;
    float temperature_amount;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} StableFluidsSourceDesc;

typedef struct StableFluidsStepDesc {
    uint32_t struct_size;
    uint32_t api_version;
    const StableFluidsSourceDesc* sources;
    uint32_t source_count;
} StableFluidsStepDesc;

typedef struct StableFluidsExportFieldDesc {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t field;
    void* destination;
} StableFluidsExportFieldDesc;

typedef struct StableFluidsGridDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
} StableFluidsGridDesc;

STABLE_FLUIDS_API int32_t stable_fluids_validate_context_create_desc(const StableFluidsContextCreateDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_scene_desc(const StableFluidsSceneDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_step_desc(const StableFluidsStepDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_validate_export_field_desc(const StableFluidsExportFieldDesc* desc);

STABLE_FLUIDS_API int32_t stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, StableFluidsContext* out_context);
STABLE_FLUIDS_API int32_t stable_fluids_destroy_context_cuda(StableFluidsContext context);
STABLE_FLUIDS_API int32_t stable_fluids_reset_context_cuda(StableFluidsContext context);
STABLE_FLUIDS_API int32_t stable_fluids_update_scene_cuda(StableFluidsContext context, const StableFluidsSceneDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_export_field_cuda(StableFluidsContext context, const StableFluidsExportFieldDesc* desc);
STABLE_FLUIDS_API int32_t stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_H
