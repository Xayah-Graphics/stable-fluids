#ifndef STABLE_FLUIDS_3D_H
#define STABLE_FLUIDS_3D_H

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

typedef enum StableFluidsResult {
    STABLE_FLUIDS_RESULT_OK               = 0,
    STABLE_FLUIDS_RESULT_INVALID_ARGUMENT = 1,
    STABLE_FLUIDS_RESULT_INVALID_CONTEXT  = 2,
    STABLE_FLUIDS_RESULT_INVALID_CONFIG   = 3,
    STABLE_FLUIDS_RESULT_INVALID_FIELD    = 4,
    STABLE_FLUIDS_RESULT_INVALID_SCENE    = 5,
    STABLE_FLUIDS_RESULT_INVALID_EXPORT   = 6,
    STABLE_FLUIDS_RESULT_OUT_OF_MEMORY    = 7,
    STABLE_FLUIDS_RESULT_BACKEND_FAILURE  = 8,
} StableFluidsResult;

typedef enum StableFluidsVelocityBoundaryType {
    STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP   = 0,
    STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP = 1,
    STABLE_FLUIDS_VELOCITY_BOUNDARY_INFLOW    = 2,
    STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW   = 3,
} StableFluidsVelocityBoundaryType;

typedef enum StableFluidsColliderType {
    STABLE_FLUIDS_COLLIDER_SPHERE = 0,
    STABLE_FLUIDS_COLLIDER_BOX    = 1,
} StableFluidsColliderType;

typedef enum StableFluidsFieldExtensionMode {
    STABLE_FLUIDS_FIELD_EXTENSION_CONSTANT    = 0,
    STABLE_FLUIDS_FIELD_EXTENSION_STREAK      = 1,
    STABLE_FLUIDS_FIELD_EXTENSION_REPEAT      = 2,
    STABLE_FLUIDS_FIELD_EXTENSION_EXTRAPOLATE = 3,
} StableFluidsFieldExtensionMode;

typedef enum StableFluidsFieldFlags {
    STABLE_FLUIDS_FIELD_ADVECT  = 1u << 0,
    STABLE_FLUIDS_FIELD_DIFFUSE = 1u << 1,
} StableFluidsFieldFlags;

typedef uint32_t StableFluidsFieldHandle;

typedef struct StableFluidsBoundaryFaceDesc {
    uint32_t type;
    float velocity;
} StableFluidsBoundaryFaceDesc;

typedef struct StableFluidsFieldCreateDesc {
    const char* name;
    uint32_t component_count;
    uint32_t flags;
    float diffusion;
    uint32_t extension_mode;
    float default_value_0;
    float default_value_1;
    float default_value_2;
    float default_value_3;
} StableFluidsFieldCreateDesc;

typedef struct StableFluidsBuoyancyDesc {
    uint32_t field_index;
    float weight;
    float ambient;
} StableFluidsBuoyancyDesc;

typedef struct StableFluidsContext_t* StableFluidsContext;

typedef struct StableFluidsDomainBoundaryDesc {
    StableFluidsBoundaryFaceDesc x_min;
    StableFluidsBoundaryFaceDesc x_max;
    StableFluidsBoundaryFaceDesc y_min;
    StableFluidsBoundaryFaceDesc y_max;
    StableFluidsBoundaryFaceDesc z_min;
    StableFluidsBoundaryFaceDesc z_max;
} StableFluidsDomainBoundaryDesc;

typedef struct StableFluidsSimulationConfig {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float viscosity;
    int32_t diffuse_iterations;
    int32_t pressure_iterations;
    float uniform_force_x;
    float uniform_force_y;
    float uniform_force_z;
    StableFluidsDomainBoundaryDesc domain_boundary;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
} StableFluidsSimulationConfig;

typedef struct StableFluidsContextCreateDesc {
    StableFluidsSimulationConfig config;
    void* stream;
    const StableFluidsFieldCreateDesc* fields;
    uint32_t field_count;
    const StableFluidsBuoyancyDesc* buoyancy_terms;
    uint32_t buoyancy_term_count;
} StableFluidsContextCreateDesc;

typedef struct StableFluidsColliderDesc {
    uint32_t collider_type;
    uint32_t velocity_boundary_type;
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
    const StableFluidsColliderDesc* colliders;
    uint32_t collider_count;
} StableFluidsSceneDesc;

typedef struct StableFluidsVelocitySourceDesc {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} StableFluidsVelocitySourceDesc;

typedef struct StableFluidsFieldSourceDesc {
    StableFluidsFieldHandle field;
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float value_0;
    float value_1;
    float value_2;
    float value_3;
} StableFluidsFieldSourceDesc;

typedef struct StableFluidsStepDesc {
    const StableFluidsVelocitySourceDesc* velocity_sources;
    uint32_t velocity_source_count;
    const StableFluidsFieldSourceDesc* field_sources;
    uint32_t field_source_count;
} StableFluidsStepDesc;

typedef struct StableFluidsGridDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
} StableFluidsGridDesc;

STABLE_FLUIDS_API StableFluidsResult stable_fluids_create_context_cuda(
    const StableFluidsContextCreateDesc* desc,
    StableFluidsContext* out_context,
    StableFluidsFieldHandle* out_field_handles,
    uint32_t out_field_handle_capacity
);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_destroy_context_cuda(StableFluidsContext context);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_reset_context_cuda(StableFluidsContext context);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_update_scene_cuda(StableFluidsContext context, const StableFluidsSceneDesc* desc);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_field_components_cuda(
    StableFluidsContext context,
    StableFluidsFieldHandle field_handle,
    uint32_t component_offset,
    uint32_t component_count,
    void* destination
);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_alpha_rgb_rgba_cuda(
    StableFluidsContext context,
    StableFluidsFieldHandle alpha_field,
    StableFluidsFieldHandle rgb_field,
    void* destination
);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_velocity_cuda(StableFluidsContext context, void* destination);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_velocity_magnitude_cuda(StableFluidsContext context, void* destination);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_solid_mask_cuda(StableFluidsContext context, void* destination);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_pressure_cuda(StableFluidsContext context, void* destination);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_divergence_cuda(StableFluidsContext context, void* destination);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_get_grid_desc_cuda(StableFluidsContext context, StableFluidsGridDesc* out_desc);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_3D_H
