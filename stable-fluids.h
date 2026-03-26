#ifndef STABLE_FLUIDS_H
#define STABLE_FLUIDS_H

#include <cstdint>

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
    STABLE_FLUIDS_EXPORT_FIELD_COMPONENTS    = 0,
    STABLE_FLUIDS_EXPORT_ALPHA_RGB_RGBA      = 1,
    STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE  = 2,
    STABLE_FLUIDS_EXPORT_SOLID_MASK          = 3,
    STABLE_FLUIDS_EXPORT_PRESSURE            = 4,
    STABLE_FLUIDS_EXPORT_DIVERGENCE          = 5,
} StableFluidsExportField;

typedef enum StableFluidsFieldBoundaryMode {
    STABLE_FLUIDS_FIELD_BOUNDARY_CONSTANT    = 0,
    STABLE_FLUIDS_FIELD_BOUNDARY_STREAK      = 1,
    STABLE_FLUIDS_FIELD_BOUNDARY_REPEAT      = 2,
    STABLE_FLUIDS_FIELD_BOUNDARY_EXTRAPOLATE = 3,
} StableFluidsFieldBoundaryMode;

typedef enum StableFluidsFieldFlags {
    STABLE_FLUIDS_FIELD_ADVECT  = 1u << 0,
    STABLE_FLUIDS_FIELD_DIFFUSE = 1u << 1,
} StableFluidsFieldFlags;

typedef uint32_t StableFluidsFieldHandle;

typedef struct StableFluidsBoundaryFaceDesc {
    uint32_t type;
    float velocity;
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
    struct StableFluidsFieldDesc* fields;
    uint32_t field_count;
    const struct StableFluidsBuoyancyDesc* buoyancy_terms;
    uint32_t buoyancy_term_count;
} StableFluidsContextCreateDesc;

typedef struct StableFluidsFieldDesc {
    const char* name;
    uint32_t component_count;
    uint32_t flags;
    float diffusion;
    uint32_t boundary_mode;
    float default_value_0;
    float default_value_1;
    float default_value_2;
    float default_value_3;
    StableFluidsFieldHandle handle;
} StableFluidsFieldDesc;

typedef struct StableFluidsBuoyancyDesc {
    StableFluidsFieldHandle field;
    float weight;
    float ambient;
} StableFluidsBuoyancyDesc;

typedef struct StableFluidsColliderDesc {
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

typedef struct StableFluidsExportFieldDesc {
    uint32_t field;
    StableFluidsFieldHandle field_handle;
    uint32_t component_offset;
    uint32_t component_count;
    StableFluidsFieldHandle alpha_field;
    StableFluidsFieldHandle rgb_field;
    void* destination;
} StableFluidsExportFieldDesc;

typedef struct StableFluidsGridDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
} StableFluidsGridDesc;

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
