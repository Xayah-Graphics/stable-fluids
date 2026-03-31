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
    STABLE_FLUIDS_RESULT_OK              = 0,
    STABLE_FLUIDS_RESULT_OUT_OF_MEMORY   = 1,
    STABLE_FLUIDS_RESULT_BACKEND_FAILURE = 2,
} StableFluidsResult;

typedef enum StableFluidsBoundaryMode {
    STABLE_FLUIDS_BOUNDARY_FIXED    = 0,
    STABLE_FLUIDS_BOUNDARY_PERIODIC = 1,
} StableFluidsBoundaryMode;

typedef uint32_t StableFluidsScalarFieldHandle;
typedef uint32_t StableFluidsVectorFieldHandle;

typedef struct StableFluidsBoundaryConfig {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} StableFluidsBoundaryConfig;

typedef struct StableFluidsSimulationConfig {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float viscosity;
    int32_t diffuse_iterations;
    int32_t pressure_iterations;
    StableFluidsBoundaryConfig boundary;
} StableFluidsSimulationConfig;
typedef struct StableFluidsScalarFieldDesc {
    const char* name;
    float diffusion;
    float dissipation;
    float initial_value;
} StableFluidsScalarFieldDesc;

typedef enum StableFluidsVectorFieldUsage {
    STABLE_FLUIDS_VECTOR_FIELD_FORCE = 0,
} StableFluidsVectorFieldUsage;

typedef struct StableFluidsVectorFieldDesc {
    const char* name;
    StableFluidsVectorFieldUsage usage;
    float initial_value_x;
    float initial_value_y;
    float initial_value_z;
} StableFluidsVectorFieldDesc;

typedef struct StableFluidsContextCreateDesc {
    StableFluidsSimulationConfig config;
    void* stream;
    const StableFluidsScalarFieldDesc* scalar_fields;
    uint32_t scalar_field_count;
    const StableFluidsVectorFieldDesc* vector_fields;
    uint32_t vector_field_count;
} StableFluidsContextCreateDesc;
STABLE_FLUIDS_API StableFluidsResult stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, void** out_context, StableFluidsScalarFieldHandle* out_scalar_field_handles, uint32_t out_scalar_field_handle_capacity, StableFluidsVectorFieldHandle* out_vector_field_handles, uint32_t out_vector_field_handle_capacity);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_destroy_context_cuda(void* context);

STABLE_FLUIDS_API StableFluidsResult stable_fluids_update_scalar_field_cuda(void* context, StableFluidsScalarFieldHandle field, const float* values);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_update_scalar_field_source_cuda(void* context, StableFluidsScalarFieldHandle field, const float* values);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_update_vector_field_cuda(void* context, StableFluidsVectorFieldHandle field, const float* values_x, const float* values_y, const float* values_z);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_step_cuda(void* context);

typedef enum StableFluidsViewKind {
    STABLE_FLUIDS_VIEW_SCALAR_FIELD_DATA   = 0,
    STABLE_FLUIDS_VIEW_SCALAR_FIELD_SOURCE = 1,
    STABLE_FLUIDS_VIEW_VECTOR_FIELD        = 2,
    STABLE_FLUIDS_VIEW_FLOW_VELOCITY       = 3,
    STABLE_FLUIDS_VIEW_FLOW_VELOCITY_MAGNITUDE = 4,
    STABLE_FLUIDS_VIEW_FLOW_PRESSURE       = 5,
    STABLE_FLUIDS_VIEW_FLOW_DIVERGENCE     = 6,
} StableFluidsViewKind;

typedef enum StableFluidsViewLayout {
    STABLE_FLUIDS_VIEW_LAYOUT_F32_3D      = 0,
    STABLE_FLUIDS_VIEW_LAYOUT_F32_3D_SOA3 = 1,
} StableFluidsViewLayout;

typedef struct StableFluidsViewRequest {
    uint32_t kind;
    StableFluidsScalarFieldHandle scalar_field;
    StableFluidsVectorFieldHandle vector_field;
    void* consumer_stream;
} StableFluidsViewRequest;

typedef struct StableFluidsView {
    uint32_t layout;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    uint64_t row_stride_bytes;
    uint64_t slice_stride_bytes;
    const float* data0;
    const float* data1;
    const float* data2;
} StableFluidsView;

STABLE_FLUIDS_API StableFluidsResult stable_fluids_get_view_cuda(void* context, const StableFluidsViewRequest* request, StableFluidsView* out_view);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_3D_H
