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

typedef uint32_t StableFluidsFieldHandle;

typedef struct StableFluidsBoundaryConfig {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} StableFluidsBoundaryConfig;

typedef struct StableFluidsFieldCreateDesc {
    const char* name;
    float diffusion;
    float dissipation;
    float initial_value;
} StableFluidsFieldCreateDesc;

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
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
} StableFluidsSimulationConfig;

typedef struct StableFluidsContext_t* StableFluidsContext;

typedef struct StableFluidsContextCreateDesc {
    StableFluidsSimulationConfig config;
    void* stream;
    const StableFluidsFieldCreateDesc* fields;
    uint32_t field_count;
} StableFluidsContextCreateDesc;

typedef struct StableFluidsFieldSourceDesc {
    StableFluidsFieldHandle field;
    const float* values;
} StableFluidsFieldSourceDesc;

typedef struct StableFluidsStepDesc {
    const float* force_x;
    const float* force_y;
    const float* force_z;
    const StableFluidsFieldSourceDesc* field_sources;
    uint32_t field_source_count;
} StableFluidsStepDesc;

typedef enum StableFluidsExportKind {
    STABLE_FLUIDS_EXPORT_FIELD              = 0,
    STABLE_FLUIDS_EXPORT_VELOCITY           = 1,
    STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE = 2,
    STABLE_FLUIDS_EXPORT_PRESSURE           = 3,
    STABLE_FLUIDS_EXPORT_DIVERGENCE         = 4,
} StableFluidsExportKind;

typedef struct StableFluidsExportDesc {
    uint32_t kind;
    StableFluidsFieldHandle field;
} StableFluidsExportDesc;

STABLE_FLUIDS_API StableFluidsResult stable_fluids_create_context_cuda(const StableFluidsContextCreateDesc* desc, StableFluidsContext* out_context, StableFluidsFieldHandle* out_field_handles, uint32_t out_field_handle_capacity);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_destroy_context_cuda(StableFluidsContext context);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_step_cuda(StableFluidsContext context, const StableFluidsStepDesc* desc);
STABLE_FLUIDS_API StableFluidsResult stable_fluids_export_cuda(StableFluidsContext context, const StableFluidsExportDesc* desc, void* destination);

#ifdef __cplusplus
}
#endif

#endif // STABLE_FLUIDS_3D_H
