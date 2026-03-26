#include "stable-fluids.h"

namespace {

    constexpr int32_t success                = 0;
    constexpr int32_t invalid_descriptor     = 1000;
    constexpr int32_t invalid_grid           = 1001;
    constexpr int32_t invalid_cell_size      = 1002;
    constexpr int32_t invalid_dt             = 1003;
    constexpr int32_t invalid_iterations     = 1004;
    constexpr int32_t invalid_boundary       = 1005;
    constexpr int32_t unsupported_api        = 1006;
    constexpr int32_t invalid_context        = 1007;
    constexpr int32_t invalid_collider       = 1008;
    constexpr int32_t invalid_export_field   = 1009;
    constexpr int32_t invalid_destination    = 2002;

    int32_t validate_base(const uint32_t struct_size, const uint32_t expected_size, const uint32_t api_version) {
        if (struct_size < expected_size) return invalid_descriptor;
        if (api_version != STABLE_FLUIDS_API_VERSION) return unsupported_api;
        return success;
    }

    int32_t validate_boundary_face(const StableFluidsBoundaryFaceDesc& face) {
        if (face.type > STABLE_FLUIDS_BOUNDARY_OUTFLOW) return invalid_boundary;
        return success;
    }

    int32_t validate_domain_boundary(const StableFluidsDomainBoundaryDesc& boundary) {
        if (const int32_t code = validate_boundary_face(boundary.x_min); code != 0) return code;
        if (const int32_t code = validate_boundary_face(boundary.x_max); code != 0) return code;
        if (const int32_t code = validate_boundary_face(boundary.y_min); code != 0) return code;
        if (const int32_t code = validate_boundary_face(boundary.y_max); code != 0) return code;
        if (const int32_t code = validate_boundary_face(boundary.z_min); code != 0) return code;
        if (const int32_t code = validate_boundary_face(boundary.z_max); code != 0) return code;
        return success;
    }

    int32_t validate_config(const StableFluidsSimulationConfig& config) {
        if (config.nx <= 0 || config.ny <= 0 || config.nz <= 0) return invalid_grid;
        if (config.cell_size <= 0.0f) return invalid_cell_size;
        if (config.dt <= 0.0f) return invalid_dt;
        if (config.diffuse_iterations <= 0 || config.pressure_iterations <= 0) return invalid_iterations;
        return validate_domain_boundary(config.domain_boundary);
    }

} // namespace

extern "C" {

int32_t stable_fluids_validate_context_create_desc(const StableFluidsContextCreateDesc* desc) {
    if (desc == nullptr) return invalid_descriptor;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsContextCreateDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_base(desc->config.struct_size, sizeof(StableFluidsSimulationConfig), desc->config.api_version); code != 0) return code;
    return validate_config(desc->config);
}

int32_t stable_fluids_validate_scene_desc(const StableFluidsSceneDesc* desc) {
    if (desc == nullptr) return invalid_descriptor;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsSceneDesc), desc->api_version); code != 0) return code;
    if (desc->collider_count == 0) return success;
    if (desc->colliders == nullptr) return invalid_collider;
    for (uint32_t index = 0; index < desc->collider_count; ++index) {
        const auto& collider = desc->colliders[index];
        if (const int32_t code = validate_base(collider.struct_size, sizeof(StableFluidsColliderDesc), collider.api_version); code != 0) return code;
        if (collider.collider_type > STABLE_FLUIDS_COLLIDER_BOX) return invalid_collider;
        if (collider.boundary_type > STABLE_FLUIDS_BOUNDARY_FREE_SLIP) return invalid_collider;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_SPHERE && collider.radius <= 0.0f) return invalid_collider;
        if (collider.collider_type == STABLE_FLUIDS_COLLIDER_BOX && (collider.half_extent_x <= 0.0f || collider.half_extent_y <= 0.0f || collider.half_extent_z <= 0.0f)) return invalid_collider;
    }
    return success;
}

int32_t stable_fluids_validate_step_desc(const StableFluidsStepDesc* desc) {
    if (desc == nullptr) return invalid_descriptor;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsStepDesc), desc->api_version); code != 0) return code;
    if (desc->source_count == 0) return success;
    if (desc->sources == nullptr) return invalid_descriptor;
    for (uint32_t index = 0; index < desc->source_count; ++index) {
        const auto& source = desc->sources[index];
        if (const int32_t code = validate_base(source.struct_size, sizeof(StableFluidsSourceDesc), source.api_version); code != 0) return code;
        if (source.radius <= 0.0f) return invalid_descriptor;
    }
    return success;
}

int32_t stable_fluids_validate_export_field_desc(const StableFluidsExportFieldDesc* desc) {
    if (desc == nullptr) return invalid_descriptor;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsExportFieldDesc), desc->api_version); code != 0) return code;
    if (desc->field > STABLE_FLUIDS_EXPORT_DIVERGENCE) return invalid_export_field;
    if (desc->destination == nullptr) return invalid_destination;
    return success;
}

} // extern "C"
