#include "stable-fluids.h"

extern "C" {

namespace {

    int32_t validate_base(const uint32_t struct_size, const uint32_t expected_size, const uint32_t api_version) {
        if (struct_size < expected_size) return 1000;
        if (api_version != STABLE_FLUIDS_API_VERSION) return 1006;
        return 0;
    }

    int32_t validate_grid(const int32_t nx, const int32_t ny, const int32_t nz, const float cell_size, const float dt) {
        if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
        if (cell_size <= 0.0f) return 1002;
        if (dt <= 0.0f) return 1003;
        return 0;
    }

    int32_t validate_boundaries(const uint32_t boundary_x_min, const uint32_t boundary_x_max, const uint32_t boundary_y_min, const uint32_t boundary_y_max, const uint32_t boundary_z_min, const uint32_t boundary_z_max) {
        if (boundary_x_min > STABLE_FLUIDS_BOUNDARY_OUTFLOW || boundary_x_max > STABLE_FLUIDS_BOUNDARY_OUTFLOW || boundary_y_min > STABLE_FLUIDS_BOUNDARY_OUTFLOW || boundary_y_max > STABLE_FLUIDS_BOUNDARY_OUTFLOW || boundary_z_min > STABLE_FLUIDS_BOUNDARY_OUTFLOW || boundary_z_max > STABLE_FLUIDS_BOUNDARY_OUTFLOW) return 1005;
        return 0;
    }

} // namespace

int32_t stable_fluids_validate_advect_velocity_desc(const StableFluidsAdvectVelocityDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAdvectVelocityDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_velocity_x == nullptr) return 2008;
    if (desc->temporary_velocity_y == nullptr) return 2009;
    if (desc->temporary_velocity_z == nullptr) return 2010;
    if (desc->temporary_previous_velocity_x == nullptr) return 2012;
    if (desc->temporary_previous_velocity_y == nullptr) return 2013;
    if (desc->temporary_previous_velocity_z == nullptr) return 2014;
    return 0;
}

int32_t stable_fluids_validate_diffuse_velocity_desc(const StableFluidsDiffuseVelocityDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsDiffuseVelocityDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (desc->diffuse_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_velocity_x == nullptr) return 2008;
    if (desc->temporary_velocity_y == nullptr) return 2009;
    if (desc->temporary_velocity_z == nullptr) return 2010;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_previous_density == nullptr) return 2011;
    return 0;
}

int32_t stable_fluids_validate_project_desc(const StableFluidsProjectDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsProjectDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->cell_size <= 0.0f) return 1002;
    if (desc->pressure_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_pressure == nullptr) return 2015;
    if (desc->temporary_divergence == nullptr) return 2016;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_previous_density == nullptr) return 2011;
    return 0;
}


int32_t stable_fluids_validate_advect_scalar_desc(const StableFluidsAdvectScalarDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAdvectScalarDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->scalar == nullptr) return 2001;
    if (desc->temporary_scalar == nullptr) return 2007;
    if (desc->temporary_previous_scalar == nullptr) return 2011;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    return 0;
}

int32_t stable_fluids_validate_diffuse_scalar_desc(const StableFluidsDiffuseScalarDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsDiffuseScalarDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (desc->diffuse_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->scalar == nullptr) return 2001;
    if (desc->temporary_scalar == nullptr) return 2007;
    if (desc->temporary_solution_storage == nullptr) return 2015;
    if (desc->temporary_rhs_storage == nullptr) return 2016;
    return 0;
}

int32_t stable_fluids_validate_add_scalar_source_desc(const StableFluidsAddScalarSourceDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAddScalarSourceDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->scalar == nullptr) return 2001;
    return 0;
}

int32_t stable_fluids_validate_add_vector_source_desc(const StableFluidsAddVectorSourceDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAddVectorSourceDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->vector_x == nullptr) return 2003;
    if (desc->vector_y == nullptr) return 2004;
    if (desc->vector_z == nullptr) return 2005;
    return 0;
}

int32_t stable_fluids_validate_add_force_desc(const StableFluidsAddForceDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAddForceDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->dt <= 0.0f) return 1003;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;

    const bool has_buoyancy = desc->density_buoyancy != 0.0f || desc->temperature_buoyancy != 0.0f;

    if (desc->force_x == nullptr && (desc->force_y != nullptr || desc->force_z != nullptr)) return 2018;
    if (desc->force_y == nullptr && (desc->force_x != nullptr || desc->force_z != nullptr)) return 2019;
    if (desc->force_z == nullptr && (desc->force_x != nullptr || desc->force_y != nullptr)) return 2020;

    if (has_buoyancy && desc->density == nullptr) return 2001;
    if (desc->temperature_buoyancy != 0.0f && desc->temperature == nullptr) return 2017;
    return 0;
}

int32_t stable_fluids_validate_compute_staggered_velocity_magnitude_desc(const StableFluidsComputeStaggeredVelocityMagnitudeDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsComputeStaggeredVelocityMagnitudeDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->destination == nullptr) return 2002;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    return 0;
}

int32_t stable_fluids_validate_pack_smoke_rgba_desc(const StableFluidsPackSmokeRgbaDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsPackSmokeRgbaDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->destination_rgba == nullptr) return 2002;
    if (desc->density == nullptr) return 2001;
    if (desc->dye_r == nullptr || desc->dye_g == nullptr || desc->dye_b == nullptr) return 2006;
    return 0;
}

} // extern "C"
