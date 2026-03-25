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
    if (boundary_x_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || boundary_x_max > STABLE_FLUIDS_BOUNDARY_PERIODIC || boundary_y_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || boundary_y_max > STABLE_FLUIDS_BOUNDARY_PERIODIC || boundary_z_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || boundary_z_max > STABLE_FLUIDS_BOUNDARY_PERIODIC) return 1005;
    return 0;
}

} // namespace

int32_t stable_fluids_validate_desc(const StableFluidsStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsStepDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (desc->diffuse_iterations <= 0 || desc->pressure_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->density == nullptr) return 2001;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_velocity_x == nullptr) return 2008;
    if (desc->temporary_velocity_y == nullptr) return 2009;
    if (desc->temporary_velocity_z == nullptr) return 2010;
    if (desc->temporary_previous_density == nullptr) return 2011;
    if (desc->temporary_previous_velocity_x == nullptr) return 2012;
    if (desc->temporary_previous_velocity_y == nullptr) return 2013;
    if (desc->temporary_previous_velocity_z == nullptr) return 2014;
    if (desc->temporary_pressure == nullptr) return 2015;
    if (desc->temporary_divergence == nullptr) return 2016;
    return 0;
}

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

int32_t stable_fluids_validate_advect_density_desc(const StableFluidsAdvectDensityDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsAdvectDensityDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->density == nullptr) return 2001;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_previous_density == nullptr) return 2011;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    return 0;
}

int32_t stable_fluids_validate_diffuse_density_desc(const StableFluidsDiffuseDensityDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(StableFluidsDiffuseDensityDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (desc->diffuse_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->density == nullptr) return 2001;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_pressure == nullptr) return 2015;
    if (desc->temporary_divergence == nullptr) return 2016;
    return 0;
}

} // extern "C"
