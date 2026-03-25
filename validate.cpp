#include "stable-fluids.h"

extern "C" {

int32_t stable_fluids_validate_desc(const StableFluidsStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(StableFluidsStepDesc)) return 1000;
    if (desc->api_version != STABLE_FLUIDS_API_VERSION) return 1006;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->cell_size <= 0.0f) return 1002;
    if (desc->dt <= 0.0f) return 1003;
    if (desc->diffuse_iterations <= 0 || desc->pressure_iterations <= 0) return 1004;
    if (desc->boundary_x_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || desc->boundary_x_max > STABLE_FLUIDS_BOUNDARY_PERIODIC || desc->boundary_y_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || desc->boundary_y_max > STABLE_FLUIDS_BOUNDARY_PERIODIC || desc->boundary_z_min > STABLE_FLUIDS_BOUNDARY_PERIODIC || desc->boundary_z_max > STABLE_FLUIDS_BOUNDARY_PERIODIC)
        return 1005;
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

} // extern "C"
