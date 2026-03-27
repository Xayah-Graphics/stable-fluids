#pragma once

#include "stable-fluids-3d.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <string_view>

namespace smoke {

    enum class FieldId : uint32_t {
        SmokeColor        = 0,
        Density           = 1,
        VelocityMagnitude = 2,
        SolidMask         = 3,
        Pressure          = 4,
        Divergence        = 5,
    };

    enum class ScenePreset : uint32_t {
        DualJetCollider = 0,
        SmokePlume      = 1,
        Custom          = 2,
    };

    enum class FieldSemantic : uint32_t {
        Density           = 0,
        VelocityMagnitude = 1,
        DyeColor          = 2,
        GenericScalar     = 3,
    };

    enum class FieldDisplayMode : uint32_t {
        Scalar = 0,
        Smoke  = 1,
    };

    struct SolverStats {
        double last_step_call_ms    = 0.0;
        double average_step_call_ms = 0.0;
        uint64_t step_count         = 0;
        float projection_max_abs_divergence = 0.0f;
        float projection_rms_divergence     = 0.0f;
    };

    struct FieldVisualPreset {
        FieldDisplayMode display_mode = FieldDisplayMode::Scalar;
        float density_scale           = 0.95f;
        float absorption              = 1.20f;
        float scalar_min              = 0.0f;
        float scalar_max              = 1.0f;
        float scalar_opacity          = 2.0f;
        float scalar_low_r            = 0.08f;
        float scalar_low_g            = 0.18f;
        float scalar_low_b            = 0.46f;
        float scalar_high_r           = 0.98f;
        float scalar_high_g           = 0.82f;
        float scalar_high_b           = 0.24f;
    };

    struct FieldInfo {
        FieldId id{};
        std::string_view label{};
        uint32_t component_count = 1;
        FieldSemantic semantic   = FieldSemantic::GenericScalar;
        FieldVisualPreset preset{};
    };

    struct ColliderOverlay {
        bool enabled     = false;
        uint32_t type    = 0;
        float center_x   = 0.0f;
        float center_y   = 0.0f;
        float center_z   = 0.0f;
        float radius     = 0.0f;
        float half_x     = 0.0f;
        float half_y     = 0.0f;
        float half_z     = 0.0f;
    };

    struct ColliderSettings {
        bool enabled        = false;
        int type            = 0;
        float center_x      = 0.50f;
        float center_y      = 0.50f;
        float center_z      = 0.50f;
        float radius        = 0.20f;
        float half_extent_x = 0.10f;
        float half_extent_y = 0.08f;
        float half_extent_z = 0.10f;
        float velocity_x    = 0.0f;
        float velocity_y    = 0.0f;
        float velocity_z    = 0.0f;
        uint32_t boundary   = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP);
    };

    struct SourceEmitterSettings {
        bool enabled         = false;
        float center_x       = 0.50f;
        float center_y       = 0.10f;
        float center_z       = 0.50f;
        float direction_x    = 0.0f;
        float direction_y    = 1.0f;
        float direction_z    = 0.0f;
        float speed          = 0.0f;
        float radius         = 0.03f;
        float density_amount = 0.0f;
        float dye_amount     = 0.0f;
        float color_r        = 1.00f;
        float color_g        = 1.00f;
        float color_b        = 1.00f;
    };

    struct Settings {
        ScenePreset scene_preset = ScenePreset::Custom;
        StableFluidsSimulationConfig config{
            .nx = 100,
            .ny = 100,
            .nz = 100,
            .cell_size = 0.01f,
            .dt = 1.0f / 120.0f,
            .viscosity = 0.f,
            .diffuse_iterations = 24,
            .pressure_iterations = 96,
            .uniform_force_x = 0.0f,
            .uniform_force_y = 0.0f,
            .uniform_force_z = 0.0f,
            .domain_boundary = {
                .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            },
            .block_x = 8,
            .block_y = 8,
            .block_z = 4,
        };
        float density_diffusion = 0.00005f;
        float dye_diffusion     = 0.00003f;
        float gravity_y         = -9.81f;
        float buoyancy_beta     = 0.10f;
        float ambient_density   = 0.0f;
        int selected_field      = 0;
        bool emit_source        = true;
        SourceEmitterSettings emitter_a{};
        SourceEmitterSettings emitter_b{};
        ColliderSettings collider{};
    };

    class StableSimulation {
    public:
        StableSimulation();
        ~StableSimulation();

        StableSimulation(const StableSimulation&) = delete;
        StableSimulation& operator=(const StableSimulation&) = delete;

        Settings& settings();
        [[nodiscard]] const Settings& settings() const;
        [[nodiscard]] const SolverStats& stats() const;
        [[nodiscard]] cudaStream_t stream() const;
        [[nodiscard]] ScenePreset scene_preset() const;
        [[nodiscard]] std::span<const FieldInfo> fields() const;
        [[nodiscard]] ColliderOverlay collider_overlay() const;

        void rebuild();
        void apply_scene_preset(ScenePreset preset);
        void step(int sim_steps);
        void export_field(FieldId field, void* destination) const;
        void export_velocity(void* destination) const;
        [[nodiscard]] StableFluidsGridDesc grid_desc() const;

    private:
        void update_scene();

        Settings settings_{};
        SolverStats stats_{};
        cudaStream_t stream_              = nullptr;
        StableFluidsContext context_      = nullptr;
        StableFluidsFieldHandle density_field_ = 0;
        StableFluidsFieldHandle dye_field_     = 0;
    };

} // namespace smoke
