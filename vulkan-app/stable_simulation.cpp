#include "stable_simulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace smoke {

    namespace {

        constexpr std::array field_catalog{
            FieldInfo{
                .id = FieldId::SmokeColor,
                .label = "Smoke Color",
                .component_count = 4,
                .semantic = FieldSemantic::DyeColor,
                .preset = {
                    .display_mode = FieldDisplayMode::Smoke,
                    .density_scale = 1.60f,
                    .absorption = 2.40f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.0f,
                    .scalar_opacity = 2.0f,
                    .scalar_low_r = 0.08f,
                    .scalar_low_g = 0.18f,
                    .scalar_low_b = 0.46f,
                    .scalar_high_r = 0.98f,
                    .scalar_high_g = 0.82f,
                    .scalar_high_b = 0.24f,
                },
            },
            FieldInfo{
                .id = FieldId::Density,
                .label = "Density",
                .component_count = 1,
                .semantic = FieldSemantic::Density,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.40f,
                    .scalar_opacity = 3.00f,
                    .scalar_low_r = 0.10f,
                    .scalar_low_g = 0.08f,
                    .scalar_low_b = 0.30f,
                    .scalar_high_r = 1.00f,
                    .scalar_high_g = 0.24f,
                    .scalar_high_b = 0.74f,
                },
            },
            FieldInfo{
                .id = FieldId::VelocityMagnitude,
                .label = "Velocity Magnitude",
                .component_count = 1,
                .semantic = FieldSemantic::VelocityMagnitude,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.20f,
                    .scalar_opacity = 1.80f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
            FieldInfo{
                .id = FieldId::SolidMask,
                .label = "Solid Mask",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.0f,
                    .scalar_opacity = 3.2f,
                    .scalar_low_r = 0.05f,
                    .scalar_low_g = 0.06f,
                    .scalar_low_b = 0.07f,
                    .scalar_high_r = 0.94f,
                    .scalar_high_g = 0.92f,
                    .scalar_high_b = 0.88f,
                },
            },
            FieldInfo{
                .id = FieldId::Pressure,
                .label = "Pressure",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = -0.12f,
                    .scalar_max = 0.12f,
                    .scalar_opacity = 2.20f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
            FieldInfo{
                .id = FieldId::Divergence,
                .label = "Divergence",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = -120.0f,
                    .scalar_max = 120.0f,
                    .scalar_opacity = 1.80f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
        };

        void check_cuda(const cudaError_t status) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string("cudaStreamCreateWithFlags") + ": " + cudaGetErrorString(status));
        }

        void check_stable(const StableFluidsResult code, const char* what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        }

        [[nodiscard]] Settings make_settings_for_preset(const ScenePreset preset) {
            Settings settings{};
            settings.scene_preset = preset;
            settings.config.domain_boundary = {
                .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
            };
            settings.density_diffusion = 0.00005f;
            settings.dye_diffusion = 0.00003f;
            settings.gravity_y = -9.81f;
            settings.buoyancy_beta = 0.10f;
            settings.ambient_density = 0.0f;
            settings.emit_source = true;
            settings.collider = {
                .enabled = true,
                .type = 0,
                .center_x = 0.50f,
                .center_y = 0.40f,
                .center_z = 0.50f,
                .radius = 0.13f,
                .half_extent_x = 0.10f,
                .half_extent_y = 0.08f,
                .half_extent_z = 0.10f,
                .velocity_x = 0.0f,
                .velocity_y = 0.0f,
                .velocity_z = 0.0f,
                .boundary = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP),
            };

            if (preset == ScenePreset::SmokePlume) {
                settings.config.dt = 1.0f / 120.0f;
                settings.config.pressure_iterations = 120;
                settings.config.domain_boundary = {
                    .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                    .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                    .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                    .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                    .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                    .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                };
                settings.density_diffusion = 0.00001f;
                settings.dye_diffusion = 0.000008f;
                settings.gravity_y = -9.81f;
                settings.buoyancy_beta = 0.65f;
                settings.ambient_density = 0.0f;
                settings.collider.enabled = false;
                settings.emitter_a = {
                    .enabled = true,
                    .center_x = 0.50f,
                    .center_y = 0.04f,
                    .center_z = 0.50f,
                    .direction_x = 0.0f,
                    .direction_y = 1.0f,
                    .direction_z = 0.0f,
                    .speed = 0.03f,
                    .radius = 0.035f,
                    .density_amount = 1.40f,
                    .dye_amount = 1.10f,
                    .color_r = 0.95f,
                    .color_g = 0.92f,
                    .color_b = 0.86f,
                };
                settings.emitter_b = {
                    .enabled = false,
                    .center_x = 0.50f,
                    .center_y = 0.10f,
                    .center_z = 0.50f,
                    .direction_x = 0.0f,
                    .direction_y = 1.0f,
                    .direction_z = 0.0f,
                    .speed = 0.0f,
                    .radius = 0.04f,
                    .density_amount = 0.0f,
                    .dye_amount = 0.0f,
                    .color_r = 1.0f,
                    .color_g = 1.0f,
                    .color_b = 1.0f,
                };
                return settings;
            }

            settings.scene_preset = ScenePreset::DualJetCollider;
            settings.config.dt = 1.0f / 120.0f;
            settings.config.pressure_iterations = 96;
            settings.density_diffusion = 0.00003f;
            settings.dye_diffusion = 0.000015f;
            settings.gravity_y = -9.81f;
            settings.buoyancy_beta = 0.12f;
            settings.ambient_density = 0.0f;
            settings.collider = {
                .enabled = true,
                .type = 0,
                .center_x = 0.50f,
                .center_y = 0.40f,
                .center_z = 0.50f,
                .radius = 0.13f,
                .half_extent_x = 0.10f,
                .half_extent_y = 0.08f,
                .half_extent_z = 0.10f,
                .velocity_x = 0.0f,
                .velocity_y = 0.0f,
                .velocity_z = 0.0f,
                .boundary = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP),
            };
            settings.emitter_a = {
                .enabled = true,
                .center_x = 0.18f,
                .center_y = 0.18f,
                .center_z = 0.76f,
                .direction_x = 0.82f,
                .direction_y = 0.34f,
                .direction_z = -0.46f,
                .speed = 0.62f,
                .radius = 0.045f,
                .density_amount = 0.80f,
                .dye_amount = 0.95f,
                .color_r = 1.00f,
                .color_g = 0.20f,
                .color_b = 0.72f,
            };
            settings.emitter_b = {
                .enabled = true,
                .center_x = 0.82f,
                .center_y = 0.18f,
                .center_z = 0.24f,
                .direction_x = -0.82f,
                .direction_y = 0.34f,
                .direction_z = 0.46f,
                .speed = 0.62f,
                .radius = 0.045f,
                .density_amount = 0.80f,
                .dye_amount = 0.95f,
                .color_r = 0.12f,
                .color_g = 0.38f,
                .color_b = 1.00f,
            };
            return settings;
        }

    } // namespace

    StableSimulation::StableSimulation() {
        settings_ = make_settings_for_preset(ScenePreset::DualJetCollider);
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        rebuild();
    }

    StableSimulation::~StableSimulation() {
        if (context_ != nullptr) stable_fluids_destroy_context_cuda(context_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    Settings& StableSimulation::settings() {
        return settings_;
    }

    const Settings& StableSimulation::settings() const {
        return settings_;
    }

    const SolverStats& StableSimulation::stats() const {
        return stats_;
    }

    cudaStream_t StableSimulation::stream() const {
        return stream_;
    }

    ScenePreset StableSimulation::scene_preset() const {
        return settings_.scene_preset;
    }

    std::span<const FieldInfo> StableSimulation::fields() const {
        return field_catalog;
    }

    const FieldInfo& StableSimulation::field_info(const FieldId field) const {
        for (const auto& info : field_catalog) {
            if (info.id == field) return info;
        }
        return field_catalog.front();
    }

    ColliderOverlay StableSimulation::collider_overlay() const {
        return ColliderOverlay{
            .enabled = settings_.collider.enabled,
            .type = static_cast<uint32_t>(settings_.collider.type),
            .center_x = settings_.collider.center_x,
            .center_y = settings_.collider.center_y,
            .center_z = settings_.collider.center_z,
            .radius = settings_.collider.radius,
            .half_x = settings_.collider.half_extent_x,
            .half_y = settings_.collider.half_extent_y,
            .half_z = settings_.collider.half_extent_z,
        };
    }

    void StableSimulation::apply_scene_preset(const ScenePreset preset) {
        const int selected_field = settings_.selected_field;
        settings_ = make_settings_for_preset(preset);
        settings_.selected_field = selected_field;
    }

    void StableSimulation::rebuild() {
        const float extent_x = static_cast<float>(settings_.config.nx) * settings_.config.cell_size;
        const float extent_y = static_cast<float>(settings_.config.ny) * settings_.config.cell_size;
        const float extent_z = static_cast<float>((std::max)(settings_.config.nz, 1)) * settings_.config.cell_size;
        const float min_extent = (std::min)({extent_x, extent_y, extent_z});
        auto clamp_emitter = [&](SourceEmitterSettings& emitter) {
            emitter.center_x = std::clamp(emitter.center_x, 0.0f, extent_x);
            emitter.center_y = std::clamp(emitter.center_y, 0.0f, extent_y);
            emitter.center_z = std::clamp(emitter.center_z, 0.0f, extent_z);
            emitter.radius = std::clamp(emitter.radius, settings_.config.cell_size, min_extent * 0.25f);
            emitter.speed = std::max(emitter.speed, 0.0f);
        };
        clamp_emitter(settings_.emitter_a);
        clamp_emitter(settings_.emitter_b);
        settings_.collider.center_x = std::clamp(settings_.collider.center_x, 0.0f, extent_x);
        settings_.collider.center_y = std::clamp(settings_.collider.center_y, 0.0f, extent_y);
        settings_.collider.center_z = std::clamp(settings_.collider.center_z, 0.0f, extent_z);
        settings_.collider.radius = std::clamp(settings_.collider.radius, settings_.config.cell_size, min_extent * 0.45f);
        settings_.collider.half_extent_x = std::clamp(settings_.collider.half_extent_x, settings_.config.cell_size, extent_x * 0.45f);
        settings_.collider.half_extent_y = std::clamp(settings_.collider.half_extent_y, settings_.config.cell_size, extent_y * 0.45f);
        settings_.collider.half_extent_z = std::clamp(settings_.collider.half_extent_z, settings_.config.cell_size, extent_z * 0.45f);

        if (context_ != nullptr) {
            check_stable(stable_fluids_destroy_context_cuda(context_), "stable_fluids_destroy_context_cuda");
            context_ = nullptr;
        }
        density_field_ = 0;
        dye_field_ = 0;

        std::array fields{
            StableFluidsFieldCreateDesc{
                .name = "density",
                .component_count = 1,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = settings_.density_diffusion,
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
            StableFluidsFieldCreateDesc{
                .name = "dye",
                .component_count = 3,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = settings_.dye_diffusion,
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
        };
        const float buoyancy_weight = -settings_.gravity_y * settings_.buoyancy_beta;
        std::array buoyancy_terms{
            StableFluidsBuoyancyDesc{
                .field_index = 0,
                .weight = buoyancy_weight,
                .ambient = settings_.ambient_density,
            },
        };
        const uint32_t buoyancy_term_count = std::abs(buoyancy_weight) > 1.0e-6f ? static_cast<uint32_t>(buoyancy_terms.size()) : 0u;
        std::array<StableFluidsFieldHandle, 2> field_handles{};
        StableFluidsContextCreateDesc create_desc{
            .config = settings_.config,
            .stream = stream_,
            .fields = fields.data(),
            .field_count = static_cast<uint32_t>(fields.size()),
            .buoyancy_terms = buoyancy_term_count > 0 ? buoyancy_terms.data() : nullptr,
            .buoyancy_term_count = buoyancy_term_count,
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &context_, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda");
        density_field_ = field_handles[0];
        dye_field_ = field_handles[1];
        update_scene();
        stats_ = {};
    }

    void StableSimulation::update_scene() {
        StableFluidsSceneDesc scene_desc{
            .colliders = nullptr,
            .collider_count = 0,
        };
        StableFluidsColliderDesc collider{
            .collider_type = static_cast<uint32_t>(settings_.collider.type == 0 ? STABLE_FLUIDS_COLLIDER_SPHERE : STABLE_FLUIDS_COLLIDER_BOX),
            .velocity_boundary_type = settings_.collider.boundary,
            .center_x = settings_.collider.center_x,
            .center_y = settings_.collider.center_y,
            .center_z = settings_.collider.center_z,
            .radius = settings_.collider.radius,
            .half_extent_x = settings_.collider.half_extent_x,
            .half_extent_y = settings_.collider.half_extent_y,
            .half_extent_z = settings_.collider.half_extent_z,
            .linear_velocity_x = settings_.collider.velocity_x,
            .linear_velocity_y = settings_.collider.velocity_y,
            .linear_velocity_z = settings_.collider.velocity_z,
        };
        if (settings_.collider.enabled) {
            scene_desc.colliders = &collider;
            scene_desc.collider_count = 1;
        }
        check_stable(stable_fluids_update_scene_cuda(context_, &scene_desc), "stable_fluids_update_scene_cuda");
    }

    void StableSimulation::step(const int sim_steps) {
        std::array<StableFluidsVelocitySourceDesc, 2> velocity_sources{};
        std::array<StableFluidsFieldSourceDesc, 4> field_sources{};
        uint32_t velocity_source_count = 0;
        uint32_t field_source_count = 0;
        const auto append_emitter = [&](const SourceEmitterSettings& emitter) {
            if (!emitter.enabled) return;
            const float dir_len = std::sqrt(emitter.direction_x * emitter.direction_x + emitter.direction_y * emitter.direction_y + emitter.direction_z * emitter.direction_z);
            const float inv_len = dir_len > 1.0e-5f ? 1.0f / dir_len : 1.0f;
            const float dir_x = dir_len > 1.0e-5f ? emitter.direction_x * inv_len : 0.0f;
            const float dir_y = dir_len > 1.0e-5f ? emitter.direction_y * inv_len : 1.0f;
            const float dir_z = dir_len > 1.0e-5f ? emitter.direction_z * inv_len : 0.0f;
            velocity_sources[velocity_source_count++] = {
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .velocity_x = dir_x * emitter.speed,
                .velocity_y = dir_y * emitter.speed,
                .velocity_z = dir_z * emitter.speed,
            };
            field_sources[field_source_count++] = {
                .field = density_field_,
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .value_0 = emitter.density_amount,
                .value_1 = 0.0f,
                .value_2 = 0.0f,
                .value_3 = 0.0f,
            };
            field_sources[field_source_count++] = {
                .field = dye_field_,
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .value_0 = emitter.dye_amount * emitter.color_r,
                .value_1 = emitter.dye_amount * emitter.color_g,
                .value_2 = emitter.dye_amount * emitter.color_b,
                .value_3 = 0.0f,
            };
        };
        append_emitter(settings_.emitter_a);
        append_emitter(settings_.emitter_b);

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            StableFluidsStepDesc step_desc{
                .velocity_sources = settings_.emit_source ? velocity_sources.data() : nullptr,
                .velocity_source_count = settings_.emit_source ? velocity_source_count : 0u,
                .field_sources = settings_.emit_source ? field_sources.data() : nullptr,
                .field_source_count = settings_.emit_source ? field_source_count : 0u,
            };
            const auto begin = std::chrono::steady_clock::now();
            check_stable(stable_fluids_step_cuda(context_, &step_desc), "stable_fluids_step_cuda");
            const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            stats_.last_step_call_ms = elapsed_ms;
            ++stats_.step_count;
            stats_.average_step_call_ms += (elapsed_ms - stats_.average_step_call_ms) / static_cast<double>(stats_.step_count);
        }
        if (sim_steps <= 0) return;
        StableFluidsProjectionMetrics metrics{};
        check_stable(stable_fluids_get_projection_metrics_cuda(context_, &metrics), "stable_fluids_get_projection_metrics_cuda");
        stats_.projection_max_abs_divergence = metrics.max_abs_divergence;
        stats_.projection_rms_divergence = metrics.rms_divergence;
    }

    void StableSimulation::export_field(const FieldId field, void* destination) const {
        if (field == FieldId::SmokeColor) {
            check_stable(stable_fluids_export_alpha_rgb_rgba_cuda(context_, density_field_, dye_field_, destination), "stable_fluids_export_alpha_rgb_rgba_cuda");
            return;
        }
        if (field == FieldId::Density) {
            check_stable(stable_fluids_export_field_components_cuda(context_, density_field_, 0, 1, destination), "stable_fluids_export_field_components_cuda");
            return;
        }
        if (field == FieldId::VelocityMagnitude) {
            check_stable(stable_fluids_export_velocity_magnitude_cuda(context_, destination), "stable_fluids_export_velocity_magnitude_cuda");
            return;
        }
        if (field == FieldId::SolidMask) {
            check_stable(stable_fluids_export_solid_mask_cuda(context_, destination), "stable_fluids_export_solid_mask_cuda");
            return;
        }
        if (field == FieldId::Pressure) {
            check_stable(stable_fluids_export_pressure_cuda(context_, destination), "stable_fluids_export_pressure_cuda");
            return;
        }
        check_stable(stable_fluids_export_divergence_cuda(context_, destination), "stable_fluids_export_divergence_cuda");
    }

    void StableSimulation::export_velocity(void* destination) const {
        check_stable(stable_fluids_export_velocity_cuda(context_, destination), "stable_fluids_export_velocity_cuda");
    }

    StableFluidsGridDesc StableSimulation::grid_desc() const {
        StableFluidsGridDesc desc{};
        check_stable(stable_fluids_get_grid_desc_cuda(context_, &desc), "stable_fluids_get_grid_desc_cuda");
        return desc;
    }

} // namespace smoke
