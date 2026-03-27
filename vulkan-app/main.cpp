#include "stable_simulation.hpp"

#include <cuda_runtime.h>
#include <imgui.h>
#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

import app;
import std;
import viz.snapshot;

namespace {

    constexpr uint32_t snapshot_slot_count = 4;

    struct PlaybackState {
        bool paused             = false;
        bool step_once          = false;
        int sim_steps_per_frame = 1;
        int snapshot_interval   = 2;
    };

    constexpr std::array boundary_labels{
        "No-slip",
        "Free-slip",
        "Inflow",
        "Outflow",
    };

    constexpr std::array collider_type_labels{
        "Sphere",
        "Box",
    };

    constexpr std::array scene_preset_labels{
        "Dual Jet Collider",
        "Smoke Plume",
        "Custom",
    };

} // namespace

int main() {
    try {
        app::VisualizationApp renderer;
        smoke::StableSimulation simulation{};
        viz::snapshot::SnapshotSet snapshots{};
        PlaybackState playback{};

        auto current_field = [&]() -> const smoke::FieldInfo& {
            auto& selected = simulation.settings().selected_field;
            const auto fields = simulation.fields();
            selected = std::clamp(selected, 0, static_cast<int>(fields.size()) - 1);
            return fields[static_cast<size_t>(selected)];
        };
        auto to_app_semantic = [](const smoke::FieldSemantic semantic) {
            if (semantic == smoke::FieldSemantic::Density) return app::FieldSemantic::Density;
            if (semantic == smoke::FieldSemantic::VelocityMagnitude) return app::FieldSemantic::VelocityMagnitude;
            if (semantic == smoke::FieldSemantic::DyeColor) return app::FieldSemantic::DyeColor;
            return app::FieldSemantic::GenericScalar;
        };
        auto to_app_collider = [&]() {
            const auto collider = simulation.collider_overlay();
            return app::ColliderOverlay{
                .enabled = collider.enabled,
                .type = collider.type,
                .center_x = collider.center_x,
                .center_y = collider.center_y,
                .center_z = collider.center_z,
                .radius = collider.radius,
                .half_x = collider.half_x,
                .half_y = collider.half_y,
                .half_z = collider.half_z,
            };
        };
        auto make_capture_request = [&]() {
            const auto grid = simulation.grid_desc();
            return viz::snapshot::CaptureRequest{
                .grid = {
                    .nx = static_cast<uint32_t>(grid.nx),
                    .ny = static_cast<uint32_t>(grid.ny),
                    .nz = static_cast<uint32_t>((std::max)(grid.nz, 1)),
                    .cell_size = grid.cell_size,
                },
                .field_component_count = current_field().component_count,
                .semantic = to_app_semantic(current_field().semantic),
                .label = current_field().label,
                .export_velocity_host = renderer.settings().show_velocity_plane,
            };
        };
        auto apply_field_defaults = [&](const smoke::FieldInfo& field) {
            auto& settings = renderer.settings();
            settings.render_mode = field.preset.display_mode == smoke::FieldDisplayMode::Smoke ? app::RenderMode::Smoke : app::RenderMode::Scalar;
            settings.density_scale = field.preset.density_scale;
            settings.absorption = field.preset.absorption;
            settings.scalar_min = field.preset.scalar_min;
            settings.scalar_max = field.preset.scalar_max;
            settings.scalar_opacity = field.preset.scalar_opacity;
            settings.scalar_low_r = field.preset.scalar_low_r;
            settings.scalar_low_g = field.preset.scalar_low_g;
            settings.scalar_low_b = field.preset.scalar_low_b;
            settings.scalar_high_r = field.preset.scalar_high_r;
            settings.scalar_high_g = field.preset.scalar_high_g;
            settings.scalar_high_b = field.preset.scalar_high_b;
        };
        auto check_interop_support = [&]() {
            const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
            if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("stable-fluids visualizer requires Vulkan timeline semaphore support");
            int cuda_device_index = 0;
            if (cudaGetDevice(&cuda_device_index) != cudaSuccess) throw std::runtime_error("cudaGetDevice failed");
            int timeline_supported = 0;
            if (cudaDeviceGetAttribute(&timeline_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index) != cudaSuccess) throw std::runtime_error("cudaDeviceGetAttribute failed");
            if (timeline_supported == 0) throw std::runtime_error("CUDA timeline semaphore interop is required");
        };
        auto capture_snapshot = [&](const int slot_index, const char* tag) {
            nvtx3::scoped_range range{tag};
            const auto request = make_capture_request();
            const auto begin = std::chrono::steady_clock::now();
            const auto capture = snapshots.begin_capture(slot_index);
            simulation.export_field(current_field().id, capture.field_cuda_ptr);
            if (request.export_velocity_host) {
                simulation.export_velocity(capture.velocity_cuda_ptr);
                if (cudaMemcpyAsync(capture.velocity_host_ptr, capture.velocity_cuda_ptr, snapshots.stats().velocity_bytes, cudaMemcpyDeviceToHost, simulation.stream()) != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync velocity snapshot failed");
            }
            cudaExternalSemaphoreSignalParams signal_params{};
            signal_params.params.fence.value = capture.ready_generation;
            if (cudaSignalExternalSemaphoresAsync(&capture.external_semaphore, &signal_params, 1, simulation.stream()) != cudaSuccess) throw std::runtime_error("cudaSignalExternalSemaphoresAsync failed");
            if (cudaStreamSynchronize(simulation.stream()) != cudaSuccess) throw std::runtime_error("cudaStreamSynchronize failed");
            snapshots.complete_capture(slot_index, request, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        };
        auto ensure_snapshot_storage = [&]() {
            const auto request = make_capture_request();
            if (snapshots.matches(request)) return false;
            renderer.vk_context().device.waitIdle();
            auto descriptor_sets = renderer.allocate_field_descriptor_sets(snapshot_slot_count);
            snapshots.reset(renderer.vk_context(), descriptor_sets, request);
            return true;
        };
        auto reset_backend = [&]() {
            nvtx3::scoped_range range{"stable_fluids.reset_backend"};
            check_interop_support();
            renderer.vk_context().device.waitIdle();
            simulation.rebuild();
            if (simulation.settings().emit_source) simulation.step(1);
            ensure_snapshot_storage();
            const int slot_index = snapshots.find_available_slot(renderer.frames_in_flight());
            if (slot_index >= 0) capture_snapshot(slot_index, "stable_fluids.initial_snapshot");
            if (const auto snapshot = snapshots.active_snapshot(to_app_collider())) renderer.frame_content(*snapshot);
        };

        apply_field_defaults(current_field());
        reset_backend();

        while (!renderer.should_close()) {
            nvtx3::scoped_range frame_range{"stable_fluids.frame"};
            renderer.begin_frame();

            bool reset_requested = false;
            bool field_changed = false;

            ImGui::Begin("Simulation");
            auto& settings = simulation.settings();
            const float extent_x = static_cast<float>(settings.config.nx) * settings.config.cell_size;
            const float extent_y = static_cast<float>(settings.config.ny) * settings.config.cell_size;
            const float extent_z = static_cast<float>(settings.config.nz) * settings.config.cell_size;
            const float min_extent = (std::min)({extent_x, extent_y, extent_z});
            auto mark_scene_custom = [&]() {
                if (settings.scene_preset != smoke::ScenePreset::Custom) settings.scene_preset = smoke::ScenePreset::Custom;
            };
            auto request_scene_reset = [&]() {
                mark_scene_custom();
                reset_requested = true;
            };

            if (ImGui::BeginCombo("Field", current_field().label.data())) {
                for (int i = 0; i < static_cast<int>(simulation.fields().size()); ++i) {
                    const bool is_selected = settings.selected_field == i;
                    if (ImGui::Selectable(simulation.fields()[static_cast<size_t>(i)].label.data(), is_selected)) {
                        settings.selected_field = i;
                        field_changed = true;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            int scene_preset = std::clamp(static_cast<int>(simulation.scene_preset()), 0, static_cast<int>(scene_preset_labels.size()) - 1);
            if (ImGui::BeginCombo("Scene Preset", scene_preset_labels[static_cast<size_t>(scene_preset)])) {
                for (int i = 0; i < static_cast<int>(scene_preset_labels.size()); ++i) {
                    const bool is_selected = scene_preset == i;
                    if (ImGui::Selectable(scene_preset_labels[static_cast<size_t>(i)], is_selected)) {
                        if (i < static_cast<int>(smoke::ScenePreset::Custom)) {
                            simulation.apply_scene_preset(static_cast<smoke::ScenePreset>(i));
                            reset_requested = true;
                        } else {
                            settings.scene_preset = smoke::ScenePreset::Custom;
                        }
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Pause Simulation", &playback.paused);
            if (ImGui::Button("Single Step")) playback.step_once = true;
            ImGui::SameLine();
            if (ImGui::Button("Reset Backend")) reset_requested = true;
            ImGui::SliderInt("Sim Steps / Frame", &playback.sim_steps_per_frame, 1, 8);
            ImGui::SliderInt("Snapshot Interval", &playback.snapshot_interval, 1, 8);

            ImGui::Separator();
            ImGui::TextUnformatted("Solver");
            ImGui::Text("Step calls: %llu", static_cast<unsigned long long>(simulation.stats().step_count));
            ImGui::Text("Last step call: %.3f ms", simulation.stats().last_step_call_ms);
            ImGui::Text("Avg step call: %.3f ms", simulation.stats().average_step_call_ms);
            ImGui::Text("Projection max |div|: %.6g", simulation.stats().projection_max_abs_divergence);
            ImGui::Text("Projection RMS div: %.6g", simulation.stats().projection_rms_divergence);
            ImGui::Text("Snapshot commits: %llu", static_cast<unsigned long long>(snapshots.stats().snapshot_count));
            ImGui::Text("Last snapshot: %.3f ms", snapshots.stats().last_snapshot_ms);
            ImGui::Text("Avg snapshot: %.3f ms", snapshots.stats().average_snapshot_ms);

            ImGui::Separator();
            ImGui::TextUnformatted("Grid / Time");
            ImGui::Text("Domain Size: %.3f m x %.3f m x %.3f m", extent_x, extent_y, extent_z);
            if (ImGui::SliderInt("Grid X (cells)", &settings.config.nx, 16, 512)) request_scene_reset();
            if (ImGui::SliderInt("Grid Y (cells)", &settings.config.ny, 16, 512)) request_scene_reset();
            if (ImGui::SliderInt("Grid Z (cells)", &settings.config.nz, 16, 512)) request_scene_reset();
            if (ImGui::SliderFloat("Dt (s)", &settings.config.dt, 1.0f / 480.0f, 1.0f / 24.0f, "%.5f")) request_scene_reset();
            if (ImGui::SliderFloat("Cell Size (m)", &settings.config.cell_size, 0.0025f, 0.05f, "%.4f")) request_scene_reset();
            if (ImGui::SliderFloat("Viscosity (m^2/s)", &settings.config.viscosity, 0.0f, 0.002f, "%.5f")) request_scene_reset();
            if (ImGui::SliderFloat("Density Diffusion (m^2/s)", &settings.density_diffusion, 0.0f, 0.002f, "%.5f")) request_scene_reset();
            if (ImGui::SliderFloat("Dye Diffusion (m^2/s)", &settings.dye_diffusion, 0.0f, 0.002f, "%.5f")) request_scene_reset();
            if (ImGui::SliderInt("Diffuse Iterations", &settings.config.diffuse_iterations, 1, 64)) request_scene_reset();
            if (ImGui::SliderInt("Pressure Iterations", &settings.config.pressure_iterations, 4, 192)) request_scene_reset();

            auto draw_boundary_combo = [&](const char* label, StableFluidsBoundaryFaceDesc& face) {
                int boundary = std::clamp(static_cast<int>(face.type), 0, static_cast<int>(boundary_labels.size()) - 1);
                if (ImGui::BeginCombo(label, boundary_labels[static_cast<size_t>(boundary)])) {
                    for (int i = 0; i < static_cast<int>(boundary_labels.size()); ++i) {
                        const bool is_selected = boundary == i;
                        if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                            boundary = i;
                            face.type = static_cast<uint32_t>(i);
                            request_scene_reset();
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            };

            ImGui::Separator();
            ImGui::TextUnformatted("Domain Boundary");
            draw_boundary_combo("Boundary X-", settings.config.domain_boundary.x_min);
            draw_boundary_combo("Boundary X+", settings.config.domain_boundary.x_max);
            draw_boundary_combo("Boundary Y-", settings.config.domain_boundary.y_min);
            draw_boundary_combo("Boundary Y+", settings.config.domain_boundary.y_max);
            draw_boundary_combo("Boundary Z-", settings.config.domain_boundary.z_min);
            draw_boundary_combo("Boundary Z+", settings.config.domain_boundary.z_max);
            if (ImGui::SliderFloat("Inflow Vel X- (m/s)", &settings.config.domain_boundary.x_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat("Inflow Vel X+ (m/s)", &settings.config.domain_boundary.x_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat("Inflow Vel Y- (m/s)", &settings.config.domain_boundary.y_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat("Inflow Vel Y+ (m/s)", &settings.config.domain_boundary.y_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat("Inflow Vel Z- (m/s)", &settings.config.domain_boundary.z_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat("Inflow Vel Z+ (m/s)", &settings.config.domain_boundary.z_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();

            ImGui::Separator();
            ImGui::TextUnformatted("Forces");
            if (ImGui::SliderFloat("Gravity Y (m/s^2)", &settings.gravity_y, -20.0f, 20.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Buoyancy Beta", &settings.buoyancy_beta, 0.0f, 2.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Ambient Density", &settings.ambient_density, 0.0f, 2.0f, "%.3f")) request_scene_reset();
            ImGui::Text("Buoyancy accel / density: %.3f m/s^2", -settings.gravity_y * settings.buoyancy_beta);
            if (ImGui::SliderFloat("Uniform Force X (m/s^2)", &settings.config.uniform_force_x, -20.0f, 20.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Uniform Force Y (m/s^2)", &settings.config.uniform_force_y, -20.0f, 20.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Uniform Force Z (m/s^2)", &settings.config.uniform_force_z, -20.0f, 20.0f, "%.3f")) request_scene_reset();

            ImGui::Separator();
            ImGui::TextUnformatted("Sources");
            if (ImGui::Checkbox("Emit Source", &settings.emit_source)) request_scene_reset();
            auto draw_emitter_controls = [&](const char* label, smoke::SourceEmitterSettings& emitter) {
                if (!ImGui::TreeNode(label)) return;
                if (ImGui::Checkbox((std::string("Enabled##") + label).c_str(), &emitter.enabled)) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Center X (m)##") + label).c_str(), &emitter.center_x, 0.0f, extent_x, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Center Y (m)##") + label).c_str(), &emitter.center_y, 0.0f, extent_y, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Center Z (m)##") + label).c_str(), &emitter.center_z, 0.0f, extent_z, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Dir X##") + label).c_str(), &emitter.direction_x, -1.0f, 1.0f, "%.2f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Dir Y##") + label).c_str(), &emitter.direction_y, -1.0f, 1.0f, "%.2f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Dir Z##") + label).c_str(), &emitter.direction_z, -1.0f, 1.0f, "%.2f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Speed (m/s)##") + label).c_str(), &emitter.speed, 0.0f, 5.0f, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Radius (m)##") + label).c_str(), &emitter.radius, settings.config.cell_size, min_extent * 0.25f, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Density##") + label).c_str(), &emitter.density_amount, 0.0f, 3.0f, "%.2f")) request_scene_reset();
                if (ImGui::SliderFloat((std::string("Dye##") + label).c_str(), &emitter.dye_amount, 0.0f, 3.0f, "%.2f")) request_scene_reset();
                if (ImGui::ColorEdit3((std::string("Color##") + label).c_str(), &emitter.color_r)) request_scene_reset();
                ImGui::TreePop();
            };
            draw_emitter_controls("Emitter A", settings.emitter_a);
            draw_emitter_controls("Emitter B", settings.emitter_b);

            ImGui::Separator();
            ImGui::TextUnformatted("Collider");
            if (ImGui::Checkbox("Enable Collider", &settings.collider.enabled)) request_scene_reset();
            int collider_type = std::clamp(settings.collider.type, 0, static_cast<int>(collider_type_labels.size()) - 1);
            if (ImGui::BeginCombo("Collider Type", collider_type_labels[static_cast<size_t>(collider_type)])) {
                for (int i = 0; i < static_cast<int>(collider_type_labels.size()); ++i) {
                    const bool is_selected = collider_type == i;
                    if (ImGui::Selectable(collider_type_labels[static_cast<size_t>(i)], is_selected)) {
                        settings.collider.type = i;
                        request_scene_reset();
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            int collider_boundary = std::clamp(static_cast<int>(settings.collider.boundary), 0, 1);
            if (ImGui::BeginCombo("Collider Boundary", boundary_labels[static_cast<size_t>(collider_boundary)])) {
                for (int i = 0; i < 2; ++i) {
                    const bool is_selected = collider_boundary == i;
                    if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                        settings.collider.boundary = static_cast<uint32_t>(i);
                        request_scene_reset();
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (ImGui::SliderFloat("Collider X (m)", &settings.collider.center_x, 0.0f, extent_x, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Collider Y (m)", &settings.collider.center_y, 0.0f, extent_y, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Collider Z (m)", &settings.collider.center_z, 0.0f, extent_z, "%.3f")) request_scene_reset();
            if (settings.collider.type == 0) {
                if (ImGui::SliderFloat("Collider Radius (m)", &settings.collider.radius, settings.config.cell_size, min_extent * 0.45f, "%.3f")) request_scene_reset();
            } else {
                if (ImGui::SliderFloat("Half Extent X (m)", &settings.collider.half_extent_x, settings.config.cell_size, extent_x * 0.45f, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat("Half Extent Y (m)", &settings.collider.half_extent_y, settings.config.cell_size, extent_y * 0.45f, "%.3f")) request_scene_reset();
                if (ImGui::SliderFloat("Half Extent Z (m)", &settings.collider.half_extent_z, settings.config.cell_size, extent_z * 0.45f, "%.3f")) request_scene_reset();
            }
            if (ImGui::SliderFloat("Collider Vel X (m/s)", &settings.collider.velocity_x, -3.0f, 3.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Collider Vel Y (m/s)", &settings.collider.velocity_y, -3.0f, 3.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Collider Vel Z (m/s)", &settings.collider.velocity_z, -3.0f, 3.0f, "%.3f")) request_scene_reset();
            ImGui::End();

            if (field_changed) apply_field_defaults(current_field());

            if (reset_requested) {
                reset_backend();
            } else if (field_changed && ensure_snapshot_storage()) {
                const int slot_index = snapshots.find_available_slot(renderer.frames_in_flight());
                if (slot_index >= 0) capture_snapshot(slot_index, "stable_fluids.field_change_realloc");
                if (const auto snapshot = snapshots.active_snapshot(to_app_collider())) renderer.frame_content(*snapshot);
            }

            auto snapshot = snapshots.active_snapshot(to_app_collider());
            renderer.draw_visualization_ui(snapshot);
            if (ensure_snapshot_storage()) {
                const int slot_index = snapshots.find_available_slot(renderer.frames_in_flight());
                if (slot_index >= 0) capture_snapshot(slot_index, "stable_fluids.visual_storage_reset");
                snapshot = snapshots.active_snapshot(to_app_collider());
                if (snapshot) renderer.frame_content(*snapshot);
            }

            if (!reset_requested) {
                const bool run_simulation = !playback.paused || playback.step_once;
                if (run_simulation) {
                    simulation.step(playback.sim_steps_per_frame);
                    if (snapshots.stats().steps_since_snapshot < static_cast<uint32_t>(playback.snapshot_interval)) ++snapshots.stats().steps_since_snapshot;
                    if (snapshots.stats().steps_since_snapshot >= static_cast<uint32_t>(playback.snapshot_interval)) {
                        const int slot_index = snapshots.find_available_slot(renderer.frames_in_flight());
                        if (slot_index >= 0) capture_snapshot(slot_index, "stable_fluids.simulation_snapshot");
                    }
                }
            }

            playback.step_once = false;
            if ((field_changed || !snapshot) && !reset_requested) {
                const int slot_index = snapshots.find_available_slot(renderer.frames_in_flight());
                if (slot_index >= 0) capture_snapshot(slot_index, "stable_fluids.refresh_snapshot");
            }

            snapshot = snapshots.active_snapshot(to_app_collider());
            const bool submitted = renderer.render_frame(snapshot);
            if (submitted) snapshots.mark_submitted(renderer.frames_in_flight());
        }

        renderer.vk_context().device.waitIdle();
        snapshots.destroy();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
