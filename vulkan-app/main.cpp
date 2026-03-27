#include <cstdio>
#include <cstdint>

import app;
import std;

int main() {
    app::AppState state{};
    app::AppData data{};
    std::unique_ptr<app::VisualizationApp> renderer{};

    try {
        renderer = std::make_unique<app::VisualizationApp>();

        app::apply_scene_preset(state, state.physics.preset);
        app::create_runtime_data(data);
        app::check_interop_support(*renderer);
        app::apply_field_visual_preset(state);

        renderer->vk_context().device.waitIdle();
        app::rebuild_physics(state, data);
        if (state.physics.emit_source) app::step_physics(state, data, 1);
        app::sync_capture_storage(state, data, *renderer);
        app::capture_snapshot(state, data, *renderer, "stable_fluids.initial_snapshot");
        if (const auto snapshot = app::active_snapshot(state, data)) renderer->frame_content(state.ui.render, *snapshot);

        while (!renderer->should_close()) {
            renderer->begin_frame();

            bool reset_requested = false;
            bool field_changed = false;
            app::draw_simulation_controls(state, data, reset_requested, field_changed);
            if (field_changed) app::apply_field_visual_preset(state);

            if (reset_requested) {
                renderer->vk_context().device.waitIdle();
                app::rebuild_physics(state, data);
                if (state.physics.emit_source) app::step_physics(state, data, 1);
                app::sync_capture_storage(state, data, *renderer);
                app::capture_snapshot(state, data, *renderer, "stable_fluids.initial_snapshot");
                if (const auto snapshot = app::active_snapshot(state, data)) renderer->frame_content(state.ui.render, *snapshot);
            } else if (field_changed && app::sync_capture_storage(state, data, *renderer)) {
                app::capture_snapshot(state, data, *renderer, "stable_fluids.field_change_realloc");
                if (const auto snapshot = app::active_snapshot(state, data)) renderer->frame_content(state.ui.render, *snapshot);
            }

            auto snapshot = app::active_snapshot(state, data);
            renderer->draw_visualization_ui(state.ui.render, snapshot);
            if (app::sync_capture_storage(state, data, *renderer)) {
                app::capture_snapshot(state, data, *renderer, "stable_fluids.visual_storage_reset");
                snapshot = app::active_snapshot(state, data);
                if (snapshot) renderer->frame_content(state.ui.render, *snapshot);
            }

            if (!reset_requested) {
                const bool run_simulation = !state.ui.playback.paused || state.ui.playback.step_once;
                if (run_simulation) {
                    app::step_physics(state, data, state.ui.playback.sim_steps_per_frame);
                    if (data.capture.steps_since_snapshot < static_cast<uint32_t>(state.ui.playback.snapshot_interval)) ++data.capture.steps_since_snapshot;
                    if (data.capture.steps_since_snapshot >= static_cast<uint32_t>(state.ui.playback.snapshot_interval)) app::capture_snapshot(state, data, *renderer, "stable_fluids.simulation_snapshot");
                }
            }

            state.ui.playback.step_once = false;
            if ((field_changed || !snapshot) && !reset_requested) app::capture_snapshot(state, data, *renderer, "stable_fluids.refresh_snapshot");

            snapshot = app::active_snapshot(state, data);
            const bool submitted = renderer->render_frame(state.ui.render, snapshot);
            if (submitted) app::mark_snapshot_submitted(data);
        }

        renderer->vk_context().device.waitIdle();
        app::destroy_runtime_data(data);
        return 0;
    } catch (const std::exception& e) {
        try {
            if (renderer) renderer->vk_context().device.waitIdle();
        } catch (...) {
        }
        app::destroy_runtime_data(data);
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
