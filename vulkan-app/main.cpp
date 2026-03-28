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

        app::create_runtime_data(data);
        app::check_interop_support(*renderer);
        app::apply_field_visual_preset(state);

        renderer->vk_context().device.waitIdle();
        app::rebuild_physics(state, data);
        app::sync_capture_storage(data, *renderer);
        app::capture_snapshot(state, data, *renderer, "stable_fluids.initial_snapshot");
        if (const auto snapshot = app::active_snapshot(data)) renderer->frame_content(state.ui.render, *snapshot);

        while (!renderer->should_close()) {
            renderer->begin_frame();

            bool reset_requested = false;
            bool field_changed   = false;
            auto snapshot        = app::active_snapshot(data);
            renderer->draw_visualization_ui(state, data, reset_requested, field_changed, snapshot);

            if (reset_requested) {
                renderer->vk_context().device.waitIdle();
                app::rebuild_physics(state, data);
                app::sync_capture_storage(data, *renderer);
                app::capture_snapshot(state, data, *renderer, "stable_fluids.reset_snapshot");
                snapshot = app::active_snapshot(data);
                if (snapshot) renderer->frame_content(state.ui.render, *snapshot);
            } else {
                if (!state.ui.playback.paused) app::step_physics(state, data, 1);
                if (!state.ui.playback.paused || field_changed || !snapshot) app::capture_snapshot(state, data, *renderer, "stable_fluids.frame_snapshot");
            }

            snapshot = app::active_snapshot(data);
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
