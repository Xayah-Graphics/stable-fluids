#include <cstdint>
#include <cstdio>

import app;
import scene_plume;
import std;

int main() {
    app::AppState state{};
    app::AppData data{};
    scene_plume::Scene scene{};
    std::unique_ptr<app::VisualizationApp> renderer{};
    static_assert(app::SceneSample<scene_plume::Scene>);

    try {
        renderer = std::make_unique<app::VisualizationApp>();
        state.ui.render = scene.default_visualization();

        app::create_runtime_data(data);
        app::check_interop_support(*renderer);

        renderer->vk_context().device.waitIdle();
        scene.rebuild();
        app::sync_capture_storage(data, *renderer, scene.info().grid);
        app::capture_snapshot(state, data, scene, *renderer, "stable_fluids.initial_snapshot");
        if (const auto snapshot = app::active_snapshot(data)) renderer->frame_content(state.ui.render, *snapshot);

        while (!renderer->should_close()) {
            renderer->begin_frame();

            bool reset_requested = false;
            bool field_changed   = false;
            auto snapshot        = app::active_snapshot(data);
            renderer->draw_visualization_ui(state, scene.info(), scene.fields(), reset_requested, field_changed, snapshot);

            if (reset_requested) {
                renderer->vk_context().device.waitIdle();
                scene.rebuild();
                app::sync_capture_storage(data, *renderer, scene.info().grid);
                app::capture_snapshot(state, data, scene, *renderer, "stable_fluids.reset_snapshot");
                snapshot = app::active_snapshot(data);
                if (snapshot) renderer->frame_content(state.ui.render, *snapshot);
            } else {
                if (!state.ui.playback.paused) scene.step(1);
                if (!state.ui.playback.paused || field_changed || !snapshot) app::capture_snapshot(state, data, scene, *renderer, "stable_fluids.frame_snapshot");
            }

            snapshot             = app::active_snapshot(data);
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
