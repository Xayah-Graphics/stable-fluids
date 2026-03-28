import app;
import scene_cloud;
import scene_plume;
import std;

int main() {
    auto scenes = std::array{
        app::make_scene_entry<scene_plume::Scene>("Plume"),
        app::make_scene_entry<scene_cloud::Scene>("Cloud"),
    };
    return app::run_scene_switcher(scenes);
}
