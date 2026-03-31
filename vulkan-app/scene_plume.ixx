module;

#include "stable-fluids-3d.h"
#include <cuda_runtime.h>

export module scene_plume;

import app;
import std;

export namespace scene_plume {

    class Scene {
    public:
        Scene();
        ~Scene();

        Scene(const Scene&)                = delete;
        Scene& operator=(const Scene&)     = delete;
        Scene(Scene&&) noexcept            = delete;
        Scene& operator=(Scene&&) noexcept = delete;

        [[nodiscard]] std::span<const app::FieldInfo> fields() const;
        [[nodiscard]] app::VisualizationSettings default_visualization() const;
        [[nodiscard]] app::SceneInfo info() const;
        [[nodiscard]] cudaStream_t stream() const;

        void rebuild();
        void step(int sim_steps);
        void export_field(uint32_t field_index, void* device_destination) const;
        void export_velocity(void* device_destination, float* host_destination) const;

    private:
        StableFluidsSimulationConfig config_{
            .nx                  = 96,
            .ny                  = 128,
            .nz                  = 96,
            .cell_size           = 0.01f,
            .dt                  = 1.0f / 90.0f,
            .viscosity           = 0.00012f,
            .diffuse_iterations  = 24,
            .pressure_iterations = 96,
            .boundary =
                {
                    .x = STABLE_FLUIDS_BOUNDARY_PERIODIC,
                    .y = STABLE_FLUIDS_BOUNDARY_FIXED,
                    .z = STABLE_FLUIDS_BOUNDARY_PERIODIC,
                },
            .block_x = 8,
            .block_y = 8,
            .block_z = 4,
        };
        cudaStream_t stream_                   = nullptr;
        void* context_           = nullptr;
        StableFluidsFieldHandle density_field_ = 0;
        app::GridShape grid_{};
        float* force_x_device_        = nullptr;
        float* force_y_device_        = nullptr;
        float* force_z_device_        = nullptr;
        float* density_source_device_ = nullptr;
        std::vector<float> force_x_host_{};
        std::vector<float> force_z_host_{};
        std::vector<float> source_mask_{};
        std::vector<float> swirl_x_mask_{};
        std::vector<float> swirl_z_mask_{};
        std::vector<float> drift_mask_{};
        uint64_t animation_step_ = 0;
        app::SceneInfo info_{};
    };

} // namespace scene_plume
