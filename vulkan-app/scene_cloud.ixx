module;

#include "stable-fluids-3d.h"
#include <cuda_runtime.h>

export module scene_cloud;

import app;
import std;

export namespace scene_cloud {

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
            .nx                  = 160,
            .ny                  = 96,
            .nz                  = 96,
            .cell_size           = 0.0125f,
            .dt                  = 1.0f / 72.0f,
            .viscosity           = 0.00008f,
            .diffuse_iterations  = 20,
            .pressure_iterations = 88,
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
        StableFluidsContext context_           = nullptr;
        StableFluidsFieldHandle density_field_ = 0;
        app::GridShape grid_{};
        float* force_x_device_        = nullptr;
        float* force_y_device_        = nullptr;
        float* force_z_device_        = nullptr;
        float* density_source_device_ = nullptr;
        std::vector<float> force_x_host_{};
        std::vector<float> force_z_host_{};
        std::vector<float> wind_mask_{};
        std::vector<float> shear_mask_{};
        std::vector<float> curl_x_mask_{};
        std::vector<float> curl_z_mask_{};
        std::vector<float> pulse_mask_{};
        uint64_t animation_step_ = 0;
        app::SceneInfo info_{};
    };

} // namespace scene_cloud
