module;

#include "stable-fluids-3d.h"
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <vulkan/vulkan_raii.hpp>

export module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.pipeline;
import vk.swapchain;

export namespace app {

    enum class FieldSemantic : uint32_t {
        Density           = 0,
        VelocityMagnitude = 1,
        Pressure          = 2,
        Divergence        = 3,
    };

    enum class ViewMode : uint32_t {
        Plane  = 0,
        Volume = 1,
    };

    enum class PlaneAxis : uint32_t {
        XY = 0,
        XZ = 1,
        YZ = 2,
    };

    struct alignas(16) uvec4 {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t w;
    };

    struct GridShape {
        uint32_t nx     = 0;
        uint32_t ny     = 0;
        uint32_t nz     = 1;
        float cell_size = 1.0f;

        [[nodiscard]] float extent_x() const {
            return static_cast<float>(nx) * cell_size;
        }

        [[nodiscard]] float extent_y() const {
            return static_cast<float>(ny) * cell_size;
        }

        [[nodiscard]] float extent_z() const {
            return static_cast<float>((std::max) (nz, 1u)) * cell_size;
        }

        [[nodiscard]] float max_extent() const {
            return (std::max) ({extent_x(), extent_y(), extent_z()});
        }
    };

    struct FieldResourceView {
        vk::DescriptorSet descriptor_set{nullptr};
        vk::Semaphore timeline_semaphore{nullptr};
        uint64_t ready_generation = 0;
        uint32_t component_count  = 1;
        FieldSemantic semantic    = FieldSemantic::Density;
        std::string_view label{};
    };

    struct VisualizationSnapshotView {
        GridShape grid{};
        FieldResourceView field{};
    };

    struct alignas(16) FieldPushConstants {
        vk::math::vec4 eye{};
        vk::math::vec4 right{};
        vk::math::vec4 up{};
        vk::math::vec4 forward{};
        vk::math::vec4 volume_min{};
        vk::math::vec4 volume_max{};
        vk::math::vec4 color_a{};
        vk::math::vec4 color_b{};
        vk::math::vec4 params0{};
        uvec4 params1{};
        uvec4 params2{};
        vk::math::vec4 params3{};
    };

    struct VisualizationSettings {
        ViewMode view_mode   = ViewMode::Volume;
        PlaneAxis plane_axis = PlaneAxis::XZ;
        int march_steps      = 112;
        float slice_position = 0.42f;
        float density_scale  = 1.35f;
        float scalar_min     = 0.0f;
        float scalar_max     = 3.5f;
        float scalar_opacity = 5.4f;
        float scalar_low_r   = 0.03f;
        float scalar_low_g   = 0.04f;
        float scalar_low_b   = 0.07f;
        float scalar_high_r  = 0.94f;
        float scalar_high_g  = 0.90f;
        float scalar_high_b  = 0.84f;
    };

    enum class FieldId : uint32_t {
        Density           = 0,
        VelocityMagnitude = 1,
        Pressure          = 2,
        Divergence        = 3,
    };

    struct FieldVisualPreset {
        float density_scale  = 1.0f;
        float scalar_min     = 0.0f;
        float scalar_max     = 1.0f;
        float scalar_opacity = 3.0f;
        float scalar_low_r   = 0.06f;
        float scalar_low_g   = 0.08f;
        float scalar_low_b   = 0.12f;
        float scalar_high_r  = 0.95f;
        float scalar_high_g  = 0.86f;
        float scalar_high_b  = 0.72f;
    };

    struct FieldInfo {
        FieldId id{};
        std::string_view label{};
        FieldSemantic semantic = FieldSemantic::Density;
        uint32_t export_kind   = STABLE_FLUIDS_EXPORT_FIELD;
        FieldVisualPreset preset{};
    };

    struct SolverStats {
        double last_step_call_ms            = 0.0;
        double average_step_call_ms         = 0.0;
        uint64_t step_count                 = 0;
        float projection_max_abs_divergence = 0.0f;
        float projection_rms_divergence     = 0.0f;
    };

    struct CaptureStats {
        double last_snapshot_ms    = 0.0;
        double average_snapshot_ms = 0.0;
        uint64_t snapshot_count    = 0;
    };

    struct PlaybackSettings {
        bool paused = false;
    };

    struct AppState {
        struct {
            int selected_field = 0;
            StableFluidsSimulationConfig config{
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
        } physics{};

        struct {
            VisualizationSettings render{};
            PlaybackSettings playback{};
        } ui{};
    };

    struct CaptureSlot {
        vk::raii::Buffer buffer{nullptr};
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::Semaphore timeline_semaphore{nullptr};
        vk::raii::DescriptorSet descriptor_set{nullptr};
        cudaExternalMemory_t external_memory       = nullptr;
        cudaExternalSemaphore_t external_semaphore = nullptr;
        void* field_cuda_ptr                       = nullptr;
        uint64_t ready_generation                  = 0;
        uint64_t last_used_submit_serial           = 0;
        GridShape grid{};
        uint32_t field_component_count = 1;
        FieldSemantic semantic         = FieldSemantic::Density;
        std::string_view label{};
    };

    struct AppData {
        struct {
            cudaStream_t stream                   = nullptr;
            StableFluidsContext context           = nullptr;
            StableFluidsFieldHandle density_field = 0;
            GridShape grid{};
            float* force_x_device        = nullptr;
            float* force_y_device        = nullptr;
            float* force_z_device        = nullptr;
            float* density_source_device = nullptr;
            std::vector<float> force_x_host{};
            std::vector<float> force_z_host{};
            std::vector<float> source_mask{};
            std::vector<float> swirl_x_mask{};
            std::vector<float> swirl_z_mask{};
            std::vector<float> drift_mask{};
            uint64_t animation_step = 0;
            SolverStats stats{};
        } physics{};

        struct {
            CaptureStats stats{};
            uint64_t field_bytes   = 0;
            uint64_t generation    = 0;
            uint64_t submit_serial = 0;
            int active_slot        = -1;
            GridShape request_grid{};
            std::vector<CaptureSlot> slots{};
        } capture{};
    };

    class VisualizationApp {
    public:
        VisualizationApp();
        ~VisualizationApp();

        VisualizationApp(const VisualizationApp&)                = delete;
        VisualizationApp& operator=(const VisualizationApp&)     = delete;
        VisualizationApp(VisualizationApp&&) noexcept            = delete;
        VisualizationApp& operator=(VisualizationApp&&) noexcept = delete;

        [[nodiscard]] bool should_close() const;
        void begin_frame();
        void draw_visualization_ui(AppState& state, const AppData& data, bool& reset_requested, bool& field_changed, const std::optional<VisualizationSnapshotView>& snapshot);
        bool render_frame(const VisualizationSettings& settings, const std::optional<VisualizationSnapshotView>& snapshot);
        void frame_content(const VisualizationSettings& settings, const VisualizationSnapshotView& snapshot);

        [[nodiscard]] const vk::context::VulkanContext& vk_context() const;
        [[nodiscard]] uint32_t frames_in_flight() const;
        [[nodiscard]] std::vector<vk::raii::DescriptorSet> allocate_field_descriptor_sets(uint32_t count);

    private:
        void recreate_swapchain();

        static constexpr uint32_t frames_in_flight_value_ = 2;

        struct WindowState {
            bool* resize_requested = nullptr;
            float scroll           = 0.0f;
            bool first_mouse       = true;
            double last_x          = 0.0;
            double last_y          = 0.0;
        };

        vk::context::VulkanContext vkctx_{};
        vk::context::SurfaceContext sctx_{};
        GLFWwindow* window_ = nullptr;
        WindowState window_state_{};
        vk::swapchain::Swapchain sc_{};
        vk::frame::FrameSystem frames_{};
        vk::imgui::ImGuiSystem imgui_sys_{};
        vk::camera::Camera camera_{};
        vk::raii::DescriptorSetLayout field_set_layout_{nullptr};
        vk::raii::DescriptorPool field_descriptor_pool_{nullptr};
        vk::raii::ShaderModule plane_shader_module_{nullptr};
        vk::raii::ShaderModule volume_shader_module_{nullptr};
        vk::pipeline::GraphicsPipeline plane_pipeline_{};
        vk::pipeline::GraphicsPipeline volume_pipeline_{};
        float render_fps_                                      = 0.0f;
        uint32_t frame_index_                                  = 0;
        std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
    };

    void create_runtime_data(AppData& data);
    void destroy_runtime_data(AppData& data);
    void check_interop_support(const VisualizationApp& renderer);
    void rebuild_physics(const AppState& state, AppData& data);
    void step_physics(const AppState& state, AppData& data, int sim_steps);
    bool sync_capture_storage(AppData& data, VisualizationApp& renderer);
    bool capture_snapshot(AppState& state, AppData& data, VisualizationApp& renderer, const char* tag);
    [[nodiscard]] std::optional<VisualizationSnapshotView> active_snapshot(const AppData& data);
    void mark_snapshot_submitted(AppData& data);

} // namespace app
