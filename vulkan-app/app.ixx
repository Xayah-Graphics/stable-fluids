module;

#include <GLFW/glfw3.h>

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
        DyeColor          = 2,
        GenericScalar     = 3,
    };

    enum class ViewMode : uint32_t {
        Plane  = 0,
        Volume = 1,
    };

    enum class RenderMode : uint32_t {
        Smoke  = 0,
        Scalar = 1,
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
        uint32_t nx = 0;
        uint32_t ny = 0;
        uint32_t nz = 1;
        float cell_size = 1.0f;

        [[nodiscard]] float extent_x() const {
            return static_cast<float>(nx) * cell_size;
        }

        [[nodiscard]] float extent_y() const {
            return static_cast<float>(ny) * cell_size;
        }

        [[nodiscard]] float extent_z() const {
            return static_cast<float>((std::max)(nz, 1u)) * cell_size;
        }

        [[nodiscard]] float max_extent() const {
            return (std::max)({extent_x(), extent_y(), extent_z()});
        }
    };

    struct FieldResourceView {
        vk::DescriptorSet descriptor_set{nullptr};
        vk::Semaphore timeline_semaphore{nullptr};
        uint64_t ready_generation = 0;
        uint32_t component_count  = 1;
        FieldSemantic semantic    = FieldSemantic::GenericScalar;
        std::string_view label{};
    };

    struct ColliderOverlay {
        bool enabled       = false;
        uint32_t type      = 0;
        float center_x     = 0.0f;
        float center_y     = 0.0f;
        float center_z     = 0.0f;
        float radius       = 0.0f;
        float half_x       = 0.0f;
        float half_y       = 0.0f;
        float half_z       = 0.0f;
    };

    struct VectorFieldOverlay {
        const float* data = nullptr;
    };

    struct VisualizationSnapshotView {
        GridShape grid{};
        FieldResourceView field{};
        ColliderOverlay collider{};
        VectorFieldOverlay velocity{};
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
        ViewMode view_mode              = ViewMode::Volume;
        RenderMode render_mode          = RenderMode::Smoke;
        PlaneAxis plane_axis            = PlaneAxis::XY;
        int march_steps                 = 96;
        float slice_position            = 0.5f;
        float density_scale             = 0.95f;
        float absorption                = 1.20f;
        float scalar_min                = 0.0f;
        float scalar_max                = 1.0f;
        float scalar_opacity            = 2.0f;
        float scalar_low_r              = 0.08f;
        float scalar_low_g              = 0.18f;
        float scalar_low_b              = 0.46f;
        float scalar_high_r             = 0.98f;
        float scalar_high_g             = 0.82f;
        float scalar_high_b             = 0.24f;
        bool show_bounds                = false;
        bool show_collider              = true;
        bool show_velocity_plane        = false;
        int velocity_grid               = 24;
        int velocity_steps              = 64;
        float velocity_step             = 0.24f;
        float velocity_min_speed        = 0.02f;
        float velocity_thickness        = 1.6f;
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
        void draw_visualization_ui(const std::optional<VisualizationSnapshotView>& snapshot);
        bool render_frame(const std::optional<VisualizationSnapshotView>& snapshot);
        void frame_content(const VisualizationSnapshotView& snapshot);

        [[nodiscard]] VisualizationSettings& settings();
        [[nodiscard]] const VisualizationSettings& settings() const;
        [[nodiscard]] const vk::context::VulkanContext& vk_context() const;
        [[nodiscard]] uint32_t frames_in_flight() const;
        [[nodiscard]] std::vector<vk::raii::DescriptorSet> allocate_field_descriptor_sets(uint32_t count);

    private:
        void recreate_swapchain();
        void collect_camera_input(float dt_seconds);

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
        VisualizationSettings settings_{};
        float render_fps_                                      = 0.0f;
        uint32_t frame_index_                                  = 0;
        std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
    };

} // namespace app
