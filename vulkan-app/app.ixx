module;

#include "stable-fluids-3d.h"

#include <cuda_runtime.h>
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

    enum class ScenePreset : uint32_t {
        DualJetCollider = 0,
        SmokePlume      = 1,
        Custom          = 2,
    };

    enum class ColliderType : uint32_t {
        Sphere = 0,
        Box    = 1,
    };

    enum class FieldId : uint32_t {
        SmokeColor        = 0,
        Density           = 1,
        VelocityMagnitude = 2,
        SolidMask         = 3,
        Pressure          = 4,
        Divergence        = 5,
    };

    enum class FieldDisplayMode : uint32_t {
        Scalar = 0,
        Smoke  = 1,
    };

    struct FieldVisualPreset {
        FieldDisplayMode display_mode = FieldDisplayMode::Scalar;
        float density_scale           = 0.95f;
        float absorption              = 1.20f;
        float scalar_min              = 0.0f;
        float scalar_max              = 1.0f;
        float scalar_opacity          = 2.0f;
        float scalar_low_r            = 0.08f;
        float scalar_low_g            = 0.18f;
        float scalar_low_b            = 0.46f;
        float scalar_high_r           = 0.98f;
        float scalar_high_g           = 0.82f;
        float scalar_high_b           = 0.24f;
    };

    struct FieldInfo {
        FieldId id{};
        std::string_view label{};
        uint32_t component_count = 1;
        FieldSemantic semantic   = FieldSemantic::GenericScalar;
        FieldVisualPreset preset{};
    };

    struct SolverStats {
        double last_step_call_ms              = 0.0;
        double average_step_call_ms           = 0.0;
        uint64_t step_count                   = 0;
        float projection_max_abs_divergence   = 0.0f;
        float projection_rms_divergence       = 0.0f;
    };

    struct CaptureStats {
        double last_snapshot_ms    = 0.0;
        double average_snapshot_ms = 0.0;
        uint64_t snapshot_count    = 0;
    };

    struct ColliderSettings {
        bool enabled               = false;
        ColliderType type          = ColliderType::Sphere;
        float center_x             = 0.50f;
        float center_y             = 0.50f;
        float center_z             = 0.50f;
        float radius               = 0.20f;
        float half_extent_x        = 0.10f;
        float half_extent_y        = 0.08f;
        float half_extent_z        = 0.10f;
        float velocity_x           = 0.0f;
        float velocity_y           = 0.0f;
        float velocity_z           = 0.0f;
        uint32_t boundary          = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP);
    };

    struct SourceEmitterSettings {
        bool enabled               = false;
        float center_x             = 0.50f;
        float center_y             = 0.10f;
        float center_z             = 0.50f;
        float direction_x          = 0.0f;
        float direction_y          = 1.0f;
        float direction_z          = 0.0f;
        float speed                = 0.0f;
        float radius               = 0.03f;
        float density_amount       = 0.0f;
        float dye_amount           = 0.0f;
        float color_r              = 1.00f;
        float color_g              = 1.00f;
        float color_b              = 1.00f;
    };

    struct PlaybackSettings {
        bool paused                = false;
        bool step_once             = false;
        int sim_steps_per_frame    = 1;
        int snapshot_interval      = 2;
    };

    struct AppState {
        struct {
            ScenePreset preset      = ScenePreset::DualJetCollider;
            int selected_field      = 0;
            bool emit_source        = true;

            struct {
                StableFluidsSimulationConfig backend{
                    .nx = 100,
                    .ny = 100,
                    .nz = 100,
                    .cell_size = 0.01f,
                    .dt = 1.0f / 120.0f,
                    .viscosity = 0.0f,
                    .diffuse_iterations = 24,
                    .pressure_iterations = 96,
                    .uniform_force_x = 0.0f,
                    .uniform_force_y = 0.0f,
                    .uniform_force_z = 0.0f,
                    .domain_boundary = {
                        .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                        .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                        .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                        .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                        .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                        .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                    },
                    .block_x = 8,
                    .block_y = 8,
                    .block_z = 4,
                };
                float density_diffusion = 0.00005f;
                float dye_diffusion     = 0.00003f;
                float gravity_y         = -9.81f;
                float buoyancy_beta     = 0.10f;
                float ambient_density   = 0.0f;
            } solver{};

            struct {
                SourceEmitterSettings a{};
                SourceEmitterSettings b{};
            } emitters{};

            struct {
                ColliderSettings collider{};
            } scene{};
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
        void* velocity_cuda_ptr                    = nullptr;
        std::vector<float> velocity_host{};
        uint64_t ready_generation                  = 0;
        uint64_t last_used_submit_serial           = 0;
        GridShape grid{};
        uint32_t field_component_count             = 1;
        FieldSemantic semantic                     = FieldSemantic::GenericScalar;
        std::string_view label{};
        bool has_velocity_host                     = false;
    };

    struct AppData {
        struct {
            cudaStream_t stream                   = nullptr;
            StableFluidsContext context           = nullptr;
            StableFluidsFieldHandle density_field = 0;
            StableFluidsFieldHandle dye_field     = 0;
            SolverStats stats{};
        } physics{};

        struct {
            CaptureStats stats{};
            uint64_t field_bytes                  = 0;
            uint64_t velocity_bytes               = 0;
            uint64_t generation                   = 0;
            uint64_t submit_serial                = 0;
            uint32_t steps_since_snapshot         = 0;
            int active_slot                       = -1;
            GridShape request_grid{};
            uint32_t request_field_component_count = 1;
            bool request_export_velocity_host     = false;
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
        void draw_visualization_ui(VisualizationSettings& settings, const std::optional<VisualizationSnapshotView>& snapshot);
        bool render_frame(const VisualizationSettings& settings, const std::optional<VisualizationSnapshotView>& snapshot);
        void frame_content(const VisualizationSettings& settings, const VisualizationSnapshotView& snapshot);

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
        float render_fps_                                      = 0.0f;
        uint32_t frame_index_                                  = 0;
        std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
    };

    [[nodiscard]] std::span<const FieldInfo> field_catalog();
    [[nodiscard]] const FieldInfo& current_field_info(AppState& state);
    void apply_scene_preset(AppState& state, ScenePreset preset);
    void apply_field_visual_preset(AppState& state);
    void create_runtime_data(AppData& data);
    void destroy_runtime_data(AppData& data);
    void check_interop_support(const VisualizationApp& renderer);
    void rebuild_physics(AppState& state, AppData& data);
    void step_physics(const AppState& state, AppData& data, int sim_steps);
    bool sync_capture_storage(AppState& state, AppData& data, VisualizationApp& renderer);
    bool capture_snapshot(AppState& state, AppData& data, VisualizationApp& renderer, const char* tag);
    [[nodiscard]] std::optional<VisualizationSnapshotView> active_snapshot(const AppState& state, const AppData& data);
    void mark_snapshot_submitted(AppData& data);
    void draw_simulation_controls(AppState& state, const AppData& data, bool& reset_requested, bool& field_changed);

} // namespace app
