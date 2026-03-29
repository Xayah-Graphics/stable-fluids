module;

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

    enum class ViewMode : uint32_t {
        Plane  = 0,
        Volume = 1,
    };

    enum class PlaneAxis : uint32_t {
        XY = 0,
        XZ = 1,
        YZ = 2,
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
    };

    struct VisualizationSnapshotView {
        GridShape grid{};
        FieldResourceView field{};
        const float* velocity = nullptr;
    };

    struct alignas(16) FieldPushConstants {
        vk::math::vec4 eye{};
        vk::math::vec4 right{};
        vk::math::vec4 up{};
        vk::math::vec4 forward{};
        vk::math::vec4 volume_min{};
        vk::math::vec4 volume_max{};
        vk::math::vec4 background_bottom{};
        vk::math::vec4 background_top{};
        vk::math::vec4 color_a{};
        vk::math::vec4 color_b{};
        vk::math::vec4 params0{};
        vk::math::uvec4 params1{};
        vk::math::uvec4 params2{};
        vk::math::vec4 params3{};
    };

    struct VisualizationSettings {
        ViewMode view_mode               = ViewMode::Volume;
        PlaneAxis plane_axis             = PlaneAxis::XY;
        int march_steps                  = 112;
        float slice_position             = 0.42f;
        bool show_velocity_plane         = false;
        int velocity_plane_seed_count    = 40;
        float velocity_plane_arrow_cells = 2.0f;
        float velocity_plane_min_speed   = 0.003f;
        float velocity_plane_thickness   = 0.8f;
        float density_scale              = 1.35f;
        float scalar_min                 = 0.0f;
        float scalar_max                 = 3.5f;
        float scalar_opacity             = 5.4f;
        float scalar_low_r               = 0.03f;
        float scalar_low_g               = 0.04f;
        float scalar_low_b               = 0.07f;
        float scalar_high_r              = 0.94f;
        float scalar_high_g              = 0.90f;
        float scalar_high_b              = 0.84f;
        float background_bottom_r        = 0.035f;
        float background_bottom_g        = 0.04f;
        float background_bottom_b        = 0.05f;
        float background_top_r           = 0.05f;
        float background_top_g           = 0.06f;
        float background_top_b           = 0.08f;
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
        std::string_view label{};
        FieldVisualPreset preset{};
    };

    struct SceneInfo {
        GridShape grid{};
        float dt                 = 0.0f;
        uint64_t step_count      = 0;
        double last_step_call_ms = 0.0;
    };

    struct AppState {
        int selected_field = 0;
        bool paused        = false;
        VisualizationSettings render{};
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
        uint64_t ready_generation        = 0;
        uint64_t last_used_submit_serial = 0;
        bool has_velocity_host           = false;
    };

    struct AppData {
        struct {
            uint64_t generation       = 0;
            uint64_t submit_serial    = 0;
            int active_slot           = -1;
            bool has_velocity_storage = false;
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
        void draw_visualization_ui(AppState& state, const SceneInfo& scene, std::span<const FieldInfo> fields, bool& reset_requested, bool& field_changed, const std::optional<VisualizationSnapshotView>& snapshot);
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
        std::filesystem::path shader_dir_{};
        vk::raii::ShaderModule background_shader_module_{nullptr};
        vk::raii::ShaderModule plane_shader_module_{nullptr};
        vk::raii::ShaderModule volume_shader_module_{nullptr};
        vk::pipeline::GraphicsPipeline background_pipeline_{};
        vk::pipeline::GraphicsPipeline plane_pipeline_{};
        vk::pipeline::GraphicsPipeline volume_pipeline_{};
        uint32_t frame_index_ = 0;
        std::chrono::steady_clock::time_point last_frame_time_{}; // default-initialized
    };

    template <typename TScene>
    concept SceneSample = requires(TScene scene, const TScene const_scene, const uint32_t field_index, const int sim_steps, void* device_destination, float* host_destination) {
        { const_scene.fields() } -> std::convertible_to<std::span<const FieldInfo>>;
        { const_scene.default_visualization() } -> std::same_as<VisualizationSettings>;
        { const_scene.info() } -> std::same_as<SceneInfo>;
        { scene.rebuild() } -> std::same_as<void>;
        { scene.step(sim_steps) } -> std::same_as<void>;
        { const_scene.export_field(field_index, device_destination) } -> std::same_as<void>;
        { const_scene.export_velocity(device_destination, host_destination) } -> std::same_as<void>;
        { const_scene.stream() } -> std::same_as<cudaStream_t>;
    };

    void destroy_runtime_data(AppData& data);
    void check_interop_support(const VisualizationApp& renderer);
    void apply_field_preset(VisualizationSettings& settings, const FieldVisualPreset& preset);
    bool sync_capture_storage(AppData& data, VisualizationApp& renderer, const GridShape& grid, bool with_velocity_plane);

    template <SceneSample TScene>
    bool capture_snapshot(AppState& state, AppData& data, TScene& scene, VisualizationApp& renderer) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        int slot_index = -1;
        for (uint32_t i = 0; i < data.capture.slots.size(); ++i) {
            const auto& slot = data.capture.slots[i];
            if (static_cast<int>(i) == data.capture.active_slot) continue;
            if (slot.ready_generation != 0 && data.capture.submit_serial < slot.last_used_submit_serial + renderer.frames_in_flight() + 1) continue;
            slot_index = static_cast<int>(i);
            break;
        }
        if (slot_index < 0) return false;

        const auto& const_scene = scene;
        const auto fields       = const_scene.fields();
        if (fields.empty()) throw std::runtime_error("scene must expose at least one field");
        state.selected_field = std::clamp(state.selected_field, 0, static_cast<int>(fields.size()) - 1);

        auto& slot             = data.capture.slots.at(static_cast<size_t>(slot_index));
        const auto field_index = static_cast<uint32_t>(state.selected_field);
        scene.export_field(field_index, slot.field_cuda_ptr);
        if (state.render.show_velocity_plane && slot.velocity_cuda_ptr != nullptr && !slot.velocity_host.empty()) scene.export_velocity(slot.velocity_cuda_ptr, slot.velocity_host.data());
        cudaExternalSemaphoreSignalParams signal_params{};
        signal_params.params.fence.value = data.capture.generation + 1;
        check_cuda(cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, scene.stream()), "cudaSignalExternalSemaphoresAsync");
        check_cuda(cudaStreamSynchronize(scene.stream()), "cudaStreamSynchronize");
        slot.ready_generation    = data.capture.generation + 1;
        slot.has_velocity_host   = state.render.show_velocity_plane && slot.velocity_cuda_ptr != nullptr && !slot.velocity_host.empty();
        data.capture.generation  = slot.ready_generation;
        data.capture.active_slot = slot_index;
        return true;
    }

    [[nodiscard]] std::optional<VisualizationSnapshotView> active_snapshot(const AppData& data);
    void mark_snapshot_submitted(AppData& data);

    template <SceneSample TScene>
    int run_scene() {
        AppState state{};
        AppData data{};
        std::unique_ptr<VisualizationApp> renderer{};
        try {
            renderer = std::make_unique<VisualizationApp>();
            check_interop_support(*renderer);

            TScene scene{};
            renderer->vk_context().device.waitIdle();
            scene.rebuild();

            state.selected_field = 0;
            state.render         = scene.default_visualization();
            sync_capture_storage(data, *renderer, scene.info().grid, state.render.show_velocity_plane);
            capture_snapshot(state, data, scene, *renderer);
            if (const auto snapshot = active_snapshot(data)) renderer->frame_content(state.render, *snapshot);

            while (!renderer->should_close()) {
                renderer->begin_frame();

                bool reset_requested = false;
                bool field_changed   = false;
                auto snapshot        = active_snapshot(data);
                renderer->draw_visualization_ui(state, scene.info(), scene.fields(), reset_requested, field_changed, snapshot);

                if (reset_requested) {
                    renderer->vk_context().device.waitIdle();
                    scene.rebuild();
                    sync_capture_storage(data, *renderer, scene.info().grid, state.render.show_velocity_plane);
                    capture_snapshot(state, data, scene, *renderer);
                    snapshot = active_snapshot(data);
                    if (snapshot) renderer->frame_content(state.render, *snapshot);
                } else {
                    sync_capture_storage(data, *renderer, scene.info().grid, state.render.show_velocity_plane);
                    if (!state.paused) scene.step(1);
                    if (!state.paused || field_changed || !snapshot) capture_snapshot(state, data, scene, *renderer);
                }

                snapshot             = active_snapshot(data);
                const bool submitted = renderer->render_frame(state.render, snapshot);
                if (submitted) mark_snapshot_submitted(data);
            }

            renderer->vk_context().device.waitIdle();
            destroy_runtime_data(data);
            return 0;
        } catch (const std::exception& e) {
            try {
                if (renderer) renderer->vk_context().device.waitIdle();
            } catch (...) {
            }
            destroy_runtime_data(data);
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }

} // namespace app
