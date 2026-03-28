module;

#include "stable-fluids-3d.h"

#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <imgui.h>

#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan_raii.hpp>

module app;

import std;
import vk.camera;
import vk.context;
import vk.frame;
import vk.imgui;
import vk.math;
import vk.memory;
import vk.pipeline;
import vk.swapchain;

namespace app {

    namespace {

        constexpr std::array field_catalog_storage{
            FieldInfo{
                .id          = FieldId::Density,
                .label       = "Density",
                .semantic    = FieldSemantic::Density,
                .export_kind = STABLE_FLUIDS_EXPORT_FIELD,
                .preset =
                    {
                        .density_scale  = 1.35f,
                        .scalar_min     = 0.0f,
                        .scalar_max     = 3.5f,
                        .scalar_opacity = 5.4f,
                        .scalar_low_r   = 0.03f,
                        .scalar_low_g   = 0.04f,
                        .scalar_low_b   = 0.07f,
                        .scalar_high_r  = 0.94f,
                        .scalar_high_g  = 0.90f,
                        .scalar_high_b  = 0.84f,
                    },
            },
            FieldInfo{
                .id          = FieldId::VelocityMagnitude,
                .label       = "Velocity Magnitude",
                .semantic    = FieldSemantic::VelocityMagnitude,
                .export_kind = STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE,
                .preset =
                    {
                        .density_scale  = 1.0f,
                        .scalar_min     = 0.0f,
                        .scalar_max     = 1.3f,
                        .scalar_opacity = 2.2f,
                        .scalar_low_r   = 0.06f,
                        .scalar_low_g   = 0.10f,
                        .scalar_low_b   = 0.24f,
                        .scalar_high_r  = 0.24f,
                        .scalar_high_g  = 0.88f,
                        .scalar_high_b  = 1.00f,
                    },
            },
            FieldInfo{
                .id          = FieldId::Pressure,
                .label       = "Pressure",
                .semantic    = FieldSemantic::Pressure,
                .export_kind = STABLE_FLUIDS_EXPORT_PRESSURE,
                .preset =
                    {
                        .density_scale  = 1.0f,
                        .scalar_min     = -0.18f,
                        .scalar_max     = 0.18f,
                        .scalar_opacity = 2.3f,
                        .scalar_low_r   = 0.08f,
                        .scalar_low_g   = 0.22f,
                        .scalar_low_b   = 0.62f,
                        .scalar_high_r  = 0.96f,
                        .scalar_high_g  = 0.58f,
                        .scalar_high_b  = 0.18f,
                    },
            },
            FieldInfo{
                .id          = FieldId::Divergence,
                .label       = "Divergence",
                .semantic    = FieldSemantic::Divergence,
                .export_kind = STABLE_FLUIDS_EXPORT_DIVERGENCE,
                .preset =
                    {
                        .density_scale  = 1.0f,
                        .scalar_min     = -24.0f,
                        .scalar_max     = 24.0f,
                        .scalar_opacity = 2.3f,
                        .scalar_low_r   = 0.05f,
                        .scalar_low_g   = 0.14f,
                        .scalar_low_b   = 0.50f,
                        .scalar_high_r  = 0.94f,
                        .scalar_high_g  = 0.28f,
                        .scalar_high_b  = 0.22f,
                    },
            },
        };

    } // namespace

    VisualizationApp::VisualizationApp() {
        using namespace vk;

        auto [vkctx, sctx] = context::setup_vk_context_glfw("stable-fluids", "stable-fluids-viz");
        vkctx_             = std::move(vkctx);
        sctx_              = std::move(sctx);
        window_            = sctx_.window.get();

        window_state_.resize_requested = &sctx_.resize_requested;
        glfwSetWindowUserPointer(window_, &window_state_);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* raw_window, int, int) {
            auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr && state->resize_requested != nullptr) *state->resize_requested = true;
        });
        glfwSetScrollCallback(window_, [](GLFWwindow* raw_window, double, double yoffset) {
            auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(raw_window));
            if (state != nullptr) state->scroll += static_cast<float>(yoffset);
        });
        glfwGetCursorPos(window_, &window_state_.last_x, &window_state_.last_y);

        sc_        = swapchain::setup_swapchain(vkctx_, sctx_);
        frames_    = frame::create_frame_system(vkctx_, sc_, frames_in_flight_value_);
        imgui_sys_ = imgui::create(vkctx_, window_, sc_.format, frames_in_flight_value_, static_cast<uint32_t>(sc_.images.size()));

        camera::CameraConfig camera_config{};
        camera_config.orbit_rotate_sens = 0.005f;
        camera_config.orbit_zoom_sens   = 0.12f;
        camera_.set_config(camera_config);
        camera_.home();

        DescriptorSetLayoutBinding field_binding{
            .binding         = 0,
            .descriptorType  = DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags      = ShaderStageFlagBits::eFragment,
        };
        DescriptorSetLayoutCreateInfo field_layout_ci{
            .bindingCount = 1,
            .pBindings    = &field_binding,
        };
        field_set_layout_ = raii::DescriptorSetLayout{vkctx_.device, field_layout_ci};

        DescriptorPoolSize field_pool_size{
            .type            = DescriptorType::eStorageBuffer,
            .descriptorCount = 128,
        };
        DescriptorPoolCreateInfo field_pool_ci{
            .flags         = DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 128,
            .poolSizeCount = 1,
            .pPoolSizes    = &field_pool_size,
        };
        field_descriptor_pool_ = raii::DescriptorPool{vkctx_.device, field_pool_ci};

        const std::filesystem::path shader_dir = std::filesystem::path(SMOKE_SIM_SHADER_DIR);
        const auto plane_shader_spv            = pipeline::read_file_bytes((shader_dir / "field_plane.spv").string());
        const auto volume_shader_spv           = pipeline::read_file_bytes((shader_dir / "field_volume.spv").string());
        plane_shader_module_                   = pipeline::load_shader_module(vkctx_.device, plane_shader_spv);
        volume_shader_module_                  = pipeline::load_shader_module(vkctx_.device, volume_shader_spv);

        std::array<DescriptorSetLayout, 1> pipeline_set_layouts{*field_set_layout_};
        pipeline::GraphicsPipelineDesc pipeline_desc{
            .color_format         = sc_.format,
            .use_depth            = false,
            .use_blend            = false,
            .topology             = PrimitiveTopology::eTriangleList,
            .cull                 = CullModeFlagBits::eNone,
            .push_constant_bytes  = sizeof(FieldPushConstants),
            .push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment,
            .set_layouts          = pipeline_set_layouts,
        };

        pipeline::VertexInput empty_vertex_input{};
        plane_pipeline_  = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, plane_shader_module_, "vs_main", "fs_main");
        volume_pipeline_ = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, volume_shader_module_, "vs_main", "fs_main");
        last_frame_time_ = std::chrono::steady_clock::now();
    }

    VisualizationApp::~VisualizationApp() {
        try {
            vkctx_.device.waitIdle();
        } catch (...) {
        }

        if (imgui_sys_.initialized) vk::imgui::shutdown(imgui_sys_);
    }

    bool VisualizationApp::should_close() const {
        return glfwWindowShouldClose(window_) != 0;
    }

    void VisualizationApp::begin_frame() {
        nvtx3::scoped_range range{"viz.begin_frame"};
        glfwPollEvents();

        const auto now         = std::chrono::steady_clock::now();
        const float dt_seconds = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_       = now;
        if (dt_seconds > 0.0f) {
            const float instantaneous_fps = 1.0f / dt_seconds;
            render_fps_                   = render_fps_ > 0.0f ? std::lerp(render_fps_, instantaneous_fps, 0.1f) : instantaneous_fps;
        }

        if (sctx_.resize_requested) recreate_swapchain();

        vk::imgui::begin_frame();

        double mouse_x = 0.0;
        double mouse_y = 0.0;
        glfwGetCursorPos(window_, &mouse_x, &mouse_y);

        float mouse_dx = 0.0f;
        float mouse_dy = 0.0f;
        if (window_state_.first_mouse) {
            window_state_.first_mouse = false;
        } else {
            mouse_dx = static_cast<float>(mouse_x - window_state_.last_x);
            mouse_dy = static_cast<float>(mouse_y - window_state_.last_y);
        }
        window_state_.last_x = mouse_x;
        window_state_.last_y = mouse_y;

        auto& io = ImGui::GetIO();
        vk::camera::CameraInput camera_input{};
        if (!io.WantCaptureMouse) {
            camera_input.lmb      = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            camera_input.mmb      = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
            camera_input.rmb      = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
            camera_input.mouse_dx = mouse_dx;
            camera_input.mouse_dy = mouse_dy;
            camera_input.scroll   = window_state_.scroll;
        }
        if (!io.WantCaptureKeyboard) {
            camera_input.forward  = glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS;
            camera_input.backward = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
            camera_input.left     = glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS;
            camera_input.right    = glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS;
            camera_input.up       = glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS;
            camera_input.down     = glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS;
            camera_input.shift    = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
            camera_input.ctrl     = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            camera_input.alt      = glfwGetKey(window_, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
            camera_input.space    = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
        }
        window_state_.scroll = 0.0f;
        camera_.update(dt_seconds, sc_.extent.width, sc_.extent.height, camera_input);
    }

    void VisualizationApp::draw_visualization_ui(AppState& state, const AppData& data, bool& reset_requested, bool& field_changed, const std::optional<VisualizationSnapshotView>& snapshot) {
        bool reframe_requested       = false;
        auto& settings               = state.ui.render;
        state.physics.selected_field = std::clamp(state.physics.selected_field, 0, static_cast<int>(field_catalog_storage.size()) - 1);
        const auto& field            = field_catalog_storage[static_cast<size_t>(state.physics.selected_field)];

        ImGui::Begin("Smoke");
        if (ImGui::BeginCombo("Field", field.label.data())) {
            for (int i = 0; i < static_cast<int>(field_catalog_storage.size()); ++i) {
                const bool is_selected = state.physics.selected_field == i;
                if (ImGui::Selectable(field_catalog_storage[static_cast<size_t>(i)].label.data(), is_selected)) {
                    state.physics.selected_field = i;
                    const auto& preset           = field_catalog_storage[static_cast<size_t>(i)].preset;
                    settings.density_scale       = preset.density_scale;
                    settings.scalar_min          = preset.scalar_min;
                    settings.scalar_max          = preset.scalar_max;
                    settings.scalar_opacity      = preset.scalar_opacity;
                    settings.scalar_low_r        = preset.scalar_low_r;
                    settings.scalar_low_g        = preset.scalar_low_g;
                    settings.scalar_low_b        = preset.scalar_low_b;
                    settings.scalar_high_r       = preset.scalar_high_r;
                    settings.scalar_high_g       = preset.scalar_high_g;
                    settings.scalar_high_b       = preset.scalar_high_b;
                    field_changed                = true;
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("Pause", &state.ui.playback.paused);
        ImGui::SameLine();
        if (ImGui::Button("Reset")) reset_requested = true;

        int view_mode = static_cast<int>(settings.view_mode);
        constexpr std::array view_labels{
            "Plane",
            "Volume",
        };
        if (ImGui::Combo("View", &view_mode, view_labels.data(), static_cast<int>(view_labels.size()))) {
            settings.view_mode = static_cast<ViewMode>(view_mode);
            reframe_requested  = true;
        }

        if (settings.view_mode == ViewMode::Plane) {
            int plane_axis = static_cast<int>(settings.plane_axis);
            constexpr std::array plane_labels{
                "XY",
                "XZ",
                "YZ",
            };
            if (ImGui::Combo("Slice Axis", &plane_axis, plane_labels.data(), static_cast<int>(plane_labels.size()))) {
                settings.plane_axis = static_cast<PlaneAxis>(plane_axis);
                reframe_requested   = true;
            }
            ImGui::SliderFloat("Slice", &settings.slice_position, 0.0f, 1.0f, "%.3f");
        }

        ImGui::Separator();
        ImGui::Text("Grid: %d x %d x %d", state.physics.config.nx, state.physics.config.ny, state.physics.config.nz);
        ImGui::Text("dt: %.5f  h: %.4f", state.physics.config.dt, state.physics.config.cell_size);
        ImGui::Text("Field: %.*s", static_cast<int>(field.label.size()), field.label.data());
        ImGui::Text("Steps: %llu", static_cast<unsigned long long>(data.physics.stats.step_count));
        ImGui::Text("Step Call: %.3f ms", data.physics.stats.last_step_call_ms);
        ImGui::Text("Max |div|: %.6g", data.physics.stats.projection_max_abs_divergence);
        ImGui::Text("RMS div: %.6g", data.physics.stats.projection_rms_divergence);
        if (snapshot) ImGui::Text("Generation: %llu", static_cast<unsigned long long>(snapshot->field.ready_generation));
        ImGui::End();

        if (reframe_requested && snapshot) frame_content(settings, *snapshot);

        if (const ImGuiViewport* viewport = ImGui::GetMainViewport()) {
            ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + 12.0f, viewport->Pos.y + 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
            ImGui::Begin("Render Stats Overlay", nullptr, overlay_flags);
            ImGui::Text("Render: %.1f FPS", render_fps_);
            ImGui::End();
        }
    }

    bool VisualizationApp::render_frame(const VisualizationSettings& settings, const std::optional<VisualizationSnapshotView>& snapshot) {
        nvtx3::scoped_range range{"viz.render_frame"};
        using namespace vk;

        const auto acquire_result = frame::begin_frame(vkctx_, sc_, frames_, frame_index_);
        if (acquire_result.need_recreate) {
            recreate_swapchain();
            vk::imgui::end_frame();
            return false;
        }

        frame::begin_commands(frames_, frame_index_);
        auto& cmd = frame::cmd(frames_, frame_index_);

        const uint32_t image_index        = acquire_result.image_index;
        const ImageLayout previous_layout = frames_.swapchain_image_layout[image_index];
        const ImageMemoryBarrier2 to_color_barrier{
            .srcStageMask     = previous_layout == ImageLayout::eUndefined ? PipelineStageFlagBits2::eNone : PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask    = previous_layout == ImageLayout::eUndefined ? AccessFlags2{} : (AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite),
            .dstStageMask     = PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask    = AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout        = previous_layout,
            .newLayout        = ImageLayout::eColorAttachmentOptimal,
            .image            = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &to_color_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::eColorAttachmentOptimal;

        ClearValue clear_value{};
        clear_value.color = ClearColorValue{std::array<float, 4>{0.035f, 0.04f, 0.05f, 1.0f}};
        RenderingAttachmentInfo color_attachment{
            .imageView   = *sc_.image_views[image_index],
            .imageLayout = ImageLayout::eColorAttachmentOptimal,
            .loadOp      = AttachmentLoadOp::eClear,
            .storeOp     = AttachmentStoreOp::eStore,
            .clearValue  = clear_value,
        };
        RenderingInfo rendering_info{
            .renderArea           = Rect2D{Offset2D{0, 0}, sc_.extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &color_attachment,
        };

        cmd.beginRendering(rendering_info);
        cmd.setViewport(0, Viewport{
                               0.0f,
                               0.0f,
                               static_cast<float>(sc_.extent.width),
                               static_cast<float>(sc_.extent.height),
                               0.0f,
                               1.0f,
                           });
        cmd.setScissor(0, Rect2D{{0, 0}, sc_.extent});

        if (snapshot) {
            const auto& matrices      = camera_.matrices();
            const auto& camera_config = camera_.config();
            const float aspect        = static_cast<float>(sc_.extent.width) / static_cast<float>((std::max) (sc_.extent.height, 1u));
            const float half_fov_tan  = std::tan(camera_config.fov_y_rad * 0.5f);
            FieldPushConstants push{};
            push.eye        = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
            push.right      = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
            push.up         = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
            push.forward    = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
            push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
            push.volume_max = {snapshot->grid.extent_x(), snapshot->grid.extent_y(), snapshot->grid.extent_z(), 0.0f};
            push.color_a    = {settings.scalar_low_r, settings.scalar_low_g, settings.scalar_low_b, 1.0f};
            push.color_b    = {settings.scalar_high_r, settings.scalar_high_g, settings.scalar_high_b, 1.0f};
            push.params0    = {
                aspect,
                half_fov_tan,
                settings.density_scale,
                settings.scalar_opacity,
            };
            push.params1 = {
                snapshot->grid.nx,
                snapshot->grid.ny,
                snapshot->grid.nz,
                static_cast<uint32_t>(settings.march_steps),
            };
            push.params2 = {
                1u,
                snapshot->field.component_count,
                static_cast<uint32_t>(settings.plane_axis),
                static_cast<uint32_t>(camera_config.projection),
            };
            push.params3 = {
                settings.scalar_min,
                settings.scalar_max,
                settings.slice_position,
                camera_config.ortho_height,
            };

            const auto& pipeline = settings.view_mode == ViewMode::Volume ? volume_pipeline_ : plane_pipeline_;
            cmd.bindPipeline(PipelineBindPoint::eGraphics, *pipeline.pipeline);
            cmd.bindDescriptorSets(PipelineBindPoint::eGraphics, *pipeline.layout, 0, {snapshot->field.descriptor_set}, {});
            const ArrayProxy<const FieldPushConstants> push_block(1, &push);
            cmd.pushConstants(*pipeline.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();

        vk::math::mat4 gizmo_c2w{};
        gizmo_c2w.c0 = {camera_.matrices().right.x, camera_.matrices().right.y, camera_.matrices().right.z, 0.0f};
        gizmo_c2w.c1 = {camera_.matrices().up.x, camera_.matrices().up.y, camera_.matrices().up.z, 0.0f};
        gizmo_c2w.c2 = {camera_.matrices().forward.x, camera_.matrices().forward.y, camera_.matrices().forward.z, 0.0f};
        gizmo_c2w.c3 = {camera_.matrices().eye.x, camera_.matrices().eye.y, camera_.matrices().eye.z, 1.0f};
        imgui::draw_mini_axis_gizmo(gizmo_c2w);
        imgui::render(imgui_sys_, cmd, sc_.extent, *sc_.image_views[image_index], ImageLayout::eColorAttachmentOptimal);

        const ImageMemoryBarrier2 to_present_barrier{
            .srcStageMask     = PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask    = AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask     = PipelineStageFlagBits2::eBottomOfPipe,
            .oldLayout        = ImageLayout::eColorAttachmentOptimal,
            .newLayout        = ImageLayout::ePresentSrcKHR,
            .image            = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &to_present_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::ePresentSrcKHR;

        std::array<SemaphoreSubmitInfo, 1> volume_waits{};
        std::span<const SemaphoreSubmitInfo> extra_waits{};
        if (snapshot && snapshot->field.timeline_semaphore) {
            volume_waits[0] = SemaphoreSubmitInfo{
                .semaphore = snapshot->field.timeline_semaphore,
                .value     = snapshot->field.ready_generation,
                .stageMask = PipelineStageFlagBits2::eFragmentShader,
            };
            extra_waits = std::span<const SemaphoreSubmitInfo>(volume_waits.data(), volume_waits.size());
        }

        const bool need_recreate = frame::end_frame(vkctx_, sc_, frames_, frame_index_, image_index, extra_waits);
        if (need_recreate || sctx_.resize_requested) recreate_swapchain();

        frame_index_ = (frame_index_ + 1) % frames_.frames_in_flight;
        vk::imgui::end_frame();
        return true;
    }

    void VisualizationApp::frame_content(const VisualizationSettings& settings, const VisualizationSnapshotView& snapshot) {
        auto update_camera_config = [&](const vk::camera::Projection projection, const float ortho_height) {
            auto camera_config         = camera_.config();
            camera_config.projection   = projection;
            camera_config.ortho_height = ortho_height;
            camera_.set_config(camera_config);
        };

        const float center_x                 = snapshot.grid.extent_x() * 0.5f;
        const float center_y                 = snapshot.grid.extent_y() * 0.5f;
        const float center_z                 = snapshot.grid.extent_z() * 0.5f;
        vk::camera::CameraState camera_state = camera_.state();
        camera_state.mode                    = vk::camera::Mode::Orbit;
        camera_state.orbit.target            = {center_x, center_y, center_z, 0.0f};
        camera_state.orbit.distance          = snapshot.grid.max_extent() * 1.35f;

        if (settings.view_mode == ViewMode::Plane) {
            update_camera_config(vk::camera::Projection::Orthographic, snapshot.grid.max_extent() * 1.1f);
            if (settings.plane_axis == PlaneAxis::XY) {
                camera_state.orbit.yaw_rad   = 0.0f;
                camera_state.orbit.pitch_rad = 0.0f;
            }
            if (settings.plane_axis == PlaneAxis::XZ) {
                camera_state.orbit.yaw_rad   = 0.0f;
                camera_state.orbit.pitch_rad = -1.55334303427f;
            }
            if (settings.plane_axis == PlaneAxis::YZ) {
                camera_state.orbit.yaw_rad   = 1.57079632679f;
                camera_state.orbit.pitch_rad = 0.0f;
            }
        } else {
            update_camera_config(vk::camera::Projection::Perspective, snapshot.grid.max_extent());
            camera_state.orbit.distance  = snapshot.grid.max_extent() * 1.10f;
            camera_state.orbit.yaw_rad   = 0.0f;
            camera_state.orbit.pitch_rad = 0.0f;
        }
        camera_.set_state(camera_state);
    }

    const vk::context::VulkanContext& VisualizationApp::vk_context() const {
        return vkctx_;
    }

    uint32_t VisualizationApp::frames_in_flight() const {
        return frames_.frames_in_flight;
    }

    std::vector<vk::raii::DescriptorSet> VisualizationApp::allocate_field_descriptor_sets(const uint32_t count) {
        std::vector<vk::DescriptorSetLayout> field_layouts(count, *field_set_layout_);
        vk::DescriptorSetAllocateInfo field_alloc_info{
            .descriptorPool     = *field_descriptor_pool_,
            .descriptorSetCount = count,
            .pSetLayouts        = field_layouts.data(),
        };
        return vkctx_.device.allocateDescriptorSets(field_alloc_info);
    }

    void VisualizationApp::recreate_swapchain() {
        vk::swapchain::recreate_swapchain(vkctx_, sctx_, sc_);
        vk::frame::on_swapchain_recreated(vkctx_, sc_, frames_);
        const auto image_count = static_cast<uint32_t>(sc_.images.size());
        if (imgui_sys_.image_count != image_count || imgui_sys_.min_image_count != image_count || imgui_sys_.color_format != sc_.format) {
            const bool docking   = imgui_sys_.docking;
            const bool viewports = imgui_sys_.viewports;
            vk::imgui::shutdown(imgui_sys_);
            imgui_sys_ = vk::imgui::create(vkctx_, window_, sc_.format, image_count, image_count, docking, viewports);
        }
        sctx_.resize_requested = false;
    }

    namespace {

        constexpr uint32_t snapshot_slot_count = 4;

    } // namespace

    void create_runtime_data(AppData& data) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        destroy_runtime_data(data);
        check_cuda(cudaStreamCreateWithFlags(&data.physics.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    void destroy_runtime_data(AppData& data) {
        for (auto& slot : data.capture.slots) {
            if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
            if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
            if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
            slot = {};
        }
        data.capture = {};
        if (data.physics.context != nullptr) stable_fluids_destroy_context_cuda(data.physics.context);
        if (data.physics.force_x_device != nullptr) cudaFree(data.physics.force_x_device);
        if (data.physics.force_y_device != nullptr) cudaFree(data.physics.force_y_device);
        if (data.physics.force_z_device != nullptr) cudaFree(data.physics.force_z_device);
        if (data.physics.density_source_device != nullptr) cudaFree(data.physics.density_source_device);
        data.physics.force_x_device        = nullptr;
        data.physics.force_y_device        = nullptr;
        data.physics.force_z_device        = nullptr;
        data.physics.density_source_device = nullptr;
        data.physics.force_x_host.clear();
        data.physics.force_z_host.clear();
        data.physics.source_mask.clear();
        data.physics.swirl_x_mask.clear();
        data.physics.swirl_z_mask.clear();
        data.physics.drift_mask.clear();
        if (data.physics.stream != nullptr) cudaStreamDestroy(data.physics.stream);
        data.physics = {};
    }

    void check_interop_support(const VisualizationApp& renderer) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
        if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("stable-fluids visualizer requires Vulkan timeline semaphore support");
        int cuda_device_index = 0;
        check_cuda(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
        int timeline_supported = 0;
        check_cuda(cudaDeviceGetAttribute(&timeline_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index), "cudaDeviceGetAttribute");
        if (timeline_supported == 0) throw std::runtime_error("CUDA timeline semaphore interop is required");
    }

    void rebuild_physics(const AppState& state, AppData& data) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        if (data.physics.context != nullptr) {
            check_stable(stable_fluids_destroy_context_cuda(data.physics.context), "stable_fluids_destroy_context_cuda");
            data.physics.context = nullptr;
        }
        if (data.physics.force_x_device != nullptr) cudaFree(data.physics.force_x_device);
        if (data.physics.force_y_device != nullptr) cudaFree(data.physics.force_y_device);
        if (data.physics.force_z_device != nullptr) cudaFree(data.physics.force_z_device);
        if (data.physics.density_source_device != nullptr) cudaFree(data.physics.density_source_device);
        data.physics.force_x_device        = nullptr;
        data.physics.force_y_device        = nullptr;
        data.physics.force_z_device        = nullptr;
        data.physics.density_source_device = nullptr;
        data.physics.force_x_host.clear();
        data.physics.force_z_host.clear();
        data.physics.source_mask.clear();
        data.physics.swirl_x_mask.clear();
        data.physics.swirl_z_mask.clear();
        data.physics.drift_mask.clear();

        const std::array fields{
            StableFluidsFieldCreateDesc{
                .name          = "density",
                .diffusion     = 0.00005f,
                .dissipation   = 0.35f,
                .initial_value = 0.0f,
            },
        };
        std::array<StableFluidsFieldHandle, 1> field_handles{};
        const StableFluidsContextCreateDesc create_desc{
            .config      = state.physics.config,
            .stream      = data.physics.stream,
            .fields      = fields.data(),
            .field_count = static_cast<uint32_t>(fields.size()),
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &data.physics.context, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda");
        data.physics.density_field = field_handles[0];

        const auto nx           = state.physics.config.nx;
        const auto ny           = state.physics.config.ny;
        const auto nz           = state.physics.config.nz;
        const auto cell_count   = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
        const auto scalar_bytes = cell_count * sizeof(float);
        const float h           = state.physics.config.cell_size;
        const float extent_x    = static_cast<float>(nx) * h;
        const float extent_y    = static_cast<float>(ny) * h;
        const float extent_z    = static_cast<float>(nz) * h;
        data.physics.grid       = {
                  .nx        = static_cast<uint32_t>(nx),
                  .ny        = static_cast<uint32_t>(ny),
                  .nz        = static_cast<uint32_t>(nz),
                  .cell_size = h,
        };
        const float source_x = extent_x * 0.50f;
        const float source_y = extent_y * 0.13f;
        const float source_z = extent_z * 0.50f;
        const float source_r = h * 6.0f;
        const float swirl_y  = source_y + source_r * 0.90f;
        const float drift_y  = source_y + source_r * 1.35f;

        data.physics.force_x_host.assign(cell_count, 0.0f);
        data.physics.force_z_host.assign(cell_count, 0.0f);
        data.physics.source_mask.assign(cell_count, 0.0f);
        data.physics.swirl_x_mask.assign(cell_count, 0.0f);
        data.physics.swirl_z_mask.assign(cell_count, 0.0f);
        data.physics.drift_mask.assign(cell_count, 0.0f);

        std::vector<float> force_y_host(cell_count, 0.0f);
        std::vector<float> density_source_host(cell_count, 0.0f);
        auto radial_weight = [](const float px, const float py, const float pz, const float cx, const float cy, const float cz, const float radius) {
            const float dx      = px - cx;
            const float dy      = py - cy;
            const float dz      = pz - cz;
            const float radius2 = radius * radius;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            if (dist2 >= radius2 || radius2 <= 0.0f) return 0.0f;
            return 1.0f - dist2 / radius2;
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto index                 = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                    const float px                   = (static_cast<float>(x) + 0.5f) * h;
                    const float py                   = (static_cast<float>(y) + 0.5f) * h;
                    const float pz                   = (static_cast<float>(z) + 0.5f) * h;
                    const float source_weight        = radial_weight(px, py, pz, source_x, source_y, source_z, source_r);
                    const float swirl_weight         = radial_weight(px, py, pz, source_x, swirl_y, source_z, source_r * 1.65f);
                    const float drift_weight         = radial_weight(px, py, pz, source_x, drift_y, source_z, source_r * 2.10f);
                    const float dx                   = px - source_x;
                    const float dz                   = pz - source_z;
                    const float radial               = std::sqrt(dx * dx + dz * dz);
                    const float inv_radial           = radial > 1.0e-5f ? 1.0f / radial : 0.0f;
                    data.physics.source_mask[index]  = source_weight;
                    data.physics.swirl_x_mask[index] = -dz * inv_radial * swirl_weight;
                    data.physics.swirl_z_mask[index] = dx * inv_radial * swirl_weight;
                    data.physics.drift_mask[index]   = drift_weight;
                    density_source_host[index]       = 32.0f * source_weight;
                    force_y_host[index]              = 7.6f * source_weight;
                }
            }
        }

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data.physics.force_x_device), scalar_bytes), "cudaMalloc force_x_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data.physics.force_y_device), scalar_bytes), "cudaMalloc force_y_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data.physics.force_z_device), scalar_bytes), "cudaMalloc force_z_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data.physics.density_source_device), scalar_bytes), "cudaMalloc density_source_device");
        check_cuda(cudaMemsetAsync(data.physics.force_x_device, 0, scalar_bytes, data.physics.stream), "cudaMemsetAsync force_x_device");
        check_cuda(cudaMemsetAsync(data.physics.force_z_device, 0, scalar_bytes, data.physics.stream), "cudaMemsetAsync force_z_device");
        check_cuda(cudaMemcpyAsync(data.physics.force_y_device, force_y_host.data(), scalar_bytes, cudaMemcpyHostToDevice, data.physics.stream), "cudaMemcpyAsync force_y_device");
        check_cuda(cudaMemcpyAsync(data.physics.density_source_device, density_source_host.data(), scalar_bytes, cudaMemcpyHostToDevice, data.physics.stream), "cudaMemcpyAsync density_source_device");
        data.physics.animation_step = 0;
        data.physics.stats          = {};
    }

    void step_physics(const AppState&, AppData& data, const int sim_steps) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        if (sim_steps <= 0) return;
        const auto scalar_bytes = data.physics.force_x_host.size() * sizeof(float);
        const StableFluidsFieldSourceDesc field_source{
            .field  = data.physics.density_field,
            .values = data.physics.density_source_device,
        };

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            const float phase   = static_cast<float>(data.physics.animation_step) * 0.045f;
            const float drift_x = 1.15f * std::sin(phase);
            const float drift_z = 0.85f * std::cos(phase * 0.71f);
            const float swirl   = 1.65f * std::cos(phase * 0.53f);
            for (size_t i = 0; i < data.physics.force_x_host.size(); ++i) {
                data.physics.force_x_host[i] = drift_x * data.physics.drift_mask[i] + swirl * data.physics.swirl_x_mask[i];
                data.physics.force_z_host[i] = drift_z * data.physics.drift_mask[i] + swirl * data.physics.swirl_z_mask[i];
            }

            const auto begin = std::chrono::steady_clock::now();
            check_cuda(cudaMemcpyAsync(data.physics.force_x_device, data.physics.force_x_host.data(), scalar_bytes, cudaMemcpyHostToDevice, data.physics.stream), "cudaMemcpyAsync force_x_device");
            check_cuda(cudaMemcpyAsync(data.physics.force_z_device, data.physics.force_z_host.data(), scalar_bytes, cudaMemcpyHostToDevice, data.physics.stream), "cudaMemcpyAsync force_z_device");
            const StableFluidsStepDesc step_desc{
                .force_x            = data.physics.force_x_device,
                .force_y            = data.physics.force_y_device,
                .force_z            = data.physics.force_z_device,
                .field_sources      = &field_source,
                .field_source_count = 1,
            };
            check_stable(stable_fluids_step_cuda(data.physics.context, &step_desc), "stable_fluids_step_cuda");
            const auto elapsed_ms                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            data.physics.stats.last_step_call_ms = elapsed_ms;
            ++data.physics.stats.step_count;
            data.physics.stats.average_step_call_ms += (elapsed_ms - data.physics.stats.average_step_call_ms) / static_cast<double>(data.physics.stats.step_count);
            ++data.physics.animation_step;
        }

        data.physics.stats.projection_max_abs_divergence = 0.0f;
        data.physics.stats.projection_rms_divergence     = 0.0f;
    }

    bool sync_capture_storage(AppData& data, VisualizationApp& renderer) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        const GridShape request_grid{
            .nx        = data.physics.grid.nx,
            .ny        = data.physics.grid.ny,
            .nz        = data.physics.grid.nz,
            .cell_size = data.physics.grid.cell_size,
        };
        const bool matches = !data.capture.slots.empty() && data.capture.request_grid.nx == request_grid.nx && data.capture.request_grid.ny == request_grid.ny && data.capture.request_grid.nz == request_grid.nz && data.capture.request_grid.cell_size == request_grid.cell_size;
        if (matches) return false;

        renderer.vk_context().device.waitIdle();
        for (auto& slot : data.capture.slots) {
            if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
            if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
            if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
            slot = {};
        }
        data.capture              = {};
        data.capture.request_grid = request_grid;
        data.capture.field_bytes  = static_cast<uint64_t>(request_grid.nx) * static_cast<uint64_t>(request_grid.ny) * static_cast<uint64_t>((std::max) (request_grid.nz, 1u)) * sizeof(float);

        auto descriptor_sets = renderer.allocate_field_descriptor_sets(snapshot_slot_count);
        data.capture.slots.reserve(descriptor_sets.size());
        for (size_t slot_index = 0; slot_index < descriptor_sets.size(); ++slot_index) {
            auto& slot          = data.capture.slots.emplace_back();
            slot.descriptor_set = std::move(descriptor_sets[slot_index]);
#if defined(_WIN32)
            constexpr auto memory_handle_type    = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
            constexpr auto semaphore_handle_type = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
            constexpr auto memory_handle_type    = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
            constexpr auto semaphore_handle_type = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif
            vk::SemaphoreTypeCreateInfo timeline_semaphore_ci{
                .semaphoreType = vk::SemaphoreType::eTimeline,
                .initialValue  = 0,
            };
            vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                .pNext       = &timeline_semaphore_ci,
                .handleTypes = semaphore_handle_type,
            };
            vk::SemaphoreCreateInfo semaphore_ci{
                .pNext = &export_semaphore_ci,
            };
            slot.timeline_semaphore = vk::raii::Semaphore{renderer.vk_context().device, semaphore_ci};

            vk::ExternalMemoryBufferCreateInfo external_buffer_ci{
                .handleTypes = memory_handle_type,
            };
            vk::BufferCreateInfo buffer_ci{
                .pNext       = &external_buffer_ci,
                .size        = data.capture.field_bytes,
                .usage       = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                .sharingMode = vk::SharingMode::eExclusive,
            };
            slot.buffer                               = vk::raii::Buffer{renderer.vk_context().device, buffer_ci};
            const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
            vk::ExportMemoryAllocateInfo export_memory_ci{
                .handleTypes = memory_handle_type,
            };
            vk::MemoryAllocateInfo alloc_ci{
                .pNext           = &export_memory_ci,
                .allocationSize  = requirements.size,
                .memoryTypeIndex = vk::memory::find_memory_type(renderer.vk_context().physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
            };
            slot.memory = vk::raii::DeviceMemory{renderer.vk_context().device, alloc_ci};
            slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
            vk::MemoryGetWin32HandleInfoKHR memory_handle_info{
                .memory     = *slot.memory,
                .handleType = memory_handle_type,
            };
            HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueWin32,
                .handle =
                    {
                        .win32 =
                            {
                                .handle = memory_handle,
                            },
                    },
                .size = requirements.size,
            };
            check_cuda(cudaImportExternalMemory(&slot.external_memory, &external_memory_desc), "cudaImportExternalMemory");
            CloseHandle(memory_handle);

            vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                .semaphore  = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32,
                .handle =
                    {
                        .win32 =
                            {
                                .handle = semaphore_handle,
                            },
                    },
            };
            check_cuda(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
            CloseHandle(semaphore_handle);
#else
            vk::MemoryGetFdInfoKHR memory_handle_info{
                .memory     = *slot.memory,
                .handleType = memory_handle_type,
            };
            const int memory_fd = renderer.vk_context().device.getMemoryFdKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueFd,
                .handle =
                    {
                        .fd = memory_fd,
                    },
                .size = requirements.size,
            };
            check_cuda(cudaImportExternalMemory(&slot.external_memory, &external_memory_desc), "cudaImportExternalMemory");

            vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                .semaphore  = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            const int semaphore_fd = renderer.vk_context().device.getSemaphoreFdKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
                .handle =
                    {
                        .fd = semaphore_fd,
                    },
            };
            check_cuda(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
#endif

            cudaExternalMemoryBufferDesc buffer_desc{
                .offset = 0,
                .size   = data.capture.field_bytes,
            };
            check_cuda(cudaExternalMemoryGetMappedBuffer(&slot.field_cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");

            vk::DescriptorBufferInfo field_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range  = data.capture.field_bytes,
            };
            vk::WriteDescriptorSet field_write{
                .dstSet          = *slot.descriptor_set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &field_info,
            };
            renderer.vk_context().device.updateDescriptorSets(field_write, {});
        }
        return true;
    }

    bool capture_snapshot(AppState& state, AppData& data, VisualizationApp& renderer, const char* tag) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
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
        nvtx3::scoped_range range{tag};

        auto& slot                   = data.capture.slots.at(static_cast<size_t>(slot_index));
        state.physics.selected_field = std::clamp(state.physics.selected_field, 0, static_cast<int>(field_catalog_storage.size()) - 1);
        const auto& field            = field_catalog_storage[static_cast<size_t>(state.physics.selected_field)];
        const StableFluidsExportDesc export_desc{
            .kind  = field.export_kind,
            .field = field.id == FieldId::Density ? data.physics.density_field : 0u,
        };

        const auto begin = std::chrono::steady_clock::now();
        check_stable(stable_fluids_export_cuda(data.physics.context, &export_desc, slot.field_cuda_ptr), "stable_fluids_export_cuda");
        cudaExternalSemaphoreSignalParams signal_params{};
        signal_params.params.fence.value = data.capture.generation + 1;
        check_cuda(cudaSignalExternalSemaphoresAsync(&slot.external_semaphore, &signal_params, 1, data.physics.stream), "cudaSignalExternalSemaphoresAsync");
        check_cuda(cudaStreamSynchronize(data.physics.stream), "cudaStreamSynchronize");

        slot.ready_generation               = data.capture.generation + 1;
        slot.grid                           = data.capture.request_grid;
        slot.field_component_count          = 1;
        slot.semantic                       = field.semantic;
        slot.label                          = field.label;
        data.capture.generation             = slot.ready_generation;
        data.capture.active_slot            = slot_index;
        data.capture.stats.last_snapshot_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        ++data.capture.stats.snapshot_count;
        data.capture.stats.average_snapshot_ms += (data.capture.stats.last_snapshot_ms - data.capture.stats.average_snapshot_ms) / static_cast<double>(data.capture.stats.snapshot_count);
        return true;
    }

    std::optional<VisualizationSnapshotView> active_snapshot(const AppData& data) {
        if (data.capture.active_slot < 0) return std::nullopt;
        const auto& slot = data.capture.slots.at(static_cast<size_t>(data.capture.active_slot));
        return VisualizationSnapshotView{
            .grid = slot.grid,
            .field =
                {
                    .descriptor_set     = *slot.descriptor_set,
                    .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                    .ready_generation   = slot.ready_generation,
                    .component_count    = slot.field_component_count,
                    .semantic           = slot.semantic,
                    .label              = slot.label,
                },
        };
    }

    void mark_snapshot_submitted(AppData& data) {
        const uint64_t next_submit_serial = data.capture.submit_serial + 1;
        if (data.capture.active_slot >= 0) data.capture.slots[static_cast<size_t>(data.capture.active_slot)].last_used_submit_serial = next_submit_serial;
        data.capture.submit_serial = next_submit_serial;
    }

} // namespace app
