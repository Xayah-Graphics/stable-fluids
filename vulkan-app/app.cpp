module;

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
        imgui_sys_ = imgui::create(vkctx_, window_, sc_.format, frames_in_flight_value_, static_cast<uint32_t>(sc_.images.size()), true, false);

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

        shader_dir_                      = std::filesystem::path(SMOKE_SIM_SHADER_DIR);
        const auto background_shader_spv = pipeline::read_file_bytes((shader_dir_ / "background.spv").string());
        const auto plane_shader_spv      = pipeline::read_file_bytes((shader_dir_ / "field_plane.spv").string());
        const auto volume_shader_spv     = pipeline::read_file_bytes((shader_dir_ / "field_volume.spv").string());
        background_shader_module_        = pipeline::load_shader_module(vkctx_.device, background_shader_spv);
        plane_shader_module_             = pipeline::load_shader_module(vkctx_.device, plane_shader_spv);
        volume_shader_module_            = pipeline::load_shader_module(vkctx_.device, volume_shader_spv);

        std::array<DescriptorSetLayout, 1> pipeline_set_layouts{*field_set_layout_};
        pipeline::GraphicsPipelineDesc background_pipeline_desc{
            .color_format         = sc_.format,
            .use_depth            = false,
            .use_blend            = false,
            .topology             = PrimitiveTopology::eTriangleList,
            .cull                 = CullModeFlagBits::eNone,
            .push_constant_bytes  = sizeof(FieldPushConstants),
            .push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment,
        };
        pipeline::GraphicsPipelineDesc pipeline_desc{
            .color_format         = sc_.format,
            .use_depth            = false,
            .use_blend            = true,
            .topology             = PrimitiveTopology::eTriangleList,
            .cull                 = CullModeFlagBits::eNone,
            .push_constant_bytes  = sizeof(FieldPushConstants),
            .push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment,
            .set_layouts          = pipeline_set_layouts,
        };

        pipeline::VertexInput empty_vertex_input{};
        background_pipeline_ = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, background_pipeline_desc, background_shader_module_, "vs_main", "fs_main");
        plane_pipeline_      = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, plane_shader_module_, "vs_main", "fs_main");
        volume_pipeline_     = pipeline::create_graphics_pipeline(vkctx_.device, empty_vertex_input, pipeline_desc, volume_shader_module_, "vs_main", "fs_main");
        last_frame_time_     = std::chrono::steady_clock::now();
    }

    VisualizationApp::~VisualizationApp() {
        try {
            vkctx_.device.waitIdle();
        } catch (...) {
        }

        if (imgui_sys_.initialized) vk::imgui::shutdown(imgui_sys_);
        volume_pipeline_          = {};
        plane_pipeline_           = {};
        background_pipeline_      = {};
        volume_shader_module_     = nullptr;
        plane_shader_module_      = nullptr;
        background_shader_module_ = nullptr;
        field_descriptor_pool_    = nullptr;
        field_set_layout_         = nullptr;
        frames_                   = {};
        sc_                       = {};
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

    void VisualizationApp::draw_visualization_ui(AppState& state, const SceneInfo& scene, const std::span<const FieldInfo> fields, bool& reset_requested, bool& field_changed, const std::optional<VisualizationSnapshotView>& snapshot) {
        bool reframe_requested = false;
        auto& settings         = state.render;
        if (fields.empty()) throw std::runtime_error("scene must expose at least one field");
        state.selected_field = std::clamp(state.selected_field, 0, static_cast<int>(fields.size()) - 1);
        const auto& field    = fields[static_cast<size_t>(state.selected_field)];

        ImGui::Begin("Stable Fluids");
        if (ImGui::BeginCombo("Field", field.label.data())) {
            for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                const bool is_selected = state.selected_field == i;
                if (ImGui::Selectable(fields[static_cast<size_t>(i)].label.data(), is_selected)) {
                    state.selected_field = i;
                    apply_field_preset(settings, fields[static_cast<size_t>(i)].preset);
                    field_changed = true;
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("Pause", &state.paused);
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

        if (settings.view_mode == ViewMode::Plane || settings.show_velocity_plane) {
            int plane_axis = static_cast<int>(settings.plane_axis);
            constexpr std::array plane_labels{
                "XY",
                "XZ",
                "YZ",
            };
            if (ImGui::Combo("Axis", &plane_axis, plane_labels.data(), static_cast<int>(plane_labels.size()))) {
                settings.plane_axis = static_cast<PlaneAxis>(plane_axis);
                reframe_requested   = true;
            }
            ImGui::SliderFloat("Slice", &settings.slice_position, 0.0f, 1.0f, "%.3f");
        }
        if (ImGui::Checkbox("Velocity Plane", &settings.show_velocity_plane)) field_changed = true;
        if (settings.show_velocity_plane) {
            ImGui::SliderInt("Arrow Seeds", &settings.velocity_plane_seed_count, 4, 48);
            ImGui::SliderFloat("Arrow Scale", &settings.velocity_plane_arrow_cells, 0.1f, 2.0f, "%.2f");
            ImGui::SliderFloat("Min Speed", &settings.velocity_plane_min_speed, 0.001f, 0.5f, "%.3f");
            ImGui::SliderFloat("Arrow Width", &settings.velocity_plane_thickness, 0.5f, 4.0f, "%.2f");
        }

        ImGui::Separator();
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Grid: %u x %u x %u", scene.grid.nx, scene.grid.ny, scene.grid.nz);
        ImGui::Text("dt: %.5f  h: %.4f", scene.dt, scene.grid.cell_size);
        ImGui::Text("Field: %.*s", static_cast<int>(field.label.size()), field.label.data());
        ImGui::Text("Steps: %llu", static_cast<unsigned long long>(scene.step_count));
        ImGui::Text("Step Call: %.3f ms", scene.last_step_call_ms);
        ImGui::End();

        if (reframe_requested && snapshot) frame_content(settings, *snapshot);
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
        clear_value.color = ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
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

        const auto& matrices      = camera_.matrices();
        const auto& camera_config = camera_.config();
        const float aspect        = static_cast<float>(sc_.extent.width) / static_cast<float>((std::max) (sc_.extent.height, 1u));
        const float half_fov_tan  = std::tan(camera_config.fov_y_rad * 0.5f);
        FieldPushConstants push{};
        push.eye               = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
        push.right             = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
        push.up                = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
        push.forward           = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
        push.background_bottom = {settings.background_bottom_r, settings.background_bottom_g, settings.background_bottom_b, 1.0f};
        push.background_top    = {settings.background_top_r, settings.background_top_g, settings.background_top_b, 1.0f};
        push.params0           = {
            aspect,
            half_fov_tan,
            0.0f,
            0.0f,
        };
        push.params2 = {
            static_cast<uint32_t>(camera_config.projection),
            static_cast<uint32_t>(settings.plane_axis),
            0u,
            0u,
        };
        push.params3 = {
            0.0f,
            0.0f,
            settings.slice_position,
            camera_config.ortho_height,
        };

        const ArrayProxy<const FieldPushConstants> push_block(1, &push);
        cmd.bindPipeline(PipelineBindPoint::eGraphics, *background_pipeline_.pipeline);
        cmd.pushConstants(*background_pipeline_.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
        cmd.draw(3, 1, 0, 0);

        if (snapshot) {
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
                static_cast<uint32_t>(camera_config.projection),
                static_cast<uint32_t>(settings.plane_axis),
                0u,
                0u,
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
            cmd.pushConstants(*pipeline.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();

        if (snapshot && settings.show_velocity_plane && snapshot->velocity != nullptr) {
            if (ImGuiViewport* viewport = ImGui::GetMainViewport()) {
                ImDrawList* draw_list = ImGui::GetForegroundDrawList(viewport);
                const auto& view_proj = camera_.matrices().view_proj;
                auto project_point    = [&](const vk::math::vec3& point, ImVec2& out) {
                    const auto clip = vk::math::mul(view_proj, vk::math::vec4{point.x, point.y, point.z, 1.0f});
                    if (clip.w <= 1.0e-4f) return false;
                    const float inv_w = 1.0f / clip.w;
                    const float ndc_x = clip.x * inv_w;
                    const float ndc_y = clip.y * inv_w;
                    const float ndc_z = clip.z * inv_w;
                    if (ndc_z < -0.25f || ndc_z > 1.25f) return false;
                    out.x = viewport->Pos.x + (ndc_x * 0.5f + 0.5f) * viewport->Size.x;
                    out.y = viewport->Pos.y + (1.0f - (ndc_y * 0.5f + 0.5f)) * viewport->Size.y;
                    return true;
                };
                auto draw_segment = [&](const vk::math::vec3& a, const vk::math::vec3& b, const ImU32 color) {
                    ImVec2 screen_a{};
                    ImVec2 screen_b{};
                    if (!project_point(a, screen_a)) return;
                    if (!project_point(b, screen_b)) return;
                    draw_list->AddLine(screen_a, screen_b, color, settings.velocity_plane_thickness);
                };
                auto sample_velocity = [&](const float px, const float py, const float pz) {
                    const auto nx         = static_cast<int>(snapshot->grid.nx);
                    const auto ny         = static_cast<int>(snapshot->grid.ny);
                    const auto nz         = static_cast<int>(snapshot->grid.nz);
                    const auto cell_count = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
                    const float gx        = std::clamp(px / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(nx - 1));
                    const float gy        = std::clamp(py / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(ny - 1));
                    const float gz        = std::clamp(pz / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(nz - 1));
                    const int x0          = static_cast<int>(std::floor(gx));
                    const int y0          = static_cast<int>(std::floor(gy));
                    const int z0          = static_cast<int>(std::floor(gz));
                    const int x1          = (std::min) (x0 + 1, nx - 1);
                    const int y1          = (std::min) (y0 + 1, ny - 1);
                    const int z1          = (std::min) (z0 + 1, nz - 1);
                    const float tx        = gx - static_cast<float>(x0);
                    const float ty        = gy - static_cast<float>(y0);
                    const float tz        = gz - static_cast<float>(z0);
                    auto load             = [&](const int x, const int y, const int z) {
                        const auto index = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                        return vk::math::vec3{
                            snapshot->velocity[index],
                            snapshot->velocity[cell_count + index],
                            snapshot->velocity[cell_count * 2u + index],
                            0.0f,
                        };
                    };
                    auto lerp3 = [&](const vk::math::vec3& a, const vk::math::vec3& b, const float t) {
                        return vk::math::vec3{
                            std::lerp(a.x, b.x, t),
                            std::lerp(a.y, b.y, t),
                            std::lerp(a.z, b.z, t),
                            0.0f,
                        };
                    };
                    const auto c00 = lerp3(load(x0, y0, z0), load(x1, y0, z0), tx);
                    const auto c10 = lerp3(load(x0, y1, z0), load(x1, y1, z0), tx);
                    const auto c01 = lerp3(load(x0, y0, z1), load(x1, y0, z1), tx);
                    const auto c11 = lerp3(load(x0, y1, z1), load(x1, y1, z1), tx);
                    return lerp3(lerp3(c00, c10, ty), lerp3(c01, c11, ty), tz);
                };

                const PlaneAxis plane_axis = settings.plane_axis;
                const float max_x          = snapshot->grid.extent_x();
                const float max_y          = snapshot->grid.extent_y();
                const float max_z          = snapshot->grid.extent_z();
                const float slice_position = std::clamp(settings.slice_position, 0.0f, 1.0f);
                const int seed_count       = (std::max) (settings.velocity_plane_seed_count, 2);
                const float min_speed      = (std::max) (settings.velocity_plane_min_speed, 1.0e-5f);
                const float arrow_scale    = settings.velocity_plane_arrow_cells * snapshot->grid.cell_size;
                std::array<vk::math::vec3, 4> plane_corners{};
                if (plane_axis == PlaneAxis::XY) {
                    const float z = slice_position * max_z;
                    plane_corners = {
                        vk::math::vec3{0.0f, 0.0f, z, 0.0f},
                        vk::math::vec3{max_x, 0.0f, z, 0.0f},
                        vk::math::vec3{max_x, max_y, z, 0.0f},
                        vk::math::vec3{0.0f, max_y, z, 0.0f},
                    };
                }
                if (plane_axis == PlaneAxis::XZ) {
                    const float y = slice_position * max_y;
                    plane_corners = {
                        vk::math::vec3{0.0f, y, 0.0f, 0.0f},
                        vk::math::vec3{max_x, y, 0.0f, 0.0f},
                        vk::math::vec3{max_x, y, max_z, 0.0f},
                        vk::math::vec3{0.0f, y, max_z, 0.0f},
                    };
                }
                if (plane_axis == PlaneAxis::YZ) {
                    const float x = slice_position * max_x;
                    plane_corners = {
                        vk::math::vec3{x, 0.0f, 0.0f, 0.0f},
                        vk::math::vec3{x, max_y, 0.0f, 0.0f},
                        vk::math::vec3{x, max_y, max_z, 0.0f},
                        vk::math::vec3{x, 0.0f, max_z, 0.0f},
                    };
                }
                draw_segment(plane_corners[0], plane_corners[1], IM_COL32(112, 220, 255, 120));
                draw_segment(plane_corners[1], plane_corners[2], IM_COL32(112, 220, 255, 120));
                draw_segment(plane_corners[2], plane_corners[3], IM_COL32(112, 220, 255, 120));
                draw_segment(plane_corners[3], plane_corners[0], IM_COL32(112, 220, 255, 120));

                for (int j = 0; j < seed_count; ++j) {
                    for (int i = 0; i < seed_count; ++i) {
                        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(seed_count);
                        const float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(seed_count);
                        vk::math::vec3 pos{};
                        if (plane_axis == PlaneAxis::XY) pos = {u * max_x, v * max_y, slice_position * max_z, 0.0f};
                        if (plane_axis == PlaneAxis::XZ) pos = {u * max_x, slice_position * max_y, v * max_z, 0.0f};
                        if (plane_axis == PlaneAxis::YZ) pos = {slice_position * max_x, u * max_y, v * max_z, 0.0f};
                        const auto velocity = sample_velocity(pos.x, pos.y, pos.z);
                        vk::math::vec3 plane_velocity{};
                        if (plane_axis == PlaneAxis::XY) plane_velocity = {velocity.x, velocity.y, 0.0f, 0.0f};
                        if (plane_axis == PlaneAxis::XZ) plane_velocity = {velocity.x, 0.0f, velocity.z, 0.0f};
                        if (plane_axis == PlaneAxis::YZ) plane_velocity = {0.0f, velocity.y, velocity.z, 0.0f};
                        const float speed = std::sqrt(plane_velocity.x * plane_velocity.x + plane_velocity.y * plane_velocity.y + plane_velocity.z * plane_velocity.z);
                        if (speed < min_speed) continue;
                        const float inv_speed = 1.0f / speed;
                        const vk::math::vec3 direction{
                            plane_velocity.x * inv_speed,
                            plane_velocity.y * inv_speed,
                            plane_velocity.z * inv_speed,
                            0.0f,
                        };
                        vk::math::vec3 side{};
                        if (plane_axis == PlaneAxis::XY) side = {-direction.y, direction.x, 0.0f, 0.0f};
                        if (plane_axis == PlaneAxis::XZ) side = {-direction.z, 0.0f, direction.x, 0.0f};
                        if (plane_axis == PlaneAxis::YZ) side = {0.0f, -direction.z, direction.y, 0.0f};
                        const float speed_t      = std::clamp(speed / (min_speed * 8.0f), 0.0f, 1.0f);
                        const float glyph_length = arrow_scale * std::lerp(0.35f, 1.0f, speed_t);
                        const float head_length  = glyph_length * 0.34f;
                        const float wing_span    = head_length * 0.55f;
                        vk::math::vec3 tip{
                            pos.x + direction.x * glyph_length,
                            pos.y + direction.y * glyph_length,
                            pos.z + direction.z * glyph_length,
                            0.0f,
                        };
                        tip.x = std::clamp(tip.x, 0.0f, max_x);
                        tip.y = std::clamp(tip.y, 0.0f, max_y);
                        tip.z = std::clamp(tip.z, 0.0f, max_z);
                        if (plane_axis == PlaneAxis::XY) tip.z = pos.z;
                        if (plane_axis == PlaneAxis::XZ) tip.y = pos.y;
                        if (plane_axis == PlaneAxis::YZ) tip.x = pos.x;
                        const vk::math::vec3 head_base{
                            tip.x - direction.x * head_length,
                            tip.y - direction.y * head_length,
                            tip.z - direction.z * head_length,
                            0.0f,
                        };
                        const vk::math::vec3 head_left{
                            head_base.x + side.x * wing_span,
                            head_base.y + side.y * wing_span,
                            head_base.z + side.z * wing_span,
                            0.0f,
                        };
                        const vk::math::vec3 head_right{
                            head_base.x - side.x * wing_span,
                            head_base.y - side.y * wing_span,
                            head_base.z - side.z * wing_span,
                            0.0f,
                        };
                        const ImU32 color = IM_COL32(static_cast<int>(std::lerp(72.0f, 255.0f, speed_t)), static_cast<int>(std::lerp(196.0f, 212.0f, speed_t)), static_cast<int>(std::lerp(255.0f, 96.0f, speed_t)), static_cast<int>(std::lerp(112.0f, 224.0f, speed_t)));
                        draw_segment(pos, tip, color);
                        draw_segment(head_left, tip, color);
                        draw_segment(head_right, tip, color);
                    }
                }
            }
        }

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

    void destroy_runtime_data(AppData& data) {
        for (auto& slot : data.capture.slots) {
            if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
            if (slot.velocity_cuda_ptr != nullptr) cudaFree(slot.velocity_cuda_ptr);
            if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
            if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
            slot = {};
        }
        data.capture = {};
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

    void apply_field_preset(VisualizationSettings& settings, const FieldVisualPreset& preset) {
        settings.density_scale  = preset.density_scale;
        settings.scalar_min     = preset.scalar_min;
        settings.scalar_max     = preset.scalar_max;
        settings.scalar_opacity = preset.scalar_opacity;
        settings.scalar_low_r   = preset.scalar_low_r;
        settings.scalar_low_g   = preset.scalar_low_g;
        settings.scalar_low_b   = preset.scalar_low_b;
        settings.scalar_high_r  = preset.scalar_high_r;
        settings.scalar_high_g  = preset.scalar_high_g;
        settings.scalar_high_b  = preset.scalar_high_b;
    }

    bool sync_capture_storage(AppData& data, VisualizationApp& renderer, const GridShape& grid, const bool with_velocity_plane) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        const GridShape request_grid{
            .nx        = grid.nx,
            .ny        = grid.ny,
            .nz        = grid.nz,
            .cell_size = grid.cell_size,
        };
        const bool matches = !data.capture.slots.empty() && data.capture.request_grid.nx == request_grid.nx && data.capture.request_grid.ny == request_grid.ny && data.capture.request_grid.nz == request_grid.nz && data.capture.request_grid.cell_size == request_grid.cell_size && data.capture.has_velocity_storage == with_velocity_plane;
        if (matches) return false;

        renderer.vk_context().device.waitIdle();
        for (auto& slot : data.capture.slots) {
            if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
            if (slot.velocity_cuda_ptr != nullptr) cudaFree(slot.velocity_cuda_ptr);
            if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
            if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
            slot = {};
        }
        data.capture                      = {};
        data.capture.request_grid         = request_grid;
        data.capture.has_velocity_storage = with_velocity_plane;
        const auto field_bytes            = static_cast<uint64_t>(request_grid.nx) * static_cast<uint64_t>(request_grid.ny) * static_cast<uint64_t>((std::max) (request_grid.nz, 1u)) * sizeof(float);
        const auto velocity_bytes         = field_bytes * 3u;

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
                .size        = field_bytes,
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
                .size   = field_bytes,
            };
            check_cuda(cudaExternalMemoryGetMappedBuffer(&slot.field_cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");
            if (with_velocity_plane) {
                check_cuda(cudaMalloc(&slot.velocity_cuda_ptr, velocity_bytes), "cudaMalloc velocity snapshot");
                slot.velocity_host.resize(static_cast<size_t>(velocity_bytes / sizeof(float)));
            }

            vk::DescriptorBufferInfo field_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range  = field_bytes,
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

    std::optional<VisualizationSnapshotView> active_snapshot(const AppData& data) {
        if (data.capture.active_slot < 0) return std::nullopt;
        const auto& slot = data.capture.slots.at(static_cast<size_t>(data.capture.active_slot));
        return VisualizationSnapshotView{
            .grid = data.capture.request_grid,
            .field =
                {
                    .descriptor_set     = *slot.descriptor_set,
                    .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                    .ready_generation   = slot.ready_generation,
                },
            .velocity = slot.has_velocity_host && !slot.velocity_host.empty() ? slot.velocity_host.data() : nullptr,
        };
    }

    void mark_snapshot_submitted(AppData& data) {
        const uint64_t next_submit_serial = data.capture.submit_serial + 1;
        if (data.capture.active_slot >= 0) data.capture.slots[static_cast<size_t>(data.capture.active_slot)].last_used_submit_serial = next_submit_serial;
        data.capture.submit_serial = next_submit_serial;
    }

} // namespace app
