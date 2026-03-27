module;

#include <GLFW/glfw3.h>
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
        imgui_sys_ = imgui::create(vkctx_, window_, sc_.format, frames_in_flight_value_, static_cast<uint32_t>(sc_.images.size()));

        camera::CameraConfig camera_config{};
        camera_config.orbit_rotate_sens = 0.005f;
        camera_config.orbit_zoom_sens   = 0.12f;
        camera_.set_config(camera_config);
        camera_.home();

        DescriptorSetLayoutBinding field_binding{
            .binding = 0,
            .descriptorType = DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = ShaderStageFlagBits::eFragment,
        };
        DescriptorSetLayoutCreateInfo field_layout_ci{
            .bindingCount = 1,
            .pBindings = &field_binding,
        };
        field_set_layout_ = raii::DescriptorSetLayout{vkctx_.device, field_layout_ci};

        DescriptorPoolSize field_pool_size{
            .type = DescriptorType::eStorageBuffer,
            .descriptorCount = 128,
        };
        DescriptorPoolCreateInfo field_pool_ci{
            .flags = DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 128,
            .poolSizeCount = 1,
            .pPoolSizes = &field_pool_size,
        };
        field_descriptor_pool_ = raii::DescriptorPool{vkctx_.device, field_pool_ci};

        const std::filesystem::path shader_dir = std::filesystem::path(SMOKE_SIM_SHADER_DIR);
        const auto plane_shader_spv            = pipeline::read_file_bytes((shader_dir / "field_plane.spv").string());
        const auto volume_shader_spv           = pipeline::read_file_bytes((shader_dir / "field_volume.spv").string());
        plane_shader_module_                   = pipeline::load_shader_module(vkctx_.device, plane_shader_spv);
        volume_shader_module_                  = pipeline::load_shader_module(vkctx_.device, volume_shader_spv);

        std::array<DescriptorSetLayout, 1> pipeline_set_layouts{*field_set_layout_};
        pipeline::GraphicsPipelineDesc pipeline_desc{
            .color_format = sc_.format,
            .use_depth = false,
            .use_blend = false,
            .topology = PrimitiveTopology::eTriangleList,
            .cull = CullModeFlagBits::eNone,
            .push_constant_bytes = sizeof(FieldPushConstants),
            .push_constant_stages = ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment,
            .set_layouts = pipeline_set_layouts,
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

    FrameInfo VisualizationApp::begin_frame() {
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
        collect_camera_input(dt_seconds);
        return FrameInfo{.dt_seconds = dt_seconds, .render_fps = render_fps_};
    }

    void VisualizationApp::draw_visualization_ui(const std::optional<VisualizationSnapshotView>& snapshot) {
        auto smoke_capable = [&](const VisualizationSnapshotView& view) {
            return view.field.semantic == FieldSemantic::DyeColor && view.field.component_count == 4;
        };

        bool reframe_requested = false;
        if (snapshot && !smoke_capable(*snapshot) && settings_.render_mode == RenderMode::Smoke) settings_.render_mode = RenderMode::Scalar;

        ImGui::Begin("Visualization");
        if (snapshot) {
            ImGui::Text("Field: %.*s", static_cast<int>(snapshot->field.label.size()), snapshot->field.label.data());
            ImGui::Text("Grid: %u x %u x %u", snapshot->grid.nx, snapshot->grid.ny, snapshot->grid.nz);
            ImGui::Text("Generation: %llu", static_cast<unsigned long long>(snapshot->field.ready_generation));

            int view_mode = static_cast<int>(settings_.view_mode);
            const char* labels[] = {"Plane", "Volume"};
            if (ImGui::Combo("View", &view_mode, labels, 2)) {
                settings_.view_mode = static_cast<ViewMode>(view_mode);
                reframe_requested = true;
            }

            const auto resolved_view = settings_.view_mode;
            if (resolved_view == ViewMode::Plane) {
                int plane_axis = static_cast<int>(settings_.plane_axis);
                const char* axis_labels[] = {"XY", "XZ", "YZ"};
                if (ImGui::Combo("Plane Axis", &plane_axis, axis_labels, 3)) {
                    settings_.plane_axis = static_cast<PlaneAxis>(plane_axis);
                    reframe_requested = true;
                }
                ImGui::SliderFloat("Slice", &settings_.slice_position, 0.0f, 1.0f, "%.3f");
            }

            if (smoke_capable(*snapshot)) {
                int render_mode = static_cast<int>(settings_.render_mode);
                const char* labels[] = {"Smoke", "Scalar"};
                if (ImGui::Combo("Field Mode", &render_mode, labels, 2)) settings_.render_mode = static_cast<RenderMode>(render_mode);
            } else {
                ImGui::TextUnformatted("Field Mode: Scalar");
            }

            ImGui::SliderFloat("Density Scale", &settings_.density_scale, 0.05f, 8.0f, "%.2f");
            if (resolved_view == ViewMode::Volume) ImGui::SliderInt("March Steps", &settings_.march_steps, 24, 224);
            if (settings_.render_mode == RenderMode::Smoke) {
                ImGui::SliderFloat("Absorption", &settings_.absorption, 0.05f, 8.0f, "%.2f");
            } else {
                ImGui::SliderFloat("Value Min", &settings_.scalar_min, -200.0f, 200.0f, "%.3f");
                ImGui::SliderFloat("Value Max", &settings_.scalar_max, -200.0f, 200.0f, "%.3f");
                ImGui::SliderFloat("Opacity", &settings_.scalar_opacity, 0.05f, 8.0f, "%.2f");
                ImGui::ColorEdit3("Low Color", &settings_.scalar_low_r);
                ImGui::ColorEdit3("High Color", &settings_.scalar_high_r);
            }

            ImGui::Separator();
            ImGui::Checkbox("Show Bounds", &settings_.show_bounds);
            ImGui::Checkbox("Show Collider", &settings_.show_collider);
            ImGui::Checkbox("Show Velocity Plane", &settings_.show_velocity_plane);
            if (settings_.show_velocity_plane) {
                ImGui::SliderInt("Vector Grid", &settings_.velocity_grid, 4, 48);
                ImGui::SliderInt("Vector Steps", &settings_.velocity_steps, 4, 96);
                ImGui::SliderFloat("Vector Step", &settings_.velocity_step, 0.10f, 3.0f, "%.2f");
                ImGui::SliderFloat("Min Speed", &settings_.velocity_min_speed, 0.0f, 4.0f, "%.3f");
                ImGui::SliderFloat("Line Width", &settings_.velocity_thickness, 0.5f, 4.0f, "%.2f");
            }
        } else {
            ImGui::TextUnformatted("Field: None");
        }
        ImGui::End();

        if (reframe_requested && snapshot) frame_content(*snapshot);

        if (const ImGuiViewport* viewport = ImGui::GetMainViewport()) {
            ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + 12.0f, viewport->Pos.y + 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
            ImGui::Begin("Render Stats Overlay", nullptr, overlay_flags);
            ImGui::Text("Render: %.1f FPS", render_fps_);
            ImGui::End();
        }
    }

    bool VisualizationApp::render_frame(const std::optional<VisualizationSnapshotView>& snapshot) {
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
            .srcStageMask = previous_layout == ImageLayout::eUndefined ? PipelineStageFlagBits2::eNone : PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = previous_layout == ImageLayout::eUndefined ? AccessFlags2{} : (AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite),
            .dstStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = previous_layout,
            .newLayout = ImageLayout::eColorAttachmentOptimal,
            .image = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_color_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::eColorAttachmentOptimal;

        ClearValue clear_value{};
        clear_value.color = ClearColorValue{std::array<float, 4>{0.035f, 0.04f, 0.05f, 1.0f}};
        RenderingAttachmentInfo color_attachment{
            .imageView = *sc_.image_views[image_index],
            .imageLayout = ImageLayout::eColorAttachmentOptimal,
            .loadOp = AttachmentLoadOp::eClear,
            .storeOp = AttachmentStoreOp::eStore,
            .clearValue = clear_value,
        };
        RenderingInfo rendering_info{
            .renderArea = Rect2D{Offset2D{0, 0}, sc_.extent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment,
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

        ViewMode resolved_view = ViewMode::Plane;
        PlaneAxis resolved_plane = settings_.plane_axis;
        if (snapshot) {
            resolved_view = settings_.view_mode;
            const auto& matrices = camera_.matrices();
            const auto& camera_config = camera_.config();
            const float aspect = static_cast<float>(sc_.extent.width) / static_cast<float>((std::max)(sc_.extent.height, 1u));
            const float half_fov_tan = std::tan(camera_config.fov_y_rad * 0.5f);
            const float slice_position = std::clamp(settings_.slice_position, 0.0f, 1.0f);
            FieldPushConstants push{};
            push.eye        = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
            push.right      = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
            push.up         = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
            push.forward    = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
            push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
            push.volume_max = {snapshot->grid.extent_x(), snapshot->grid.extent_y(), snapshot->grid.extent_z(), 0.0f};
            push.color_a    = settings_.render_mode == RenderMode::Smoke ? vk::math::vec4{} : vk::math::vec4{settings_.scalar_low_r, settings_.scalar_low_g, settings_.scalar_low_b, 1.0f};
            push.color_b    = settings_.render_mode == RenderMode::Smoke ? vk::math::vec4{} : vk::math::vec4{settings_.scalar_high_r, settings_.scalar_high_g, settings_.scalar_high_b, 1.0f};
            push.params0    = {
                aspect,
                half_fov_tan,
                settings_.density_scale,
                settings_.render_mode == RenderMode::Smoke ? settings_.absorption : settings_.scalar_opacity,
            };
            push.params1 = {
                snapshot->grid.nx,
                snapshot->grid.ny,
                snapshot->grid.nz,
                static_cast<uint32_t>(settings_.march_steps),
            };
            push.params2 = {
                static_cast<uint32_t>(settings_.render_mode),
                snapshot->field.component_count,
                static_cast<uint32_t>(resolved_plane),
                static_cast<uint32_t>(camera_config.projection),
            };
            push.params3 = {
                settings_.scalar_min,
                settings_.scalar_max,
                slice_position,
                camera_config.ortho_height,
            };

            const auto& pipeline = resolved_view == ViewMode::Volume ? volume_pipeline_ : plane_pipeline_;
            cmd.bindPipeline(PipelineBindPoint::eGraphics, *pipeline.pipeline);
            cmd.bindDescriptorSets(PipelineBindPoint::eGraphics, *pipeline.layout, 0, {snapshot->field.descriptor_set}, {});
            const ArrayProxy<const FieldPushConstants> push_block(1, &push);
            cmd.pushConstants(*pipeline.layout, ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eFragment, 0, push_block);
            cmd.draw(3, 1, 0, 0);
        }
        cmd.endRendering();

        if (snapshot && (settings_.show_bounds || (settings_.show_collider && snapshot->collider.enabled) || (settings_.show_velocity_plane && snapshot->velocity.data != nullptr))) {
            if (ImGuiViewport* viewport = ImGui::GetMainViewport()) {
                ImDrawList* draw_list = ImGui::GetForegroundDrawList(viewport);
                const auto& view_proj = camera_.matrices().view_proj;
                auto project_point = [&](const vk::math::vec3& point, ImVec2& out) {
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
                auto draw_segment = [&](const vk::math::vec3& a, const vk::math::vec3& b, const ImU32 color, const float thickness) {
                    ImVec2 screen_a{};
                    ImVec2 screen_b{};
                    if (!project_point(a, screen_a)) return;
                    if (!project_point(b, screen_b)) return;
                    draw_list->AddLine(screen_a, screen_b, color, thickness);
                };
                auto draw_box = [&](const std::array<vk::math::vec3, 8>& corners, const ImU32 color, const float thickness) {
                    constexpr std::array<std::array<int, 2>, 12> edges{{
                        {0, 1}, {1, 2}, {2, 3}, {3, 0},
                        {4, 5}, {5, 6}, {6, 7}, {7, 4},
                        {0, 4}, {1, 5}, {2, 6}, {3, 7},
                    }};
                    for (const auto& edge : edges) draw_segment(corners[static_cast<size_t>(edge[0])], corners[static_cast<size_t>(edge[1])], color, thickness);
                };
                auto sample_velocity = [&](const float px, const float py, const float pz) {
                    const auto nx = static_cast<int>(snapshot->grid.nx);
                    const auto ny = static_cast<int>(snapshot->grid.ny);
                    const auto nz = static_cast<int>(snapshot->grid.nz);
                    const float gx = std::clamp(px / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(nx - 1));
                    const float gy = std::clamp(py / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(ny - 1));
                    const float gz = std::clamp(pz / snapshot->grid.cell_size - 0.5f, 0.0f, static_cast<float>(nz - 1));
                    const int x0 = static_cast<int>(std::floor(gx));
                    const int y0 = static_cast<int>(std::floor(gy));
                    const int z0 = static_cast<int>(std::floor(gz));
                    const int x1 = (std::min)(x0 + 1, nx - 1);
                    const int y1 = (std::min)(y0 + 1, ny - 1);
                    const int z1 = (std::min)(z0 + 1, nz - 1);
                    const float tx = gx - static_cast<float>(x0);
                    const float ty = gy - static_cast<float>(y0);
                    const float tz = gz - static_cast<float>(z0);
                    const auto load = [&](const int x, const int y, const int z) {
                        const auto index = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                        return vk::math::vec3{
                            snapshot->velocity.data[index * 3u + 0u],
                            snapshot->velocity.data[index * 3u + 1u],
                            snapshot->velocity.data[index * 3u + 2u],
                            0.0f,
                        };
                    };
                    const auto lerp3 = [&](const vk::math::vec3& a, const vk::math::vec3& b, const float t) {
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

                const float max_x = snapshot->grid.extent_x();
                const float max_y = snapshot->grid.extent_y();
                const float max_z = snapshot->grid.extent_z();
                if (settings_.show_bounds) {
                    const std::array bounds_corners{
                        vk::math::vec3{0.0f, 0.0f, 0.0f, 0.0f},
                        vk::math::vec3{max_x, 0.0f, 0.0f, 0.0f},
                        vk::math::vec3{max_x, max_y, 0.0f, 0.0f},
                        vk::math::vec3{0.0f, max_y, 0.0f, 0.0f},
                        vk::math::vec3{0.0f, 0.0f, max_z, 0.0f},
                        vk::math::vec3{max_x, 0.0f, max_z, 0.0f},
                        vk::math::vec3{max_x, max_y, max_z, 0.0f},
                        vk::math::vec3{0.0f, max_y, max_z, 0.0f},
                    };
                    draw_box(bounds_corners, IM_COL32(236, 238, 244, 196), 1.6f);
                }

                if (settings_.show_collider && snapshot->collider.enabled) {
                    if (snapshot->collider.type == 0u) {
                        constexpr int ring_segments = 48;
                        constexpr float tau = 6.28318530718f;
                        const vk::math::vec3 center{snapshot->collider.center_x, snapshot->collider.center_y, snapshot->collider.center_z, 0.0f};
                        auto draw_ring = [&](const int plane) {
                            for (int i = 0; i < ring_segments; ++i) {
                                const float angle_a = tau * static_cast<float>(i) / static_cast<float>(ring_segments);
                                const float angle_b = tau * static_cast<float>(i + 1) / static_cast<float>(ring_segments);
                                const float cos_a = std::cos(angle_a);
                                const float sin_a = std::sin(angle_a);
                                const float cos_b = std::cos(angle_b);
                                const float sin_b = std::sin(angle_b);
                                vk::math::vec3 point_a{};
                                vk::math::vec3 point_b{};
                                if (plane == 0) {
                                    point_a = {center.x + cos_a * snapshot->collider.radius, center.y + sin_a * snapshot->collider.radius, center.z, 0.0f};
                                    point_b = {center.x + cos_b * snapshot->collider.radius, center.y + sin_b * snapshot->collider.radius, center.z, 0.0f};
                                } else if (plane == 1) {
                                    point_a = {center.x + cos_a * snapshot->collider.radius, center.y, center.z + sin_a * snapshot->collider.radius, 0.0f};
                                    point_b = {center.x + cos_b * snapshot->collider.radius, center.y, center.z + sin_b * snapshot->collider.radius, 0.0f};
                                } else {
                                    point_a = {center.x, center.y + cos_a * snapshot->collider.radius, center.z + sin_a * snapshot->collider.radius, 0.0f};
                                    point_b = {center.x, center.y + cos_b * snapshot->collider.radius, center.z + sin_b * snapshot->collider.radius, 0.0f};
                                }
                                draw_segment(point_a, point_b, IM_COL32(255, 176, 92, 224), 2.0f);
                            }
                        };
                        draw_ring(0);
                        draw_ring(1);
                        draw_ring(2);
                    } else {
                        const float min_x = snapshot->collider.center_x - snapshot->collider.half_x;
                        const float min_y = snapshot->collider.center_y - snapshot->collider.half_y;
                        const float min_z = snapshot->collider.center_z - snapshot->collider.half_z;
                        const float max_cx = snapshot->collider.center_x + snapshot->collider.half_x;
                        const float max_cy = snapshot->collider.center_y + snapshot->collider.half_y;
                        const float max_cz = snapshot->collider.center_z + snapshot->collider.half_z;
                        const std::array collider_corners{
                            vk::math::vec3{min_x, min_y, min_z, 0.0f},
                            vk::math::vec3{max_cx, min_y, min_z, 0.0f},
                            vk::math::vec3{max_cx, max_cy, min_z, 0.0f},
                            vk::math::vec3{min_x, max_cy, min_z, 0.0f},
                            vk::math::vec3{min_x, min_y, max_cz, 0.0f},
                            vk::math::vec3{max_cx, min_y, max_cz, 0.0f},
                            vk::math::vec3{max_cx, max_cy, max_cz, 0.0f},
                            vk::math::vec3{min_x, max_cy, max_cz, 0.0f},
                        };
                        draw_box(collider_corners, IM_COL32(255, 176, 92, 224), 2.0f);
                    }
                }

                if (settings_.show_velocity_plane && snapshot->velocity.data != nullptr) {
                    const float slice_position = std::clamp(settings_.slice_position, 0.0f, 1.0f);
                    const int seed_count = (std::max)(settings_.velocity_grid, 2);
                    const int step_count = (std::max)(settings_.velocity_steps, 1);
                    const float step_scale = settings_.velocity_step * snapshot->grid.cell_size;
                    std::array<vk::math::vec3, 4> plane_corners{};
                    if (resolved_plane == PlaneAxis::XY) {
                        const float z = slice_position * max_z;
                        plane_corners = {
                            vk::math::vec3{0.0f, 0.0f, z, 0.0f},
                            vk::math::vec3{max_x, 0.0f, z, 0.0f},
                            vk::math::vec3{max_x, max_y, z, 0.0f},
                            vk::math::vec3{0.0f, max_y, z, 0.0f},
                        };
                    } else if (resolved_plane == PlaneAxis::XZ) {
                        const float y = slice_position * max_y;
                        plane_corners = {
                            vk::math::vec3{0.0f, y, 0.0f, 0.0f},
                            vk::math::vec3{max_x, y, 0.0f, 0.0f},
                            vk::math::vec3{max_x, y, max_z, 0.0f},
                            vk::math::vec3{0.0f, y, max_z, 0.0f},
                        };
                    } else {
                        const float x = slice_position * max_x;
                        plane_corners = {
                            vk::math::vec3{x, 0.0f, 0.0f, 0.0f},
                            vk::math::vec3{x, max_y, 0.0f, 0.0f},
                            vk::math::vec3{x, max_y, max_z, 0.0f},
                            vk::math::vec3{x, 0.0f, max_z, 0.0f},
                        };
                    }
                    draw_segment(plane_corners[0], plane_corners[1], IM_COL32(112, 220, 255, 120), 1.0f);
                    draw_segment(plane_corners[1], plane_corners[2], IM_COL32(112, 220, 255, 120), 1.0f);
                    draw_segment(plane_corners[2], plane_corners[3], IM_COL32(112, 220, 255, 120), 1.0f);
                    draw_segment(plane_corners[3], plane_corners[0], IM_COL32(112, 220, 255, 120), 1.0f);

                    for (int j = 0; j < seed_count; ++j) {
                        for (int i = 0; i < seed_count; ++i) {
                            const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(seed_count);
                            const float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(seed_count);
                            vk::math::vec3 pos{};
                            if (resolved_plane == PlaneAxis::XY) pos = {u * max_x, v * max_y, slice_position * max_z, 0.0f};
                            if (resolved_plane == PlaneAxis::XZ) pos = {u * max_x, slice_position * max_y, v * max_z, 0.0f};
                            if (resolved_plane == PlaneAxis::YZ) pos = {slice_position * max_x, u * max_y, v * max_z, 0.0f};
                            for (int step = 0; step < step_count; ++step) {
                                const auto velocity = sample_velocity(pos.x, pos.y, pos.z);
                                vk::math::vec3 plane_velocity{};
                                if (resolved_plane == PlaneAxis::XY) plane_velocity = {velocity.x, velocity.y, 0.0f, 0.0f};
                                if (resolved_plane == PlaneAxis::XZ) plane_velocity = {velocity.x, 0.0f, velocity.z, 0.0f};
                                if (resolved_plane == PlaneAxis::YZ) plane_velocity = {0.0f, velocity.y, velocity.z, 0.0f};
                                const float speed = std::sqrt(plane_velocity.x * plane_velocity.x + plane_velocity.y * plane_velocity.y + plane_velocity.z * plane_velocity.z);
                                if (speed < settings_.velocity_min_speed) break;
                                const float inv_speed = 1.0f / speed;
                                vk::math::vec3 next{
                                    pos.x + plane_velocity.x * inv_speed * step_scale,
                                    pos.y + plane_velocity.y * inv_speed * step_scale,
                                    pos.z + plane_velocity.z * inv_speed * step_scale,
                                    0.0f,
                                };
                                next.x = std::clamp(next.x, 0.0f, max_x);
                                next.y = std::clamp(next.y, 0.0f, max_y);
                                next.z = std::clamp(next.z, 0.0f, max_z);
                                if (resolved_plane == PlaneAxis::XY) next.z = pos.z;
                                if (resolved_plane == PlaneAxis::XZ) next.y = pos.y;
                                if (resolved_plane == PlaneAxis::YZ) next.x = pos.x;
                                const float speed_t = std::clamp(speed / ((std::max)(settings_.velocity_min_speed, 1.0e-4f) * 8.0f), 0.0f, 1.0f);
                                const ImU32 color = IM_COL32(
                                    static_cast<int>(std::lerp(72.0f, 255.0f, speed_t)),
                                    static_cast<int>(std::lerp(196.0f, 212.0f, speed_t)),
                                    static_cast<int>(std::lerp(255.0f, 96.0f, speed_t)),
                                    static_cast<int>(std::lerp(112.0f, 224.0f, speed_t))
                                );
                                draw_segment(pos, next, color, settings_.velocity_thickness);
                                pos = next;
                            }
                        }
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
            .srcStageMask = PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = PipelineStageFlagBits2::eBottomOfPipe,
            .oldLayout = ImageLayout::eColorAttachmentOptimal,
            .newLayout = ImageLayout::ePresentSrcKHR,
            .image = sc_.images[image_index],
            .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_present_barrier,
        });
        frames_.swapchain_image_layout[image_index] = ImageLayout::ePresentSrcKHR;

        std::array<SemaphoreSubmitInfo, 1> volume_waits{};
        std::span<const SemaphoreSubmitInfo> extra_waits{};
        if (snapshot && snapshot->field.timeline_semaphore) {
            volume_waits[0] = SemaphoreSubmitInfo{
                .semaphore = snapshot->field.timeline_semaphore,
                .value = snapshot->field.ready_generation,
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

    void VisualizationApp::frame_content(const VisualizationSnapshotView& snapshot) {
        auto update_camera_config = [&](const vk::camera::Projection projection, const float ortho_height) {
            auto camera_config = camera_.config();
            camera_config.projection = projection;
            camera_config.ortho_height = ortho_height;
            camera_.set_config(camera_config);
        };

        const auto resolved_view = settings_.view_mode;
        const float center_x = snapshot.grid.extent_x() * 0.5f;
        const float center_y = snapshot.grid.extent_y() * 0.5f;
        const float center_z = snapshot.grid.extent_z() * 0.5f;
        vk::camera::CameraState camera_state = camera_.state();
        camera_state.mode = vk::camera::Mode::Orbit;
        camera_state.orbit.target = {center_x, center_y, center_z, 0.0f};
        camera_state.orbit.distance = snapshot.grid.max_extent() * 1.5f;

        if (resolved_view == ViewMode::Plane) {
            update_camera_config(vk::camera::Projection::Orthographic, snapshot.grid.max_extent() * 1.1f);
            const PlaneAxis plane_axis = settings_.plane_axis;
            if (plane_axis == PlaneAxis::XY) {
                camera_state.orbit.yaw_rad = 0.0f;
                camera_state.orbit.pitch_rad = 0.0f;
            } else if (plane_axis == PlaneAxis::XZ) {
                camera_state.orbit.yaw_rad = 0.0f;
                camera_state.orbit.pitch_rad = -1.55334303427f;
            } else {
                camera_state.orbit.yaw_rad = 1.57079632679f;
                camera_state.orbit.pitch_rad = 0.0f;
            }
        } else {
            update_camera_config(vk::camera::Projection::Perspective, snapshot.grid.max_extent());
            camera_state.orbit.distance = snapshot.grid.max_extent() * 1.15f;
            camera_state.orbit.yaw_rad = 0.0f;
            camera_state.orbit.pitch_rad = 0.0f;
        }
        camera_.set_state(camera_state);
    }

    VisualizationSettings& VisualizationApp::settings() {
        return settings_;
    }

    const VisualizationSettings& VisualizationApp::settings() const {
        return settings_;
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
            .descriptorPool = *field_descriptor_pool_,
            .descriptorSetCount = count,
            .pSetLayouts = field_layouts.data(),
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

    void VisualizationApp::collect_camera_input(const float dt_seconds) {
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

} // namespace app
