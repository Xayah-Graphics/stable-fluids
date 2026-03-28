module;

#include "stable-fluids-3d.h"

#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include <cuda_runtime.h>
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
        collect_camera_input(dt_seconds);
    }

    void VisualizationApp::draw_visualization_ui(VisualizationSettings& settings, const std::optional<VisualizationSnapshotView>& snapshot) {
        auto smoke_capable = [&](const VisualizationSnapshotView& view) {
            return view.field.semantic == FieldSemantic::DyeColor && view.field.component_count == 4;
        };

        bool reframe_requested = false;
        if (snapshot && !smoke_capable(*snapshot) && settings.render_mode == RenderMode::Smoke) settings.render_mode = RenderMode::Scalar;

        ImGui::Begin("Visualization");
        if (snapshot) {
            ImGui::Text("Field: %.*s", static_cast<int>(snapshot->field.label.size()), snapshot->field.label.data());
            ImGui::Text("Grid: %u x %u x %u", snapshot->grid.nx, snapshot->grid.ny, snapshot->grid.nz);
            ImGui::Text("Generation: %llu", static_cast<unsigned long long>(snapshot->field.ready_generation));

            int view_mode = static_cast<int>(settings.view_mode);
            const char* labels[] = {"Plane", "Volume"};
            if (ImGui::Combo("View", &view_mode, labels, 2)) {
                settings.view_mode = static_cast<ViewMode>(view_mode);
                reframe_requested = true;
            }

            const auto resolved_view = settings.view_mode;
            if (resolved_view == ViewMode::Plane) {
                int plane_axis = static_cast<int>(settings.plane_axis);
                const char* axis_labels[] = {"XY", "XZ", "YZ"};
                if (ImGui::Combo("Plane Axis", &plane_axis, axis_labels, 3)) {
                    settings.plane_axis = static_cast<PlaneAxis>(plane_axis);
                    reframe_requested = true;
                }
                ImGui::SliderFloat("Slice", &settings.slice_position, 0.0f, 1.0f, "%.3f");
            }

            if (smoke_capable(*snapshot)) {
                int render_mode = static_cast<int>(settings.render_mode);
                const char* labels[] = {"Smoke", "Scalar"};
                if (ImGui::Combo("Field Mode", &render_mode, labels, 2)) settings.render_mode = static_cast<RenderMode>(render_mode);
            } else {
                ImGui::TextUnformatted("Field Mode: Scalar");
            }

            ImGui::SliderFloat("Density Scale", &settings.density_scale, 0.05f, 8.0f, "%.2f");
            if (resolved_view == ViewMode::Volume) ImGui::SliderInt("March Steps", &settings.march_steps, 24, 224);
            if (settings.render_mode == RenderMode::Smoke) {
                ImGui::SliderFloat("Absorption", &settings.absorption, 0.05f, 8.0f, "%.2f");
            } else {
                ImGui::SliderFloat("Value Min", &settings.scalar_min, -200.0f, 200.0f, "%.3f");
                ImGui::SliderFloat("Value Max", &settings.scalar_max, -200.0f, 200.0f, "%.3f");
                ImGui::SliderFloat("Opacity", &settings.scalar_opacity, 0.05f, 8.0f, "%.2f");
                ImGui::ColorEdit3("Low Color", &settings.scalar_low_r);
                ImGui::ColorEdit3("High Color", &settings.scalar_high_r);
            }

            ImGui::Separator();
            ImGui::Checkbox("Show Bounds", &settings.show_bounds);
            ImGui::Checkbox("Show Collider", &settings.show_collider);
            ImGui::Checkbox("Show Velocity Plane", &settings.show_velocity_plane);
            if (settings.show_velocity_plane) {
                ImGui::SliderInt("Vector Grid", &settings.velocity_grid, 4, 48);
                ImGui::SliderInt("Vector Steps", &settings.velocity_steps, 4, 96);
                ImGui::SliderFloat("Vector Step", &settings.velocity_step, 0.10f, 3.0f, "%.2f");
                ImGui::SliderFloat("Min Speed", &settings.velocity_min_speed, 0.01f, 4.0f, "%.3f");
                ImGui::SliderFloat("Line Width", &settings.velocity_thickness, 0.5f, 4.0f, "%.2f");
            }
        } else {
            ImGui::TextUnformatted("Field: None");
        }
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
        PlaneAxis resolved_plane = settings.plane_axis;
        if (snapshot) {
            resolved_view = settings.view_mode;
            const auto& matrices = camera_.matrices();
            const auto& camera_config = camera_.config();
            const float aspect = static_cast<float>(sc_.extent.width) / static_cast<float>((std::max)(sc_.extent.height, 1u));
            const float half_fov_tan = std::tan(camera_config.fov_y_rad * 0.5f);
            const float slice_position = std::clamp(settings.slice_position, 0.0f, 1.0f);
            FieldPushConstants push{};
            push.eye        = {matrices.eye.x, matrices.eye.y, matrices.eye.z, 1.0f};
            push.right      = {matrices.right.x, matrices.right.y, matrices.right.z, 0.0f};
            push.up         = {matrices.up.x, matrices.up.y, matrices.up.z, 0.0f};
            push.forward    = {matrices.forward.x, matrices.forward.y, matrices.forward.z, 0.0f};
            push.volume_min = {0.0f, 0.0f, 0.0f, 0.0f};
            push.volume_max = {snapshot->grid.extent_x(), snapshot->grid.extent_y(), snapshot->grid.extent_z(), 0.0f};
            push.color_a    = settings.render_mode == RenderMode::Smoke ? vk::math::vec4{} : vk::math::vec4{settings.scalar_low_r, settings.scalar_low_g, settings.scalar_low_b, 1.0f};
            push.color_b    = settings.render_mode == RenderMode::Smoke ? vk::math::vec4{} : vk::math::vec4{settings.scalar_high_r, settings.scalar_high_g, settings.scalar_high_b, 1.0f};
            push.params0    = {
                aspect,
                half_fov_tan,
                settings.density_scale,
                settings.render_mode == RenderMode::Smoke ? settings.absorption : settings.scalar_opacity,
            };
            push.params1 = {
                snapshot->grid.nx,
                snapshot->grid.ny,
                snapshot->grid.nz,
                static_cast<uint32_t>(settings.march_steps),
            };
            push.params2 = {
                static_cast<uint32_t>(settings.render_mode),
                snapshot->field.component_count,
                static_cast<uint32_t>(resolved_plane),
                static_cast<uint32_t>(camera_config.projection),
            };
            push.params3 = {
                settings.scalar_min,
                settings.scalar_max,
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

        if (snapshot && (settings.show_bounds || (settings.show_collider && snapshot->collider.enabled) || (settings.show_velocity_plane && snapshot->velocity.data != nullptr))) {
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
                if (settings.show_bounds) {
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

                if (settings.show_collider && snapshot->collider.enabled) {
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

                if (settings.show_velocity_plane && snapshot->velocity.data != nullptr) {
                    const float slice_position = std::clamp(settings.slice_position, 0.0f, 1.0f);
                    const int seed_count = (std::max)(settings.velocity_grid, 2);
                    const int step_count = (std::max)(settings.velocity_steps, 1);
                    const float step_scale = settings.velocity_step * snapshot->grid.cell_size;
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
                                if (speed < settings.velocity_min_speed) break;
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
                                const float speed_t = std::clamp(speed / ((std::max)(settings.velocity_min_speed, 1.0e-4f) * 8.0f), 0.0f, 1.0f);
                                const ImU32 color = IM_COL32(
                                    static_cast<int>(std::lerp(72.0f, 255.0f, speed_t)),
                                    static_cast<int>(std::lerp(196.0f, 212.0f, speed_t)),
                                    static_cast<int>(std::lerp(255.0f, 96.0f, speed_t)),
                                    static_cast<int>(std::lerp(112.0f, 224.0f, speed_t))
                                );
                                draw_segment(pos, next, color, settings.velocity_thickness);
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

    void VisualizationApp::frame_content(const VisualizationSettings& settings, const VisualizationSnapshotView& snapshot) {
        auto update_camera_config = [&](const vk::camera::Projection projection, const float ortho_height) {
            auto camera_config = camera_.config();
            camera_config.projection = projection;
            camera_config.ortho_height = ortho_height;
            camera_.set_config(camera_config);
        };

        const auto resolved_view = settings.view_mode;
        const float center_x = snapshot.grid.extent_x() * 0.5f;
        const float center_y = snapshot.grid.extent_y() * 0.5f;
        const float center_z = snapshot.grid.extent_z() * 0.5f;
        vk::camera::CameraState camera_state = camera_.state();
        camera_state.mode = vk::camera::Mode::Orbit;
        camera_state.orbit.target = {center_x, center_y, center_z, 0.0f};
        camera_state.orbit.distance = snapshot.grid.max_extent() * 1.5f;

        if (resolved_view == ViewMode::Plane) {
            update_camera_config(vk::camera::Projection::Orthographic, snapshot.grid.max_extent() * 1.1f);
            const PlaneAxis plane_axis = settings.plane_axis;
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

    namespace {

        constexpr uint32_t snapshot_slot_count = 4;

        constexpr std::array boundary_labels{
            "No-slip",
            "Free-slip",
            "Inflow",
            "Outflow",
        };

        constexpr std::array collider_type_labels{
            "Sphere",
            "Box",
        };

        constexpr std::array scene_preset_labels{
            "Dual Jet Collider",
            "Smoke Plume",
            "Custom",
        };

        constexpr std::array field_catalog_storage{
            FieldInfo{
                .id = FieldId::SmokeColor,
                .label = "Smoke Color",
                .component_count = 4,
                .semantic = FieldSemantic::DyeColor,
                .preset = {
                    .display_mode = FieldDisplayMode::Smoke,
                    .density_scale = 10.0f,
                    .absorption = 3.0f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.0f,
                    .scalar_opacity = 2.0f,
                    .scalar_low_r = 0.08f,
                    .scalar_low_g = 0.18f,
                    .scalar_low_b = 0.46f,
                    .scalar_high_r = 0.98f,
                    .scalar_high_g = 0.82f,
                    .scalar_high_b = 0.24f,
                },
            },
            FieldInfo{
                .id = FieldId::Density,
                .label = "Density",
                .component_count = 1,
                .semantic = FieldSemantic::Density,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 10.0f,
                    .absorption = 3.0f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.40f,
                    .scalar_opacity = 3.00f,
                    .scalar_low_r = 0.10f,
                    .scalar_low_g = 0.08f,
                    .scalar_low_b = 0.30f,
                    .scalar_high_r = 1.00f,
                    .scalar_high_g = 0.24f,
                    .scalar_high_b = 0.74f,
                },
            },
            FieldInfo{
                .id = FieldId::VelocityMagnitude,
                .label = "Velocity Magnitude",
                .component_count = 1,
                .semantic = FieldSemantic::VelocityMagnitude,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 10.0f,
                    .absorption = 3.0f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.20f,
                    .scalar_opacity = 1.80f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
            FieldInfo{
                .id = FieldId::SolidMask,
                .label = "Solid Mask",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = 0.0f,
                    .scalar_max = 1.0f,
                    .scalar_opacity = 3.2f,
                    .scalar_low_r = 0.05f,
                    .scalar_low_g = 0.06f,
                    .scalar_low_b = 0.07f,
                    .scalar_high_r = 0.94f,
                    .scalar_high_g = 0.92f,
                    .scalar_high_b = 0.88f,
                },
            },
            FieldInfo{
                .id = FieldId::Pressure,
                .label = "Pressure",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = -0.12f,
                    .scalar_max = 0.12f,
                    .scalar_opacity = 2.20f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
            FieldInfo{
                .id = FieldId::Divergence,
                .label = "Divergence",
                .component_count = 1,
                .semantic = FieldSemantic::GenericScalar,
                .preset = {
                    .display_mode = FieldDisplayMode::Scalar,
                    .density_scale = 1.0f,
                    .absorption = 1.20f,
                    .scalar_min = -120.0f,
                    .scalar_max = 120.0f,
                    .scalar_opacity = 1.80f,
                    .scalar_low_r = 0.06f,
                    .scalar_low_g = 0.10f,
                    .scalar_low_b = 0.42f,
                    .scalar_high_r = 0.18f,
                    .scalar_high_g = 0.88f,
                    .scalar_high_b = 1.00f,
                },
            },
        };

        struct CaptureRequest {
            GridShape grid{};
            uint32_t field_component_count = 1;
            FieldSemantic semantic         = FieldSemantic::GenericScalar;
            std::string_view label{};
            bool export_velocity_host      = false;
        };

        struct CaptureResources {
            void* field_cuda_ptr                       = nullptr;
            void* velocity_cuda_ptr                    = nullptr;
            float* velocity_host_ptr                   = nullptr;
            cudaExternalSemaphore_t external_semaphore = nullptr;
            uint64_t ready_generation                  = 0;
        };

        void check_cuda(const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }

        void check_stable(const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        }

        uint64_t field_bytes_for(const CaptureRequest& request) {
            const uint64_t nx = request.grid.nx;
            const uint64_t ny = request.grid.ny;
            const uint64_t nz = static_cast<uint64_t>((std::max)(request.grid.nz, 1u));
            return nx * ny * nz * static_cast<uint64_t>((std::max)(request.field_component_count, 1u)) * sizeof(float);
        }

        uint64_t velocity_bytes_for(const CaptureRequest& request) {
            if (!request.export_velocity_host) return 0;
            const uint64_t nx = request.grid.nx;
            const uint64_t ny = request.grid.ny;
            const uint64_t nz = static_cast<uint64_t>((std::max)(request.grid.nz, 1u));
            return nx * ny * nz * 3ull * sizeof(float);
        }

        StableFluidsGridDesc physics_grid_desc(const AppData& data) {
            StableFluidsGridDesc desc{};
            check_stable(stable_fluids_get_grid_desc_cuda(data.physics.context, &desc), "stable_fluids_get_grid_desc_cuda");
            return desc;
        }

        ColliderOverlay collider_overlay(const AppState& state) {
            return ColliderOverlay{
                .enabled = state.physics.scene.collider.enabled,
                .type = static_cast<uint32_t>(state.physics.scene.collider.type),
                .center_x = state.physics.scene.collider.center_x,
                .center_y = state.physics.scene.collider.center_y,
                .center_z = state.physics.scene.collider.center_z,
                .radius = state.physics.scene.collider.radius,
                .half_x = state.physics.scene.collider.half_extent_x,
                .half_y = state.physics.scene.collider.half_extent_y,
                .half_z = state.physics.scene.collider.half_extent_z,
            };
        }

        void export_field(const AppState& state, const AppData& data, const FieldId field, void* destination) {
            if (field == FieldId::SmokeColor) {
                const StableFluidsExportDesc desc{
                    .kind = STABLE_FLUIDS_EXPORT_ALPHA_RGB_RGBA,
                    .field_a = data.physics.density_field,
                    .field_b = data.physics.dye_field,
                    .component_offset = 0,
                    .component_count = 0,
                };
                check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
                return;
            }
            if (field == FieldId::Density) {
                const StableFluidsExportDesc desc{
                    .kind = STABLE_FLUIDS_EXPORT_FIELD_COMPONENTS,
                    .field_a = data.physics.density_field,
                    .field_b = 0,
                    .component_offset = 0,
                    .component_count = 1,
                };
                check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
                return;
            }
            if (field == FieldId::VelocityMagnitude) {
                const StableFluidsExportDesc desc{
                    .kind = STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE,
                    .field_a = 0,
                    .field_b = 0,
                    .component_offset = 0,
                    .component_count = 0,
                };
                check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
                return;
            }
            if (field == FieldId::SolidMask) {
                const StableFluidsExportDesc desc{
                    .kind = STABLE_FLUIDS_EXPORT_SOLID_MASK,
                    .field_a = 0,
                    .field_b = 0,
                    .component_offset = 0,
                    .component_count = 0,
                };
                check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
                return;
            }
            if (field == FieldId::Pressure) {
                const StableFluidsExportDesc desc{
                    .kind = STABLE_FLUIDS_EXPORT_PRESSURE,
                    .field_a = 0,
                    .field_b = 0,
                    .component_offset = 0,
                    .component_count = 0,
                };
                check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
                return;
            }
            const StableFluidsExportDesc desc{
                .kind = STABLE_FLUIDS_EXPORT_DIVERGENCE,
                .field_a = 0,
                .field_b = 0,
                .component_offset = 0,
                .component_count = 0,
            };
            check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
        }

        void export_velocity(const AppData& data, void* destination) {
            const StableFluidsExportDesc desc{
                .kind = STABLE_FLUIDS_EXPORT_VELOCITY,
                .field_a = 0,
                .field_b = 0,
                .component_offset = 0,
                .component_count = 0,
            };
            check_stable(stable_fluids_export_cuda(data.physics.context, &desc, destination), "stable_fluids_export_cuda");
        }

        CaptureRequest make_capture_request(AppState& state, const AppData& data) {
            const auto grid = physics_grid_desc(data);
            const auto& field = current_field_info(state);
            return CaptureRequest{
                .grid = {
                    .nx = static_cast<uint32_t>(grid.nx),
                    .ny = static_cast<uint32_t>(grid.ny),
                    .nz = static_cast<uint32_t>((std::max)(grid.nz, 1)),
                    .cell_size = grid.cell_size,
                },
                .field_component_count = field.component_count,
                .semantic = field.semantic,
                .label = field.label,
                .export_velocity_host = state.ui.render.show_velocity_plane,
            };
        }

        bool capture_matches_request(const AppData& data, const CaptureRequest& request) {
            return !data.capture.slots.empty()
                && data.capture.request_grid.nx == request.grid.nx
                && data.capture.request_grid.ny == request.grid.ny
                && data.capture.request_grid.nz == request.grid.nz
                && data.capture.request_grid.cell_size == request.grid.cell_size
                && data.capture.request_field_component_count == request.field_component_count
                && data.capture.request_export_velocity_host == request.export_velocity_host;
        }

        int find_available_capture_slot(const AppData& data, const uint32_t frames_in_flight) {
            for (uint32_t slot_index = 0; slot_index < data.capture.slots.size(); ++slot_index) {
                const auto& slot = data.capture.slots[slot_index];
                if (static_cast<int>(slot_index) == data.capture.active_slot) continue;
                if (slot.ready_generation != 0 && data.capture.submit_serial < slot.last_used_submit_serial + frames_in_flight + 1) continue;
                return static_cast<int>(slot_index);
            }
            return -1;
        }

        CaptureResources begin_capture(AppData& data, const int slot_index) {
            auto& slot = data.capture.slots.at(static_cast<size_t>(slot_index));
            return CaptureResources{
                .field_cuda_ptr = slot.field_cuda_ptr,
                .velocity_cuda_ptr = slot.velocity_cuda_ptr,
                .velocity_host_ptr = slot.velocity_host.empty() ? nullptr : slot.velocity_host.data(),
                .external_semaphore = slot.external_semaphore,
                .ready_generation = data.capture.generation + 1,
            };
        }

        void complete_capture(AppData& data, const int slot_index, const CaptureRequest& request, const double capture_ms) {
            auto& slot = data.capture.slots.at(static_cast<size_t>(slot_index));
            slot.ready_generation = data.capture.generation + 1;
            slot.grid = request.grid;
            slot.field_component_count = request.field_component_count;
            slot.semantic = request.semantic;
            slot.label = request.label;
            slot.has_velocity_host = request.export_velocity_host;
            data.capture.generation = slot.ready_generation;
            data.capture.active_slot = slot_index;
            data.capture.steps_since_snapshot = 0;
            data.capture.stats.last_snapshot_ms = capture_ms;
            ++data.capture.stats.snapshot_count;
            data.capture.stats.average_snapshot_ms += (capture_ms - data.capture.stats.average_snapshot_ms) / static_cast<double>(data.capture.stats.snapshot_count);
        }

        void destroy_capture_storage(AppData& data) {
            for (auto& slot : data.capture.slots) {
                if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
                if (slot.velocity_cuda_ptr != nullptr) cudaFree(slot.velocity_cuda_ptr);
                if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
                if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
                slot = {};
            }
            data.capture = {};
        }

        void upload_scene(const AppState& state, AppData& data) {
            StableFluidsSceneDesc scene_desc{
                .colliders = nullptr,
                .collider_count = 0,
            };
            StableFluidsColliderDesc collider{
                .collider_type = static_cast<uint32_t>(state.physics.scene.collider.type == ColliderType::Sphere ? STABLE_FLUIDS_COLLIDER_SPHERE : STABLE_FLUIDS_COLLIDER_BOX),
                .velocity_boundary_type = state.physics.scene.collider.boundary,
                .center_x = state.physics.scene.collider.center_x,
                .center_y = state.physics.scene.collider.center_y,
                .center_z = state.physics.scene.collider.center_z,
                .radius = state.physics.scene.collider.radius,
                .half_extent_x = state.physics.scene.collider.half_extent_x,
                .half_extent_y = state.physics.scene.collider.half_extent_y,
                .half_extent_z = state.physics.scene.collider.half_extent_z,
                .linear_velocity_x = state.physics.scene.collider.velocity_x,
                .linear_velocity_y = state.physics.scene.collider.velocity_y,
                .linear_velocity_z = state.physics.scene.collider.velocity_z,
            };
            if (state.physics.scene.collider.enabled) {
                scene_desc.colliders = &collider;
                scene_desc.collider_count = 1;
            }
            check_stable(stable_fluids_update_scene_cuda(data.physics.context, &scene_desc), "stable_fluids_update_scene_cuda");
        }

    } // namespace

    std::span<const FieldInfo> field_catalog() {
        return field_catalog_storage;
    }

    const FieldInfo& current_field_info(AppState& state) {
        auto& selected = state.physics.selected_field;
        selected = std::clamp(selected, 0, static_cast<int>(field_catalog_storage.size()) - 1);
        return field_catalog_storage[static_cast<size_t>(selected)];
    }

    void apply_scene_preset(AppState& state, const ScenePreset preset) {
        const int selected_field = state.physics.selected_field;
        state.physics = {};
        state.physics.preset = preset;
        state.physics.selected_field = selected_field;
        state.physics.scene.collider = ColliderSettings{
            .enabled = true,
            .type = ColliderType::Sphere,
            .center_x = 0.50f,
            .center_y = 0.40f,
            .center_z = 0.50f,
            .radius = 0.15f,
            .half_extent_x = 0.10f,
            .half_extent_y = 0.08f,
            .half_extent_z = 0.10f,
            .velocity_x = 0.0f,
            .velocity_y = 0.0f,
            .velocity_z = 0.0f,
            .boundary = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP),
        };

        if (preset == ScenePreset::SmokePlume) {
            state.physics.solver.backend.pressure_iterations = 120;
            state.physics.solver.backend.domain_boundary = {
                .x_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                .x_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                .y_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_NO_SLIP), .velocity = 0.0f, },
                .y_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_OUTFLOW), .velocity = 0.0f, },
                .z_min = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
                .z_max = { .type = static_cast<uint32_t>(STABLE_FLUIDS_VELOCITY_BOUNDARY_FREE_SLIP), .velocity = 0.0f, },
            };
            state.physics.solver.density_diffusion = 0.00001f;
            state.physics.solver.dye_diffusion = 0.000008f;
            state.physics.solver.buoyancy_beta = 0.65f;
            state.physics.scene.collider.enabled = false;
            state.physics.emitters.a = SourceEmitterSettings{
                .enabled = true,
                .center_x = 0.50f,
                .center_y = 0.04f,
                .center_z = 0.50f,
                .speed = 0.03f,
                .radius = 0.035f,
                .density_amount = 1.40f,
                .dye_amount = 1.10f,
                .color_r = 0.95f,
                .color_g = 0.92f,
                .color_b = 0.86f,
            };
            state.physics.emitters.b = {};
            return;
        }

        state.physics.solver.density_diffusion = 0.00003f;
        state.physics.solver.dye_diffusion = 0.000015f;
        state.physics.solver.buoyancy_beta = 0.12f;
        state.physics.emitters.a = SourceEmitterSettings{
            .enabled = true,
            .center_x = 0.1f,
            .center_y = 0.1f,
            .center_z = 0.5f,
            .direction_x = 1.0f,
            .direction_y = 0.0f,
            .direction_z = 0.0f,
            .speed = 0.62f,
            .radius = 0.045f,
            .density_amount = 0.80f,
            .dye_amount = 0.95f,
            .color_r = 1.00f,
            .color_g = 0.20f,
            .color_b = 0.72f,
        };
        state.physics.emitters.b = SourceEmitterSettings{
            .enabled = true,
            .center_x = 0.9f,
            .center_y = 0.1f,
            .center_z = 0.5f,
            .direction_x = -1.0f,
            .direction_y = 0.0f,
            .direction_z = 0.0f,
            .speed = 0.62f,
            .radius = 0.045f,
            .density_amount = 0.80f,
            .dye_amount = 0.95f,
            .color_r = 0.12f,
            .color_g = 0.38f,
            .color_b = 1.00f,
        };
    }

    void apply_field_visual_preset(AppState& state) {
        const auto& preset = current_field_info(state).preset;
        auto& render = state.ui.render;
        render.render_mode = preset.display_mode == FieldDisplayMode::Smoke ? RenderMode::Smoke : RenderMode::Scalar;
        render.density_scale = preset.density_scale;
        render.absorption = preset.absorption;
        render.scalar_min = preset.scalar_min;
        render.scalar_max = preset.scalar_max;
        render.scalar_opacity = preset.scalar_opacity;
        render.scalar_low_r = preset.scalar_low_r;
        render.scalar_low_g = preset.scalar_low_g;
        render.scalar_low_b = preset.scalar_low_b;
        render.scalar_high_r = preset.scalar_high_r;
        render.scalar_high_g = preset.scalar_high_g;
        render.scalar_high_b = preset.scalar_high_b;
    }

    void create_runtime_data(AppData& data) {
        destroy_runtime_data(data);
        check_cuda(cudaStreamCreateWithFlags(&data.physics.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    void destroy_runtime_data(AppData& data) {
        destroy_capture_storage(data);
        if (data.physics.context != nullptr) stable_fluids_destroy_context_cuda(data.physics.context);
        if (data.physics.stream != nullptr) cudaStreamDestroy(data.physics.stream);
        data.physics = {};
    }

    void check_interop_support(const VisualizationApp& renderer) {
        const auto timeline_features = renderer.vk_context().physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
        if (!timeline_features.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore) throw std::runtime_error("stable-fluids visualizer requires Vulkan timeline semaphore support");
        int cuda_device_index = 0;
        check_cuda(cudaGetDevice(&cuda_device_index), "cudaGetDevice");
        int timeline_supported = 0;
        check_cuda(cudaDeviceGetAttribute(&timeline_supported, cudaDevAttrTimelineSemaphoreInteropSupported, cuda_device_index), "cudaDeviceGetAttribute");
        if (timeline_supported == 0) throw std::runtime_error("CUDA timeline semaphore interop is required");
    }

    void rebuild_physics(AppState& state, AppData& data) {
        const float extent_x = static_cast<float>(state.physics.solver.backend.nx) * state.physics.solver.backend.cell_size;
        const float extent_y = static_cast<float>(state.physics.solver.backend.ny) * state.physics.solver.backend.cell_size;
        const float extent_z = static_cast<float>((std::max)(state.physics.solver.backend.nz, 1)) * state.physics.solver.backend.cell_size;
        const float min_extent = (std::min)({extent_x, extent_y, extent_z});

        auto clamp_emitter = [&](SourceEmitterSettings& emitter) {
            emitter.center_x = std::clamp(emitter.center_x, 0.0f, extent_x);
            emitter.center_y = std::clamp(emitter.center_y, 0.0f, extent_y);
            emitter.center_z = std::clamp(emitter.center_z, 0.0f, extent_z);
            emitter.radius = std::clamp(emitter.radius, state.physics.solver.backend.cell_size, min_extent * 0.25f);
            emitter.speed = std::max(emitter.speed, 0.0f);
        };
        clamp_emitter(state.physics.emitters.a);
        clamp_emitter(state.physics.emitters.b);
        auto& collider = state.physics.scene.collider;
        collider.center_x = std::clamp(collider.center_x, 0.0f, extent_x);
        collider.center_y = std::clamp(collider.center_y, 0.0f, extent_y);
        collider.center_z = std::clamp(collider.center_z, 0.0f, extent_z);
        collider.radius = std::clamp(collider.radius, state.physics.solver.backend.cell_size, min_extent * 0.45f);
        collider.half_extent_x = std::clamp(collider.half_extent_x, state.physics.solver.backend.cell_size, extent_x * 0.45f);
        collider.half_extent_y = std::clamp(collider.half_extent_y, state.physics.solver.backend.cell_size, extent_y * 0.45f);
        collider.half_extent_z = std::clamp(collider.half_extent_z, state.physics.solver.backend.cell_size, extent_z * 0.45f);

        if (data.physics.context != nullptr) {
            check_stable(stable_fluids_destroy_context_cuda(data.physics.context), "stable_fluids_destroy_context_cuda");
            data.physics.context = nullptr;
        }
        data.physics.density_field = 0;
        data.physics.dye_field = 0;

        std::array fields{
            StableFluidsFieldCreateDesc{
                .name = "density",
                .component_count = 1,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = state.physics.solver.density_diffusion,
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
            StableFluidsFieldCreateDesc{
                .name = "dye",
                .component_count = 3,
                .flags = STABLE_FLUIDS_FIELD_ADVECT | STABLE_FLUIDS_FIELD_DIFFUSE,
                .diffusion = state.physics.solver.dye_diffusion,
                .extension_mode = static_cast<uint32_t>(STABLE_FLUIDS_FIELD_EXTENSION_STREAK),
                .default_value_0 = 0.0f,
                .default_value_1 = 0.0f,
                .default_value_2 = 0.0f,
                .default_value_3 = 0.0f,
            },
        };
        const float buoyancy_weight = -state.physics.solver.gravity_y * state.physics.solver.buoyancy_beta;
        std::array buoyancy_terms{
            StableFluidsBuoyancyDesc{
                .field_index = 0,
                .weight = buoyancy_weight,
                .ambient = state.physics.solver.ambient_density,
            },
        };
        const uint32_t buoyancy_term_count = std::abs(buoyancy_weight) > 1.0e-6f ? static_cast<uint32_t>(buoyancy_terms.size()) : 0u;
        std::array<StableFluidsFieldHandle, 2> field_handles{};
        StableFluidsContextCreateDesc create_desc{
            .config = state.physics.solver.backend,
            .stream = data.physics.stream,
            .fields = fields.data(),
            .field_count = static_cast<uint32_t>(fields.size()),
            .buoyancy_terms = buoyancy_term_count > 0 ? buoyancy_terms.data() : nullptr,
            .buoyancy_term_count = buoyancy_term_count,
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &data.physics.context, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda");
        data.physics.density_field = field_handles[0];
        data.physics.dye_field = field_handles[1];
        data.physics.stats = {};

        upload_scene(state, data);
    }

    void step_physics(const AppState& state, AppData& data, const int sim_steps) {
        std::array<StableFluidsVelocitySourceDesc, 2> velocity_sources{};
        std::array<StableFluidsFieldSourceDesc, 4> field_sources{};
        uint32_t velocity_source_count = 0;
        uint32_t field_source_count = 0;
        const auto append_emitter = [&](const SourceEmitterSettings& emitter) {
            if (!emitter.enabled) return;
            const float dir_len = std::sqrt(emitter.direction_x * emitter.direction_x + emitter.direction_y * emitter.direction_y + emitter.direction_z * emitter.direction_z);
            const float inv_len = dir_len > 1.0e-5f ? 1.0f / dir_len : 1.0f;
            const float dir_x = dir_len > 1.0e-5f ? emitter.direction_x * inv_len : 0.0f;
            const float dir_y = dir_len > 1.0e-5f ? emitter.direction_y * inv_len : 1.0f;
            const float dir_z = dir_len > 1.0e-5f ? emitter.direction_z * inv_len : 0.0f;
            velocity_sources[velocity_source_count++] = {
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .velocity_x = dir_x * emitter.speed,
                .velocity_y = dir_y * emitter.speed,
                .velocity_z = dir_z * emitter.speed,
            };
            field_sources[field_source_count++] = {
                .field = data.physics.density_field,
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .value_0 = emitter.density_amount,
                .value_1 = 0.0f,
                .value_2 = 0.0f,
                .value_3 = 0.0f,
            };
            field_sources[field_source_count++] = {
                .field = data.physics.dye_field,
                .center_x = emitter.center_x,
                .center_y = emitter.center_y,
                .center_z = emitter.center_z,
                .radius = emitter.radius,
                .value_0 = emitter.dye_amount * emitter.color_r,
                .value_1 = emitter.dye_amount * emitter.color_g,
                .value_2 = emitter.dye_amount * emitter.color_b,
                .value_3 = 0.0f,
            };
        };
        append_emitter(state.physics.emitters.a);
        append_emitter(state.physics.emitters.b);

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            StableFluidsStepDesc step_desc{
                .velocity_sources = state.physics.emit_source ? velocity_sources.data() : nullptr,
                .velocity_source_count = state.physics.emit_source ? velocity_source_count : 0u,
                .field_sources = state.physics.emit_source ? field_sources.data() : nullptr,
                .field_source_count = state.physics.emit_source ? field_source_count : 0u,
            };
            const auto begin = std::chrono::steady_clock::now();
            check_stable(stable_fluids_step_cuda(data.physics.context, &step_desc), "stable_fluids_step_cuda");
            const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            data.physics.stats.last_step_call_ms = elapsed_ms;
            ++data.physics.stats.step_count;
            data.physics.stats.average_step_call_ms += (elapsed_ms - data.physics.stats.average_step_call_ms) / static_cast<double>(data.physics.stats.step_count);
        }
        if (sim_steps <= 0) return;
        StableFluidsProjectionMetrics metrics{};
        check_stable(stable_fluids_get_projection_metrics_cuda(data.physics.context, &metrics), "stable_fluids_get_projection_metrics_cuda");
        data.physics.stats.projection_max_abs_divergence = metrics.max_abs_divergence;
        data.physics.stats.projection_rms_divergence = metrics.rms_divergence;
    }

    bool sync_capture_storage(AppState& state, AppData& data, VisualizationApp& renderer) {
        const auto request = make_capture_request(state, data);
        if (capture_matches_request(data, request)) return false;
        renderer.vk_context().device.waitIdle();
        destroy_capture_storage(data);
        data.capture.request_grid = request.grid;
        data.capture.request_field_component_count = request.field_component_count;
        data.capture.request_export_velocity_host = request.export_velocity_host;
        data.capture.field_bytes = field_bytes_for(request);
        data.capture.velocity_bytes = velocity_bytes_for(request);

        auto descriptor_sets = renderer.allocate_field_descriptor_sets(snapshot_slot_count);
        data.capture.slots.reserve(descriptor_sets.size());
        for (size_t slot_index = 0; slot_index < descriptor_sets.size(); ++slot_index) {
            auto& slot = data.capture.slots.emplace_back();
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
                .initialValue = 0,
            };
            vk::ExportSemaphoreCreateInfo export_semaphore_ci{
                .pNext = &timeline_semaphore_ci,
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
                .pNext = &external_buffer_ci,
                .size = data.capture.field_bytes,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                .sharingMode = vk::SharingMode::eExclusive,
            };
            slot.buffer = vk::raii::Buffer{renderer.vk_context().device, buffer_ci};
            const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
            vk::ExportMemoryAllocateInfo export_memory_ci{
                .handleTypes = memory_handle_type,
            };
            vk::MemoryAllocateInfo alloc_ci{
                .pNext = &export_memory_ci,
                .allocationSize = requirements.size,
                .memoryTypeIndex = vk::memory::find_memory_type(renderer.vk_context().physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
            };
            slot.memory = vk::raii::DeviceMemory{renderer.vk_context().device, alloc_ci};
            slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
            vk::MemoryGetWin32HandleInfoKHR memory_handle_info{
                .memory = *slot.memory,
                .handleType = memory_handle_type,
            };
            HANDLE memory_handle = renderer.vk_context().device.getMemoryWin32HandleKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueWin32,
                .handle = { .win32 = { .handle = memory_handle, }, },
                .size = requirements.size,
            };
            check_cuda(cudaImportExternalMemory(&slot.external_memory, &external_memory_desc), "cudaImportExternalMemory");
            CloseHandle(memory_handle);

            vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            HANDLE semaphore_handle = renderer.vk_context().device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32,
                .handle = { .win32 = { .handle = semaphore_handle, }, },
            };
            check_cuda(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
            CloseHandle(semaphore_handle);
#else
            vk::MemoryGetFdInfoKHR memory_handle_info{
                .memory = *slot.memory,
                .handleType = memory_handle_type,
            };
            const int memory_fd = renderer.vk_context().device.getMemoryFdKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueFd,
                .handle = { .fd = memory_fd, },
                .size = requirements.size,
            };
            check_cuda(cudaImportExternalMemory(&slot.external_memory, &external_memory_desc), "cudaImportExternalMemory");

            vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            const int semaphore_fd = renderer.vk_context().device.getSemaphoreFdKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
                .handle = { .fd = semaphore_fd, },
            };
            check_cuda(cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc), "cudaImportExternalSemaphore");
#endif

            cudaExternalMemoryBufferDesc buffer_desc{
                .offset = 0,
                .size = data.capture.field_bytes,
            };
            check_cuda(cudaExternalMemoryGetMappedBuffer(&slot.field_cuda_ptr, slot.external_memory, &buffer_desc), "cudaExternalMemoryGetMappedBuffer");
            if (data.capture.velocity_bytes != 0) {
                check_cuda(cudaMalloc(&slot.velocity_cuda_ptr, data.capture.velocity_bytes), "cudaMalloc velocity snapshot");
                slot.velocity_host.resize(static_cast<size_t>(data.capture.velocity_bytes / sizeof(float)));
            }

            vk::DescriptorBufferInfo field_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range = data.capture.field_bytes,
            };
            vk::WriteDescriptorSet field_write{
                .dstSet = *slot.descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &field_info,
            };
            renderer.vk_context().device.updateDescriptorSets(field_write, {});
        }
        return true;
    }

    bool capture_snapshot(AppState& state, AppData& data, VisualizationApp& renderer, const char* tag) {
        const int slot_index = find_available_capture_slot(data, renderer.frames_in_flight());
        if (slot_index < 0) return false;
        nvtx3::scoped_range range{tag};
        const auto request = make_capture_request(state, data);
        const auto begin = std::chrono::steady_clock::now();
        const auto capture = begin_capture(data, slot_index);
        export_field(state, data, current_field_info(state).id, capture.field_cuda_ptr);
        if (request.export_velocity_host) {
            export_velocity(data, capture.velocity_cuda_ptr);
            check_cuda(cudaMemcpyAsync(capture.velocity_host_ptr, capture.velocity_cuda_ptr, data.capture.velocity_bytes, cudaMemcpyDeviceToHost, data.physics.stream), "cudaMemcpyAsync velocity snapshot");
        }
        cudaExternalSemaphoreSignalParams signal_params{};
        signal_params.params.fence.value = capture.ready_generation;
        check_cuda(cudaSignalExternalSemaphoresAsync(&capture.external_semaphore, &signal_params, 1, data.physics.stream), "cudaSignalExternalSemaphoresAsync");
        check_cuda(cudaStreamSynchronize(data.physics.stream), "cudaStreamSynchronize");
        complete_capture(data, slot_index, request, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        return true;
    }

    std::optional<VisualizationSnapshotView> active_snapshot(const AppState& state, const AppData& data) {
        if (data.capture.active_slot < 0) return std::nullopt;
        const auto& slot = data.capture.slots.at(static_cast<size_t>(data.capture.active_slot));
        return VisualizationSnapshotView{
            .grid = slot.grid,
            .field = {
                .descriptor_set = *slot.descriptor_set,
                .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                .ready_generation = slot.ready_generation,
                .component_count = slot.field_component_count,
                .semantic = slot.semantic,
                .label = slot.label,
            },
            .collider = collider_overlay(state),
            .velocity = {
                .data = slot.has_velocity_host ? slot.velocity_host.data() : nullptr,
            },
        };
    }

    void mark_snapshot_submitted(AppData& data) {
        const uint64_t next_submit_serial = data.capture.submit_serial + 1;
        if (data.capture.active_slot >= 0) data.capture.slots[static_cast<size_t>(data.capture.active_slot)].last_used_submit_serial = next_submit_serial;
        data.capture.submit_serial = next_submit_serial;
    }

    void draw_simulation_controls(AppState& state, const AppData& data, bool& reset_requested, bool& field_changed) {
        ImGui::Begin("Simulation");
        auto& physics = state.physics;
        auto& solver = physics.solver;
        auto& playback = state.ui.playback;
        const float extent_x = static_cast<float>(solver.backend.nx) * solver.backend.cell_size;
        const float extent_y = static_cast<float>(solver.backend.ny) * solver.backend.cell_size;
        const float extent_z = static_cast<float>(solver.backend.nz) * solver.backend.cell_size;
        const float min_extent = (std::min)({extent_x, extent_y, extent_z});
        auto request_scene_reset = [&]() {
            if (physics.preset != ScenePreset::Custom) physics.preset = ScenePreset::Custom;
            reset_requested = true;
        };

        if (ImGui::BeginCombo("Field", current_field_info(state).label.data())) {
            for (int i = 0; i < static_cast<int>(field_catalog_storage.size()); ++i) {
                const bool is_selected = physics.selected_field == i;
                if (ImGui::Selectable(field_catalog_storage[static_cast<size_t>(i)].label.data(), is_selected)) {
                    physics.selected_field = i;
                    field_changed = true;
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        int scene_preset = std::clamp(static_cast<int>(physics.preset), 0, static_cast<int>(scene_preset_labels.size()) - 1);
        if (ImGui::BeginCombo("Scene Preset", scene_preset_labels[static_cast<size_t>(scene_preset)])) {
            for (int i = 0; i < static_cast<int>(scene_preset_labels.size()); ++i) {
                const bool is_selected = scene_preset == i;
                if (ImGui::Selectable(scene_preset_labels[static_cast<size_t>(i)], is_selected)) {
                    if (i < static_cast<int>(ScenePreset::Custom)) {
                        apply_scene_preset(state, static_cast<ScenePreset>(i));
                        reset_requested = true;
                    } else {
                        physics.preset = ScenePreset::Custom;
                    }
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("Pause Simulation", &playback.paused);
        if (ImGui::Button("Single Step")) playback.step_once = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset Backend")) reset_requested = true;
        ImGui::SliderInt("Sim Steps / Frame", &playback.sim_steps_per_frame, 1, 8);
        ImGui::SliderInt("Snapshot Interval", &playback.snapshot_interval, 1, 8);

        ImGui::Separator();
        ImGui::TextUnformatted("Solver");
        ImGui::Text("Step calls: %llu", static_cast<unsigned long long>(data.physics.stats.step_count));
        ImGui::Text("Last step call: %.3f ms", data.physics.stats.last_step_call_ms);
        ImGui::Text("Avg step call: %.3f ms", data.physics.stats.average_step_call_ms);
        ImGui::Text("Projection max |div|: %.6g", data.physics.stats.projection_max_abs_divergence);
        ImGui::Text("Projection RMS div: %.6g", data.physics.stats.projection_rms_divergence);
        ImGui::Text("Snapshot commits: %llu", static_cast<unsigned long long>(data.capture.stats.snapshot_count));
        ImGui::Text("Last snapshot: %.3f ms", data.capture.stats.last_snapshot_ms);
        ImGui::Text("Avg snapshot: %.3f ms", data.capture.stats.average_snapshot_ms);

        ImGui::Separator();
        ImGui::TextUnformatted("Grid / Time");
        ImGui::Text("Domain Size: %.3f m x %.3f m x %.3f m", extent_x, extent_y, extent_z);
        if (ImGui::SliderInt("Grid X (cells)", &solver.backend.nx, 16, 512)) request_scene_reset();
        if (ImGui::SliderInt("Grid Y (cells)", &solver.backend.ny, 16, 512)) request_scene_reset();
        if (ImGui::SliderInt("Grid Z (cells)", &solver.backend.nz, 16, 512)) request_scene_reset();
        if (ImGui::SliderFloat("Dt (s)", &solver.backend.dt, 1.0f / 480.0f, 1.0f / 24.0f, "%.5f")) request_scene_reset();
        if (ImGui::SliderFloat("Cell Size (m)", &solver.backend.cell_size, 0.0025f, 0.05f, "%.4f")) request_scene_reset();
        if (ImGui::SliderFloat("Viscosity (m^2/s)", &solver.backend.viscosity, 0.0f, 0.002f, "%.5f")) request_scene_reset();
        if (ImGui::SliderFloat("Density Diffusion (m^2/s)", &solver.density_diffusion, 0.0f, 0.002f, "%.5f")) request_scene_reset();
        if (ImGui::SliderFloat("Dye Diffusion (m^2/s)", &solver.dye_diffusion, 0.0f, 0.002f, "%.5f")) request_scene_reset();
        if (ImGui::SliderInt("Diffuse Iterations", &solver.backend.diffuse_iterations, 1, 64)) request_scene_reset();
        if (ImGui::SliderInt("Pressure Iterations", &solver.backend.pressure_iterations, 4, 192)) request_scene_reset();

        auto draw_boundary_combo = [&](const char* label, StableFluidsBoundaryFaceDesc& face) {
            int boundary = std::clamp(static_cast<int>(face.type), 0, static_cast<int>(boundary_labels.size()) - 1);
            if (ImGui::BeginCombo(label, boundary_labels[static_cast<size_t>(boundary)])) {
                for (int i = 0; i < static_cast<int>(boundary_labels.size()); ++i) {
                    const bool is_selected = boundary == i;
                    if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                        boundary = i;
                        face.type = static_cast<uint32_t>(i);
                        request_scene_reset();
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        };

        ImGui::Separator();
        ImGui::TextUnformatted("Domain Boundary");
        draw_boundary_combo("Boundary X-", solver.backend.domain_boundary.x_min);
        draw_boundary_combo("Boundary X+", solver.backend.domain_boundary.x_max);
        draw_boundary_combo("Boundary Y-", solver.backend.domain_boundary.y_min);
        draw_boundary_combo("Boundary Y+", solver.backend.domain_boundary.y_max);
        draw_boundary_combo("Boundary Z-", solver.backend.domain_boundary.z_min);
        draw_boundary_combo("Boundary Z+", solver.backend.domain_boundary.z_max);
        if (ImGui::SliderFloat("Inflow Vel X- (m/s)", &solver.backend.domain_boundary.x_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
        if (ImGui::SliderFloat("Inflow Vel X+ (m/s)", &solver.backend.domain_boundary.x_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
        if (ImGui::SliderFloat("Inflow Vel Y- (m/s)", &solver.backend.domain_boundary.y_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
        if (ImGui::SliderFloat("Inflow Vel Y+ (m/s)", &solver.backend.domain_boundary.y_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
        if (ImGui::SliderFloat("Inflow Vel Z- (m/s)", &solver.backend.domain_boundary.z_min.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();
        if (ImGui::SliderFloat("Inflow Vel Z+ (m/s)", &solver.backend.domain_boundary.z_max.velocity, -4.0f, 4.0f, "%.2f")) request_scene_reset();

        ImGui::Separator();
        ImGui::TextUnformatted("Forces");
        if (ImGui::SliderFloat("Gravity Y (m/s^2)", &solver.gravity_y, -20.0f, 20.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Buoyancy Beta", &solver.buoyancy_beta, 0.0f, 2.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Ambient Density", &solver.ambient_density, 0.0f, 2.0f, "%.3f")) request_scene_reset();
        ImGui::Text("Buoyancy accel / density: %.3f m/s^2", -solver.gravity_y * solver.buoyancy_beta);
        if (ImGui::SliderFloat("Uniform Force X (m/s^2)", &solver.backend.uniform_force_x, -20.0f, 20.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Uniform Force Y (m/s^2)", &solver.backend.uniform_force_y, -20.0f, 20.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Uniform Force Z (m/s^2)", &solver.backend.uniform_force_z, -20.0f, 20.0f, "%.3f")) request_scene_reset();

        ImGui::Separator();
        ImGui::TextUnformatted("Sources");
        if (ImGui::Checkbox("Emit Source", &physics.emit_source)) request_scene_reset();
        auto draw_emitter_controls = [&](const char* label, SourceEmitterSettings& emitter) {
            if (!ImGui::TreeNode(label)) return;
            if (ImGui::Checkbox((std::string("Enabled##") + label).c_str(), &emitter.enabled)) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Center X (m)##") + label).c_str(), &emitter.center_x, 0.0f, extent_x, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Center Y (m)##") + label).c_str(), &emitter.center_y, 0.0f, extent_y, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Center Z (m)##") + label).c_str(), &emitter.center_z, 0.0f, extent_z, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Dir X##") + label).c_str(), &emitter.direction_x, -1.0f, 1.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Dir Y##") + label).c_str(), &emitter.direction_y, -1.0f, 1.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Dir Z##") + label).c_str(), &emitter.direction_z, -1.0f, 1.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Speed (m/s)##") + label).c_str(), &emitter.speed, 0.0f, 5.0f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Radius (m)##") + label).c_str(), &emitter.radius, solver.backend.cell_size, min_extent * 0.25f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Density##") + label).c_str(), &emitter.density_amount, 0.0f, 3.0f, "%.2f")) request_scene_reset();
            if (ImGui::SliderFloat((std::string("Dye##") + label).c_str(), &emitter.dye_amount, 0.0f, 3.0f, "%.2f")) request_scene_reset();
            if (ImGui::ColorEdit3((std::string("Color##") + label).c_str(), &emitter.color_r)) request_scene_reset();
            ImGui::TreePop();
        };
        draw_emitter_controls("Emitter A", physics.emitters.a);
        draw_emitter_controls("Emitter B", physics.emitters.b);

        ImGui::Separator();
        ImGui::TextUnformatted("Collider");
        if (ImGui::Checkbox("Enable Collider", &physics.scene.collider.enabled)) request_scene_reset();
        int collider_type = std::clamp(static_cast<int>(physics.scene.collider.type), 0, static_cast<int>(collider_type_labels.size()) - 1);
        if (ImGui::BeginCombo("Collider Type", collider_type_labels[static_cast<size_t>(collider_type)])) {
            for (int i = 0; i < static_cast<int>(collider_type_labels.size()); ++i) {
                const bool is_selected = collider_type == i;
                if (ImGui::Selectable(collider_type_labels[static_cast<size_t>(i)], is_selected)) {
                    physics.scene.collider.type = static_cast<ColliderType>(i);
                    request_scene_reset();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        int collider_boundary = std::clamp(static_cast<int>(physics.scene.collider.boundary), 0, 1);
        if (ImGui::BeginCombo("Collider Boundary", boundary_labels[static_cast<size_t>(collider_boundary)])) {
            for (int i = 0; i < 2; ++i) {
                const bool is_selected = collider_boundary == i;
                if (ImGui::Selectable(boundary_labels[static_cast<size_t>(i)], is_selected)) {
                    physics.scene.collider.boundary = static_cast<uint32_t>(i);
                    request_scene_reset();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::SliderFloat("Collider X (m)", &physics.scene.collider.center_x, 0.0f, extent_x, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Collider Y (m)", &physics.scene.collider.center_y, 0.0f, extent_y, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Collider Z (m)", &physics.scene.collider.center_z, 0.0f, extent_z, "%.3f")) request_scene_reset();
        if (physics.scene.collider.type == ColliderType::Sphere) {
            if (ImGui::SliderFloat("Collider Radius (m)", &physics.scene.collider.radius, solver.backend.cell_size, min_extent * 0.45f, "%.3f")) request_scene_reset();
        } else {
            if (ImGui::SliderFloat("Half Extent X (m)", &physics.scene.collider.half_extent_x, solver.backend.cell_size, extent_x * 0.45f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Half Extent Y (m)", &physics.scene.collider.half_extent_y, solver.backend.cell_size, extent_y * 0.45f, "%.3f")) request_scene_reset();
            if (ImGui::SliderFloat("Half Extent Z (m)", &physics.scene.collider.half_extent_z, solver.backend.cell_size, extent_z * 0.45f, "%.3f")) request_scene_reset();
        }
        if (ImGui::SliderFloat("Collider Vel X (m/s)", &physics.scene.collider.velocity_x, -3.0f, 3.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Collider Vel Y (m/s)", &physics.scene.collider.velocity_y, -3.0f, 3.0f, "%.3f")) request_scene_reset();
        if (ImGui::SliderFloat("Collider Vel Z (m/s)", &physics.scene.collider.velocity_z, -3.0f, 3.0f, "%.3f")) request_scene_reset();
        ImGui::End();
    }

} // namespace app
