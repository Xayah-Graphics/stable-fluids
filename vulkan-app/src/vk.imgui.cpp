module;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <vulkan/vulkan_raii.hpp>
module vk.imgui;
import std;

namespace vk::imgui {

    static raii::DescriptorPool make_descriptor_pool(const raii::Device& device) {
        constexpr std::array<VkDescriptorPoolSize, 11> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
        }};

        DescriptorPoolCreateInfo ci{};
        ci.flags         = DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        ci.maxSets       = 1000u * static_cast<uint32_t>(sizes.size());
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes    = reinterpret_cast<const DescriptorPoolSize*>(sizes.data());

        return raii::DescriptorPool{device, ci};
    }

    ImGuiSystem create(const context::VulkanContext& vkctx, GLFWwindow* window, Format color_format, const uint32_t min_image_count, const uint32_t image_count, const bool enable_docking, const bool enable_viewports) {
        if (!window) throw std::runtime_error("vk.imgui.create: window is null");

        ImGuiSystem sys{};
        sys.window          = window;
        sys.color_format    = color_format;
        sys.min_image_count = min_image_count;
        sys.image_count     = image_count;
        sys.docking         = enable_docking;
        sys.viewports       = enable_viewports;
        sys.descriptor_pool = make_descriptor_pool(vkctx.device);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        auto& io = ImGui::GetIO();
        if (enable_docking) io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        if (enable_viewports) io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        ImGui::StyleColorsDark();

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            auto& style                       = ImGui::GetStyle();
            style.WindowRounding              = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
            throw std::runtime_error("ImGui_ImplGlfw_InitForVulkan failed");
        }

        ImGui_ImplVulkan_InitInfo init{};
        init.Instance            = static_cast<VkInstance>(*vkctx.instance);
        init.PhysicalDevice      = static_cast<VkPhysicalDevice>(*vkctx.physical_device);
        init.Device              = static_cast<VkDevice>(*vkctx.device);
        init.QueueFamily         = vkctx.graphics_queue_index;
        init.Queue               = static_cast<VkQueue>(*vkctx.graphics_queue);
        init.DescriptorPool      = static_cast<VkDescriptorPool>(*sys.descriptor_pool);
        init.MinImageCount       = min_image_count;
        init.ImageCount          = image_count;
        init.MSAASamples         = VK_SAMPLE_COUNT_1_BIT;
        init.UseDynamicRendering = VK_TRUE;

        auto vk_format = static_cast<VkFormat>(color_format);
        const VkPipelineRenderingCreateInfo pri{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &vk_format};

        init.PipelineRenderingCreateInfo = pri;

        if (!ImGui_ImplVulkan_Init(&init)) {
            throw std::runtime_error("ImGui_ImplVulkan_Init failed");
        }

        sys.initialized = true;
        return sys;
    }

    void shutdown(ImGuiSystem& sys) {
        if (!sys.initialized) return;

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        sys = {};
    }

    void begin_frame() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void end_frame() {
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }
    }

    void render(ImGuiSystem&, const raii::CommandBuffer& cmd, const Extent2D extent, const ImageView target_view, ImageLayout target_layout) {
        ImGui::Render();

        VkRenderingAttachmentInfo color{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = static_cast<VkImageView>(target_view), .imageLayout = static_cast<VkImageLayout>(target_layout), .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};

        const VkRenderingInfo ri{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {extent.width, extent.height}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &color};

        vkCmdBeginRendering(*cmd, &ri);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);
        vkCmdEndRendering(*cmd);
    }

    void set_min_image_count(ImGuiSystem& sys, const uint32_t min_image_count) {
        sys.min_image_count = min_image_count;
        ImGui_ImplVulkan_SetMinImageCount(static_cast<int>(min_image_count));
    }

    void draw_mini_axis_gizmo(const math::mat4& c2w) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        if (!viewport) return;

        ImDrawList* draw_list = ImGui::GetForegroundDrawList(viewport);
        if (!draw_list) return;

        constexpr float size   = 80.0f;
        constexpr float margin = 16.0f;
        constexpr float radius = size * 0.42f;

        const ImVec2 center(viewport->Pos.x + viewport->Size.x - margin - size * 0.5f, viewport->Pos.y + margin + size * 0.5f);

        draw_list->AddCircleFilled(center, size * 0.5f, IM_COL32(30, 32, 36, 180), 48);
        draw_list->AddCircle(center, size * 0.5f, IM_COL32(255, 255, 255, 60), 48, 1.5f);

        const math::vec3 cam_right{c2w.c0.x, c2w.c0.y, c2w.c0.z, 0.0f};
        const math::vec3 cam_up{c2w.c1.x, c2w.c1.y, c2w.c1.z, 0.0f};
        const math::vec3 cam_forward{c2w.c2.x, c2w.c2.y, c2w.c2.z, 0.0f};

        struct Axis {
            math::vec3 world_dir;
            ImU32 color;
            const char* label;
        };

        constexpr Axis axes[3] = {
            {math::vec3{1, 0, 0, 0}, IM_COL32(255, 80, 80, 255), "X"},
            {math::vec3{0, 1, 0, 0}, IM_COL32(80, 255, 80, 255), "Y"},
            {math::vec3{0, 0, 1, 0}, IM_COL32(100, 140, 255, 255), "Z"},
        };

        struct ProjectedAxis {
            math::vec3 v;
            Axis axis;
        };

        ProjectedAxis p[3];

        for (int i = 0; i < 3; ++i) {
            const math::vec3 d = vk::math::normalize(axes[i].world_dir);

            const float x = vk::math::dot(d, cam_right);
            const float y = vk::math::dot(d, cam_up);
            const float z = vk::math::dot(d, cam_forward);

            p[i] = {vk::math::normalize(math::vec3{x, y, z, 0.0f}), axes[i]};
        }

        auto draw_axis = [&](const ProjectedAxis& a, bool back) {
            const float thickness = back ? 2.0f : 3.0f;

            ImU32 color = a.axis.color;
            if (back) {
                color = IM_COL32((color >> IM_COL32_R_SHIFT) & 0xFF, (color >> IM_COL32_G_SHIFT) & 0xFF, (color >> IM_COL32_B_SHIFT) & 0xFF, 120);
            }

            const ImVec2 end(center.x + a.v.x * radius, center.y - a.v.y * radius);

            draw_list->AddLine(center, end, color, thickness);
            draw_list->AddCircleFilled(end, back ? 3.0f : 4.5f, color, 12);

            if (!back) {
                const float ox = a.v.x >= 0.0f ? 8.0f : -20.0f;
                const float oy = a.v.y >= 0.0f ? -18.0f : 4.0f;
                draw_list->AddText(ImVec2(end.x + ox, end.y + oy), color, a.axis.label);
            }
        };

        constexpr float eps = 1e-4f;

        for (const auto& a : p) {
            if (a.v.z > eps) draw_axis(a, true);
        }
        for (const auto& a : p) {
            if (a.v.z <= eps) draw_axis(a, false);
        }
    }
} // namespace vk::imgui
