module;
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <vulkan/vulkan_raii.hpp>

export module vk.imgui;
import vk.context;
import vk.math;
import std;

namespace vk::imgui {

    export struct ImGuiSystem {
        raii::DescriptorPool descriptor_pool{nullptr};
        Format color_format      = Format::eUndefined;
        uint32_t min_image_count = 2;
        uint32_t image_count     = 2;
        bool docking             = true;
        bool viewports           = true;
        GLFWwindow* window       = nullptr;
        bool initialized         = false;
    };

    export [[nodiscard]]
    ImGuiSystem create(const context::VulkanContext& vkctx, GLFWwindow* window, Format color_format, uint32_t min_image_count, uint32_t image_count, bool enable_docking = true, bool enable_viewports = true);

    export void shutdown(ImGuiSystem& sys);

    export void begin_frame();
    export void end_frame();

    export void render(ImGuiSystem& sys, const raii::CommandBuffer& cmd, Extent2D extent, ImageView target_view, ImageLayout target_layout = ImageLayout::eColorAttachmentOptimal);

    export void set_min_image_count(ImGuiSystem& sys, uint32_t min_image_count);

    export void draw_mini_axis_gizmo(const math::mat4& view);
} // namespace vk::imgui
