module;
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_raii.hpp>
export module vk.context;
import std;


namespace vk::context {
    export struct VulkanContext {
        raii::Context context;
        raii::Instance instance{nullptr};
        raii::DebugUtilsMessengerEXT debug_messenger{nullptr};
        raii::PhysicalDevice physical_device{nullptr};
        raii::Device device{nullptr};
        raii::Queue graphics_queue{nullptr};
        uint32_t graphics_queue_index{0};
        raii::CommandPool command_pool{nullptr};

        VulkanContext() {}
        ~VulkanContext()                                   = default;
        VulkanContext(const VulkanContext&)                = delete;
        VulkanContext& operator=(const VulkanContext&)     = delete;
        VulkanContext(VulkanContext&&) noexcept            = default;
        VulkanContext& operator=(VulkanContext&&) noexcept = default;
    };

    export struct SurfaceContext {
        std::shared_ptr<GLFWwindow> window{nullptr};
        raii::SurfaceKHR surface{nullptr};
        Extent2D extent{};
        bool resize_requested{false};

        SurfaceContext() {}
        ~SurfaceContext()                                    = default;
        SurfaceContext(const SurfaceContext&)                = delete;
        SurfaceContext& operator=(const SurfaceContext&)     = delete;
        SurfaceContext(SurfaceContext&&) noexcept            = default;
        SurfaceContext& operator=(SurfaceContext&&) noexcept = default;
    };

    export [[nodiscard]] std::pair<VulkanContext, SurfaceContext> setup_vk_context_glfw(const std::string& app_name, const std::string& engine_name);
} // namespace vk::context
