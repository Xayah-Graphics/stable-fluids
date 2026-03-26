module;
#include <vulkan/vulkan_raii.hpp>
export module vk.swapchain;
import vk.context;
import std;


namespace vk::swapchain {
    export struct Swapchain {
        raii::SwapchainKHR handle{nullptr};
        Format format{};
        ColorSpaceKHR color_space{};
        Extent2D extent{};
        std::vector<Image> images{};
        std::vector<raii::ImageView> image_views{};

        raii::Image depth_image{nullptr};
        raii::DeviceMemory depth_memory{nullptr};
        raii::ImageView depth_view{nullptr};
        Format depth_format{};
        ImageAspectFlags depth_aspect{};
        ImageLayout depth_layout{ImageLayout::eUndefined};

        Swapchain() {}
        ~Swapchain()                               = default;
        Swapchain(Swapchain&&) noexcept            = default;
        Swapchain& operator=(Swapchain&&) noexcept = default;
        Swapchain(const Swapchain&)                = delete;
        Swapchain& operator=(const Swapchain&)     = delete;
    };

    export [[nodiscard]] Swapchain setup_swapchain(const context::VulkanContext& vkctx, const context::SurfaceContext& sctx, const Swapchain* old = nullptr);
    export void recreate_swapchain(const context::VulkanContext& vkctx, context::SurfaceContext& sctx, Swapchain& sc);
} // namespace vk::swapchain
