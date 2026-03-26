module;
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan_raii.hpp>
module vk.swapchain;
import vk.context;
import std;


namespace {

    [[nodiscard]] vk::SurfaceFormatKHR choose_surface_format(const std::span<const vk::SurfaceFormatKHR> formats) {
        if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
            return vk::SurfaceFormatKHR{vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear};
        }

        auto pick = [&](const vk::Format fmt, const vk::ColorSpaceKHR cs) -> std::optional<vk::SurfaceFormatKHR> {
            for (const auto& f : formats) {
                if (f.format == fmt && f.colorSpace == cs) return f;
            }
            return std::nullopt;
        };

        constexpr auto cs = vk::ColorSpaceKHR::eSrgbNonlinear;

        if (const auto v = pick(vk::Format::eB8G8R8A8Srgb, cs)) return *v;
        if (const auto v = pick(vk::Format::eR8G8B8A8Srgb, cs)) return *v;
        if (const auto v = pick(vk::Format::eB8G8R8A8Unorm, cs)) return *v;
        if (const auto v = pick(vk::Format::eR8G8B8A8Unorm, cs)) return *v;

        return formats.front();
    }

    [[nodiscard]] vk::PresentModeKHR choose_present_mode(const std::span<const vk::PresentModeKHR> modes) {
        for (const auto m : modes) {
            if (m == vk::PresentModeKHR::eMailbox) return m;
        }
        for (const auto m : modes) {
            if (m == vk::PresentModeKHR::eImmediate) return m;
        }
        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR& caps, const vk::Extent2D requested) {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return caps.currentExtent;
        }
        return vk::Extent2D{
            std::clamp(requested.width, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp(requested.height, caps.minImageExtent.height, caps.maxImageExtent.height),
        };
    }

    [[nodiscard]] uint32_t choose_image_count(const vk::SurfaceCapabilitiesKHR& caps) {
        uint32_t count = caps.minImageCount + 1;
        if (count < 2) count = 2;
        if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
        return count;
    }

    [[nodiscard]] vk::CompositeAlphaFlagBitsKHR choose_composite_alpha(const vk::SurfaceCapabilitiesKHR& caps) {
        constexpr vk::CompositeAlphaFlagBitsKHR preferred[] = {
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
            vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
            vk::CompositeAlphaFlagBitsKHR::eInherit,
        };
        for (const auto v : preferred) {
            if (caps.supportedCompositeAlpha & v) return v;
        }
        throw std::runtime_error("No supported compositeAlpha");
    }

    [[nodiscard]] vk::SurfaceTransformFlagBitsKHR choose_pre_transform(const vk::SurfaceCapabilitiesKHR& caps) {
        if (caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
            return vk::SurfaceTransformFlagBitsKHR::eIdentity;
        }
        return caps.currentTransform;
    }

    [[nodiscard]] vk::ImageUsageFlags choose_swapchain_usage(const vk::SurfaceCapabilitiesKHR& caps) {
        const vk::ImageUsageFlags supported = caps.supportedUsageFlags;
        if ((supported & vk::ImageUsageFlagBits::eColorAttachment) != vk::ImageUsageFlagBits::eColorAttachment) throw std::runtime_error("Swapchain must support eColorAttachment usage");
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
        if ((supported & vk::ImageUsageFlagBits::eTransferDst) == vk::ImageUsageFlagBits::eTransferDst) usage |= vk::ImageUsageFlagBits::eTransferDst;
        if ((supported & vk::ImageUsageFlagBits::eTransferSrc) == vk::ImageUsageFlagBits::eTransferSrc) usage |= vk::ImageUsageFlagBits::eTransferSrc;
        return usage;
    }

    struct SwapchainSharing {
        vk::SharingMode mode{vk::SharingMode::eExclusive};
        std::array<uint32_t, 2> indices{};
        uint32_t count{0};
    };

    [[nodiscard]] SwapchainSharing choose_sharing(const uint32_t graphics_q, const uint32_t present_q) {
        SwapchainSharing out{};
        if (graphics_q != present_q) {
            out.mode    = vk::SharingMode::eConcurrent;
            out.indices = {graphics_q, present_q};
            out.count   = 2;
        }
        return out;
    }

    [[nodiscard]] uint32_t find_memory_type(const vk::raii::PhysicalDevice& pd, const uint32_t type_bits, const vk::MemoryPropertyFlags props) {
        const auto mp = pd.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            const bool ok_bits  = (type_bits & (1u << i)) != 0;
            const bool ok_props = (mp.memoryTypes[i].propertyFlags & props) == props;
            if (ok_bits && ok_props) return i;
        }
        throw std::runtime_error("No suitable memory type found");
    }

    [[nodiscard]] bool supports_depth_attachment(const vk::raii::PhysicalDevice& pd, const vk::Format fmt) {
        const auto p = pd.getFormatProperties(fmt);
        return (p.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) != vk::FormatFeatureFlags{};
    }

    [[nodiscard]] vk::Format choose_depth_format(const vk::raii::PhysicalDevice& pd) {
        constexpr vk::Format candidates[] = {
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint,
        };
        for (const auto f : candidates) {
            if (supports_depth_attachment(pd, f)) return f;
        }
        throw std::runtime_error("No supported depth format found");
    }

    [[nodiscard]] vk::ImageAspectFlags depth_aspect(const vk::Format fmt) {
        switch (fmt) {
        case vk::Format::eD32SfloatS8Uint:
        case vk::Format::eD24UnormS8Uint: return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        default: return vk::ImageAspectFlagBits::eDepth;
        }
    }

    struct DepthResources {
        vk::raii::Image image{nullptr};
        vk::raii::DeviceMemory memory{nullptr};
        vk::raii::ImageView view{nullptr};
    };

    [[nodiscard]] DepthResources create_depth_resources(const vk::raii::Device& device, const vk::raii::PhysicalDevice& pd, const vk::Extent2D extent, const vk::Format format, const vk::ImageAspectFlags aspect) {
        DepthResources out{};

        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        usage |= vk::ImageUsageFlagBits::eTransientAttachment;

        const vk::ImageCreateInfo image_ci{
            .imageType     = vk::ImageType::e2D,
            .format        = format,
            .extent        = vk::Extent3D{extent.width, extent.height, 1},
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = vk::SampleCountFlagBits::e1,
            .tiling        = vk::ImageTiling::eOptimal,
            .usage         = usage,
            .sharingMode   = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        out.image = vk::raii::Image{device, image_ci};

        const vk::MemoryRequirements req = out.image.getMemoryRequirements();
        const vk::MemoryAllocateInfo alloc_ci{
            .allocationSize  = req.size,
            .memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
        };

        out.memory = vk::raii::DeviceMemory{device, alloc_ci};
        out.image.bindMemory(*out.memory, 0);

        const vk::ImageViewCreateInfo view_ci{
            .image            = *out.image,
            .viewType         = vk::ImageViewType::e2D,
            .format           = format,
            .subresourceRange = vk::ImageSubresourceRange{aspect, 0, 1, 0, 1},
        };

        out.view = vk::raii::ImageView{device, view_ci};

        return out;
    }

    [[nodiscard]] vk::Extent2D wait_nonzero_framebuffer_extent(GLFWwindow* w) {
        int fbw = 0;
        int fbh = 0;
        while (fbw == 0 || fbh == 0) {
            glfwGetFramebufferSize(w, &fbw, &fbh);
            glfwWaitEvents();
        }
        return vk::Extent2D{static_cast<uint32_t>(fbw), static_cast<uint32_t>(fbh)};
    }

} // namespace

vk::swapchain::Swapchain vk::swapchain::setup_swapchain(const context::VulkanContext& vkctx, const context::SurfaceContext& sctx, const Swapchain* old) {
    const auto caps  = vkctx.physical_device.getSurfaceCapabilitiesKHR(*sctx.surface);
    const auto fmts  = vkctx.physical_device.getSurfaceFormatsKHR(*sctx.surface);
    const auto modes = vkctx.physical_device.getSurfacePresentModesKHR(*sctx.surface);

    if (fmts.empty()) throw std::runtime_error("Surface has no formats");
    if (modes.empty()) throw std::runtime_error("Surface has no present modes");

    const auto [format, colorSpace] = choose_surface_format(fmts);
    const auto present_mode         = choose_present_mode(modes);
    const auto extent               = choose_extent(caps, sctx.extent);

    if (extent.width == 0 || extent.height == 0) throw std::runtime_error("Cannot create swapchain with zero extent");

    const auto image_count = choose_image_count(caps);
    const auto usage       = choose_swapchain_usage(caps);

    const auto composite = choose_composite_alpha(caps);
    const auto transform = choose_pre_transform(caps);

    const uint32_t graphics_q         = vkctx.graphics_queue_index;
    const uint32_t present_q          = vkctx.graphics_queue_index;
    const auto [mode, indices, count] = choose_sharing(graphics_q, present_q);

    SwapchainCreateInfoKHR ci{
        .surface          = *sctx.surface,
        .minImageCount    = image_count,
        .imageFormat      = format,
        .imageColorSpace  = colorSpace,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = usage,
        .imageSharingMode = mode,
        .preTransform     = transform,
        .compositeAlpha   = composite,
        .presentMode      = present_mode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = old ? *old->handle : VK_NULL_HANDLE,
    };

    if (count) {
        ci.queueFamilyIndexCount = count;
        ci.pQueueFamilyIndices   = indices.data();
    }

    Swapchain sc{};

    sc.handle      = raii::SwapchainKHR{vkctx.device, ci};
    sc.images      = sc.handle.getImages();
    sc.format      = format;
    sc.color_space = colorSpace;
    sc.extent      = extent;

    sc.image_views.clear();
    sc.image_views.reserve(sc.images.size());

    ImageViewCreateInfo ivci{
        .image            = VK_NULL_HANDLE,
        .viewType         = ImageViewType::e2D,
        .format           = sc.format,
        .components       = ComponentMapping{},
        .subresourceRange = ImageSubresourceRange{ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };

    for (auto img : sc.images) {
        ivci.image = img;
        sc.image_views.emplace_back(vkctx.device, ivci);
    }

    sc.depth_format = choose_depth_format(vkctx.physical_device);
    sc.depth_aspect = depth_aspect(sc.depth_format);

    {
        auto [image, memory, view] = create_depth_resources(vkctx.device, vkctx.physical_device, sc.extent, sc.depth_format, sc.depth_aspect);
        sc.depth_image             = std::move(image);
        sc.depth_memory            = std::move(memory);
        sc.depth_view              = std::move(view);
        sc.depth_layout            = ImageLayout::eUndefined;
    }

    return sc;
}

void vk::swapchain::recreate_swapchain(const context::VulkanContext& vkctx, context::SurfaceContext& sctx, Swapchain& sc) {
    const auto extent = wait_nonzero_framebuffer_extent(sctx.window.get());

    vkctx.device.waitIdle();

    sctx.extent = extent;

    const Swapchain old = std::move(sc);
    sc                  = setup_swapchain(vkctx, sctx, &old);

    sctx.resize_requested = false;
}
