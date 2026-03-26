module;
#include <vulkan/vulkan_raii.hpp>
export module vk.texture;

import vk.context;
import std;

namespace vk::texture {

    export struct Texture2D {
        Format format = Format::eUndefined;
        Extent2D extent{};
        uint32_t mip_levels = 1;
        uint32_t layers     = 1;

        raii::Image image{nullptr};
        raii::DeviceMemory memory{nullptr};
        raii::ImageView view{nullptr};
        raii::Sampler sampler{nullptr};
    };

    export struct Texture2DArray {
        Format format = Format::eUndefined;
        Extent2D extent{};
        uint32_t mip_levels = 1;
        uint32_t layers     = 1;

        raii::Image image{nullptr};
        raii::DeviceMemory memory{nullptr};
        raii::ImageView view{nullptr};
        raii::Sampler sampler{nullptr};
    };

    export enum class MipMode : uint8_t {
        None,
        Generate,
    };

    export struct Texture2DDesc {
        uint32_t width   = 1;
        uint32_t height  = 1;
        uint32_t layers  = 1;
        bool srgb        = false;
        MipMode mip_mode = MipMode::Generate;

        Filter min_filter             = Filter::eLinear;
        Filter mag_filter             = Filter::eLinear;
        SamplerMipmapMode mipmap_mode = SamplerMipmapMode::eLinear;

        SamplerAddressMode address_u = SamplerAddressMode::eRepeat;
        SamplerAddressMode address_v = SamplerAddressMode::eRepeat;
        SamplerAddressMode address_w = SamplerAddressMode::eRepeat;

        float max_anisotropy = 1.0f;
    };

    export [[nodiscard]] Texture2D create_texture_2d_rgba8(const context::VulkanContext& vkctx, std::span<const std::byte> rgba8, Texture2DDesc desc);
    export [[nodiscard]] Texture2DArray create_texture_2d_array_rgba8(const context::VulkanContext& vkctx, std::span<const std::byte> rgba8, Texture2DDesc desc);
    export [[nodiscard]] raii::DescriptorSetLayout make_texture_set_layout(const raii::Device& device);
} // namespace vk::texture
