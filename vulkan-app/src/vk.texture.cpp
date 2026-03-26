module;
#include <vulkan/vulkan_raii.hpp>
module vk.texture;
import vk.context;
import std;

namespace vk::texture {

    static uint32_t mip_count_for(uint32_t w, uint32_t h) {
        uint32_t levels = 1;
        while (w > 1 || h > 1) {
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
            ++levels;
        }
        return levels;
    }

    static uint32_t find_memory_type(const raii::PhysicalDevice& pd, uint32_t type_bits, MemoryPropertyFlags props) {
        const auto mp = pd.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            const bool ok_bits  = (type_bits & (1u << i)) != 0;
            const bool ok_props = (mp.memoryTypes[i].propertyFlags & props) == props;
            if (ok_bits && ok_props) return i;
        }
        throw std::runtime_error("vk.texture: no suitable memory type");
    }

    struct BufferWithMemory {
        raii::Buffer buffer{nullptr};
        raii::DeviceMemory memory{nullptr};
        DeviceSize size = 0;
    };

    static BufferWithMemory create_buffer(const raii::PhysicalDevice& pd, const raii::Device& dev, DeviceSize size, BufferUsageFlags usage, MemoryPropertyFlags props) {
        BufferWithMemory out{};
        out.size = size;

        BufferCreateInfo bci{};
        bci.size        = size;
        bci.usage       = usage;
        bci.sharingMode = SharingMode::eExclusive;

        out.buffer = raii::Buffer{dev, bci};

        const auto req    = out.buffer.getMemoryRequirements();
        const uint32_t mt = find_memory_type(pd, req.memoryTypeBits, props);

        MemoryAllocateInfo mai{};
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = mt;

        out.memory = raii::DeviceMemory{dev, mai};
        out.buffer.bindMemory(*out.memory, 0);

        return out;
    }

    struct ImageWithMemory {
        raii::Image image{nullptr};
        raii::DeviceMemory memory{nullptr};
    };

    static ImageWithMemory create_image_2d(const raii::PhysicalDevice& pd, const raii::Device& dev, uint32_t w, uint32_t h, uint32_t mip_levels, uint32_t layers, Format format, ImageUsageFlags usage) {
        ImageWithMemory out{};

        ImageCreateInfo ici{};
        ici.imageType     = ImageType::e2D;
        ici.format        = format;
        ici.extent        = Extent3D{w, h, 1};
        ici.mipLevels     = mip_levels;
        ici.arrayLayers   = layers;
        ici.samples       = SampleCountFlagBits::e1;
        ici.tiling        = ImageTiling::eOptimal;
        ici.usage         = usage;
        ici.sharingMode   = SharingMode::eExclusive;
        ici.initialLayout = ImageLayout::eUndefined;

        out.image = raii::Image{dev, ici};

        const auto req    = out.image.getMemoryRequirements();
        const uint32_t mt = find_memory_type(pd, req.memoryTypeBits, MemoryPropertyFlagBits::eDeviceLocal);

        MemoryAllocateInfo mai{};
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = mt;

        out.memory = raii::DeviceMemory{dev, mai};
        out.image.bindMemory(*out.memory, 0);

        return out;
    }

    static raii::CommandBuffer begin_one_time(const raii::Device& dev, const raii::CommandPool& pool) {
        CommandBufferAllocateInfo ai{};
        ai.commandPool        = *pool;
        ai.level              = CommandBufferLevel::ePrimary;
        ai.commandBufferCount = 1;

        raii::CommandBuffers cbs{dev, ai};
        raii::CommandBuffer cmd = std::move(cbs.front());

        CommandBufferBeginInfo bi{};
        bi.flags = CommandBufferUsageFlagBits::eOneTimeSubmit;
        cmd.begin(bi);
        return cmd;
    }

    static void end_one_time(const raii::Queue& q, raii::CommandBuffer& cmd) {
        cmd.end();

        SubmitInfo si{};
        si.commandBufferCount   = 1;
        const CommandBuffer raw = *cmd;
        si.pCommandBuffers      = &raw;

        q.submit(si);
        q.waitIdle();
    }

    static void barrier_image(const raii::CommandBuffer& cmd, Image image, ImageAspectFlags aspect, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count, ImageLayout old_layout, ImageLayout new_layout, PipelineStageFlags2 src_stage, AccessFlags2 src_access, PipelineStageFlags2 dst_stage, AccessFlags2 dst_access) {
        ImageMemoryBarrier2 b{};
        b.srcStageMask                    = src_stage;
        b.srcAccessMask                   = src_access;
        b.dstStageMask                    = dst_stage;
        b.dstAccessMask                   = dst_access;
        b.oldLayout                       = old_layout;
        b.newLayout                       = new_layout;
        b.image                           = image;
        b.subresourceRange.aspectMask     = aspect;
        b.subresourceRange.baseMipLevel   = base_mip;
        b.subresourceRange.levelCount     = mip_count;
        b.subresourceRange.baseArrayLayer = base_layer;
        b.subresourceRange.layerCount     = layer_count;

        DependencyInfo dep{};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;

        cmd.pipelineBarrier2(dep);
    }

    // Legacy convenience overload (single-layer)
    static void barrier_image(const raii::CommandBuffer& cmd, Image image, ImageAspectFlags aspect, uint32_t base_mip, uint32_t mip_count, ImageLayout old_layout, ImageLayout new_layout, PipelineStageFlags2 src_stage, AccessFlags2 src_access, PipelineStageFlags2 dst_stage, AccessFlags2 dst_access) {
        barrier_image(cmd, image, aspect, base_mip, mip_count, 0, 1, old_layout, new_layout, src_stage, src_access, dst_stage, dst_access);
    }

    static bool supports_linear_blit(const raii::PhysicalDevice& pd, Format format) {
        const auto props = pd.getFormatProperties(format);
        return (props.optimalTilingFeatures & FormatFeatureFlagBits::eSampledImageFilterLinear) == FormatFeatureFlagBits::eSampledImageFilterLinear;
    }

    static raii::ImageView create_view_2d(const raii::Device& dev, Image image, Format format, ImageAspectFlags aspect, uint32_t mip_levels) {
        ImageViewCreateInfo vci{};
        vci.image                           = image;
        vci.viewType                        = ImageViewType::e2D;
        vci.format                          = format;
        vci.subresourceRange.aspectMask     = aspect;
        vci.subresourceRange.baseMipLevel   = 0;
        vci.subresourceRange.levelCount     = mip_levels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount     = 1;
        return raii::ImageView{dev, vci};
    }

    static raii::ImageView create_view_2d_array(const raii::Device& dev, Image image, Format format, ImageAspectFlags aspect, uint32_t mip_levels, uint32_t layers) {
        ImageViewCreateInfo vci{};
        vci.image                           = image;
        vci.viewType                        = ImageViewType::e2DArray;
        vci.format                          = format;
        vci.subresourceRange.aspectMask     = aspect;
        vci.subresourceRange.baseMipLevel   = 0;
        vci.subresourceRange.levelCount     = mip_levels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount     = layers;
        return raii::ImageView{dev, vci};
    }

    static raii::Sampler create_sampler_2d(const raii::Device& dev, const Texture2DDesc& desc, uint32_t mip_levels) {
        SamplerCreateInfo sci{};
        sci.magFilter  = desc.mag_filter;
        sci.minFilter  = desc.min_filter;
        sci.mipmapMode = desc.mipmap_mode;

        sci.addressModeU = desc.address_u;
        sci.addressModeV = desc.address_v;
        sci.addressModeW = desc.address_w;

        sci.mipLodBias       = 0.0f;
        sci.anisotropyEnable = desc.max_anisotropy > 1.0f ? VK_TRUE : VK_FALSE;
        sci.maxAnisotropy    = std::max(1.0f, desc.max_anisotropy);

        sci.compareEnable = VK_FALSE;
        sci.compareOp     = CompareOp::eNever;

        sci.minLod                  = 0.0f;
        sci.maxLod                  = float(mip_levels);
        sci.borderColor             = BorderColor::eFloatTransparentBlack;
        sci.unnormalizedCoordinates = VK_FALSE;

        return raii::Sampler{dev, sci};
    }

    Texture2D create_texture_2d_rgba8(const context::VulkanContext& vkctx, std::span<const std::byte> rgba8, Texture2DDesc desc) {
        if (desc.width == 0 || desc.height == 0) throw std::runtime_error("vk.texture: invalid extent");
        if (desc.layers != 1) throw std::runtime_error("vk.texture: create_texture_2d_rgba8 expects layers == 1");
        const size_t expected = size_t(desc.width) * size_t(desc.height) * size_t(desc.layers) * 4u;
        if (rgba8.size_bytes() != expected) throw std::runtime_error("vk.texture: rgba8 size mismatch");

        const Format format = desc.srgb ? Format::eR8G8B8A8Srgb : Format::eR8G8B8A8Unorm;

        uint32_t mip_levels = 1;
        if (desc.mip_mode == MipMode::Generate) {
            mip_levels = mip_count_for(desc.width, desc.height);
            if (!supports_linear_blit(vkctx.physical_device, format)) {
                mip_levels = 1;
            }
        }

        const DeviceSize upload_size = DeviceSize(rgba8.size_bytes());

        auto staging = create_buffer(vkctx.physical_device, vkctx.device, upload_size, BufferUsageFlagBits::eTransferSrc, MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent);

        {
            void* dst = staging.memory.mapMemory(0, upload_size);
            std::memcpy(dst, rgba8.data(), rgba8.size_bytes());
            staging.memory.unmapMemory();
        }

        ImageUsageFlags usage = ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled;
        if (mip_levels > 1) usage |= ImageUsageFlagBits::eTransferSrc;

        auto img = create_image_2d(vkctx.physical_device, vkctx.device, desc.width, desc.height, mip_levels, desc.layers, format, usage);

        auto cmd = begin_one_time(vkctx.device, vkctx.command_pool);

        barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1, ImageLayout::eUndefined, ImageLayout::eTransferDstOptimal, PipelineStageFlagBits2::eTopOfPipe, AccessFlags2{}, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite);

        BufferImageCopy bic{};
        bic.bufferOffset                    = 0;
        bic.bufferRowLength                 = 0;
        bic.bufferImageHeight               = 0;
        bic.imageSubresource.aspectMask     = ImageAspectFlagBits::eColor;
        bic.imageSubresource.mipLevel       = 0;
        bic.imageSubresource.baseArrayLayer = 0;
        bic.imageSubresource.layerCount     = 1;
        bic.imageOffset                     = Offset3D{0, 0, 0};
        bic.imageExtent                     = Extent3D{desc.width, desc.height, 1};

        cmd.copyBufferToImage(*staging.buffer, *img.image, ImageLayout::eTransferDstOptimal, bic);

        if (mip_levels == 1) {
            barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, 0, 1, 0, 1, ImageLayout::eTransferDstOptimal, ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eFragmentShader, AccessFlagBits2::eShaderSampledRead);
        } else {
            uint32_t w = desc.width;
            uint32_t h = desc.height;

            for (uint32_t level = 1; level < mip_levels; ++level) {
                barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, level - 1, 1, 0, 1, ImageLayout::eTransferDstOptimal, ImageLayout::eTransferSrcOptimal, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferRead);

                ImageBlit blit{};
                blit.srcSubresource.aspectMask     = ImageAspectFlagBits::eColor;
                blit.srcSubresource.mipLevel       = level - 1;
                blit.srcSubresource.baseArrayLayer = 0;
                blit.srcSubresource.layerCount     = 1;
                blit.srcOffsets[0]                 = Offset3D{0, 0, 0};
                blit.srcOffsets[1]                 = Offset3D{int32_t(w), int32_t(h), 1};

                const uint32_t nw = std::max(1u, w / 2);
                const uint32_t nh = std::max(1u, h / 2);

                blit.dstSubresource.aspectMask     = ImageAspectFlagBits::eColor;
                blit.dstSubresource.mipLevel       = level;
                blit.dstSubresource.baseArrayLayer = 0;
                blit.dstSubresource.layerCount     = 1;
                blit.dstOffsets[0]                 = Offset3D{0, 0, 0};
                blit.dstOffsets[1]                 = Offset3D{int32_t(nw), int32_t(nh), 1};

                cmd.blitImage(*img.image, ImageLayout::eTransferSrcOptimal, *img.image, ImageLayout::eTransferDstOptimal, blit, Filter::eLinear);

                barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, level - 1, 1, 0, 1, ImageLayout::eTransferSrcOptimal, ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferRead, PipelineStageFlagBits2::eFragmentShader, AccessFlagBits2::eShaderSampledRead);

                w = nw;
                h = nh;
            }

            barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, mip_levels - 1, 1, 0, 1, ImageLayout::eTransferDstOptimal, ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eFragmentShader, AccessFlagBits2::eShaderSampledRead);
        }

        end_one_time(vkctx.graphics_queue, cmd);

        Texture2D out{};
        out.format     = format;
        out.extent     = Extent2D{desc.width, desc.height};
        out.layers     = desc.layers;
        out.mip_levels = mip_levels;
        out.image      = std::move(img.image);
        out.memory     = std::move(img.memory);
        out.view       = create_view_2d(vkctx.device, *out.image, out.format, ImageAspectFlagBits::eColor, out.mip_levels);
        out.sampler    = create_sampler_2d(vkctx.device, desc, out.mip_levels);

        return out;
    }

    Texture2DArray create_texture_2d_array_rgba8(const context::VulkanContext& vkctx, std::span<const std::byte> rgba8, Texture2DDesc desc) {
        if (desc.width == 0 || desc.height == 0 || desc.layers == 0) throw std::runtime_error("vk.texture: invalid extent/layers");
        if (desc.mip_mode != MipMode::None) throw std::runtime_error("vk.texture: mip generation for arrays not implemented");
        const size_t expected = size_t(desc.width) * size_t(desc.height) * size_t(desc.layers) * 4u;
        if (rgba8.size_bytes() != expected) throw std::runtime_error("vk.texture: rgba8 size mismatch for array");

        const Format format = desc.srgb ? Format::eR8G8B8A8Srgb : Format::eR8G8B8A8Unorm;
        const uint32_t mip_levels = 1;

        const DeviceSize upload_size = DeviceSize(rgba8.size_bytes());

        auto staging = create_buffer(vkctx.physical_device, vkctx.device, upload_size, BufferUsageFlagBits::eTransferSrc, MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent);

        {
            void* dst = staging.memory.mapMemory(0, upload_size);
            std::memcpy(dst, rgba8.data(), rgba8.size_bytes());
            staging.memory.unmapMemory();
        }

        ImageUsageFlags usage = ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled;

        auto img = create_image_2d(vkctx.physical_device, vkctx.device, desc.width, desc.height, mip_levels, desc.layers, format, usage);

        auto cmd = begin_one_time(vkctx.device, vkctx.command_pool);

        barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, 0, mip_levels, 0, desc.layers, ImageLayout::eUndefined, ImageLayout::eTransferDstOptimal, PipelineStageFlagBits2::eTopOfPipe, AccessFlags2{}, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite);

        std::vector<BufferImageCopy> regions;
        regions.reserve(desc.layers);
        const DeviceSize layer_stride = DeviceSize(size_t(desc.width) * size_t(desc.height) * 4u);
        for (uint32_t layer = 0; layer < desc.layers; ++layer) {
            BufferImageCopy bic{};
            bic.bufferOffset                    = layer_stride * layer;
            bic.bufferRowLength                 = 0;
            bic.bufferImageHeight               = 0;
            bic.imageSubresource.aspectMask     = ImageAspectFlagBits::eColor;
            bic.imageSubresource.mipLevel       = 0;
            bic.imageSubresource.baseArrayLayer = layer;
            bic.imageSubresource.layerCount     = 1;
            bic.imageOffset                     = Offset3D{0, 0, 0};
            bic.imageExtent                     = Extent3D{desc.width, desc.height, 1};
            regions.push_back(bic);
        }

        cmd.copyBufferToImage(*staging.buffer, *img.image, ImageLayout::eTransferDstOptimal, regions);

        barrier_image(cmd, *img.image, ImageAspectFlagBits::eColor, 0, mip_levels, 0, desc.layers, ImageLayout::eTransferDstOptimal, ImageLayout::eShaderReadOnlyOptimal, PipelineStageFlagBits2::eTransfer, AccessFlagBits2::eTransferWrite, PipelineStageFlagBits2::eFragmentShader, AccessFlagBits2::eShaderSampledRead);

        end_one_time(vkctx.graphics_queue, cmd);

        Texture2DArray out{};
        out.format     = format;
        out.extent     = Extent2D{desc.width, desc.height};
        out.layers     = desc.layers;
        out.mip_levels = mip_levels;
        out.image      = std::move(img.image);
        out.memory     = std::move(img.memory);
        out.view       = create_view_2d_array(vkctx.device, *out.image, out.format, ImageAspectFlagBits::eColor, out.mip_levels, out.layers);
        out.sampler    = create_sampler_2d(vkctx.device, desc, out.mip_levels);

        return out;
    }

    raii::DescriptorSetLayout make_texture_set_layout(const raii::Device& device) {

        const DescriptorSetLayoutBinding bindings[] = {{
                                                           .binding         = 0,
                                                           .descriptorType  = DescriptorType::eSampledImage,
                                                           .descriptorCount = 1,
                                                           .stageFlags      = ShaderStageFlagBits::eFragment,
                                                       },
            {
                .binding         = 1,
                .descriptorType  = DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags      = ShaderStageFlagBits::eFragment,
            }};

        const DescriptorSetLayoutCreateInfo ci{
            .bindingCount = 2,
            .pBindings    = bindings,
        };

        return raii::DescriptorSetLayout{device, ci};
    }
} // namespace vk::texture
