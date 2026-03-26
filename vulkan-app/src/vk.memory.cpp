module;
#include <vulkan/vulkan_raii.hpp>
module vk.memory;


uint32_t vk::memory::find_memory_type(const raii::PhysicalDevice& physical_device, const uint32_t type_bits, const MemoryPropertyFlags required) {
    const auto mem = physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        const bool type_ok = (type_bits & 1u << i) != 0;
        if (const bool props_ok = (mem.memoryTypes[i].propertyFlags & required) == required; type_ok && props_ok) return i;
    }
    throw std::runtime_error("No suitable memory type");
}
vk::memory::Buffer vk::memory::create_buffer(const raii::PhysicalDevice& physical_device, const raii::Device& device, const DeviceSize size, const BufferUsageFlags usage, const MemoryPropertyFlags mem_props) {
    Buffer out{};
    out.size = size;

    const BufferCreateInfo bci{
        .size        = size,
        .usage       = usage,
        .sharingMode = SharingMode::eExclusive,
    };
    out.buffer = raii::Buffer{device, bci};

    const auto req          = out.buffer.getMemoryRequirements();
    const uint32_t mem_type = find_memory_type(physical_device, req.memoryTypeBits, mem_props);

    const MemoryAllocateInfo mai{
        .allocationSize  = req.size,
        .memoryTypeIndex = mem_type,
    };
    out.memory = raii::DeviceMemory{device, mai};

    out.buffer.bindMemory(*out.memory, 0);
    return out;
}
void vk::memory::write_mapped(const Buffer& dst, const std::span<const std::byte> bytes) {
    if (bytes.size_bytes() > static_cast<size_t>(dst.size)) throw std::runtime_error("write_mapped overflow");
    void* ptr = dst.memory.mapMemory(0, dst.size);
    std::memcpy(ptr, bytes.data(), bytes.size_bytes());
    dst.memory.unmapMemory();
}
void vk::memory::copy_buffer_immediate(const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, const Buffer& src, const Buffer& dst, const DeviceSize size) {
    const CommandBufferAllocateInfo ai{
        .commandPool        = *command_pool,
        .level              = CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    const auto cmd = std::move(device.allocateCommandBuffers(ai).front());

    cmd.begin(CommandBufferBeginInfo{.flags = CommandBufferUsageFlagBits::eOneTimeSubmit});

    BufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size};
    cmd.copyBuffer(*src.buffer, *dst.buffer, {region});

    cmd.end();

    CommandBufferSubmitInfo cbsi{.commandBuffer = *cmd};
    SubmitInfo2 submit{
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &cbsi,
    };

    raii::Fence fence{device, FenceCreateInfo{}};
    queue.submit2({submit}, *fence);
    (void) device.waitForFences(*fence, true, UINT64_MAX);
    queue.waitIdle();
}
vk::memory::Buffer vk::memory::upload_to_device_local_buffer(const raii::PhysicalDevice& physical_device, const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, const std::span<const std::byte> bytes, const BufferUsageFlags final_usage) {
    const DeviceSize size = bytes.size_bytes();

    const Buffer staging = create_buffer(physical_device, device, size, BufferUsageFlagBits::eTransferSrc, MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent);
    write_mapped(staging, bytes);

    Buffer gpu = create_buffer(physical_device, device, size, final_usage | BufferUsageFlagBits::eTransferDst, MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer_immediate(device, command_pool, queue, staging, gpu, size);
    return gpu;
}
