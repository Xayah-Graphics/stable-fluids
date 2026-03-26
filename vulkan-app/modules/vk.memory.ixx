module;
#include <vulkan/vulkan_raii.hpp>
export module vk.memory;


namespace vk::memory {
    export struct Buffer {
        raii::Buffer buffer{nullptr};
        raii::DeviceMemory memory{nullptr};
        DeviceSize size{0};
    };

    export struct MeshGPU {
        Buffer vertex_buffer;
        Buffer index_buffer;
        uint32_t index_count{0};
    };

    export template <typename VertexT>
    struct MeshCPU {
        std::vector<VertexT> vertices;
        std::vector<uint32_t> indices;
    };

    export [[nodiscard]] uint32_t find_memory_type(const raii::PhysicalDevice& physical_device, uint32_t type_bits, MemoryPropertyFlags required);
    export [[nodiscard]] Buffer create_buffer(const raii::PhysicalDevice& physical_device, const raii::Device& device, DeviceSize size, BufferUsageFlags usage, MemoryPropertyFlags mem_props);
    export void write_mapped(const Buffer& dst, std::span<const std::byte> bytes);
    export void copy_buffer_immediate(const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, const Buffer& src, const Buffer& dst, DeviceSize size);
    export [[nodiscard]] Buffer upload_to_device_local_buffer(const raii::PhysicalDevice& physical_device, const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, std::span<const std::byte> bytes, BufferUsageFlags final_usage);
    export template <typename VertexT>
    [[nodiscard]] MeshGPU upload_mesh(const raii::PhysicalDevice& physical_device, const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, const MeshCPU<VertexT>& mesh);
} // namespace vk::memory

template <typename VertexT>
vk::memory::MeshGPU vk::memory::upload_mesh(const raii::PhysicalDevice& physical_device, const raii::Device& device, const raii::CommandPool& command_pool, const raii::Queue& queue, const MeshCPU<VertexT>& mesh) {
    static_assert(std::is_standard_layout_v<VertexT>);
    static_assert(std::is_trivially_copyable_v<VertexT>);

    if (mesh.vertices.empty() || mesh.indices.empty()) throw std::runtime_error("MeshCPU is empty");

    const std::span<const std::byte> vertex_bytes{reinterpret_cast<const std::byte*>(mesh.vertices.data()), mesh.vertices.size() * sizeof(VertexT)};
    const std::span<const std::byte> index_bytes{reinterpret_cast<const std::byte*>(mesh.indices.data()), mesh.indices.size() * sizeof(uint32_t)};

    MeshGPU gpu{};
    gpu.vertex_buffer = upload_to_device_local_buffer(physical_device, device, command_pool, queue, vertex_bytes, BufferUsageFlagBits::eVertexBuffer);
    gpu.index_buffer  = upload_to_device_local_buffer(physical_device, device, command_pool, queue, index_bytes, BufferUsageFlagBits::eIndexBuffer);
    gpu.index_count   = static_cast<uint32_t>(mesh.indices.size());
    return gpu;
}
