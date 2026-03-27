module;

#if defined(_WIN32)
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include <cuda_runtime.h>

#include <vulkan/vulkan_raii.hpp>

module viz.snapshot;

import std;

namespace viz::snapshot {

    namespace {

        [[nodiscard]] uint64_t field_bytes_for(const CaptureRequest& request) {
            const uint64_t nx = request.grid.nx;
            const uint64_t ny = request.grid.ny;
            const uint64_t nz = static_cast<uint64_t>((std::max)(request.grid.nz, 1u));
            return nx * ny * nz * static_cast<uint64_t>((std::max)(request.field_component_count, 1u)) * sizeof(float);
        }

        [[nodiscard]] uint64_t velocity_bytes_for(const CaptureRequest& request) {
            if (!request.export_velocity_host) return 0;
            const uint64_t nx = request.grid.nx;
            const uint64_t ny = request.grid.ny;
            const uint64_t nz = static_cast<uint64_t>((std::max)(request.grid.nz, 1u));
            return nx * ny * nz * 3ull * sizeof(float);
        }

    } // namespace

    SnapshotSet::~SnapshotSet() {
        destroy();
    }

    void SnapshotSet::destroy() {
        for (auto& slot : slots_) {
            if (slot.field_cuda_ptr != nullptr) cudaFree(slot.field_cuda_ptr);
            if (slot.velocity_cuda_ptr != nullptr) cudaFree(slot.velocity_cuda_ptr);
            if (slot.external_semaphore != nullptr) cudaDestroyExternalSemaphore(slot.external_semaphore);
            if (slot.external_memory != nullptr) cudaDestroyExternalMemory(slot.external_memory);
            slot = {};
        }
        slots_.clear();
        request_ = {};
        stats_ = {};
    }

    bool SnapshotSet::matches(const CaptureRequest& request) const {
        return !slots_.empty()
            && request_.grid.nx == request.grid.nx
            && request_.grid.ny == request.grid.ny
            && request_.grid.nz == request.grid.nz
            && request_.grid.cell_size == request.grid.cell_size
            && request_.field_component_count == request.field_component_count
            && request_.export_velocity_host == request.export_velocity_host;
    }

    void SnapshotSet::reset(const vk::context::VulkanContext& vkctx, const std::span<vk::raii::DescriptorSet> descriptor_sets, const CaptureRequest& request) {
        destroy();
        request_ = request;
        stats_.field_bytes = field_bytes_for(request);
        stats_.velocity_bytes = velocity_bytes_for(request);

        slots_.reserve(descriptor_sets.size());
        for (size_t slot_index = 0; slot_index < descriptor_sets.size(); ++slot_index) {
            Slot slot{};
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
            slot.timeline_semaphore = vk::raii::Semaphore{vkctx.device, semaphore_ci};

            vk::ExternalMemoryBufferCreateInfo external_buffer_ci{
                .handleTypes = memory_handle_type,
            };
            vk::BufferCreateInfo buffer_ci{
                .pNext = &external_buffer_ci,
                .size = stats_.field_bytes,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                .sharingMode = vk::SharingMode::eExclusive,
            };
            slot.buffer = vk::raii::Buffer{vkctx.device, buffer_ci};
            const vk::MemoryRequirements requirements = slot.buffer.getMemoryRequirements();
            vk::ExportMemoryAllocateInfo export_memory_ci{
                .handleTypes = memory_handle_type,
            };
            vk::MemoryAllocateInfo alloc_ci{
                .pNext = &export_memory_ci,
                .allocationSize = requirements.size,
                .memoryTypeIndex = vk::memory::find_memory_type(vkctx.physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
            };
            slot.memory = vk::raii::DeviceMemory{vkctx.device, alloc_ci};
            slot.buffer.bindMemory(*slot.memory, 0);

#if defined(_WIN32)
            vk::MemoryGetWin32HandleInfoKHR memory_handle_info{
                .memory = *slot.memory,
                .handleType = memory_handle_type,
            };
            HANDLE memory_handle = vkctx.device.getMemoryWin32HandleKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueWin32,
                .handle = { .win32 = { .handle = memory_handle, }, },
                .size = requirements.size,
            };
            if (cudaImportExternalMemory(&slot.external_memory, &external_memory_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalMemory failed");
            CloseHandle(memory_handle);

            vk::SemaphoreGetWin32HandleInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            HANDLE semaphore_handle = vkctx.device.getSemaphoreWin32HandleKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32,
                .handle = { .win32 = { .handle = semaphore_handle, }, },
            };
            if (cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalSemaphore failed");
            CloseHandle(semaphore_handle);
#else
            vk::MemoryGetFdInfoKHR memory_handle_info{
                .memory = *slot.memory,
                .handleType = memory_handle_type,
            };
            const int memory_fd = vkctx.device.getMemoryFdKHR(memory_handle_info);
            cudaExternalMemoryHandleDesc external_memory_desc{
                .type = cudaExternalMemoryHandleTypeOpaqueFd,
                .handle = { .fd = memory_fd, },
                .size = requirements.size,
            };
            if (cudaImportExternalMemory(&slot.external_memory, &external_memory_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalMemory failed");

            vk::SemaphoreGetFdInfoKHR semaphore_handle_info{
                .semaphore = *slot.timeline_semaphore,
                .handleType = semaphore_handle_type,
            };
            const int semaphore_fd = vkctx.device.getSemaphoreFdKHR(semaphore_handle_info);
            cudaExternalSemaphoreHandleDesc external_semaphore_desc{
                .type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
                .handle = { .fd = semaphore_fd, },
            };
            if (cudaImportExternalSemaphore(&slot.external_semaphore, &external_semaphore_desc) != cudaSuccess) throw std::runtime_error("cudaImportExternalSemaphore failed");
#endif

            cudaExternalMemoryBufferDesc buffer_desc{
                .offset = 0,
                .size = stats_.field_bytes,
            };
            if (cudaExternalMemoryGetMappedBuffer(&slot.field_cuda_ptr, slot.external_memory, &buffer_desc) != cudaSuccess) throw std::runtime_error("cudaExternalMemoryGetMappedBuffer failed");
            if (stats_.velocity_bytes != 0) {
                if (cudaMalloc(&slot.velocity_cuda_ptr, stats_.velocity_bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc velocity snapshot failed");
                slot.velocity_host.resize(static_cast<size_t>(stats_.velocity_bytes / sizeof(float)));
            }

            vk::DescriptorBufferInfo field_info{
                .buffer = *slot.buffer,
                .offset = 0,
                .range = stats_.field_bytes,
            };
            vk::WriteDescriptorSet field_write{
                .dstSet = *slot.descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &field_info,
            };
            vkctx.device.updateDescriptorSets(field_write, {});
            slots_.push_back(std::move(slot));
        }
    }

    int SnapshotSet::find_available_slot(const uint32_t frames_in_flight) const {
        for (uint32_t slot_index = 0; slot_index < slots_.size(); ++slot_index) {
            const auto& slot = slots_[slot_index];
            if (static_cast<int>(slot_index) == stats_.active_slot) continue;
            if (slot.ready_generation != 0 && stats_.submit_serial < slot.last_used_submit_serial + frames_in_flight + 1) continue;
            return static_cast<int>(slot_index);
        }
        return -1;
    }

    CaptureResources SnapshotSet::begin_capture(const int slot_index) {
        auto& slot = slots_.at(static_cast<size_t>(slot_index));
        return CaptureResources{
            .field_cuda_ptr = slot.field_cuda_ptr,
            .velocity_cuda_ptr = slot.velocity_cuda_ptr,
            .velocity_host_ptr = slot.velocity_host.empty() ? nullptr : slot.velocity_host.data(),
            .external_semaphore = slot.external_semaphore,
            .ready_generation = stats_.snapshot_generation + 1,
        };
    }

    void SnapshotSet::complete_capture(const int slot_index, const CaptureRequest& request, const double capture_ms) {
        auto& slot = slots_.at(static_cast<size_t>(slot_index));
        slot.ready_generation = stats_.snapshot_generation + 1;
        slot.grid = request.grid;
        slot.field_component_count = request.field_component_count;
        slot.semantic = request.semantic;
        slot.label = request.label;
        slot.has_velocity_host = request.export_velocity_host;
        stats_.snapshot_generation = slot.ready_generation;
        stats_.active_slot = slot_index;
        stats_.steps_since_snapshot = 0;
        stats_.last_snapshot_ms = capture_ms;
        ++stats_.snapshot_count;
        stats_.average_snapshot_ms += (capture_ms - stats_.average_snapshot_ms) / static_cast<double>(stats_.snapshot_count);
    }

    std::optional<app::VisualizationSnapshotView> SnapshotSet::active_snapshot(const app::ColliderOverlay& collider) const {
        if (stats_.active_slot < 0) return std::nullopt;
        const auto& slot = slots_.at(static_cast<size_t>(stats_.active_slot));
        return app::VisualizationSnapshotView{
            .grid = slot.grid,
            .field = {
                .descriptor_set = *slot.descriptor_set,
                .timeline_semaphore = slot.external_semaphore != nullptr ? *slot.timeline_semaphore : vk::Semaphore{},
                .ready_generation = slot.ready_generation,
                .component_count = slot.field_component_count,
                .semantic = slot.semantic,
                .label = slot.label,
            },
            .collider = collider,
            .velocity = {
                .data = slot.has_velocity_host ? slot.velocity_host.data() : nullptr,
            },
        };
    }

    void SnapshotSet::mark_submitted(const uint32_t) {
        const uint64_t next_submit_serial = stats_.submit_serial + 1;
        if (stats_.active_slot >= 0) slots_[static_cast<size_t>(stats_.active_slot)].last_used_submit_serial = next_submit_serial;
        stats_.submit_serial = next_submit_serial;
    }

    void SnapshotSet::reset_snapshot_interval() {
        stats_.steps_since_snapshot = 0;
    }

    SnapshotStats& SnapshotSet::stats() {
        return stats_;
    }

    const SnapshotStats& SnapshotSet::stats() const {
        return stats_;
    }

} // namespace viz::snapshot
