module;

#include <cuda_runtime.h>

#include <vulkan/vulkan_raii.hpp>

export module viz.snapshot;

import app;
import std;
import vk.context;
import vk.memory;

export namespace viz::snapshot {

    struct CaptureRequest {
        app::GridShape grid{};
        app::FieldFormat field_format = app::FieldFormat::Scalar1F32;
        app::FieldSemantic semantic   = app::FieldSemantic::GenericScalar;
        std::string_view label{};
        bool export_velocity_host     = false;
        uint32_t velocity_components  = 3;
    };

    struct CaptureResources {
        void* field_cuda_ptr                             = nullptr;
        void* velocity_cuda_ptr                          = nullptr;
        float* velocity_host_ptr                         = nullptr;
        cudaExternalSemaphore_t external_semaphore       = nullptr;
        uint64_t ready_generation                        = 0;
    };

    struct SnapshotStats {
        uint64_t field_bytes          = 0;
        uint64_t velocity_bytes       = 0;
        uint64_t snapshot_generation  = 0;
        uint64_t submit_serial        = 0;
        uint32_t steps_since_snapshot = 0;
        int active_slot               = -1;
        double last_snapshot_ms       = 0.0;
        double average_snapshot_ms    = 0.0;
        uint64_t snapshot_count       = 0;
    };

    class SnapshotSet {
    public:
        SnapshotSet() = default;
        ~SnapshotSet();

        SnapshotSet(const SnapshotSet&)                = delete;
        SnapshotSet& operator=(const SnapshotSet&)     = delete;
        SnapshotSet(SnapshotSet&&) noexcept            = delete;
        SnapshotSet& operator=(SnapshotSet&&) noexcept = delete;

        void reset(const vk::context::VulkanContext& vkctx, std::span<vk::raii::DescriptorSet> descriptor_sets, const CaptureRequest& request);
        void destroy();

        [[nodiscard]] bool matches(const CaptureRequest& request) const;
        [[nodiscard]] int find_available_slot(uint32_t frames_in_flight) const;
        [[nodiscard]] CaptureResources begin_capture(int slot_index);
        void complete_capture(int slot_index, const CaptureRequest& request, double capture_ms);
        [[nodiscard]] std::optional<app::VisualizationSnapshotView> active_snapshot(const app::ColliderOverlay& collider) const;
        void mark_submitted(uint32_t frames_in_flight);
        void reset_snapshot_interval();

        [[nodiscard]] SnapshotStats& stats();
        [[nodiscard]] const SnapshotStats& stats() const;

    private:
        struct Slot {
            vk::raii::Buffer buffer{nullptr};
            vk::raii::DeviceMemory memory{nullptr};
            vk::raii::Semaphore timeline_semaphore{nullptr};
            vk::raii::DescriptorSet descriptor_set{nullptr};
            cudaExternalMemory_t external_memory       = nullptr;
            cudaExternalSemaphore_t external_semaphore = nullptr;
            void* field_cuda_ptr                       = nullptr;
            void* velocity_cuda_ptr                    = nullptr;
            std::vector<float> velocity_host{};
            uint64_t ready_generation                  = 0;
            uint64_t last_used_submit_serial           = 0;
            app::GridShape grid{};
            app::FieldFormat field_format              = app::FieldFormat::Scalar1F32;
            app::FieldSemantic semantic                = app::FieldSemantic::GenericScalar;
            std::string_view label{};
            uint32_t velocity_components               = 0;
        };

        std::vector<Slot> slots_{};
        CaptureRequest request_{};
        SnapshotStats stats_{};
    };

} // namespace viz::snapshot
