module;
#include <vulkan/vulkan_raii.hpp>
export module vk.frame;

import vk.context;
import vk.swapchain;
import std;

namespace vk::frame {

    export struct FrameSystem {
        uint32_t frames_in_flight = 0;

        std::vector<raii::CommandBuffer> command_buffers;

        std::vector<raii::Semaphore> image_acquired; // per-frame
        std::vector<raii::Fence> in_flight; // per-frame

        std::vector<raii::Semaphore> render_finished; // per-image
        std::vector<uint32_t> image_in_flight_frame;

        std::vector<ImageLayout> swapchain_image_layout;

        FrameSystem() {}
        ~FrameSystem()                                 = default;
        FrameSystem(FrameSystem&&) noexcept            = default;
        FrameSystem& operator=(FrameSystem&&) noexcept = default;
        FrameSystem(const FrameSystem&)                = delete;
        FrameSystem& operator=(const FrameSystem&)     = delete;
    };

    export struct AcquireResult {
        bool need_recreate{false};
        uint32_t image_index{0};
    };

    export [[nodiscard]] FrameSystem create_frame_system(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, uint32_t frames_in_flight);
    export void on_swapchain_recreated(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames);

    export [[nodiscard]] AcquireResult begin_frame(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames, uint32_t frame_index);
    export void begin_commands(FrameSystem& frames, uint32_t frame_index);
    export [[nodiscard]] bool end_frame(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames, uint32_t frame_index, uint32_t image_index, const std::span<const vk::SemaphoreSubmitInfo> extra_waits = {});

    export [[nodiscard]] raii::CommandBuffer& cmd(FrameSystem& frames, uint32_t frame_index);
    export [[nodiscard]] const raii::CommandBuffer& cmd(const FrameSystem& frames, uint32_t frame_index);
} // namespace vk::frame
