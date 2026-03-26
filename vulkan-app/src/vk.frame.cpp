module;
#include <vulkan/vulkan_raii.hpp>
module vk.frame;

import std;

namespace vk::frame {

    namespace {
        constexpr uint32_t invalid_u32 = 0xFFFF'FFFFu;

        [[nodiscard]] Semaphore acquire_semaphore(const FrameSystem& frames, uint32_t frame_index) {
            return *frames.image_acquired.at(frame_index);
        }

        [[nodiscard]] Fence in_flight_fence(const FrameSystem& frames, uint32_t frame_index) {
            return *frames.in_flight.at(frame_index);
        }

        [[nodiscard]] Semaphore render_finished_semaphore(const FrameSystem& frames, uint32_t image_index) {
            return *frames.render_finished.at(image_index);
        }

        [[nodiscard]] bool out_of_date_or_suboptimal(Result r) {
            return r == Result::eErrorOutOfDateKHR || r == Result::eSuboptimalKHR;
        }
    } // namespace

    raii::CommandBuffer& cmd(FrameSystem& frames, const uint32_t frame_index) {
        return frames.command_buffers.at(frame_index);
    }

    const raii::CommandBuffer& cmd(const FrameSystem& frames, const uint32_t frame_index) {
        return frames.command_buffers.at(frame_index);
    }

    FrameSystem create_frame_system(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, const uint32_t frames_in_flight) {
        if (frames_in_flight == 0) throw std::runtime_error("frames_in_flight must be > 0");

        FrameSystem out{};
        out.frames_in_flight = frames_in_flight;

        {
            const CommandBufferAllocateInfo ai{
                .commandPool        = *vkctx.command_pool,
                .level              = CommandBufferLevel::ePrimary,
                .commandBufferCount = frames_in_flight,
            };
            out.command_buffers = vkctx.device.allocateCommandBuffers(ai);
        }

        out.image_acquired.clear();
        out.in_flight.clear();

        out.image_acquired.reserve(frames_in_flight);
        out.in_flight.reserve(frames_in_flight);

        for (uint32_t i = 0; i < frames_in_flight; ++i) {
            out.image_acquired.emplace_back(vkctx.device, SemaphoreCreateInfo{});
            out.in_flight.emplace_back(vkctx.device, FenceCreateInfo{.flags = FenceCreateFlagBits::eSignaled});
        }

        on_swapchain_recreated(vkctx, sc, out);
        return out;
    }

    void on_swapchain_recreated(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames) {
        const auto image_count = static_cast<uint32_t>(sc.images.size());
        if (image_count == 0) throw std::runtime_error("swapchain has 0 images");

        frames.render_finished.clear();
        frames.render_finished.reserve(image_count);
        for (uint32_t i = 0; i < image_count; ++i) {
            frames.render_finished.emplace_back(vkctx.device, SemaphoreCreateInfo{});
        }

        frames.image_in_flight_frame.assign(image_count, invalid_u32);
        frames.swapchain_image_layout.assign(image_count, ImageLayout::eUndefined);
    }

    AcquireResult begin_frame(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames, const uint32_t frame_index) {
        AcquireResult out{};

        const Fence fence = in_flight_fence(frames, frame_index);

        (void) vkctx.device.waitForFences(fence, VK_TRUE, UINT64_MAX);
        vkctx.device.resetFences(fence);

        const ResultValue<uint32_t> acquired = sc.handle.acquireNextImage(UINT64_MAX, acquire_semaphore(frames, frame_index), nullptr);

        if (acquired.result == Result::eErrorOutOfDateKHR) {
            out.need_recreate = true;
            return out;
        }

        if (acquired.result != Result::eSuccess && acquired.result != Result::eSuboptimalKHR) {
            throw std::runtime_error("acquireNextImage failed");
        }

        out.image_index   = acquired.value;
        out.need_recreate = acquired.result == Result::eSuboptimalKHR;

        const uint32_t prev_frame = frames.image_in_flight_frame.at(out.image_index);
        if (prev_frame != invalid_u32) {
            (void) vkctx.device.waitForFences(in_flight_fence(frames, prev_frame), VK_TRUE, UINT64_MAX);
        }

        frames.image_in_flight_frame.at(out.image_index) = frame_index;

        return out;
    }

    void begin_commands(FrameSystem& frames, const uint32_t frame_index) {
        auto& c = cmd(frames, frame_index);
        c.reset();
        c.begin(CommandBufferBeginInfo{.flags = CommandBufferUsageFlagBits::eOneTimeSubmit});
    }

    bool end_frame(const context::VulkanContext& vkctx, const swapchain::Swapchain& sc, FrameSystem& frames, const uint32_t frame_index, const uint32_t image_index, const std::span<const SemaphoreSubmitInfo> extra_waits) {
        auto& c = cmd(frames, frame_index);
        c.end();

        const Semaphore wait_sem   = acquire_semaphore(frames, frame_index);
        const Semaphore signal_sem = render_finished_semaphore(frames, image_index);
        const Fence fence          = in_flight_fence(frames, frame_index);

        std::vector<SemaphoreSubmitInfo> waits;
        waits.reserve(1 + extra_waits.size());
        waits.push_back(SemaphoreSubmitInfo{
            .semaphore = wait_sem,
            .stageMask = PipelineStageFlagBits2::eAllCommands,
        });
        for (const auto& w : extra_waits) waits.push_back(w);

        const SemaphoreSubmitInfo signal{
            .semaphore = signal_sem,
            .stageMask = PipelineStageFlagBits2::eAllCommands,
        };

        const CommandBufferSubmitInfo cb{
            .commandBuffer = *c,
        };

        const SubmitInfo2 submit{
            .waitSemaphoreInfoCount   = static_cast<uint32_t>(waits.size()),
            .pWaitSemaphoreInfos      = waits.data(),
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &cb,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos    = &signal,
        };

        vkctx.graphics_queue.submit2(submit, fence);

        const SwapchainKHR swapchains[] = {*sc.handle};
        const uint32_t indices[]        = {image_index};

        const PresentInfoKHR present{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &signal_sem,
            .swapchainCount     = 1,
            .pSwapchains        = swapchains,
            .pImageIndices      = indices,
        };

        try {
            const Result r = vkctx.graphics_queue.presentKHR(present);
            if (out_of_date_or_suboptimal(r)) return true;
            if (r == Result::eErrorSurfaceLostKHR) return true;
            if (r != Result::eSuccess) throw std::runtime_error("presentKHR failed");
        } catch (const OutOfDateKHRError&) {
            return true;
        } catch (const SystemError& e) {
            if (e.code().value() == static_cast<int>(Result::eErrorOutOfDateKHR)) return true;
            if (e.code().value() == static_cast<int>(Result::eSuboptimalKHR)) return true;
            throw;
        }

        return false;
    }

} // namespace vk::frame
