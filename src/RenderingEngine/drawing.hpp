#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "pipeline.hpp"

#include <vector>
#include <array>

namespace HTN {
	class Drawing {
	public:
		Drawing(Device& _device, Pipeline& _pipeline);
		~Drawing();
		Drawing(const Drawing&) = delete;
		Drawing& operator=(const Drawing&) = delete;

		std::vector<VkFramebuffer>& getFramebuffers() { return swapchainFramebuffers; }
		VkCommandBuffer getCommandBuffer(u32 frame) { return commandBuffers[frame]; }
		void recordCommandBuffer(VkCommandBuffer commandBuffer, u32 swapchainImageIndex);
		void createFramebuffer();

	private:
		Device& device;
		Pipeline& pipeline;

		std::vector<VkFramebuffer> swapchainFramebuffers;
		std::vector<VkCommandBuffer> commandBuffers;

		void createCommandBuffer();
	};
} // namespace HTN
