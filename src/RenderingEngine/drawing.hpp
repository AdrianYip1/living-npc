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

	private:
		Device& device;
		Pipeline& pipeline;

		std::vector<VkFramebuffer> swapchainFramebuffers;

		void createFramebuffer();
	};
} // namespace HTN
