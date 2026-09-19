#include "drawing.hpp"

HTN::Drawing::Drawing(Device& _device, Pipeline& _pipeline) :
	device(_device), pipeline(_pipeline) {
	createFramebuffer();
}

HTN::Drawing::~Drawing() {
	for (auto framebuffer : swapchainFramebuffers) {
		vkDestroyFramebuffer(device.getDevice(), framebuffer, nullptr);
	}
}

void HTN::Drawing::createFramebuffer() {
	swapchainFramebuffers.resize(device.getSwapchainImageViews().size());

	for (size_t i = 0; i < device.getSwapchainImageViews().size(); i++) {
		std::array<VkImageView, 3> attachments = {
			device.getColorImageView(),
			device.getDepthImageView(),
			device.getSwapchainImageViews()[i]
		};

		VkFramebufferCreateInfo framebufferCreateInfo{};
		framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferCreateInfo.renderPass = pipeline.getRenderpass();
		framebufferCreateInfo.attachmentCount = static_cast<u32>(attachments.size());
		framebufferCreateInfo.pAttachments = attachments.data();
		framebufferCreateInfo.width = device.getExtent().width;
		framebufferCreateInfo.height = device.getExtent().height;
		framebufferCreateInfo.layers = 1;

		if (vkCreateFramebuffer(device.getDevice(), &framebufferCreateInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: Failed to create framebuffer");
		}
	}
}
