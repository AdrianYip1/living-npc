#include "drawing.hpp"
#include <imgui.h>
#include <imgui_impl_vulkan.h>

HTN::Drawing::Drawing(Device& _device, Pipeline& _pipeline) :
	device(_device), pipeline(_pipeline) {
	createFramebuffer();
	createCommandBuffer();
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

void HTN::Drawing::createCommandBuffer() {
	commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = device.getCommandPool();
	allocInfo.commandBufferCount = static_cast<u32>(commandBuffers.size());
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

	if (vkAllocateCommandBuffers(device.getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to allocate command buffer");
	}
}

void HTN::Drawing::recordCommandBuffer(VkCommandBuffer commandBuffer, u32 swapchainImageIndex,
										VkDescriptorSet descriptorSet, Model& model) {
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to begin recording command buffer");
	}

	VkRenderPassBeginInfo renderpassBeginInfo{};
	renderpassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderpassBeginInfo.renderPass = pipeline.getRenderpass();
	renderpassBeginInfo.framebuffer = swapchainFramebuffers[swapchainImageIndex];
	renderpassBeginInfo.renderArea.extent = device.getExtent();
	renderpassBeginInfo.renderArea.offset = { 0, 0 };

	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	clearValues[1].depthStencil = { 1.0f, 0 };

	renderpassBeginInfo.clearValueCount = static_cast<u32>(clearValues.size());
	renderpassBeginInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderpassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<f32>(device.getExtent().width);
	viewport.height = static_cast<f32>(device.getExtent().height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = device.getExtent();
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getGraphicsPipeline());
	model.bind(commandBuffer);
	model.draw(commandBuffer, pipeline.getPipelineLayout(), descriptorSet, 0);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to record command buffer");
	}
}
