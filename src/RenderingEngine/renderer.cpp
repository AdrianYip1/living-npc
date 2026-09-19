#include "renderer.hpp"

HTN::Renderer::Renderer(Window& _window) :
	window(_window),
	device(_window),
	pipeline(device, "shaders/shader.vert.spv", "shaders/shader.frag.spv"),
	drawing(device, pipeline) {
	createSyncObjects();
}

HTN::Renderer::~Renderer() {
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device.getDevice(), imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(device.getDevice(), inFlightFences[i], nullptr);
	}
	for (size_t i = 0; i < renderFinishedSemaphores.size(); i++) {
		vkDestroySemaphore(device.getDevice(), renderFinishedSemaphores[i], nullptr);
	}
}

void HTN::Renderer::drawFrame() {
	vkWaitForFences(device.getDevice(), 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

	u32 swapchainImageIndex;
	VkResult result = vkAcquireNextImageKHR(device.getDevice(), device.getSwapchain(), UINT64_MAX,
											imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &swapchainImageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		device.recreateSwapchain();
		drawing.createFramebuffer();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("ERROR: Failed to acquire swap chain image");
	}

	vkResetFences(device.getDevice(), 1, &inFlightFences[currentFrame]);
	vkResetCommandBuffer(drawing.getCommandBuffer(currentFrame), 0);

	drawing.recordCommandBuffer(drawing.getCommandBuffer(currentFrame), swapchainImageIndex);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = drawing.getCommandBuffer(currentFrame);
	submitInfo.pCommandBuffers = &commandBuffer;

	VkSemaphore signalSemaphore[] = { renderFinishedSemaphores[swapchainImageIndex] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphore;

	if (vkQueueSubmit(device.getGraphicsQueue(), 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to submit draw command buffer");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphore;

	VkSwapchainKHR swapchains[] = { device.getSwapchain() };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapchains;
	presentInfo.pImageIndices = &swapchainImageIndex;

	result = vkQueuePresentKHR(device.getPresentQueue(), &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.getFramebufferResized()) {
		window.setFramebufferResized(false);
		device.recreateSwapchain();
		drawing.createFramebuffer();
	}
	else if (result != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to present swap chain image");
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void HTN::Renderer::createSyncObjects() {
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(device.getSwapchainImageViews().size());
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(device.getDevice(), &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: Failed to create synchronization objects");
		}
	}
	for (size_t i = 0; i < device.getSwapchainImageViews().size(); i++) {
		if (vkCreateSemaphore(device.getDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: Failed to create synchronization object");
		}
	}
}

void HTN::Renderer::wait() {
	vkDeviceWaitIdle(device.getDevice());
}
