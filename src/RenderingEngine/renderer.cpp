#include "renderer.hpp"
#include "Model/modelLoading.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <GLFW/glfw3.h>

#include <enginemath/mat4.hpp>

HTN::Renderer::Renderer(Window& _window) :
	window(_window),
	device(_window),
	pipeline(device, "shaders/shader.vert.spv", "shaders/shader.frag.spv"),
	drawing(device, pipeline),
	uniform(device) {

	std::vector<Vertex> vertices;
	std::vector<u32> indices;
	Loader::loadModel("models/face1.gltf", vertices, indices);

	Model::createModel(device, vertices, indices, &model);
	createDescriptors();
	initImGui();
	createSyncObjects();
}

HTN::Renderer::~Renderer() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	vkDestroyDescriptorPool(device.getDevice(), imguiPool, nullptr);
	vkDestroyDescriptorPool(device.getDevice(), descriptorPool, nullptr);

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

	UBO ubo{};
	ubo.view = enginemath::Mat4::lookAtM(
		enginemath::Vec3(0.0f, 1.56f, 0.5f),
		enginemath::Vec3(0.0f, 1.56f, 0.0f),
		enginemath::Vec3(0.0f, 1.0f, 0.0f));

	f32 aspect = static_cast<f32>(device.getExtent().width) / static_cast<f32>(device.getExtent().height);
	ubo.proj = enginemath::Mat4::projectionM(0.7854f, aspect, 0.1f, 100.0f);
	ubo.proj.m[1][1] *= -1.0f;

	uniform.updateUniformBuffer(currentFrame, ubo);
	uniform.updateLightBuffer(currentFrame, light);

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const ImGuiViewport* vp = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x, vp->WorkPos.y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.35f, vp->WorkSize.y * 0.25f), ImGuiCond_Always);
	ImGui::Begin("UI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
	ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color", &light.color.x);
	ImGui::End();

	ImGui::Render();

	drawing.recordCommandBuffer(drawing.getCommandBuffer(currentFrame), swapchainImageIndex,
								descriptorSets[currentFrame], model);

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

void HTN::Renderer::createDescriptors() {
	Descriptor::createDescriptorPool(device,
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
		descriptorPool);

	Descriptor::createDescriptorSets(device,
		pipeline.getUboSetLayout(),
		descriptorPool,
		{uniform.getUniformBuffers(), uniform.getLightUniformBuffers()},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
		descriptorSets);
}

void HTN::Renderer::initImGui() {
	VkDescriptorPoolSize poolSizes[] = {
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10}
	};

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	poolInfo.maxSets = 10;
	poolInfo.poolSizeCount = 3;
	poolInfo.pPoolSizes = poolSizes;

	if (vkCreateDescriptorPool(device.getDevice(), &poolInfo, nullptr, &imguiPool) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create ImGui descriptor pool");
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(window.getWindow(), true);

	QueueFamilyIndices queueIndices = device.findQueueFamilies(device.getPhysicalDevice());

	ImGui_ImplVulkan_InitInfo initInfo{};
	initInfo.ApiVersion = VK_API_VERSION_1_0;
	initInfo.Instance = device.getInstance();
	initInfo.PhysicalDevice = device.getPhysicalDevice();
	initInfo.Device = device.getDevice();
	initInfo.QueueFamily = queueIndices.graphicsAndComputeFamily.value();
	initInfo.Queue = device.getGraphicsQueue();
	initInfo.DescriptorPool = imguiPool;
	initInfo.MinImageCount = 2;
	initInfo.ImageCount = static_cast<u32>(device.getSwapchainImageViews().size());
	initInfo.PipelineInfoMain.RenderPass = pipeline.getRenderpass();
	initInfo.PipelineInfoMain.Subpass = 0;
	initInfo.PipelineInfoMain.MSAASamples = device.getMSAASampleCount();

	ImGui_ImplVulkan_Init(&initInfo);
}

void HTN::Renderer::wait() {
	vkDeviceWaitIdle(device.getDevice());
}
