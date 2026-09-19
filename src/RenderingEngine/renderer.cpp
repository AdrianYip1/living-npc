#include "renderer.hpp"
#include "Model/modelLoading.hpp"
#include "Helpers/buffers.hpp"
#include "stb_image.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <GLFW/glfw3.h>

#include <enginemath/mat4.hpp>
#include <fstream>
#include <chrono>
#include <cstring>
#include <future>
#include <vector>
#include <iostream>

HTN::Renderer::Renderer(Window& _window, Camera& _camera) :
	window(_window),
	camera(_camera),
	device(_window),
	pipeline(device, "shaders/shader.vert.spv", "shaders/shader.frag.spv"),
	scenePipeline(device, "shaders/shader.vert.spv", "shaders/shader.frag.spv",
				  pipeline.getRenderpass(), pipeline.getUboSetLayout(), VK_CULL_MODE_BACK_BIT),
	skyboxPipeline(device, "shaders/skybox.vert.spv", "shaders/skybox.frag.spv",
				   pipeline.getRenderpass()),
	drawing(device, pipeline),
	uniform(device) {

	fModel fmodel;
	Loader::loadModel("models/combined.gltf", fmodel, skeleton);

	inverseBindMatrices = fmodel.inverseBindMatrix;
	Model::createModel(device, std::move(fmodel), &model);
	loadScene();
	createTextures();
	createCubemap();
	createDescriptors();
	initImGui();
	createSyncObjects();

	skeleton.loadAnimation(AnimState::IDLE, "models/idle_anim.gltf");
	skeleton.loadAnimation(AnimState::TALK, "models/talking_anim.gltf",
		{"Head", "Neck", "HeadTop_End"});

	u32 count = npcCount();
	faces.resize(count);
	faceWeights.resize(count);
	npcTransforms.resize(count, enginemath::Mat4::identity());
	npcAnimStates.resize(count, AnimState::IDLE);
	npcPrevAnimStates.resize(count, AnimState::IDLE);
	npcAnimTimes.resize(count, 0.0f);
	npcBlendTimers.resize(count, 0.0f);
	f32 spacing = 1.5f;
	f32 startX = -spacing * (count - 1) * 0.5f;
	for (u32 i = 0; i < count; i++) {
		faceWeights[i].assign(MAX_WEIGHTS, 0.0f);
		npcTransforms[i] = enginemath::Mat4::translationM(startX + i * spacing, 0.0f, 0.0f);
		faces[i] = std::make_unique<faceAnim>();
		faces[i]->setupSpeech();
	}

	const WorldBounds& b = WORLD_BOUNDS;
	std::string dir = CONVO_LOG_DIR;
	std::ofstream boundsOut(dir + "/bounds.json", std::ios::trunc);
	if (boundsOut) {
		boundsOut << "{\"minX\":" << b.minX << ",\"maxX\":" << b.maxX
				  << ",\"minZ\":" << b.minZ << ",\"maxZ\":" << b.maxZ
				  << ",\"obstacles\":[";
		for (size_t i = 0; i < OBSTACLES.size(); i++) {
			if (i) boundsOut << ",";
			boundsOut << "{\"x\":" << OBSTACLES[i].x
					  << ",\"z\":" << OBSTACLES[i].z
					  << ",\"r\":" << OBSTACLES[i].radius << "}";
		}
		boundsOut << "]}";
	}
}

HTN::Renderer::~Renderer() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	textures.clear();
	for (u32 i = 0; i < SKYBOX_COUNT; i++) {
		vkDestroySampler(device.getDevice(), cubemapSamplers[i], nullptr);
		vkDestroyImageView(device.getDevice(), cubemapViews[i], nullptr);
		vkDestroyImage(device.getDevice(), cubemapImages[i], nullptr);
		vkFreeMemory(device.getDevice(), cubemapMemories[i], nullptr);
	}
	vkDestroyDescriptorPool(device.getDevice(), imguiPool, nullptr);
	vkDestroyDescriptorPool(device.getDevice(), descriptorPool, nullptr);
	if (skyboxPool != VK_NULL_HANDLE)
		vkDestroyDescriptorPool(device.getDevice(), skyboxPool, nullptr);

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

	camera.setAspect(static_cast<f32>(device.getExtent().width) / static_cast<f32>(device.getExtent().height));

	static auto startTime = std::chrono::steady_clock::now();
	f32 elapsed = std::chrono::duration<f32>(std::chrono::steady_clock::now() - startTime).count();

	UBO ubo{};
	ubo.view = camera.getView();
	ubo.proj = camera.getProj();
	ubo.time = elapsed;

	u32 count = npcCount();
	f32 dt = animClock.elapsedMs() / 1000.0f;
	animClock.resetTime();

	for (u32 s = 0; s < count; s++) {
		npcAnimTimes[s] += dt;
		faceWeights[s] = faces[s]->sample();
		uniform.updateWeightBuffer(currentFrame, s, faceWeights[s]);
	}

	std::vector<enginemath::Mat4> allJoints;
	allJoints.reserve(MAX_JOINTS * count);
	for (u32 i = 0; i < count; i++) {
		auto pal = skeleton.computePalette(npcAnimStates[i], npcAnimTimes[i], inverseBindMatrices);
		if (npcBlendTimers[i] > 0.0f) {
			npcBlendTimers[i] -= dt;
			f32 t = 1.0f - npcBlendTimers[i] / BLEND_DURATION;
			if (t > 1.0f) t = 1.0f;
			auto prev = skeleton.computePalette(npcPrevAnimStates[i], npcAnimTimes[i], inverseBindMatrices);
			for (u32 j = 0; j < pal.size(); j++)
				for (int c = 0; c < 4; c++)
					for (int r = 0; r < 4; r++)
						pal[j].m[c][r] = prev[j].m[c][r] * (1.0f - t) + pal[j].m[c][r] * t;
		}
		allJoints.insert(allJoints.end(), pal.begin(), pal.end());
	}

	uniform.updateUniformBuffer(currentFrame, ubo);
	uniform.updateLightBuffer(currentFrame, light);
	uniform.updateJointBuffer(currentFrame, allJoints);

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const ImGuiViewport* vp = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x, vp->WorkPos.y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.35f, vp->WorkSize.y * 0.25f), ImGuiCond_Always);
	ImGui::Begin("UI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
	ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color", &light.color.x);

	static char ttsText[512] = "Hello, I am a living NPC. Nice to meet you!";
	ImGui::InputText("Text", ttsText, sizeof(ttsText));
	if (ImGui::Button("Speak") && !faces[0]->isBusy()) {
		faces[0]->startSpeaking(std::string(ttsText));
	}
	ImGui::SameLine();
	if (faces[0]->isBusy()) ImGui::Text("Speaking...");

	ImGui::End();

	ImGui::Render();

	drawing.recordCommandBuffer(drawing.getCommandBuffer(currentFrame), swapchainImageIndex,
								materialSets, currentFrame, model, npcTransforms,
								hasScene ? &sceneModel : nullptr,
								hasScene ? &scenePipeline : nullptr,
								&skyboxPipeline, skyboxSets);

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

void HTN::Renderer::loadScene() {
	const std::string scenePath = "models/scene_opt/background3.gltf";
	std::ifstream test(scenePath);
	if (!test.good()) return;
	test.close();

	fModel sceneFmodel;
	Skeleton throwaway;
	Loader::loadModel(scenePath, sceneFmodel, throwaway, true);
	collisionMesh.append(sceneFmodel.collisionVertices, sceneFmodel.collisionIndices);
	groundGrid.build(collisionMesh);
	Model::createModel(device, std::move(sceneFmodel), &sceneModel);
	hasScene = true;
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

void HTN::Renderer::createTextures() {
	struct TextureJob { std::string key; std::string path; };
	std::vector<TextureJob> jobs;

	for (const submesh& s : model.getPrimitives()) {
		if (!s.textureUri.empty() && textures.find(s.textureUri) == textures.end()) {
			textures[s.textureUri] = nullptr;
			jobs.push_back({s.textureUri, "models/" + s.textureUri});
		}
	}
	if (hasScene) {
		for (const submesh& s : sceneModel.getPrimitives()) {
			if (!s.textureUri.empty() && textures.find(s.textureUri) == textures.end()) {
				textures[s.textureUri] = nullptr;
				jobs.push_back({s.textureUri, "models/scene_opt/" + s.textureUri});
			}
		}
	}
	if (textures.find("") == textures.end()) {
		textures[""] = nullptr;
		jobs.push_back({"", "models/white.png"});
	}

	std::vector<std::future<DecodedImage>> futures;
	futures.reserve(jobs.size());
	for (const auto& job : jobs) {
		std::string p = job.path;
		futures.push_back(std::async(std::launch::async, [p] {
			return DecodedImage::fromFile(p);
		}));
	}

	for (size_t i = 0; i < jobs.size(); i++) {
		DecodedImage decoded = futures[i].get();
		if (!decoded)
			throw std::runtime_error("ERROR: Failed to load texture image: " + jobs[i].path);
		textures[jobs[i].key] = std::make_unique<Texture>(device, std::move(decoded));
	}
}

void HTN::Renderer::createCubemap() {
	cubemapImages.resize(SKYBOX_COUNT);
	cubemapMemories.resize(SKYBOX_COUNT);
	cubemapViews.resize(SKYBOX_COUNT);
	cubemapSamplers.resize(SKYBOX_COUNT);

	for (u32 s = 0; s < SKYBOX_COUNT; s++) {
		int w, h, ch;
		stbi_uc* pixels[6];
		for (int j = 0; j < 6; j++) {
			pixels[j] = stbi_load(cubemapTexturePaths[s][j], &w, &h, &ch, STBI_rgb_alpha);
			if (!pixels[j]) throw std::runtime_error("Failed to load skybox face");
		}

		VkDeviceSize faceSize = w * h * 4;
		VkDeviceSize cubeTotalSize = faceSize * 6;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		Buffer::createBuffer(device, cubeTotalSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device.getDevice(), stagingBufferMemory, 0, cubeTotalSize, 0, &data);
		for (int j = 0; j < 6; j++) {
			memcpy(static_cast<char*>(data) + j * faceSize, pixels[j], faceSize);
			stbi_image_free(pixels[j]);
		}
		vkUnmapMemory(device.getDevice(), stagingBufferMemory);

		VkImageCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
		ci.imageType = VK_IMAGE_TYPE_2D;
		ci.format = VK_FORMAT_R8G8B8A8_SRGB;
		ci.extent = {static_cast<u32>(w), static_cast<u32>(h), 1};
		ci.mipLevels = 1;
		ci.arrayLayers = 6;
		ci.samples = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling = VK_IMAGE_TILING_OPTIMAL;
		ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		vkCreateImage(device.getDevice(), &ci, nullptr, &cubemapImages[s]);

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device.getDevice(), cubemapImages[s], &memReq);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = Buffer::findMemoryType(device, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(device.getDevice(), &allocInfo, nullptr, &cubemapMemories[s]);
		vkBindImageMemory(device.getDevice(), cubemapImages[s], cubemapMemories[s], 0);

		VkCommandBuffer cmd = Buffer::beginSingleTimeCommands(device);

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = cubemapImages[s];
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 6;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		VkBufferImageCopy regions[6]{};
		for (int j = 0; j < 6; j++) {
			regions[j].bufferOffset = j * faceSize;
			regions[j].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			regions[j].imageSubresource.mipLevel = 0;
			regions[j].imageSubresource.baseArrayLayer = j;
			regions[j].imageSubresource.layerCount = 1;
			regions[j].imageExtent = {static_cast<u32>(w), static_cast<u32>(h), 1};
		}
		vkCmdCopyBufferToImage(cmd, stagingBuffer, cubemapImages[s],
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		Buffer::endSingleTimeCommands(device, cmd);

		vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
		vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);

		VkImageViewCreateInfo viewCI{};
		viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewCI.image = cubemapImages[s];
		viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewCI.format = VK_FORMAT_R8G8B8A8_SRGB;
		viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewCI.subresourceRange.baseMipLevel = 0;
		viewCI.subresourceRange.levelCount = 1;
		viewCI.subresourceRange.baseArrayLayer = 0;
		viewCI.subresourceRange.layerCount = 6;
		vkCreateImageView(device.getDevice(), &viewCI, nullptr, &cubemapViews[s]);

		VkSamplerCreateInfo samplerCI{};
		samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCI.maxLod = 0.0f;
		samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		vkCreateSampler(device.getDevice(), &samplerCI, nullptr, &cubemapSamplers[s]);
	}
}

void HTN::Renderer::createDescriptors() {
	u32 materialCount = static_cast<u32>(textures.size());

	Descriptor::createDescriptorPool(device,
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
		descriptorPool, materialCount);

	VkBuffer deltasBuffer = model.getDeltasBuffer();
	VkBuffer instanceBuf = hasScene ? sceneModel.getInstanceBuffer() : model.getInstanceBuffer();

	for (auto& [uri, tex] : textures) {
		VkDescriptorImageInfo imageInfo{};
		imageInfo.sampler = tex->getSampler();
		imageInfo.imageView = tex->getImageView();
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		Descriptor::createDescriptorSets(device, pipeline.getUboSetLayout(), descriptorPool,
			{0, 1, 2, 3, 4, 5, 6},
			{uniform.getUniformBuffers(), uniform.getLightUniformBuffers(),
			 {deltasBuffer, deltasBuffer}, uniform.getWeightBuffers(),
			 uniform.getJointBuffers(), {}, {instanceBuf, instanceBuf}},
			{{}, {}, {}, {}, {}, imageInfo, {}},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
			materialSets[uri]);
	}

	Descriptor::createDescriptorPool(device,
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
		skyboxPool);

	VkDescriptorImageInfo cubemapInfo[SKYBOX_COUNT];
	for (u32 i = 0; i < SKYBOX_COUNT; i++) {
		cubemapInfo[i].sampler = cubemapSamplers[i];
		cubemapInfo[i].imageView = cubemapViews[i];
		cubemapInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	Descriptor::createDescriptorSets(device, skyboxPipeline.getUboSetLayout(), skyboxPool,
		{0, 1, 2, 3},
		{uniform.getUniformBuffers(), {}, {}, {}},
		{{}, cubemapInfo[0], cubemapInfo[1], cubemapInfo[2]},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
		skyboxSets);
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

void HTN::Renderer::setNPCAnimState(u32 slot, AnimState state) {
	if (npcAnimStates[slot] != state) {
		npcPrevAnimStates[slot] = npcAnimStates[slot];
		npcAnimStates[slot] = state;
		npcBlendTimers[slot] = BLEND_DURATION;
	}
}

bool HTN::Renderer::anyBusy() const {
	for (auto& f : faces)
		if (f->isBusy()) return true;
	return false;
}

void HTN::Renderer::wait() {
	vkDeviceWaitIdle(device.getDevice());
}
