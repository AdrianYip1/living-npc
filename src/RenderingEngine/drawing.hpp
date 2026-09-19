#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "pipeline.hpp"
#include "Model/model.hpp"

#include <vector>
#include <array>
#include <map>
#include <string>

namespace HTN {
	class Drawing {
	public:
		Drawing(Device& _device, Pipeline& _pipeline);
		~Drawing();
		Drawing(const Drawing&) = delete;
		Drawing& operator=(const Drawing&) = delete;

		std::vector<VkFramebuffer>& getFramebuffers() { return swapchainFramebuffers; }
		VkCommandBuffer getCommandBuffer(u32 frame) { return commandBuffers[frame]; }
		void recordCommandBuffer(VkCommandBuffer commandBuffer, u32 swapchainImageIndex,
							 const std::map<std::string, std::vector<VkDescriptorSet>>& materialSets,
							 u32 currentFrame, Model& model,
							 const std::vector<enginemath::Mat4>& npcTransforms,
							 Model* sceneModel = nullptr, Pipeline* scenePipeline = nullptr,
							 Pipeline* skyboxPipeline = nullptr,
							 const std::vector<VkDescriptorSet>& skyboxSets = {});
		void createFramebuffer();

	private:
		Device& device;
		Pipeline& pipeline;

		std::vector<VkFramebuffer> swapchainFramebuffers;
		std::vector<VkCommandBuffer> commandBuffers;

		void createCommandBuffer();
	};
} // namespace HTN
