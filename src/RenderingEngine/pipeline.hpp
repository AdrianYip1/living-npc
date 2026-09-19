#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "descriptor.hpp"

#include <fstream>
#include <array>
#include <string>

namespace HTN {
	class Pipeline {
	public:
		Pipeline(Device& _device, const std::string& vertPath, const std::string& fragPath);
		~Pipeline();
		Pipeline(const Pipeline&) = delete;
		Pipeline& operator=(const Pipeline&) = delete;

		VkPipelineLayout getPipelineLayout() { return pipelineLayout; }
		VkPipeline getGraphicsPipeline() { return graphicsPipeline; }
		VkRenderPass getRenderpass() { return renderpass; }
		VkDescriptorSetLayout& getUboSetLayout() { return uboSetLayout; }

	private:
		Device& device;
		std::string vertPath;
		std::string fragPath;

		VkPipelineLayout pipelineLayout;
		VkRenderPass renderpass;
		VkPipeline graphicsPipeline;
		VkDescriptorSetLayout uboSetLayout;

		void createRenderPass();
		void createGraphicsPipeline();

		static std::vector<char> readFile(const std::string& filename);
		VkShaderModule createShaderModule(const std::vector<char>& shaderCode);
	};
} // namespace HTN
