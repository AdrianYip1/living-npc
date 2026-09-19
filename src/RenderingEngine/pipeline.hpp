#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"

#include <fstream>
#include <array>
#include <string>

namespace HTN {
	class Pipeline {
	public:

		// TODO: will probably need mutiple pipeline ctors for different things like rendering
		// the scene, hdri (sky?), npcs, etc
		Pipeline(Device& _device, const std::string& vertPath, const std::string& fragPath);
		~Pipeline();
		Pipeline(const Pipeline&) = delete;
		Pipeline& operator=(const Pipeline&) = delete;

		// getters
		VkPipelineLayout getPipelineLayout() { return pipelineLayout; }
		VkPipeline getGraphicsPipeline() { return graphicsPipeline; }
		VkRenderPass getRenderpass() { return renderpass; }

	private:
		Device& device;
		std::string vertPath;
		std::string fragPath;

		VkPipelineLayout pipelineLayout;
		VkRenderPass renderpass;
		VkPipeline graphicsPipeline;

		void createRenderPass();
		void createGraphicsPipeline();

		static std::vector<char> readFile(const std::string& filename);
		VkShaderModule createShaderModule(const std::vector<char>& shaderCode);
	};
} // namespace HTN
