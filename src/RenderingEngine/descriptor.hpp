#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"

#include <vector>

namespace HTN {
	class Device;
	class Descriptor {
	public:
		Descriptor() = default;
		~Descriptor() = default;

		static void createDescriptorSetLayout(Device& device, std::vector<VkDescriptorType> descriptorTypes,
											  std::vector<VkShaderStageFlags> shaderFlags,
											  std::vector<u32> bindingIndices,
											  VkDescriptorSetLayout& descriptorLayout);

		static void createDescriptorPool(Device& device, std::vector<VkDescriptorType> descriptorTypes,
										 VkDescriptorPool& descriptorPool);

		static void createDescriptorSets(Device& device, const VkDescriptorSetLayout& setLayout,
										 const VkDescriptorPool& descriptorPool,
										 const std::vector<std::vector<VkBuffer>>& buffers,
										 const std::vector<VkDescriptorType> descriptorType,
										 std::vector<VkDescriptorSet>& descriptorSets);
	};
} // namespace HTN
