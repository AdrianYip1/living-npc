#include "descriptor.hpp"
#include "device.hpp"
#include <cassert>

void HTN::Descriptor::createDescriptorSetLayout(Device& device, std::vector<VkDescriptorType> descriptorTypes,
												std::vector<VkShaderStageFlags> shaderFlags,
												std::vector<u32> bindingIndices,
												VkDescriptorSetLayout& descriptorLayout) {
	assert(descriptorTypes.size() == shaderFlags.size() && descriptorTypes.size() == bindingIndices.size());

	u32 count = static_cast<u32>(descriptorTypes.size());
	std::vector<VkDescriptorSetLayoutBinding> bindings(count);
	for (size_t i = 0; i < count; i++) {
		bindings[i].binding = bindingIndices[i];
		bindings[i].descriptorType = descriptorTypes[i];
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = shaderFlags[i];
		bindings[i].pImmutableSamplers = nullptr;
	}

	VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = count;
	layoutCreateInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device.getDevice(), &layoutCreateInfo, nullptr, &descriptorLayout) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create descriptor set layout");
	}
}

void HTN::Descriptor::createDescriptorPool(Device& device, std::vector<VkDescriptorType> descriptorTypes,
										   VkDescriptorPool& descriptorPool, u32 setGroups) {
	u32 count = static_cast<u32>(descriptorTypes.size());
	std::vector<VkDescriptorPoolSize> poolSizes(count);
	for (size_t i = 0; i < count; i++) {
		poolSizes[i].type = descriptorTypes[i];
		poolSizes[i].descriptorCount = static_cast<u32>(MAX_FRAMES_IN_FLIGHT) * (setGroups + 1);
	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = count;
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<u32>(MAX_FRAMES_IN_FLIGHT) * (setGroups + 1);

	if (vkCreateDescriptorPool(device.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create descriptor pool");
	}
}

void HTN::Descriptor::createDescriptorSets(Device& device, const VkDescriptorSetLayout& setLayout,
										   const VkDescriptorPool& descriptorPool,
										   const std::vector<u32>& poolBindings,
										   const std::vector<std::vector<VkBuffer>>& buffers,
										   const std::vector<VkDescriptorImageInfo>& images,
										   const std::vector<VkDescriptorType> descriptorType,
										   std::vector<VkDescriptorSet>& descriptorSets) {
	assert(poolBindings.size() == buffers.size() && poolBindings.size() == images.size()
		&& poolBindings.size() == descriptorType.size());

	std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, setLayout);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = static_cast<u32>(MAX_FRAMES_IN_FLIGHT);
	allocInfo.pSetLayouts = layouts.data();

	descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	if (vkAllocateDescriptorSets(device.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to allocate descriptor sets");
	}

	size_t count = poolBindings.size();

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		std::vector<VkWriteDescriptorSet> descriptorWrites(count);
		std::vector<VkDescriptorBufferInfo> bufferInfo(count);

		for (size_t d = 0; d < count; d++) {
			descriptorWrites[d].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[d].dstSet = descriptorSets[i];
			descriptorWrites[d].dstBinding = poolBindings[d];
			descriptorWrites[d].dstArrayElement = 0;
			descriptorWrites[d].descriptorType = descriptorType[d];
			descriptorWrites[d].descriptorCount = 1;

			if (descriptorType[d] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
				descriptorWrites[d].pImageInfo = &images[d];
				descriptorWrites[d].pBufferInfo = nullptr;
			} else {
				bufferInfo[d].buffer = buffers[d][i];
				bufferInfo[d].offset = 0;
				bufferInfo[d].range = VK_WHOLE_SIZE;
				descriptorWrites[d].pBufferInfo = &bufferInfo[d];
				descriptorWrites[d].pImageInfo = nullptr;
			}
		}

		vkUpdateDescriptorSets(device.getDevice(), static_cast<u32>(count), descriptorWrites.data(), 0, nullptr);
	}
}
