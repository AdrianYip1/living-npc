#pragma once
#include <vulkan/vulkan.h>

#include "../../defines.hpp"

// std
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>
#include <array>

// Buffer helper class that abstracts away stuff like buffer copying, command buffer creation
namespace HTN {
	class Device;
	class Buffer {
	public:
		Buffer();
		~Buffer();
		Buffer(const Buffer& other) = delete;
		Buffer& operator=(const Buffer& other) = delete;

		static void createBuffer(Device& device, VkDeviceSize size, VkBufferUsageFlags usage,
			VkMemoryPropertyFlags properties, VkBuffer& buffer,
			VkDeviceMemory& bufferMemory);

		static void copyBuffer(Device& device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

		static void copyBufferToImage(Device& device, VkBuffer buffer, VkImage image, u32 w, u32 h);

		static u32 findMemoryType(Device& device, u32 typeFilter, VkMemoryPropertyFlags properties);

		static VkCommandBuffer beginSingleTimeCommands(Device& device);

		static void endSingleTimeCommands(Device& device, VkCommandBuffer commandBuffer);

	private:

	};
} // namespace HTN
