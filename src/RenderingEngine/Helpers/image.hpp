#pragma once
#include <vulkan/vulkan.h>

#include "../../defines.hpp"
#include "buffers.hpp"

// std
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>
#include <array>

// Image helper class for abstracting actions like creating 
// VkImages, imageviews, copying to images
namespace HTN {
	class Image {
	public:
		Image();
		~Image();
		Image(const Image& other) = delete;
		Image& operator=(const Image& other) = delete;

		static void createImage(Device& device, u32 width, u32 height, u32 mipLevels, VkSampleCountFlagBits numSamples,
								VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
								VkMemoryPropertyFlags properties, VkImage& image,
								VkDeviceMemory& imageMemory);

		static void transitionImageLayout(Device& device, VkImage image, VkFormat format, VkImageLayout oldLayout,
										  VkImageLayout newLayout, u32 mipLevels);

		static void copyBufferToImage(Device& device, VkBuffer _buffer, VkImage image, u32 width, u32 height);

		static void createImageView(VkDevice device, VkImage image, VkFormat format,
									VkImageAspectFlags aspectFlags, VkImageView& imageView,
									u32 mipLevels);

	private:

	};
} // namespace HTN
