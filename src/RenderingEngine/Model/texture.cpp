#include "texture.hpp"
#include "stb_image.h"

HTN::Texture::Texture(Device& _device, const std::string& filepath) : device(_device) {
	createTextureImage(filepath);
	createTextureImageView();
	createTextureSampler();
}

HTN::Texture::~Texture() {
	vkDestroySampler(device.getDevice(), textureSampler, nullptr);
	vkDestroyImageView(device.getDevice(), textureImageView, nullptr);
	vkDestroyImage(device.getDevice(), textureImage, nullptr);
	vkFreeMemory(device.getDevice(), textureImageMemory, nullptr);
}

void HTN::Texture::createTextureImage(const std::string& filepath) {
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

	if (!pixels) {
		throw std::runtime_error("ERROR: Failed to load texture image: " + filepath);
	}

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	Buffer::createBuffer(device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device.getDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(device.getDevice(), stagingBufferMemory);

	stbi_image_free(pixels);

	Image::createImage(device, static_cast<u32>(texWidth), static_cast<u32>(texHeight), 1,
					   VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
					   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

	Image::transitionImageLayout(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB,
								 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);

	Image::copyBufferToImage(device, stagingBuffer, textureImage,
							 static_cast<u32>(texWidth), static_cast<u32>(texHeight));

	Image::transitionImageLayout(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB,
								 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

	vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
}

void HTN::Texture::createTextureImageView() {
	Image::createImageView(device.getDevice(), textureImage, VK_FORMAT_R8G8B8A8_SRGB,
						   VK_IMAGE_ASPECT_COLOR_BIT, textureImageView, 1);
}

void HTN::Texture::createTextureSampler() {
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &properties);

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(device.getDevice(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create texture sampler");
	}
}
