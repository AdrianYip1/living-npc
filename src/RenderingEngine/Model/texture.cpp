#include "texture.hpp"
#include "stb_image.h"

#include <cmath>
#include <algorithm>

HTN::DecodedImage& HTN::DecodedImage::operator=(DecodedImage&& o) noexcept {
	if (pixels) stbi_image_free(pixels);
	pixels = o.pixels; width = o.width; height = o.height;
	o.pixels = nullptr;
	return *this;
}

HTN::DecodedImage::~DecodedImage() {
	if (pixels) stbi_image_free(pixels);
}

HTN::DecodedImage HTN::DecodedImage::fromFile(const std::string& filepath) {
	DecodedImage img;
	int channels;
	img.pixels = stbi_load(filepath.c_str(), &img.width, &img.height, &channels, STBI_rgb_alpha);
	return img;
}

HTN::Texture::Texture(Device& _device, const std::string& filepath) : device(_device) {
	createTextureImage(filepath);
	createTextureImageView();
	createTextureSampler();
}

HTN::Texture::Texture(Device& _device, DecodedImage&& decoded) : device(_device) {
	uploadTextureImage(std::move(decoded));
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
	DecodedImage decoded = DecodedImage::fromFile(filepath);
	if (!decoded)
		throw std::runtime_error("ERROR: Failed to load texture image: " + filepath);
	uploadTextureImage(std::move(decoded));
}

void HTN::Texture::uploadTextureImage(DecodedImage&& decoded) {
	int texWidth = decoded.width;
	int texHeight = decoded.height;
	VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	Buffer::createBuffer(device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device.getDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, decoded.pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(device.getDevice(), stagingBufferMemory);

	decoded = DecodedImage{};

	mipLevels = static_cast<u32>(std::floor(std::log2((std::max)(texWidth, texHeight)))) + 1;

	Image::createImage(device, static_cast<u32>(texWidth), static_cast<u32>(texHeight), mipLevels,
					   VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
					   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

	Image::transitionImageLayout(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB,
								 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);

	Image::copyBufferToImage(device, stagingBuffer, textureImage,
							 static_cast<u32>(texWidth), static_cast<u32>(texHeight));

	Image::generateMipmaps(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB,
						   static_cast<u32>(texWidth), static_cast<u32>(texHeight), mipLevels);

	vkDestroyBuffer(device.getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device.getDevice(), stagingBufferMemory, nullptr);
}

void HTN::Texture::createTextureImageView() {
	Image::createImageView(device.getDevice(), textureImage, VK_FORMAT_R8G8B8A8_SRGB,
						   VK_IMAGE_ASPECT_COLOR_BIT, textureImageView, mipLevels);
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
	samplerInfo.maxLod = static_cast<float>(mipLevels);

	if (vkCreateSampler(device.getDevice(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create texture sampler");
	}
}
