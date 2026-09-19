#pragma once
#include <vulkan/vulkan.h>

#include "../../defines.hpp"
#include "../device.hpp"
#include "../Helpers/buffers.hpp"
#include "../Helpers/image.hpp"

#include <string>

namespace HTN {
	class Texture {
	public:
		Texture(Device& _device, const std::string& filepath);
		~Texture();
		Texture(const Texture&) = delete;
		Texture& operator=(const Texture&) = delete;

		VkImageView getImageView() { return textureImageView; }
		VkSampler getSampler() { return textureSampler; }

	private:
		Device& device;

		VkImage textureImage = VK_NULL_HANDLE;
		VkDeviceMemory textureImageMemory = VK_NULL_HANDLE;
		VkImageView textureImageView = VK_NULL_HANDLE;
		VkSampler textureSampler = VK_NULL_HANDLE;
		u32 mipLevels = 1;

		void createTextureImage(const std::string& filepath);
		void createTextureImageView();
		void createTextureSampler();
	};
} // namespace HTN
