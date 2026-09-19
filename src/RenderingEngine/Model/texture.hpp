#pragma once
#include <vulkan/vulkan.h>

#include "../../defines.hpp"
#include "../device.hpp"
#include "../Helpers/buffers.hpp"
#include "../Helpers/image.hpp"

#include <string>

namespace HTN {
	struct DecodedImage {
		unsigned char* pixels = nullptr;
		int width = 0, height = 0;

		DecodedImage() = default;
		DecodedImage(DecodedImage&& o) noexcept
			: pixels(o.pixels), width(o.width), height(o.height) { o.pixels = nullptr; }
		DecodedImage& operator=(DecodedImage&& o) noexcept;
		~DecodedImage();
		DecodedImage(const DecodedImage&) = delete;
		DecodedImage& operator=(const DecodedImage&) = delete;

		static DecodedImage fromFile(const std::string& filepath);
		explicit operator bool() const { return pixels != nullptr; }
	};

	class Texture {
	public:
		Texture(Device& _device, const std::string& filepath);
		Texture(Device& _device, DecodedImage&& decoded);
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
		void uploadTextureImage(DecodedImage&& decoded);
		void createTextureImageView();
		void createTextureSampler();
	};
} // namespace HTN
