#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "Platform/window.hpp"

#include <vector>

namespace HTN {

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	class Device {
	public:
		Device(Window& window);
		~Device();
		Device(const Device&) = delete;
		Device& operator=(const Device&) = delete;

		VkInstance getInstance() const { return instance; }

	private:
		Window& window;

		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

		void createInstance();
	};

} // namespace HTN
