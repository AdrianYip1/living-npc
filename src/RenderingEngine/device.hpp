#pragma once
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../defines.hpp"
#include "Platform/window.hpp"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>

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

		// Getter functions
		VkInstance getInstance() const { return instance; }

	private:
		Window& window;

		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;

		void createInstance();
		bool checkValidationLayerSupport();
		std::vector<const char*> getRequiredExtensions();
		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
		void setupDebugMessenger();
		void createSurface();
	};

} // namespace HTN
