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
#include <set>
#include <optional>
#include <string>

namespace HTN {

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	struct QueueFamilyIndices {
		std::optional<u32> graphicsAndComputeFamily;
		std::optional<u32> presentFamily;

		bool isComplete() {
			return graphicsAndComputeFamily.has_value() &&
				presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities{};
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	class Device {
	public:
		Device(Window& window);
		~Device();
		Device(const Device&) = delete;
		Device& operator=(const Device&) = delete;

		// Getter functions
		VkInstance getInstance() const { return instance; }
		VkSurfaceKHR getSurface() const { return surface; }

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

	private:
		Window& window;

		// vk objects
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		VkDevice device = VK_NULL_HANDLE;
		VkQueue graphicsQueue = VK_NULL_HANDLE;
		VkQueue presentQueue = VK_NULL_HANDLE;
		VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

		void createInstance();
		bool checkValidationLayerSupport();
		std::vector<const char*> getRequiredExtensions();
		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
		void setupDebugMessenger();
		void createSurface();
		void pickPhysicalDevice();
		bool isDeviceSuitable(VkPhysicalDevice device);
		bool checkDeviceExtensionSupport(VkPhysicalDevice device);
		VkSampleCountFlagBits getMaxUsableSampleCount();
		void createLogicalDevice();
	};

} // namespace HTN
