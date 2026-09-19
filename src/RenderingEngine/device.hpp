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
#include <algorithm>
#include <limits>

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
		VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
		VkDevice getDevice() const { return device; }
		VkQueue getGraphicsQueue() const { return graphicsQueue; }
		VkQueue getPresentQueue() const { return presentQueue; }

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
		VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
		bool hasStencilComponent(VkFormat format);

		VkFormat findDepthFormat();

		void createSwapchain();
		void recreateSwapchain();

	private:
		Window& window;

		// instance + surface + validation 
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;

		// devices
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		VkDevice device = VK_NULL_HANDLE;
		VkQueue graphicsQueue = VK_NULL_HANDLE;
		VkQueue presentQueue = VK_NULL_HANDLE;
		VkCommandPool commandPool;
		VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

		// swapchains
		VkSwapchainKHR swapchain;
		std::vector<VkImage> swapchainImages;
		VkFormat swapchainImageFormat;
		VkExtent2D swapchainExtent;
		std::vector<VkImageView> swapchainImageViews;

		// imagges
		VkImage depthImage;
		VkDeviceMemory depthImageMemory;
		VkImageView depthImageView;

		VkImage colorImage;
		VkDeviceMemory colorImageMemory;
		VkImageView colorImageView;

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

		void createCommandPool();

		VkSurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
		VkPresentModeKHR chooseSwapchainPresentMode(const std::vector<VkPresentModeKHR>& presentModes);
		VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities);

		void createImageViews();
		void createDepthResources();
		void createColorResources();
	};

} // namespace HTN
