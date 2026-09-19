#include "device.hpp"
#include "Helpers/image.hpp"

// Debug callback for validation
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	std::cerr << "Validation: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

// Debug messenger for validation
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

// Debug destroy for validation
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

HTN::Device::Device(Window& _window) : window(_window) {
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createCommandPool();
	createSwapchain();
	createColorResources();
	createDepthResources();
}

HTN::Device::~Device() {
	vkDestroyImageView(device, depthImageView, nullptr);
	vkDestroyImage(device, depthImage, nullptr);
	vkFreeMemory(device, depthImageMemory, nullptr);

	vkDestroyImageView(device, colorImageView, nullptr);
	vkDestroyImage(device, colorImage, nullptr);
	vkFreeMemory(device, colorImageMemory, nullptr);

	for (auto imageView : swapchainImageViews) {
		vkDestroyImageView(device, imageView, nullptr);
	}

	vkDestroySwapchainKHR(device, swapchain, nullptr);
	vkDestroyCommandPool(getDevice(), commandPool, nullptr);
	vkDestroyDevice(device, nullptr);

	if (HTN_DEBUG) {
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}

	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
}

void HTN::Device::createInstance() {
	if (HTN_DEBUG && !checkValidationLayerSupport()) {
		throw std::runtime_error("ERROR: Validation layers not available but requested");
	}

	constexpr u32 HTN_VERSION = VK_MAKE_VERSION(0, 0, 1);

	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Living NPC";
	appInfo.applicationVersion = HTN_VERSION;
	appInfo.pEngineName = "Living NPC Vulkan Engine";
	appInfo.engineVersion = HTN_VERSION;
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	auto extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<u32>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	if (HTN_DEBUG) {
		createInfo.enabledLayerCount = static_cast<u32>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		populateDebugMessengerCreateInfo(debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = nullptr;
	}

	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create instance");
	}
}

bool HTN::Device::checkValidationLayerSupport() {
	u32 layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperty : availableLayers) {
			if (strcmp(layerName, layerProperty.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) return false;
	}

	return true;
}

std::vector<const char*> HTN::Device::getRequiredExtensions() {
	u32 glfwExtensionCount = 0;
	const char** glfwExtensions;

	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (HTN_DEBUG) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

void HTN::Device::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
}

void HTN::Device::setupDebugMessenger() {
	if (!HTN_DEBUG) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo;
	populateDebugMessengerCreateInfo(createInfo);

	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to setup debug messenger");
	}
}

void HTN::Device::createSurface() {
	if (glfwCreateWindowSurface(instance, window.getWindow(), nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create window surface");
	}
}

void HTN::Device::pickPhysicalDevice() {
	u32 deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("ERROR: Failed to find GPUs with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	for (const auto& dev : devices) {
		if (!isDeviceSuitable(dev)) continue;
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(dev, &props);
		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			physicalDevice = dev;
			break;
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			physicalDevice = dev;
		}
	}

	if (physicalDevice == VK_NULL_HANDLE) {
		throw std::runtime_error("ERROR: Failed to find suitable GPU");
	}
	msaaSamples = getMaxUsableSampleCount();
}

bool HTN::Device::isDeviceSuitable(VkPhysicalDevice device) {
	QueueFamilyIndices indices = findQueueFamilies(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);

	bool swapChainAdequate = false;
	if (extensionsSupported) {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

	return indices.isComplete() &&
		extensionsSupported &&
		swapChainAdequate &&
		supportedFeatures.samplerAnisotropy &&
		supportedFeatures.sampleRateShading;
}

bool HTN::Device::checkDeviceExtensionSupport(VkPhysicalDevice device) {
	u32 extensionCount = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

// for msaa
VkSampleCountFlagBits HTN::Device::getMaxUsableSampleCount() {
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

	VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
								physicalDeviceProperties.limits.framebufferDepthSampleCounts;

	if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

	return VK_SAMPLE_COUNT_1_BIT;
}

void HTN::Device::createLogicalDevice() {
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<u32> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(),
										  indices.presentFamily.value() };
	f32 queuePriority = 1.0f;

	for (u32 queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.sampleRateShading = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = static_cast<u32>(queueCreateInfos.size());
	createInfo.pEnabledFeatures = &deviceFeatures;

	createInfo.enabledExtensionCount = static_cast<u32>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create logical device");
	}

	vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
	vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

HTN::QueueFamilyIndices HTN::Device::findQueueFamilies(VkPhysicalDevice device) {
	QueueFamilyIndices indices;

	u32 queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT &&
			queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
			indices.graphicsAndComputeFamily = i;
		}
		VkBool32 presentSupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (presentSupport) {
			indices.presentFamily = i;
		}

		i++;

		if (indices.isComplete()) {
			break;
		}
	}

	return indices;
}

HTN::SwapChainSupportDetails HTN::Device::querySwapChainSupport(VkPhysicalDevice device) {
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	u32 formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	u32 presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

void HTN::Device::createCommandPool() {
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(getPhysicalDevice());

	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

	if (vkCreateCommandPool(getDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create command pool");
	}
}

VkSurfaceFormatKHR HTN::Device::chooseSwapchainSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
	for (const auto& format : formats) {
		if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
			format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return format;
		}
	}
	return formats[0];
}

VkPresentModeKHR HTN::Device::chooseSwapchainPresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
	for (const auto& mode : presentModes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return mode;
		}
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D HTN::Device::chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != std::numeric_limits<u32>::max()) {
		return capabilities.currentExtent;
	}
	else {
		int w, h;
		glfwGetFramebufferSize(window.getWindow(), &w, &h);

		VkExtent2D actualExtent = { static_cast<u32>(w), static_cast<u32>(h) };

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

void HTN::Device::createSwapchain() {
	SwapChainSupportDetails swapchainSupport = querySwapChainSupport(physicalDevice);

	VkSurfaceFormatKHR surfaceFormat = chooseSwapchainSurfaceFormat(swapchainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapchainPresentMode(swapchainSupport.presentModes);
	VkExtent2D extent = chooseSwapchainExtent(swapchainSupport.capabilities);

	u32 imageCount = swapchainSupport.capabilities.minImageCount + 1;

	if (swapchainSupport.capabilities.maxImageCount > 0 &&
		swapchainSupport.capabilities.maxImageCount < imageCount) {
		imageCount = swapchainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	u32 queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsAndComputeFamily != indices.presentFamily) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create swapchain");
	}

	vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
	swapchainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

	swapchainImageFormat = surfaceFormat.format;
	swapchainExtent = extent;

	swapchainImageViews.resize(swapchainImages.size());
	createImageViews();
}

void HTN::Device::createImageViews() {
	swapchainImageViews.resize(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++) {
		Image::createImageView(device, swapchainImages[i], swapchainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT,
			swapchainImageViews[i], 1);
	}
}

void HTN::Device::recreateSwapchain() {
	int width = 0, height = 0;
	glfwGetFramebufferSize(window.getWindow(), &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window.getWindow(), &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(device);

	createSwapchain();
	createImageViews();
	createColorResources();
	createDepthResources();
}

void HTN::Device::createDepthResources() {
	VkFormat format = findDepthFormat();

	Image::createImage(*this, swapchainExtent.width, swapchainExtent.height, 1, msaaSamples,
					   format, VK_IMAGE_TILING_OPTIMAL,
					   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
					   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					   depthImage, depthImageMemory);
	Image::createImageView(device, depthImage, format, VK_IMAGE_ASPECT_DEPTH_BIT, depthImageView, 1);
	Image::transitionImageLayout(*this, depthImage, format, VK_IMAGE_LAYOUT_UNDEFINED,
								 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

void HTN::Device::createColorResources() {
	VkFormat format = swapchainImageFormat;

	Image::createImage(*this, swapchainExtent.width, swapchainExtent.height, 1, msaaSamples,
					   format, VK_IMAGE_TILING_OPTIMAL,
					   VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
					   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					   colorImage, colorImageMemory);
	Image::createImageView(device, colorImage, format, VK_IMAGE_ASPECT_COLOR_BIT, colorImageView, 1);
}

VkFormat HTN::Device::findDepthFormat() {
	return findSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);
}

VkFormat HTN::Device::findSupportedFormat(const std::vector<VkFormat>& candidates,
										  VkImageTiling tiling, VkFormatFeatureFlags features) {
	for (VkFormat format : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		}
	}
	throw std::runtime_error("ERROR: Failed to find supported format");
}

bool HTN::Device::hasStencilComponent(VkFormat format) {
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
