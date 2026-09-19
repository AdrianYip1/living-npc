#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "pipeline.hpp"
#include "drawing.hpp"
#include "descriptor.hpp"
#include "Model/model.hpp"
#include "Model/uniform.hpp"
#include "Model/texture.hpp"
#include "Platform/window.hpp"

namespace HTN {
	class Renderer {
	public:
		Renderer(Window& _window);
		~Renderer();
		Renderer(const Renderer&) = delete;
		Renderer& operator=(const Renderer&) = delete;

		void drawFrame();
		void wait();

	private:
		Window& window;
		Device device;
		Pipeline pipeline;
		Drawing drawing;
		Uniform uniform;
		Model model;

		VkDescriptorPool descriptorPool;
		std::vector<VkDescriptorSet> descriptorSets;

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		void createSyncObjects();
		void createDescriptors();
	};
} // namespace HTN
